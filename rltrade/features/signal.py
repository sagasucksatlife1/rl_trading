"""Walk-forward gradient-boosted cross-sectional signal (the forecast layer).

Supervised model, retrained every `refit` days on an expanding window. The label
for decision row t is the forward open-to-open return from t+1 to t+1+h (the
agent trades at the open of t+1), ranked cross-sectionally. A label at row t is
only known after open t+1+h, so training rows are purged: when refitting at row
d we use rows t <= d - 1 - h - embargo.

Adds three per-asset features to the FeatureSet:
  gbm_score - cross-sectional z-score of the prediction
  gbm_rank  - cross-sectional rank in [-0.5, 0.5]
  gbm_conf  - trailing realised rank IC of the model (lagged so it is causal)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rltrade.features.library import FeatureSet

log = logging.getLogger(__name__)


def _make_model(params: dict):
    try:
        import lightgbm as lgb

        p = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=200,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        p.update(params)
        return lgb.LGBMRegressor(**p)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=200)


def _xs_rank(a: np.ndarray) -> np.ndarray:
    """Cross-sectional rank per row in [-0.5, 0.5]; NaN-safe."""
    return pd.DataFrame(a).rank(axis=1, pct=True).sub(0.5).to_numpy()


def forward_labels(open_: pd.DataFrame, horizon: int) -> np.ndarray:
    o = open_.to_numpy()
    lab = np.full_like(o, np.nan)
    # label[t] = log(open[t+1+h] / open[t+1])
    lab[: len(o) - 1 - horizon] = np.log(o[1 + horizon:] / o[1: len(o) - horizon])
    return lab


def walk_forward_signal(fs: FeatureSet, open_: pd.DataFrame, horizon: int = 21, refit: int = 63,
                        min_train: int = 756, embargo: int = 5, params: dict | None = None,
                        conf_window: int = 126) -> dict:
    T, N, F = fs.asset.shape
    X_all = fs.asset  # raw features (trees are scale-free)
    y_rank = _xs_rank(forward_labels(open_, horizon))
    start = fs.first_valid()
    pred = np.full((T, N), np.nan)
    models = []
    for d in range(start + min_train, T, refit):
        last_train = d - 1 - horizon - embargo
        rows = np.arange(start, last_train + 1)
        X = X_all[rows].reshape(-1, F)
        y = y_rank[rows].reshape(-1)
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        model = _make_model(params or {})
        model.fit(X[ok], y[ok])
        blk = slice(d, min(d + refit, T))
        Xb = X_all[blk].reshape(-1, F)
        pred[blk] = model.predict(np.nan_to_num(Xb)).reshape(-1, N)
        models.append((d, model))
    mu = np.nanmean(pred, axis=1, keepdims=True)
    sd = np.nanstd(pred, axis=1, keepdims=True) + 1e-12
    score = (pred - mu) / sd
    rank = _xs_rank(pred)
    # Realised IC for row t is known at t+1+h; lag by h+1 before rolling.
    ic = pd.Series([_spearman(pred[t], y_rank[t]) for t in range(T)], index=fs.dates)
    conf = ic.shift(horizon + 1).rolling(conf_window, min_periods=21).mean().to_numpy()
    conf = np.where(np.isfinite(pred).any(1), np.nan_to_num(conf), np.nan)
    fs.add_asset_feature("gbm_score", score)
    fs.add_asset_feature("gbm_rank", rank)
    fs.add_asset_feature("gbm_conf", np.repeat(conf[:, None], N, axis=1))
    oos_ic = ic[np.isfinite(pred).any(1)].mean()
    log.info("GBM signal: %d refits, mean OOS rank IC = %.4f", len(models), oos_ic)
    fs.extra["gbm_models"] = models
    fs.extra["gbm_oos_ic"] = float(oos_ic)
    return {"pred": pred, "score": score, "ic": ic, "models": models}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])
