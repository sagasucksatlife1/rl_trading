"""The shipped strategy: GBM cross-sectional score -> linear long-only weights, with a
no-trade band, rebalanced every `every` trading days, under the hard risk overlay.

Why this and not the RL allocator: in the walk-forward dry run (Ticket 4) this rule
beat the RL allocator's median seed at 2x costs (0.82 vs 0.54 Sharpe) at the same
turnover (~21x/yr), and the RL allocator's best result had a deflated Sharpe of 0.004
after 9 trials. The RL code is archived under research/rl_allocator/.

The schedule is row-based (trade on rows where t % every == 0), so the rule is
stateless: the backtest and the live daily loop make identical decisions.

    fstate = fit(cfg, out_dir)                    # final GBM + frozen feature state for live use
    policy = GBMBandStrategy(**cfg["strategy"])   # callable(obs, env) -> (weights | None, band)
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

from rltrade.baselines.rules import SignalLinear
from rltrade.config import cfg_get
from rltrade.features.signal import _make_model, _xs_rank, forward_labels
from rltrade.pipeline import build_dataset, fold_envdata

log = logging.getLogger(__name__)

DEFAULTS = {"every": 5, "band": 0.02, "top_frac": 0.5}


class GBMBandStrategy(SignalLinear):
    """SignalLinear with a stateless, row-based rebalance schedule."""

    def __init__(self, every: int = 5, band: float = 0.02, top_frac: float = 0.5):
        super().__init__(every=every, band=band, top_frac=top_frac)

    def reset(self):
        pass

    def __call__(self, obs, env):
        if env.t % self.every != 0:
            return None, 0.0
        return self.rebalance_now(obs, env)

    def rebalance_now(self, obs, env):
        """Target weights regardless of the schedule (e.g. a new, all-cash deployment)."""
        self.i = 0  # make the parent's periodic check trade
        return super().__call__(obs, env)


def strategy_params(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg_get(cfg, "strategy", {}) or {})}


def fit(cfg: dict, out_dir: Path) -> dict:
    """Fit the deployable state on ALL available history (labels purged at the end):
    the final GBM, the regime HMM and the feature normaliser. Writes feature_state.pkl
    and config.json into out_dir; the live loop (rltrade/live/shadow.py) reads them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    market, fs = build_dataset(cfg)
    T, N, F = fs.asset.shape
    sc = cfg_get(cfg, "signal", {}) or {}
    h, emb = sc.get("horizon", 21), sc.get("embargo", 5)
    lab = forward_labels(market.open, h)
    tr = fs.extra.get("tradable")
    if tr is not None:
        lab = np.where(tr, lab, np.nan)
    y = _xs_rank(lab)
    # the GBM consumes the raw asset features, i.e. everything before the signal columns
    n_raw = fs.asset_names.index("gbm_score")
    first = fs.first_valid()
    rows = np.arange(first, T - 1 - h - emb)
    X = fs.asset[rows, :, :n_raw].reshape(-1, n_raw)
    yy = y[rows].reshape(-1)
    ok = np.isfinite(X).all(1) & np.isfinite(yy)
    model = _make_model(sc.get("params") or {})
    model.fit(X[ok], yy[ok])
    train_rows = (first, T)
    data, norm = fold_envdata(market, fs, train_rows, cfg)
    fstate = {"normalizer": norm.state_dict(), "gbm_model": model, "feature_names": data.feat_names,
              "glob_names": data.glob_names, "train_rows": train_rows, "assets": data.assets,
              "n_raw_features": n_raw, "strategy": strategy_params(cfg), "data_hash": market.data_hash(),
              "data_meta": market.meta, "gbm_oos_ic": fs.extra.get("gbm_oos_ic")}
    with open(out_dir / "feature_state.pkl", "wb") as f:
        pickle.dump(fstate, f)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    log.info("strategy fitted on %d rows x %d assets, %d labelled samples; data %s", len(rows), N, int(ok.sum()),
             fstate["data_hash"])
    return fstate
