"""End-to-end data pipeline from config to env-ready arrays.

    market, fs = build_dataset(cfg)               # raw features + walk-forward GBM signal
    data = fold_envdata(market, fs, train, cfg)    # regime HMM + normaliser fitted on train rows only
    env = make_env(data, rows, cfg, train=True)
"""
from __future__ import annotations

import copy
import logging

import numpy as np

from rltrade.config import cfg_get
from rltrade.data.loader import load_market
from rltrade.data.synthetic import Market
from rltrade.envs.costs import CostModel
from rltrade.envs.portfolio import EnvData, PortfolioEnv
from rltrade.features.library import FeatureSet, Normalizer, build_features
from rltrade.features.regime import add_regime_features
from rltrade.features.signal import walk_forward_signal
from rltrade.risk.overlay import RiskOverlay

log = logging.getLogger(__name__)


def build_dataset(cfg: dict, market: Market | None = None) -> tuple[Market, FeatureSet]:
    market = market or load_market(cfg)
    fs = build_features(market)
    sc = cfg_get(cfg, "signal", {}) or {}
    if sc.get("enabled", True):
        walk_forward_signal(fs, market.open, horizon=sc.get("horizon", 21), refit=sc.get("refit", 63),
                            min_train=sc.get("min_train", 756), embargo=sc.get("embargo", 5),
                            params=sc.get("params"))
    return market, fs


def fold_envdata(market: Market, fs: FeatureSet, train_rows: tuple[int, int], cfg: dict) -> tuple[EnvData, Normalizer]:
    fs = copy.deepcopy(fs)
    fs.extra.pop("gbm_models", None)
    tr = slice(*train_rows)
    if cfg_get(cfg, "features.regime", True):
        add_regime_features(fs, market.close, tr, seed=cfg_get(cfg, "seed", 0), tradable=market.tradable)
    first = fs.first_valid()
    lo = max(train_rows[0], first)
    norm = Normalizer(cfg_get(cfg, "features.clip", 5.0)).fit(fs, slice(lo, train_rows[1]))
    data = EnvData.build(market, norm.transform(fs), first, cfg_get(cfg, "env.features"))
    return data, norm


def make_cost(cfg: dict) -> CostModel:
    c = cfg_get(cfg, "costs", {}) or {}
    return CostModel(c.get("commission_bps", 1.0), c.get("half_spread_bps", 2.0), c.get("impact_coef", 0.7))


def make_env(data: EnvData, rows: tuple[int, int], cfg: dict, train: bool, seed: int | None = None,
             overlay: bool | None = None) -> PortfolioEnv:
    env_cfg = dict(cfg_get(cfg, "env", {}) or {})
    env_cfg.pop("features", None)
    if "cost_mult_range" in env_cfg:
        env_cfg["cost_mult_range"] = tuple(env_cfg["cost_mult_range"])
    use_overlay = cfg_get(cfg, "risk.in_training", False) if (overlay is None and train) else \
        (cfg_get(cfg, "risk.enabled", True) if overlay is None else overlay)
    ov = RiskOverlay.from_cfg(cfg_get(cfg, "risk", {})) if use_overlay else None
    if ov is not None:
        ov.long_only = env_cfg.get("action_mode", "long_only") == "long_only"
    return PortfolioEnv(data, rows, cfg=env_cfg, cost=make_cost(cfg), reward=cfg_get(cfg, "reward", {}),
                        overlay=ov, train=train, seed=seed)


def eval_assets(data: EnvData, cfg: dict) -> np.ndarray:
    k = cfg_get(cfg, "env.n_assets") or len(data.assets)
    return np.arange(k)
