"""The daily production loop (shadow / paper / live share the same code path).

Per run, after the close of day T:
  1. load data through T, rebuild features with the FROZEN feature state
     (normaliser, HMM, latest GBM model) of the deployed model
  2. warm the policy up by replaying the last `warmup` days (recurrent state,
     and a simulated track for live-vs-backtest tracking error)
  3. build the observation at T from the broker's ACTUAL positions
  4. policy -> no-trade band -> risk overlay (with persisted kill-switch state)
  5. orders to the broker (ShadowBroker sends nothing)
  6. log the decision; fill yesterday's reward with today's realised return
  7. drift monitoring + daily report
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from rltrade.agents.sb3_agents import Ensemble, SB3Policy
from rltrade.config import cfg_get
from rltrade.data.synthetic import Market
from rltrade.envs.portfolio import EnvData
from rltrade.eval.rollout import rollout
from rltrade.features.library import Normalizer, build_features
from rltrade.features.regime import add_regime_features
from rltrade.features.signal import _xs_rank
from rltrade.live.broker import Broker
from rltrade.live.decision_log import DecisionLog
from rltrade.live.monitor import feature_drift, tracking_error
from rltrade.pipeline import make_env

log = logging.getLogger(__name__)


def load_deployment(model_dir: Path) -> tuple[Ensemble, dict]:
    model_dir = Path(model_dir)
    with open(model_dir / "feature_state.pkl", "rb") as f:
        fstate = pickle.load(f)
    seeds = sorted(model_dir.glob("seed_*/model.zip"))
    vals = []
    for p in seeds:
        meta = json.loads((p.parent / "meta.json").read_text())
        vals.append((meta.get("val_sharpe", -np.inf), p.parent))
    vals.sort(key=lambda x: x[0], reverse=True)
    k = fstate.get("top_k", 3)
    return Ensemble([SB3Policy.load(p) for _, p in vals[:k]]), fstate


def live_envdata(market: Market, fstate: dict, cfg: dict) -> EnvData:
    fs = build_features(market)
    model = fstate.get("gbm_model")
    if model is not None:
        T, N, F = fs.asset.shape
        ok = np.isfinite(fs.asset).all(axis=(1, 2))
        pred = np.full((T, N), np.nan)
        pred[ok] = model.predict(fs.asset[ok].reshape(-1, F)).reshape(-1, N)
        mu, sd = np.nanmean(pred, 1, keepdims=True), np.nanstd(pred, 1, keepdims=True) + 1e-12
        fs.add_asset_feature("gbm_score", (pred - mu) / sd)
        fs.add_asset_feature("gbm_rank", _xs_rank(pred))
        fs.add_asset_feature("gbm_conf", np.zeros((T, N)))
    fold = fstate["fold"]
    if "p_turbulent" in fstate["glob_names"]:
        add_regime_features(fs, market.close, slice(*fold.train), seed=cfg_get(cfg, "seed", 0))
    norm = Normalizer.from_state(fstate["normalizer"])
    return EnvData.build(market, norm.transform(fs), fs.first_valid(), fstate["feature_names"])


def run_daily(cfg: dict, model_dir: Path, market: Market, broker: Broker, state_dir: Path,
              warmup: int = 126) -> dict:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    policy, fstate = load_deployment(model_dir)
    data = live_envdata(market, fstate, cfg)
    T = data.n_rows
    today = pd.Timestamp(data.dates[-1])
    dlog = DecisionLog(state_dir / "decisions.jsonl")

    # 2. warm-up replay on the last `warmup` days (decision rows up to T-3)
    env = make_env(data, (max(data.first_row, T - 3 - warmup), T), cfg, train=False)
    sim = rollout(env, policy)
    # feed T-2 so the recurrent state is current, then decide at T-1
    prices_close = {s: float(data.close[-1, i]) for i, s in enumerate(data.assets)}
    pos = broker.positions()
    equity = broker.equity(prices_close)
    a = env.assets
    w_live = np.array([pos.get(data.assets[i], 0.0) * data.close[-1, i] / equity for i in a])

    st_path = state_dir / "state.json"
    st = json.loads(st_path.read_text()) if st_path.exists() else {"peak": equity, "start": equity, "since": 0,
                                                                     "overlay": {"peak": 1.0, "killed_for": 0}}
    st["peak"] = max(st["peak"], equity)
    env.w_close = w_live
    env.nav_close, env.peak, env.nav_start = equity / st["start"], st["peak"] / st["start"], 1.0
    env.since_trade = st["since"]
    env.t = T - 2
    env._frames.append(env._obs_single(T - 2))
    policy(env._obs(), env)
    env.t = T - 1
    env._frames.append(env._obs_single(T - 1))
    obs = env._obs()
    target, band = policy(obs, env)

    # 4. overlay (persisted kill-switch state) then band: same code path as the backtest env
    target = np.asarray(target)
    flags = {}
    if env.overlay is not None:
        ov = env.overlay
        ov._peak, ov._killed_for = st["overlay"]["peak"], st["overlay"]["killed_for"]
        ov.update_nav(equity / st["start"])
    target, flags = env.execute_target(target, w_live, band, T - 1, update_nav=False)
    if env.overlay is not None:
        st["overlay"] = {"peak": env.overlay._peak, "killed_for": env.overlay._killed_for}
    target_w = {data.assets[i]: float(w) for i, w in zip(a, target) if w > 1e-6}

    # 5. orders
    orders = broker.orders_for(target_w, prices_close)
    fills = broker.submit(orders, prices_close)
    st["since"] = 0 if orders else st["since"] + 1

    # 6. decision log: fill yesterday's reward with the realised equity return
    prev = dlog.records()
    if prev and prev[-1].get("reward") is None:
        r = equity / prev[-1]["equity"] - 1.0
        dlog.fill_last_reward(float(np.log1p(r)) * 100, {"ret": r, "date": str(today.date())})
    dlog.append({"date": str(today.date()), "obs": obs, "action": target, "target": target_w, "band": band,
                 "overlay": flags, "equity": equity, "orders": orders, "n_fills": len(fills), "reward": None,
                 "ts": datetime.utcnow().isoformat()})
    st_path.write_text(json.dumps(st))

    # 7. monitoring
    fold = fstate["fold"]
    tr_rows = slice(*fold.train)
    drift = feature_drift(data.feats[tr_rows], data.feats[-63:], data.feat_names)
    live_r = np.array([r["realised"]["ret"] for r in dlog.records() if r.get("realised")])
    te = tracking_error(live_r, sim["r_net"].to_numpy()[-len(live_r):] if len(live_r) else np.array([]))
    alerts = [d["feature"] for d in drift if d["alert"]]
    report = _daily_report(today, equity, target_w, orders, flags, alerts, te, state_dir)
    return {"date": str(today.date()), "target": target_w, "orders": orders, "overlay": flags,
            "drift_alerts": alerts, "tracking_error": te, "report": str(report)}


def _daily_report(day, equity, target, orders, flags, alerts, te, state_dir: Path) -> Path:
    d = state_dir / "reports"
    d.mkdir(exist_ok=True)
    lines = [f"# Daily report {day.date()}", "", f"- Equity: {equity:,.2f}",
             f"- Overlay interventions: {flags or 'none'}",
             f"- Drift alerts (PSI above its empirical null): {alerts or 'none'}", f"- Live vs backtest tracking: {te}", "",
             "## Target weights", "", *[f"- {s}: {w:.3f}" for s, w in sorted(target.items(), key=lambda x: -x[1])],
             "", "## Orders", "", *([f"- {s}: {q:+d}" for s, q in orders.items()] or ["- none"])]
    p = d / f"{day.date()}.md"
    p.write_text("\n".join(lines))
    return p
