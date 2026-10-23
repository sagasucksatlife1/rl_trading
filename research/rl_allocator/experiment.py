"""Experiment orchestration: one fold, or full walk-forward with stitched OOS.

The pass bar: the RL ensemble must beat the GBM signal with a naive sizing rule,
net of costs, out of sample, across seeds. If it doesn't, the honest conclusion
is "use the simple rule", which is a valid result.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from research.rl_allocator.agents.sb3_agents import Ensemble, SB3Policy, train_seed
from rltrade.baselines.rules import baseline_suite
from rltrade.config import cfg_get
from rltrade.eval.dsr import deflated_sharpe
from rltrade.eval.metrics import sharpe, summarize
from rltrade.eval.rollout import equal_weight_bench, rollout
from rltrade.eval.splits import Fold
from rltrade.eval.stress import behaviour_audit, cost_stress, regime_breakdown
from rltrade.eval.tearsheet import plot_behaviour, plot_equity, write_report
from rltrade.pipeline import fold_envdata, make_env
from rltrade.tracking import Tracker, load_trials

log = logging.getLogger(__name__)
PRIMARY_BASELINE = "gbm_linear_monthly"


def run_fold(cfg: dict, market, fs, fold: Fold, out_dir: Path, tracker: Tracker | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data, norm = fold_envdata(market, fs, fold.train, cfg)
    with open(out_dir / "feature_state.pkl", "wb") as f:
        models = fs.extra.get("gbm_models") or []
        pickle.dump({"normalizer": norm.state_dict(), "gbm_model": models[-1][1] if models else None,
                     "feature_names": data.feat_names, "glob_names": data.glob_names, "fold": fold,
                     "assets": data.assets, "top_k": cfg_get(cfg, "agent.ensemble_top_k", 3)}, f)

    mk_train = lambda s: make_env(data, fold.train, cfg, train=True, seed=s)
    mk_val = lambda: make_env(data, fold.val, cfg, train=False)
    mk_test = lambda: make_env(data, fold.test, cfg, train=False)

    seeds = cfg_get(cfg, "agent.seeds", [0, 1, 2, 3, 4])
    results = []
    for s in seeds:
        r = train_seed(mk_train, mk_val, cfg, s, out_dir / f"seed_{s}")
        results.append(r)
        log.info("seed %d best val Sharpe %.3f at step %d", s, r["val_sharpe"], r["step"])
    results.sort(key=lambda r: r["val_sharpe"], reverse=True)
    top_k = cfg_get(cfg, "agent.ensemble_top_k", 3)
    members = [SB3Policy.load(r["path"]) for r in results[:top_k]]
    ens = Ensemble(members)

    # validation of the ensemble = the number that goes in the trial log
    val_rec = rollout(mk_val(), ens)
    if tracker:
        tracker.log_trial(sharpe(val_rec["r_net"].to_numpy()),
                          float(val_rec["r_net"].mean() / (val_rec["r_net"].std() + 1e-12)),
                          {"fold_test": fold.test})

    records = {"rl_ensemble": rollout(mk_test(), ens)}
    seed_test = {}
    for r in results:
        rec = rollout(mk_test(), SB3Policy.load(r["path"]))
        seed_test[r["seed"]] = rec
    for name, pol in baseline_suite(cfg_get(cfg, "baselines.band", 0.02)).items():
        records[name] = rollout(mk_test(), pol)
    stress = {m: rollout(mk_test(), ens, options={"cost_mult": m}) for m in (2.0, 3.0)}
    stress_base = {m: rollout(mk_test(), baseline_suite()[PRIMARY_BASELINE], options={"cost_mult": m})
                   for m in (2.0, 3.0)}
    test_env = mk_test()
    test_env.reset()
    return {"records": records, "seed_test": seed_test, "stress": stress, "stress_base": stress_base,
            "regimes": regime_breakdown(records["rl_ensemble"], test_env), "bench": equal_weight_bench(test_env),
            "assets": [data.assets[i] for i in test_env.fixed_assets], "seed_val": {r["seed"]: r["val_sharpe"] for r in results}}


def _stitch(parts: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(parts).sort_index()


def report(cfg: dict, fold_outs: list[dict], out_dir: Path, tracker: Tracker | None = None) -> dict:
    names = list(fold_outs[0]["records"])
    stitched = {n: _stitch([fo["records"][n] for fo in fold_outs]) for n in names}
    bench = np.concatenate([fo["bench"][: len(fo["records"]["rl_ensemble"])] for fo in fold_outs])
    summary = pd.DataFrame({n: summarize(rec, bench) for n, rec in stitched.items()}).T
    cols = ["sharpe", "cagr", "ann_vol", "max_dd", "calmar", "sortino", "turnover_ann", "cost_drag_ann",
            "avg_gross", "beta_to_bench", "info_ratio"]
    summary = summary[[c for c in cols if c in summary.columns]]

    seeds = sorted(fold_outs[0]["seed_test"])
    seed_sh = {s: sharpe(_stitch([fo["seed_test"][s] for fo in fold_outs])["r_net"].to_numpy()) for s in seeds}
    seed_arr = np.array(list(seed_sh.values()))

    trials = [t for t in load_trials() if t.get("data_hash") == (tracker.data_hash if tracker else None)]
    n_trials = max(len({t["cfg_hash"] for t in trials}), 1)
    trial_pp = np.array([t["val_sharpe_pp"] for t in trials]) if len(trials) > 1 else None
    rl = stitched["rl_ensemble"]["r_net"].to_numpy()
    dsr = deflated_sharpe(rl, n_trials, trial_pp)

    stress_rows = []
    for m in (1.0, 2.0, 3.0):
        r_rl = rl if m == 1.0 else _stitch([fo["stress"][m] for fo in fold_outs])["r_net"].to_numpy()
        r_b = stitched[PRIMARY_BASELINE]["r_net"].to_numpy() if m == 1.0 else \
            _stitch([fo["stress_base"][m] for fo in fold_outs])["r_net"].to_numpy()
        stress_rows.append({"cost_mult": m, "rl_sharpe": sharpe(r_rl), f"{PRIMARY_BASELINE}_sharpe": sharpe(r_b)})
    stress = pd.DataFrame(stress_rows).set_index("cost_mult")
    regimes = pd.concat([fo["regimes"] for fo in fold_outs if len(fo["regimes"])]) if any(
        len(fo["regimes"]) for fo in fold_outs) else pd.DataFrame()
    if len(regimes):
        regimes = regimes.groupby(level=0).mean()
    audit = behaviour_audit(stitched["rl_ensemble"], bench)
    audit["seed_sharpe_median"] = float(np.median(seed_arr))
    audit["seed_sharpe_worst"] = float(seed_arr.min())
    audit["seed_sharpe_best"] = float(seed_arr.max())

    base_sh = summary.loc[PRIMARY_BASELINE, "sharpe"]
    rl_sh = summary.loc["rl_ensemble", "sharpe"]
    best_base = summary.drop(index="rl_ensemble")["sharpe"].max()
    passed = rl_sh > base_sh and np.median(seed_arr) > base_sh and stress.loc[2.0, "rl_sharpe"] > 0
    verdict = (f"PASS - RL ensemble Sharpe {rl_sh:.2f} beats the GBM naive-sizing baseline ({base_sh:.2f}), "
               f"median seed {np.median(seed_arr):.2f}, survives 2x costs." if passed else
               f"FAIL - RL ensemble Sharpe {rl_sh:.2f} vs GBM naive-sizing baseline {base_sh:.2f} "
               f"(median seed {np.median(seed_arr):.2f}, best baseline {best_base:.2f}). "
               "Honest conclusion: use the simple rule.")
    plot_equity(stitched, out_dir / "equity.png")
    plot_behaviour(stitched["rl_ensemble"], fold_outs[0]["assets"], out_dir / "behaviour.png")
    write_report(out_dir, summary, dsr, stress, regimes, audit, verdict,
                 {"folds": len(fold_outs), "seeds": seeds, "algo": cfg_get(cfg, "agent.algo"),
                  "reward": cfg_get(cfg, "reward.kind"), "data": cfg_get(cfg, "data.source")})
    for n, rec in stitched.items():
        rec.drop(columns=["weights", "overlay"]).to_csv(out_dir / f"oos_{n}.csv")
    metrics = {"summary": summary.to_dict(orient="index"), "dsr": dsr, "stress": stress.to_dict(orient="index"),
               "audit": audit, "seed_test_sharpe": seed_sh, "passed": bool(passed), "verdict": verdict}
    if tracker:
        tracker.log_metrics(metrics, "oos")
    return metrics
