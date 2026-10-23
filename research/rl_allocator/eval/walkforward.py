"""The real experiment grid: seeds x walk-forward folds, run in parallel, every run in MLflow.

    grid = run_grid(cfg, variant="dsr_learned_band", workers=4)

For each fold (expanding window, purge + embargo between train / val / test):
  * features, HMM, normaliser are fitted on that fold's TRAIN rows only
  * every seed trains with early stopping on the fold's VALIDATION block
  * every seed is evaluated on the TEST block at 1x / 2x / 3x costs
  * the ensemble = average weights of the top-k seeds BY VALIDATION Sharpe
  * all baselines run on the same block, same env, same costs
Test blocks tile, so stitching them gives one continuous out-of-sample record.

MLflow (local SQLite store, runs/mlflow.db; `mlflow ui --backend-store-uri sqlite:///runs/mlflow.db`): one run per (fold, seed) with the full
flattened config, seed, fold, data hash, config hash and metric set, tagged
kind=seed_run; one kind=summary run per grid with the report. The deflated Sharpe
counts EVERY config ever logged on the same data hash, not just the winner.
"""
from __future__ import annotations

import concurrent.futures as cf
import copy
import json
import logging
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

from rltrade.config import ROOT, cfg_get, config_hash
from rltrade.eval.metrics import sharpe, summarize
from rltrade.eval.rollout import equal_weight_bench, rollout
from rltrade.eval.splits import chrono_split, walk_forward
from rltrade.features.regime_he import regime_labels
from rltrade.pipeline import build_dataset, fold_envdata, make_env

log = logging.getLogger(__name__)
COST_MULTS = (1.0, 2.0, 3.0)


def mlflow_setup(cfg: dict):
    import mlflow

    d = ROOT / cfg_get(cfg, "tracking.dir", "runs")
    d.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{d / 'mlflow.db'}")  # MLflow 3: file store is retired
    mlflow.set_experiment(cfg_get(cfg, "tracking.experiment", "project24"))
    return mlflow


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = str(v)[:240]
    return out


def make_folds(cfg: dict, first: int, n: int):
    sp = cfg_get(cfg, "split", {}) or {}
    if sp.get("mode", "walk_forward") == "single":
        return [chrono_split(first, n, sp.get("val_frac", 0.15), sp.get("test_frac", 0.2), sp.get("purge", 21),
                             sp.get("embargo", 5))]
    folds = walk_forward(first, n, sp.get("min_train", 756), sp.get("val_len", 252), sp.get("test_len", 252),
                         sp.get("purge", 21), sp.get("embargo", 5), sp.get("rolling"))
    max_f = sp.get("max_folds")
    return folds[-max_f:] if max_f else folds


# ----------------------------------------------------------------------------- worker
def _seed_job(job: dict) -> dict:
    """Train one (fold, seed), evaluate it on test at all cost levels, log to MLflow."""
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    from research.rl_allocator.agents.sb3_agents import SB3Policy, train_seed

    cfg, data, fold, seed, out = job["cfg"], job["data"], job["fold"], job["seed"], Path(job["out"])
    t0 = time.time()
    res = train_seed(lambda s: make_env(data, fold.train, cfg, train=True, seed=s),
                     lambda: make_env(data, fold.val, cfg, train=False), cfg, seed, out)
    pol = SB3Policy.load(out)
    recs = {m: rollout(make_env(data, fold.test, cfg, train=False), pol, options={"cost_mult": m})
            for m in COST_MULTS}
    with open(out / "test_recs.pkl", "wb") as f:
        pickle.dump(recs, f)
    s1 = summarize(recs[1.0])
    metrics = {"val_sharpe": res["val_sharpe"], "best_step": res["step"], "train_minutes": (time.time() - t0) / 60,
               **{f"test_{k}": v for k, v in s1.items() if isinstance(v, (int, float))},
               **{f"test_sharpe_{int(m)}x": sharpe(recs[m]["r_net"].to_numpy()) for m in COST_MULTS}}
    try:
        mlflow = mlflow_setup(cfg)
        with mlflow.start_run(run_name=f"{job['variant']}_f{job['fold_idx']}_s{seed}"):
            mlflow.set_tags({"kind": "seed_run", "variant": job["variant"], "data_hash": job["data_hash"],
                             "cfg_hash": job["cfg_hash"], "grid_id": job["grid_id"]})
            mlflow.log_params({**_flat(cfg), "seed": seed, "fold": job["fold_idx"], "data_hash": job["data_hash"],
                               "fold.train": str(fold.train), "fold.val": str(fold.val), "fold.test": str(fold.test)})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
            for h in res.get("history", []):
                mlflow.log_metric("val_sharpe_curve", h["val_sharpe"], step=int(h["step"]))
    except Exception as e:  # noqa: BLE001 - tracking must never kill a training job
        log.warning("mlflow logging failed: %s", e)
    return {"fold_idx": job["fold_idx"], "seed": seed, "path": str(out), **metrics}


# ----------------------------------------------------------------------------- grid
def run_grid(cfg: dict, variant: str, workers: int = 4, out_root: Path | None = None) -> dict:
    from research.rl_allocator.agents.sb3_agents import Ensemble, SB3Policy
    from rltrade.baselines.rules import baseline_suite
    from research.rl_allocator.eval.report import build_report

    market, fs = build_dataset(cfg)
    data_hash = market.data_hash()
    folds = make_folds(cfg, fs.first_valid(), len(market.dates))
    seeds = list(cfg_get(cfg, "agent.seeds", [0, 1, 2, 3, 4]))
    grid_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{variant}_{config_hash(cfg)}"
    out_root = Path(out_root or ROOT / cfg_get(cfg, "tracking.dir", "runs") / "grids") / grid_id
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(json.dumps({"cfg": cfg, "variant": variant, "data_hash": data_hash,
                                                      "data_meta": market.meta, "folds": [repr(f) for f in folds]},
                                                     indent=2, default=str))
    log.info("grid %s: %d folds x %d seeds = %d runs, %d workers, data %s", grid_id, len(folds), len(seeds),
             len(folds) * len(seeds), workers, data_hash)
    fold_data = [fold_envdata(market, fs, f.train, cfg)[0] for f in folds]
    jobs = [{"cfg": cfg, "data": fold_data[i], "fold": f, "fold_idx": i, "seed": s, "variant": variant,
             "out": str(out_root / f"fold_{i}" / f"seed_{s}"), "data_hash": data_hash,
             "cfg_hash": config_hash(cfg), "grid_id": grid_id}
            for i, f in enumerate(folds) for s in seeds]
    results = []
    ctx = mp.get_context("fork")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_seed_job, j): (j["fold_idx"], j["seed"]) for j in jobs}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            log.info("[%d/%d] fold %d seed %d: val %.2f | test 1x %.2f 2x %.2f | %.1f min", len(results), len(jobs),
                     r["fold_idx"], r["seed"], r["val_sharpe"], r["test_sharpe_1x"], r["test_sharpe_2x"],
                     r["train_minutes"])
    (out_root / "seed_runs.json").write_text(json.dumps(results, indent=2, default=float))

    # per-fold evaluation: ensemble + baselines at every cost level, regimes
    top_k = cfg_get(cfg, "agent.ensemble_top_k", 3)
    fold_outs = []
    for i, f in enumerate(folds):
        data = fold_data[i]
        rs = sorted([r for r in results if r["fold_idx"] == i], key=lambda r: r["val_sharpe"], reverse=True)
        ens = Ensemble([SB3Policy.load(r["path"]) for r in rs[:top_k]])
        mk = lambda: make_env(data, f.test, cfg, train=False)
        recs = {"rl_ensemble": {m: rollout(mk(), ens, options={"cost_mult": m}) for m in COST_MULTS}}
        for name, pol in baseline_suite(cfg_get(cfg, "baselines.band", 0.02)).items():
            recs[name] = {m: rollout(mk(), pol, options={"cost_mult": m}) for m in COST_MULTS}
        seed_recs = {}
        for r in rs:
            with open(Path(r["path"]) / "test_recs.pkl", "rb") as fh:
                seed_recs[r["seed"]] = pickle.load(fh)
        env = mk()
        env.reset()
        turb = regime_labels(market.close, market.tradable, f.train, cfg_get(cfg, "eval.regime_q", 0.6))
        fold_outs.append({"fold": f, "recs": recs, "seed_recs": seed_recs, "bench": equal_weight_bench(env),
                          "turbulent": turb[env.t0: env.t0 + len(recs["rl_ensemble"][1.0])],
                          "assets": [data.assets[j] for j in env.fixed_assets]})
    with open(out_root / "fold_outs.pkl", "wb") as fh:
        pickle.dump(fold_outs, fh)
    report = build_report(cfg, variant, fold_outs, out_root, data_hash, grid_id, market.meta)
    return {"grid_id": grid_id, "dir": str(out_root), "report": report, "seed_runs": results}


def quick_cfg(cfg: dict, **over) -> dict:
    c = copy.deepcopy(cfg)
    for k, v in over.items():
        node = c
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return c
