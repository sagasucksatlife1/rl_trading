"""Re-evaluate a finished grid's ensemble with a different combination mode, no retraining.

    python research/rl_allocator/scripts/reensemble.py runs/grids/<grid_id> --mode median_banded

Rebuilds every fold's ensemble from the saved seeds (top-k by validation, as before),
reruns it at 1x/2x/3x costs on the test blocks, and rewrites the grid report.
"""
import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")

from research.rl_allocator.agents.sb3_agents import Ensemble, SB3Policy  # noqa: E402
from rltrade.config import cfg_get  # noqa: E402
from research.rl_allocator.eval.report import build_report  # noqa: E402
from rltrade.eval.rollout import rollout  # noqa: E402
from research.rl_allocator.eval.walkforward import COST_MULTS, make_folds  # noqa: E402
from rltrade.pipeline import build_dataset, fold_envdata, make_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--mode", default="median_banded")
    a = ap.parse_args()
    g = Path(a.grid)
    meta = json.loads((g / "config.json").read_text())
    cfg = meta["cfg"]
    with open(g / "fold_outs.pkl", "rb") as f:
        fold_outs = pickle.load(f)
    runs = json.loads((g / "seed_runs.json").read_text())
    market, fs = build_dataset(cfg)
    folds = make_folds(cfg, fs.first_valid(), len(market.dates))
    top_k = cfg_get(cfg, "agent.ensemble_top_k", 3)
    for i, f in enumerate(folds):
        data, _ = fold_envdata(market, fs, f.train, cfg)
        rs = sorted([r for r in runs if r["fold_idx"] == i], key=lambda r: r["val_sharpe"], reverse=True)
        ens = Ensemble([SB3Policy.load(r["path"]) for r in rs[:top_k]], mode=a.mode)
        fold_outs[i]["recs"]["rl_ensemble"] = {
            m: rollout(make_env(data, f.test, cfg, train=False), ens, options={"cost_mult": m}) for m in COST_MULTS}
        print(f"fold {i} done")
    with open(g / f"fold_outs_{a.mode}.pkl", "wb") as fh:
        pickle.dump(fold_outs, fh)
    r = build_report(cfg, meta["variant"], fold_outs, g, meta["data_hash"], g.name, meta.get("data_meta"))
    print(r["verdict"])


if __name__ == "__main__":
    main()
