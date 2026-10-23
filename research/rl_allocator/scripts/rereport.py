"""Rebuild report(s) from saved grid results without retraining (e.g. after a report-code change).

    python research/rl_allocator/scripts/rereport.py runs/grids/<grid_id> [more grid dirs...] [--ab-out runs/ab/<stamp>]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.rl_allocator.eval.report import build_report, compare_variants  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grids", nargs="+")
    ap.add_argument("--ab-out", default=None)
    a = ap.parse_args()
    reports = []
    for g in map(Path, a.grids):
        meta = json.loads((g / "config.json").read_text())
        with open(g / "fold_outs.pkl", "rb") as f:
            fold_outs = pickle.load(f)
        r = build_report(meta["cfg"], meta["variant"], fold_outs, g, meta["data_hash"], g.name, meta.get("data_meta"))
        reports.append(r)
        print(f"{meta['variant']}: {r['verdict']}")
    if a.ab_out:
        print(compare_variants(reports, Path(a.ab_out)))


if __name__ == "__main__":
    main()
