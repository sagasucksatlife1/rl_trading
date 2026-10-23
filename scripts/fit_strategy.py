"""Fit the deployable strategy state (final GBM, regime HMM, normaliser) on all history.

    python scripts/fit_strategy.py --config configs/strategy.yaml --out deploy/current
Then:  python scripts/shadow_run.py --deploy deploy/current --broker shadow
"""
from pathlib import Path

from _common import ROOT, parse

from rltrade.strategy import fit


def main():
    args, cfg = parse(__doc__, default_config="configs/strategy.yaml",
                      extra=lambda ap: ap.add_argument("--out", default="deploy/current"))
    fstate = fit(cfg, ROOT / args.out)
    print(f"deployment written to {ROOT / args.out} (data {fstate['data_hash']}, "
          f"walk-forward GBM OOS IC {fstate['gbm_oos_ic']:.4f}, strategy {fstate['strategy']})")


if __name__ == "__main__":
    main()
