"""Ticket 3: A/B the turnover fixes on the same walk-forward grid and data snapshot.

  baseline   DSR reward, learned band                  (the 76x/yr configuration)
  band       DSR reward, fixed no-trade band b          (tunable: --band)
  penalty    log return - kappa * turnover              (tunable: --kappa)
  cvar       log return - CVaR (Rockafellar-Uryasev)    (+ learned band)
  band+pen   fixed band b AND turnover penalty kappa

Each variant re-runs the full baseline comparison (1/N, GBM-linear, GBM-MV,
GBM+no-trade-band). Output: runs/ab/<stamp>/ab_comparison.md

    python research/rl_allocator/scripts/ab_turnover.py --config research/rl_allocator/configs/full.yaml --workers 4 agent.total_steps=200000
"""
import json
import time

from _common import ROOT, parse

from research.rl_allocator.eval.report import compare_variants
from research.rl_allocator.eval.walkforward import quick_cfg, run_grid

VARIANTS = {
    "baseline_dsr_learned_band": {"reward.kind": "dsr", "env.learned_band": True},
    "band_fixed": {"reward.kind": "dsr", "env.learned_band": False},  # env.band set from --band
    "turnover_penalty": {"reward.kind": "turnover", "env.learned_band": True},  # kappa from --kappa
    "cvar_reward": {"reward.kind": "cvar", "reward.lam": 0.5, "env.learned_band": True},
    "band_plus_penalty": {"reward.kind": "turnover", "env.learned_band": False},  # --band and --kappa
}


def main():
    def extra(ap):
        ap.add_argument("--workers", type=int, default=4)
        ap.add_argument("--band", type=float, default=0.03)
        ap.add_argument("--kappa", type=float, default=0.002)
        ap.add_argument("--only", nargs="*", default=None)

    args, cfg = parse(__doc__, default_config="research/rl_allocator/configs/full.yaml", extra=extra)
    out = ROOT / "runs" / "ab" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, over in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        over = dict(over)
        if name in ("band_fixed", "band_plus_penalty"):
            over["env.band"] = args.band
        if name in ("turnover_penalty", "band_plus_penalty"):
            over["reward.turnover_penalty"] = args.kappa
        c = quick_cfg(cfg, **over)
        g = run_grid(c, name, args.workers)
        reports.append(g["report"])
        (out / f"{name}.json").write_text(json.dumps({"dir": g["dir"], **g["report"]}, indent=2, default=float))
        print(compare_variants(reports, out))
    print("A/B table:", out / "ab_comparison.md")


if __name__ == "__main__":
    main()
