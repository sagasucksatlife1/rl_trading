"""Run the full walk-forward grid (seeds x folds) with MLflow logging and the OOS report.

    python research/rl_allocator/scripts/walkforward.py --config research/rl_allocator/configs/full.yaml --variant dsr_learned_band --workers 4
    python research/rl_allocator/scripts/walkforward.py --config research/rl_allocator/configs/full.yaml agent.total_steps=600000 agent.seeds=[0,1,2,3,4]
"""
from _common import parse

from research.rl_allocator.eval.walkforward import run_grid


def main():
    def extra(ap):
        ap.add_argument("--variant", default="main")
        ap.add_argument("--workers", type=int, default=4)

    args, cfg = parse(__doc__, default_config="research/rl_allocator/configs/full.yaml", extra=extra)
    g = run_grid(cfg, args.variant, args.workers)
    print("\n" + g["report"]["verdict"])
    print("report:", g["dir"] + "/report.md")


if __name__ == "__main__":
    main()
