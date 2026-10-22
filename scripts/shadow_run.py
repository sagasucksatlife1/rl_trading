"""Daily production loop. Run after the close (cron / scheduled container).

    python scripts/shadow_run.py --model runs/<run>/fold_0 --broker shadow
    python scripts/shadow_run.py --model runs/<run>/fold_0 --broker paper
    python scripts/shadow_run.py --model runs/<run>/fold_0 --broker alpaca   # ALPACA_* env vars

`--replay-days N` replays the loop over the last N days of history (useful to
test the loop end to end, and to generate a decision log for offline RL).
"""
import json

from _common import ROOT, parse

from rltrade.config import cfg_get
from rltrade.data.loader import load_market
from rltrade.live.broker import AlpacaBroker, PaperBroker, ShadowBroker
from rltrade.live.shadow import run_daily
from rltrade.pipeline import make_cost


def main():
    def extra(ap):
        ap.add_argument("--model", required=True)
        ap.add_argument("--broker", default="shadow", choices=["shadow", "paper", "alpaca"])
        ap.add_argument("--state", default="live_state")
        ap.add_argument("--replay-days", type=int, default=0)
        ap.add_argument("--capital", type=float, default=1e6)

    args, cfg = parse(__doc__, extra=extra)
    import yaml
    from pathlib import Path

    model_dir = Path(args.model)
    run_cfg = model_dir.parent / "config.yaml"
    if run_cfg.exists():  # use the exact config the model was trained with
        cfg = yaml.safe_load(run_cfg.read_text())
    market = load_market(cfg)
    state = ROOT / args.state
    if args.broker == "paper":
        broker = PaperBroker(state / "paper_account.json", args.capital, make_cost(cfg))
    elif args.broker == "alpaca":
        broker = AlpacaBroker()
    else:
        broker = ShadowBroker(args.capital)
    ends = range(len(market.dates) - args.replay_days, len(market.dates) + 1) if args.replay_days else \
        [len(market.dates)]
    for end in ends:
        out = run_daily(cfg, model_dir, market.subset(end=market.dates[end - 1]), broker, state)
        print(json.dumps({k: out[k] for k in ("date", "orders", "overlay", "drift_alerts")}, default=str))
    print("report:", out["report"])


if __name__ == "__main__":
    main()
