"""Offline RL from the decision log (CQL/IQL via d3rlpy; behaviour cloning fallback).

    python scripts/offline_run.py --log live_state/decisions.jsonl --algo cql
    python scripts/offline_run.py --from-backtest     # dev: logs from a noisy baseline
"""
import numpy as np
from _common import ROOT, parse

from rltrade.baselines.rules import SignalLinear
from rltrade.eval.metrics import sharpe
from rltrade.eval.rollout import rollout
from rltrade.eval.splits import chrono_split
from rltrade.live.decision_log import DecisionLog
from rltrade.live.offline import bc_fit, collect_transitions, train_offline
from rltrade.pipeline import build_dataset, fold_envdata, make_env


def main():
    def extra(ap):
        ap.add_argument("--log", default="live_state/decisions.jsonl")
        ap.add_argument("--algo", default="cql")
        ap.add_argument("--from-backtest", action="store_true")
        ap.add_argument("--steps", type=int, default=20000)

    args, cfg = parse(__doc__, default_config="configs/quick.yaml", extra=extra)
    market, fs = build_dataset(cfg)
    fold = chrono_split(fs.first_valid(), len(market.dates))
    data, _ = fold_envdata(market, fs, fold.train, cfg)
    if args.from_backtest:
        env = make_env(data, fold.train, cfg, train=True, seed=0)
        ds = collect_transitions(env, SignalLinear(every=5), episodes=30, noise=0.2)
    else:
        ds = DecisionLog(ROOT / args.log).transitions()
        if not ds:
            raise SystemExit("decision log has < 2 filled records; run shadow_run.py first")
    print(f"dataset: {len(ds['rewards'])} transitions")
    try:
        model = train_offline(ds, args.algo, args.steps)
        policy = lambda obs, env: (model.predict(obs[None])[0], None)
        name = args.algo
    except ImportError:
        print("d3rlpy not installed -> behaviour-cloning fallback")
        policy, loss = bc_fit(ds)
        name = "bc"
    rec = rollout(make_env(data, fold.val, cfg, train=False), policy)
    base = rollout(make_env(data, fold.val, cfg, train=False), SignalLinear(every=5))
    print(f"[{name}] val Sharpe {sharpe(rec['r_net'].to_numpy()):.2f} vs behaviour policy "
          f"{sharpe(base['r_net'].to_numpy()):.2f}")


if __name__ == "__main__":
    main()
