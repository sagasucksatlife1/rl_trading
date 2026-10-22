"""Optuna hyperparameter search. Objective = VALIDATION Sharpe of a single fold
(never test). Every trial is logged to runs/trials.jsonl, so the deflated Sharpe
of the final model is corrected for the full search.

    python scripts/hpo.py --config configs/quick.yaml --trials 20
"""
from _common import parse

from rltrade.agents.sb3_agents import train_seed
from rltrade.config import cfg_get, set_dotted
from rltrade.eval.splits import chrono_split
from rltrade.pipeline import build_dataset, fold_envdata, make_env
from rltrade.tracking import Tracker


def main():
    args, cfg = parse(__doc__, extra=lambda ap: ap.add_argument("--trials", type=int, default=20))
    import copy

    import optuna

    market, fs = build_dataset(cfg)
    fold = chrono_split(fs.first_valid(), len(market.dates))
    data, _ = fold_envdata(market, fs, fold.train, cfg)
    algo = cfg_get(cfg, "agent.algo")

    def objective(trial: optuna.Trial) -> float:
        c = copy.deepcopy(cfg)
        set_dotted(c, f"agent.{algo}.learning_rate", trial.suggest_float("lr", 1e-5, 1e-3, log=True))
        set_dotted(c, f"agent.{algo}.gamma", trial.suggest_float("gamma", 0.9, 0.999))
        if "ppo" in algo:
            set_dotted(c, f"agent.{algo}.ent_coef", trial.suggest_float("ent", 1e-4, 0.05, log=True))
            set_dotted(c, f"agent.{algo}.clip_range", trial.suggest_float("clip", 0.1, 0.3))
        set_dotted(c, "env.temperature", trial.suggest_float("temperature", 1.0, 6.0))
        set_dotted(c, "reward.kind", trial.suggest_categorical("reward", ["log", "dsr", "risk"]))
        set_dotted(c, "agent.seeds", [0])
        tr = Tracker(c, name=f"hpo{trial.number}", data_hash=market.data_hash())
        r = train_seed(lambda s: make_env(data, fold.train, c, train=True, seed=s),
                       lambda: make_env(data, fold.val, c, train=False), c, 0, tr.dir / "seed_0")
        tr.log_trial(r["val_sharpe"], r.get("val_sharpe_pp", 0.0), {"hpo": trial.params})
        tr.close()
        return r["val_sharpe"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.trials)
    print("best val Sharpe", study.best_value, "params", study.best_params)
    print("NOTE: the test block was never touched. Retrain with these params via train.py.")


if __name__ == "__main__":
    main()
