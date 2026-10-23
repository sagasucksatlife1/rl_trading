"""Walk-forward out-of-sample backtest of the shipped strategy vs the alternatives.

Same folds, timing, cost model and risk overlay as the RL research. Costs at 1x/2x/3x,
calm vs turbulent split from the Hurst/entropy classifier (thresholds fitted per fold
on training rows). Writes results/strategy_backtest/report.md.

    python scripts/backtest_strategy.py --config configs/strategy.yaml
"""
import json

import numpy as np
import pandas as pd
from _common import ROOT, parse

from rltrade.baselines.rules import BuyAndHold, EqualWeight, MeanVariance, SignalLinear
from rltrade.config import cfg_get
from rltrade.eval.dsr import deflated_sharpe
from rltrade.eval.metrics import sharpe, summarize
from rltrade.eval.rollout import rollout
from rltrade.eval.splits import walk_forward
from rltrade.eval.tearsheet import plot_equity
from rltrade.features.regime_he import regime_labels
from rltrade.pipeline import build_dataset, fold_envdata, make_env
from rltrade.strategy import GBMBandStrategy, strategy_params


def main():
    args, cfg = parse(__doc__, default_config="configs/strategy.yaml",
                      extra=lambda ap: ap.add_argument("--out", default="results/strategy_backtest"))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    market, fs = build_dataset(cfg)
    sp = cfg_get(cfg, "split", {})
    folds = walk_forward(fs.first_valid(), len(market.dates), sp.get("min_train", 756), sp.get("val_len", 252),
                         sp.get("test_len", 252), sp.get("purge", 21), sp.get("embargo", 5))
    rules = {"SHIPPED gbm_band": lambda: GBMBandStrategy(**strategy_params(cfg)),
             "gbm_linear_monthly": lambda: SignalLinear(every=21),
             "gbm_mean_variance": lambda: MeanVariance(every=21),
             "equal_weight_monthly": lambda: EqualWeight(every=21),
             "buy_and_hold": lambda: BuyAndHold()}
    recs = {n: {m: [] for m in (1.0, 2.0, 3.0)} for n in rules}
    turb = []
    for f in folds:
        data, _ = fold_envdata(market, fs, f.train, cfg)
        for n, mk in rules.items():
            for m in (1.0, 2.0, 3.0):
                recs[n][m].append(rollout(make_env(data, f.test, cfg, train=False), mk(), options={"cost_mult": m}))
        env = make_env(data, f.test, cfg, train=False)
        env.reset()
        lab = regime_labels(market.close, market.tradable, f.train)
        turb.append(lab[env.t0: env.t0 + len(recs["equal_weight_monthly"][1.0][-1])])
    st = {n: {m: pd.concat(v).sort_index() for m, v in d.items()} for n, d in recs.items()}
    turb = np.concatenate(turb)
    summary = pd.DataFrame({n: summarize(st[n][1.0]) for n in rules}).T[
        ["sharpe", "cagr", "ann_vol", "max_dd", "turnover_ann", "cost_drag_ann", "avg_gross"]]
    stress = pd.DataFrame({n: {f"{int(m)}x": sharpe(st[n][m]["r_net"].to_numpy()) for m in (1.0, 2.0, 3.0)}
                           for n in rules}).T
    reg = {}
    for n in rules:
        r = st[n][2.0]["r_net"].to_numpy()
        k = min(len(r), len(turb))
        reg[n] = {"calm_2x": sharpe(r[:k][~turb[:k]]), "turbulent_2x": sharpe(r[:k][turb[:k]])}
    regimes = pd.DataFrame(reg).T
    # the shipped rule was chosen among these 5 candidates: deflate for that choice
    pp = np.array([st[n][1.0]["r_net"].mean() / st[n][1.0]["r_net"].std() for n in rules])
    dsr = deflated_sharpe(st["SHIPPED gbm_band"][1.0]["r_net"].to_numpy(), len(rules), pp)
    plot_equity({n: st[n][1.0] for n in rules}, out / "equity.png", "Shipped strategy vs alternatives, OOS 1x")
    plot_equity({n: st[n][2.0] for n in rules}, out / "equity_2x.png", "Shipped strategy vs alternatives, OOS 2x")
    meta = market.meta or {}
    real = meta.get("pit") is True and not str(meta.get("source", "")).startswith("synthetic")
    tbl = lambda d: "```\n" + d.round(3).to_string() + "\n```"
    md = [f"# Shipped strategy: walk-forward backtest", "",
          *([] if real else ["> DRY RUN: synthetic data. Rebuild on a vendor snapshot for real numbers.", ""]),
          f"Data `{market.data_hash()}` ({meta.get('source')}, point-in-time {meta.get('pit')}) | "
          f"{len(folds)} folds | strategy {strategy_params(cfg)}", "",
          "## Out of sample, 1x costs", "", tbl(summary), "", "## Cost stress (Sharpe)", "", tbl(stress), "",
          "## Regimes (Hurst/entropy, 2x costs, Sharpe)", "", tbl(regimes), "",
          "## Deflated Sharpe (chosen among the 5 rules above)", "",
          f"- PSR(SR>0) {dsr['psr_0']:.3f} | DSR {dsr['dsr']:.3f} | expected max SR under null "
          f"{dsr['sr0_annual']:.2f}", "", "![equity](equity.png)", "", "![equity 2x](equity_2x.png)"]
    (out / "report.md").write_text("\n".join(md))
    (out / "report.json").write_text(json.dumps({"summary": summary.to_dict(orient="index"),
                                                 "stress": stress.to_dict(orient="index"),
                                                 "regimes": regimes.to_dict(orient="index"), "dsr": dsr},
                                                indent=2, default=float))
    print("\n".join(md[:-3]))


if __name__ == "__main__":
    main()
