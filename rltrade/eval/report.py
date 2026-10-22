"""Out-of-sample report for a walk-forward grid, the pass gate, and the A/B table.

Pass bar (unchanged, Ticket 4): the RL allocator must beat the GBM + no-trade-band
baseline, net of 2x costs, at the MEDIAN seed, out of sample. Best-seed numbers are
shown for completeness and never used for a decision. If the gate fails, the
verdict is: ship the GBM + no-trade-band rule, archive the RL allocator as research.

Deflated Sharpe: N = number of grid runs ever logged to MLflow on this data hash
(every A/B variant, every rerun, every HPO trial), V[SR] from their validation Sharpes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rltrade.config import cfg_get
from rltrade.eval.dsr import deflated_sharpe
from rltrade.eval.metrics import sharpe, summarize
from rltrade.eval.stress import behaviour_audit
from rltrade.eval.tearsheet import plot_behaviour, plot_equity

GATE_BASELINE = "gbm_linear_band"
GATE_COST = 2.0


def _stitch(parts):
    return pd.concat(parts).sort_index()


def _table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(floatfmt=".3f")
    except ImportError:
        return "```\n" + df.round(3).to_string() + "\n```"


def trial_stats(cfg: dict, data_hash: str) -> dict:
    """Distinct configs + their per-period validation Sharpes, from MLflow."""
    try:
        from rltrade.eval.walkforward import mlflow_setup

        mlflow = mlflow_setup(cfg)
        runs = mlflow.search_runs(filter_string=f"tags.kind = 'seed_run' and tags.data_hash = '{data_hash}'",
                                  search_all_experiments=True)
    except Exception:  # noqa: BLE001
        return {"n_configs": 1, "n_runs": 0, "trial_sr_pp": None}
    if runs.empty:
        return {"n_configs": 1, "n_runs": 0, "trial_sr_pp": None}
    # every grid run is a trial: the same config re-run on changed code (e.g. the band-order
    # fix) is a new look at the data, so group by grid, not by config hash
    key = "tags.grid_id" if "tags.grid_id" in runs else "tags.cfg_hash"
    per_cfg = runs.groupby(key)["metrics.val_sharpe"].mean() / np.sqrt(252)
    return {"n_configs": int(per_cfg.size), "n_runs": int(len(runs)),
            "trial_sr_pp": per_cfg.to_numpy() if per_cfg.size > 1 else None}


def build_report(cfg: dict, variant: str, fold_outs: list[dict], out: Path, data_hash: str, grid_id: str,
                 data_meta: dict | None = None) -> dict:
    names = list(fold_outs[0]["recs"])
    st = {n: {m: _stitch([fo["recs"][n][m] for fo in fold_outs]) for m in fold_outs[0]["recs"][n]} for n in names}
    seeds = sorted(fold_outs[0]["seed_recs"])
    seed_st = {s: {m: _stitch([fo["seed_recs"][s][m] for fo in fold_outs]) for m in (1.0, 2.0, 3.0)} for s in seeds}
    bench = np.concatenate([fo["bench"][: len(fo["recs"]["rl_ensemble"][1.0])] for fo in fold_outs])
    turb = np.concatenate([fo["turbulent"] for fo in fold_outs])

    # 1. summary at 1x
    cols = ["sharpe", "cagr", "ann_vol", "max_dd", "turnover_ann", "cost_drag_ann", "avg_gross", "beta_to_bench"]
    summary = pd.DataFrame({n: summarize(st[n][1.0], bench) for n in names}).T
    summary = summary[[c for c in cols if c in summary.columns]]

    # 2. seeds: median / worst / best, at every cost level
    sd = pd.DataFrame({s: {f"sharpe_{int(m)}x": sharpe(seed_st[s][m]["r_net"].to_numpy()) for m in (1.0, 2.0, 3.0)}
                       | {"turnover_ann": summarize(seed_st[s][1.0])["turnover_ann"]} for s in seeds}).T
    sd.index.name = "seed"
    seed_stats = pd.DataFrame({"median": sd.median(), "worst": sd.min(), "best": sd.max()}).T

    # 3. cost stress: Sharpe at 1x / 2x / 3x for everything
    stress = pd.DataFrame({n: {f"{int(m)}x": sharpe(st[n][m]["r_net"].to_numpy()) for m in st[n]} for n in names}).T
    stress.loc["rl_median_seed"] = [seed_stats.loc["median", f"sharpe_{int(m)}x"] for m in (1.0, 2.0, 3.0)]
    stress.loc["rl_worst_seed"] = [seed_stats.loc["worst", f"sharpe_{int(m)}x"] for m in (1.0, 2.0, 3.0)]

    # 4. regimes (Hurst / entropy classifier, thresholds from each fold's train rows), at 2x
    reg_rows = []
    for n in ("rl_ensemble", GATE_BASELINE, "gbm_linear_monthly", "equal_weight_monthly"):
        r = st[n][GATE_COST]["r_net"].to_numpy()
        k = min(len(r), len(turb))
        for lab, msk in (("calm", ~turb[:k]), ("turbulent", turb[:k])):
            if msk.sum() > 20:
                reg_rows.append({"strategy": n, "regime": lab, "days": int(msk.sum()),
                                 "sharpe_2x": sharpe(r[:k][msk]), "ann_ret_2x": float(r[:k][msk].mean() * 252)})
    regimes = pd.DataFrame(reg_rows).set_index(["strategy", "regime"]) if reg_rows else pd.DataFrame()

    # 5. deflated Sharpe from the full trial count
    ts = trial_stats(cfg, data_hash)
    dsr = deflated_sharpe(st["rl_ensemble"][1.0]["r_net"].to_numpy(), ts["n_configs"], ts["trial_sr_pp"])
    dsr["n_runs_logged"] = ts["n_runs"]

    # 6. gate
    base2 = stress.loc[GATE_BASELINE, f"{int(GATE_COST)}x"]
    med2 = seed_stats.loc["median", f"sharpe_{int(GATE_COST)}x"]
    ens2 = stress.loc["rl_ensemble", f"{int(GATE_COST)}x"]
    passed = bool(med2 > base2)
    verdict = (f"PASS - median-seed Sharpe at 2x costs {med2:.2f} > GBM+no-trade-band {base2:.2f} "
               f"(ensemble {ens2:.2f}, worst seed {seed_stats.loc['worst', 'sharpe_2x']:.2f})." if passed else
               f"FAIL - median-seed Sharpe at 2x costs {med2:.2f} <= GBM+no-trade-band {base2:.2f} "
               f"(ensemble {ens2:.2f}, worst seed {seed_stats.loc['worst', 'sharpe_2x']:.2f}). "
               "Ship the GBM+no-trade-band rule; archive the RL allocator as research.")
    # A verdict only counts on real, point-in-time vendor data. Anything synthetic (or
    # survivorship-biased) is a DRY RUN of the machinery, never the Ticket 4 decision.
    src = str((data_meta or {}).get("source", cfg_get(cfg, "data.source", "?")))
    real = (data_meta or {}).get("pit") is True and not src.startswith("synthetic")
    if not real:
        verdict = (f"DRY RUN ({src} data, not a real-data verdict): would read as -> {verdict}")
    audit = behaviour_audit(st["rl_ensemble"][1.0], bench)

    # 7. write
    plot_equity({n: st[n][1.0] for n in names}, out / "equity.png")
    plot_equity({n: st[n][GATE_COST] for n in names}, out / "equity_2x.png", "Out-of-sample equity at 2x costs")
    plot_behaviour(st["rl_ensemble"][1.0], fold_outs[0]["assets"], out / "behaviour.png")
    pit = (data_meta or {}).get("pit")
    lines = [f"# Walk-forward report: {variant}", "",
             f"Grid `{grid_id}` | data hash `{data_hash}` | source `{(data_meta or {}).get('source', '?')}`"
             f" | point-in-time: **{pit}** | folds {len(fold_outs)} x seeds {len(seeds)} | "
             f"{cfg_get(cfg, 'agent.total_steps')} steps/seed | reward `{cfg_get(cfg, 'reward.kind')}`"
             f" (kappa {cfg_get(cfg, 'reward.turnover_penalty', 0)}) | band "
             f"{'learned' if cfg_get(cfg, 'env.learned_band') else cfg_get(cfg, 'env.band')}", "",
             f"## Verdict\n\n**{verdict}**", "",
             "## Strategies, out of sample, 1x costs", "", _table(summary), "",
             "## RL seeds (stitched OOS): median / worst / best", "", _table(seed_stats), "", _table(sd), "",
             "## Cost stress (Sharpe, OOS)", "", _table(stress), "",
             "## Regimes (Hurst/entropy classifier, 2x costs)", "", _table(regimes) if len(regimes) else "_n/a_", "",
             "## Deflated Sharpe (RL ensemble, 1x)", "",
             f"- trials (grid runs) on this data hash (MLflow): {dsr['n_trials']} ({dsr['n_runs_logged']} seed runs)",
             f"- expected max Sharpe under the null (annualised): {dsr['sr0_annual']:.3f}",
             f"- PSR(SR>0) {dsr['psr_0']:.3f} | **DSR {dsr['dsr']:.3f}**", "",
             "## Behaviour audit", "",
             *[f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in audit.items()], "",
             "![equity](equity.png)", "", "![equity 2x](equity_2x.png)", "", "![behaviour](behaviour.png)"]
    if pit is False:
        lines.insert(2, "> WARNING: data is NOT survivorship-free; results are biased upward.\n")
    (out / "report.md").write_text("\n".join(lines))
    for n in names:
        st[n][1.0].drop(columns=["weights", "overlay"]).to_csv(out / f"oos_{n}.csv")
    res = {"variant": variant, "grid_id": grid_id, "data_hash": data_hash, "passed": passed, "verdict": verdict,
           "real_data": real,
           "summary": summary.to_dict(orient="index"), "seed_stats": seed_stats.to_dict(orient="index"),
           "stress": stress.to_dict(orient="index"), "dsr": dsr, "audit": audit,
           "gate": {"median_seed_2x": med2, "baseline_2x": base2, "ensemble_2x": ens2}}
    (out / "report.json").write_text(json.dumps(res, indent=2, default=float))
    try:
        from rltrade.eval.walkforward import mlflow_setup

        mlflow = mlflow_setup(cfg)
        with mlflow.start_run(run_name=f"{variant}_summary"):
            mlflow.set_tags({"kind": "summary", "variant": variant, "data_hash": data_hash, "grid_id": grid_id,
                             "passed": str(passed)})
            mlflow.log_metrics({"median_seed_sharpe_2x": med2, "worst_seed_sharpe_2x":
                                seed_stats.loc["worst", "sharpe_2x"], "ensemble_sharpe_2x": ens2,
                                "baseline_sharpe_2x": base2, "dsr": dsr["dsr"],
                                "rl_turnover_ann": float(summary.loc["rl_ensemble", "turnover_ann"])})
            mlflow.log_artifact(str(out / "report.md"))
    except Exception:  # noqa: BLE001
        pass
    return res


def compare_variants(reports: list[dict], out: Path) -> str:
    """Ticket 3 A/B table: each variant vs every baseline, at 1x and 2x, median and worst seed."""
    rows = []
    for r in reports:
        s, ss, stv = r["summary"], r["seed_stats"], r["stress"]
        rows.append({"variant": r["variant"],
                     "rl_turnover_ann": s["rl_ensemble"]["turnover_ann"],
                     "rl_ens_1x": stv["rl_ensemble"]["1x"], "rl_ens_2x": stv["rl_ensemble"]["2x"],
                     "rl_median_seed_2x": ss["median"]["sharpe_2x"], "rl_worst_seed_2x": ss["worst"]["sharpe_2x"],
                     "1/N_2x": stv["equal_weight_monthly"]["2x"], "gbm_linear_2x": stv["gbm_linear_monthly"]["2x"],
                     "gbm_mv_2x": stv["gbm_mean_variance"]["2x"], "gbm_band_2x": stv[GATE_BASELINE]["2x"],
                     "dsr": r["dsr"]["dsr"],
                     "gate": ("PASS" if r["passed"] else "FAIL") + ("" if r.get("real_data") else " (dry run)")})
    df = pd.DataFrame(rows).set_index("variant")
    dry = not all(r.get("real_data") for r in reports)
    md = "# Ticket 3: turnover fixes, A/B\n\n" + ("> DRY RUN on synthetic / non-PIT data: this tests the fix "
         "mechanisms, it is NOT a verdict. Rerun on a vendor snapshot.\n\n" if dry else "") + \
         "Sharpe ratios out of sample. Gate = median seed at 2x costs must beat `gbm_band_2x` " \
         "(GBM + no-trade band).\n\n" + _table(df) + "\n"
    (out / "ab_comparison.md").write_text(md)
    df.to_csv(out / "ab_comparison.csv")
    return md
