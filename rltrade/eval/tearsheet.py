"""Tear sheet: equity curves, drawdowns, exposure and the verdict, as PNG + Markdown."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_equity(records: dict[str, pd.DataFrame], path: Path, title: str = "Out-of-sample equity (net of costs)") -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for name, rec in records.items():
        nav = (1 + rec["r_net"]).cumprod()
        lw = 2.2 if name.startswith("rl") else 1.1
        axes[0].plot(nav.index, nav.values, label=name, lw=lw)
        dd = 1 - nav / nav.cummax()
        axes[1].plot(dd.index, -dd.values, lw=lw)
    axes[0].set_title(title)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("drawdown")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_behaviour(rec: pd.DataFrame, assets: list[str], path: Path) -> None:
    W = np.stack(rec["weights"].to_numpy())
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].stackplot(rec.index, np.clip(W, 0, None).T, labels=assets[: W.shape[1]], alpha=0.85)
    axes[0].set_title("Agent weights over time (behaviour audit)")
    axes[0].set_ylim(0, max(1.0, W.clip(0).sum(1).max()))
    axes[1].plot(rec.index, rec["turnover"].rolling(21).sum(), label="21d turnover")
    axes[1].plot(rec.index, np.abs(W).sum(1), label="gross exposure")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(out: Path, summary: pd.DataFrame, dsr: dict, stress: pd.DataFrame, regimes: pd.DataFrame,
                 audit: dict, verdict: str, extra: dict | None = None) -> Path:
    def table(df):
        try:
            return df.to_markdown(floatfmt=".3f")
        except ImportError:
            return "```\n" + df.round(3).to_string() + "\n```"

    lines = ["# Project 24 - RL for Trading: out-of-sample tear sheet", "", f"**Verdict:** {verdict}", "",
             "## Agent vs baselines (net of costs, out of sample)", "", table(summary), "",
             "## Deflated Sharpe", "",
             f"- PSR(SR>0): {dsr.get('psr_0', float('nan')):.3f}",
             f"- Trials counted: {dsr.get('n_trials')}",
             f"- Expected max Sharpe under the null (annualised): {dsr.get('sr0_annual', float('nan')):.3f}",
             f"- **DSR: {dsr.get('dsr', float('nan')):.3f}** (>0.95 = significant after the multiple-testing haircut)", "",
             "## Cost stress", "", table(stress), "",
             "## Regime breakdown", "", table(regimes) if len(regimes) else "_n/a_", "",
             "## Behaviour audit", "",
             *[f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in audit.items()], "",
             "![equity](equity.png)", "", "![behaviour](behaviour.png)", ""]
    if extra:
        lines += ["## Run info", "", *[f"- {k}: {v}" for k, v in extra.items()]]
    p = out / "tearsheet.md"
    p.write_text("\n".join(lines))
    return p
