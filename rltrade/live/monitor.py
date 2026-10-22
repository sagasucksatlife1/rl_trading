"""Live monitoring: feature drift, action drift, live-vs-backtest tracking error.

PSI (population stability index) per feature, bucketed on training deciles,
judged against an empirical null of same-length training windows (see
`feature_drift`). KS two-sample p-value reported as a second opinion. Action drift = PSI on the agent's
gross exposure / turnover vs the backtest distribution.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    e = np.asarray(expected, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < bins or len(a) == 0:
        return 0.0
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def feature_drift(train: np.ndarray, live: np.ndarray, names: list[str], n_null: int = 60,
                  q: float = 0.95, seed: int = 0) -> list[dict]:
    """train: [T_train, N, F], live: [T_live, N, F].

    Slow features (63d returns, 12-1 momentum) barely move inside a short live
    window, so ANY short window looks shifted against the full training
    distribution. The alert threshold is therefore an empirical null: the q-quantile
    of PSI between the training distribution and random training windows of the
    same length. Live PSI above that = drift beyond what history itself produces.
    """
    rng = np.random.default_rng(seed)
    L = live.shape[0]
    T, N, F = train.shape
    flat_tr = train.reshape(-1, F)
    flat_lv = live.reshape(-1, F)
    starts = rng.integers(0, max(T - L, 1), size=n_null)
    out = []
    for j, n in enumerate(names):
        p = psi(flat_tr[:, j], flat_lv[:, j])
        null = np.array([psi(flat_tr[:, j], train[s:s + L, :, j].reshape(-1)) for s in starts])
        thr = float(np.quantile(null, q)) if len(null) else 0.25
        ks = ks_2samp(flat_tr[:, j], flat_lv[:, j]) if len(flat_lv) > 5 else None
        out.append({"feature": n, "psi": p, "psi_null_q": thr, "ks_p": float(ks.pvalue) if ks else None,
                    "alert": p > max(thr, 0.1)})
    return out


def tracking_error(live_r: np.ndarray, sim_r: np.ndarray) -> dict:
    n = min(len(live_r), len(sim_r))
    if n < 2:
        return {"te_ann": None, "n": n}
    d = np.asarray(live_r[:n]) - np.asarray(sim_r[:n])
    return {"te_ann": float(d.std(ddof=1) * np.sqrt(252)), "mean_diff_bps": float(d.mean() * 1e4), "n": n}
