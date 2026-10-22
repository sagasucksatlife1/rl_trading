"""Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2012/2014).

PSR(SR*) = Phi( (SR - SR*) sqrt(T-1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) )
with SR the NON-annualised per-period Sharpe, g3 skew, g4 (non-excess) kurtosis.

DSR = PSR(SR0), where SR0 is the expected maximum Sharpe among N independent
trials with zero true skill:
    SR0 = sqrt(V[SR_trials]) * ((1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)))
g = Euler-Mascheroni. N must be the honest count of configs tried (MLflow /
the trial log keeps it).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649015329


def psr(r: np.ndarray, sr_star: float = 0.0) -> float:
    r = np.asarray(r, dtype=np.float64)
    T = len(r)
    if T < 3 or r.std() == 0:
        return 0.5
    sr = r.mean() / r.std(ddof=1)
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(norm.cdf((sr - sr_star) * np.sqrt(T - 1) / den))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected max of n_trials per-period Sharpe estimates under the null."""
    if n_trials <= 1:
        return 0.0
    return float(np.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials)
                                    + EULER * norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(r: np.ndarray, n_trials: int, trial_sharpes: np.ndarray | None = None) -> dict:
    """`trial_sharpes` are per-period Sharpes of all trials; if absent, V[SR] ~ 1/T."""
    r = np.asarray(r, dtype=np.float64)
    T = len(r)
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        var_sr = float(np.var(trial_sharpes, ddof=1))
    else:
        var_sr = 1.0 / max(T - 1, 1)
    sr0 = expected_max_sharpe(n_trials, var_sr)
    return {"psr_0": psr(r, 0.0), "sr0_per_period": sr0, "sr0_annual": sr0 * np.sqrt(252),
            "dsr": psr(r, sr0), "n_trials": n_trials}
