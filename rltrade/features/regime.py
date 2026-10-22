"""Gaussian HMM regime model, written from scratch (Baum-Welch + forward filter).

Fit on the training window only. At run time we use FILTERED probabilities
P(regime_t | obs_1..t), never smoothed ones: smoothing uses the future.
Observation vector: [market log return, log realised vol (21d)].
"""
from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal


class GaussianHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 100, tol: float = 1e-5, seed: int = 0):
        self.k = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)

    def _emission(self, X: np.ndarray) -> np.ndarray:
        B = np.column_stack([multivariate_normal(self.mu[i], self.cov[i], allow_singular=True).pdf(X)
                             for i in range(self.k)])
        return np.maximum(B, 1e-300)

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        T, d = X.shape
        # Init: sort by the vol column so state 0 = calm, state k-1 = turbulent.
        q = np.quantile(X[:, -1], np.linspace(0, 1, self.k + 1))
        lab = np.clip(np.searchsorted(q[1:-1], X[:, -1]), 0, self.k - 1)
        self.mu = np.array([X[lab == i].mean(0) for i in range(self.k)])
        self.cov = np.array([np.cov(X[lab == i].T) + 1e-6 * np.eye(d) for i in range(self.k)])
        self.A = np.full((self.k, self.k), 0.05 / (self.k - 1))
        np.fill_diagonal(self.A, 0.95)
        self.pi = np.full(self.k, 1.0 / self.k)
        prev = -np.inf
        for _ in range(self.n_iter):
            B = self._emission(X)
            alpha, c = self._forward(B)
            beta = np.zeros((T, self.k))
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            xi = np.zeros((self.k, self.k))
            for t in range(T - 1):
                m = alpha[t][:, None] * self.A * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
                xi += m
            self.pi = gamma[0]
            self.A = xi / xi.sum(1, keepdims=True)
            w = gamma.sum(0)
            self.mu = (gamma.T @ X) / w[:, None]
            for i in range(self.k):
                D = X - self.mu[i]
                self.cov[i] = (gamma[:, i, None] * D).T @ D / w[i] + 1e-6 * np.eye(d)
            ll = np.log(c).sum()
            if ll - prev < self.tol:
                break
            prev = ll
        order = np.argsort(self.mu[:, -1])  # relabel by vol: 0 = calm
        self.mu, self.cov = self.mu[order], self.cov[order]
        self.A = self.A[np.ix_(order, order)]
        self.pi = self.pi[order]
        return self

    def _forward(self, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = B.shape[0]
        alpha = np.zeros((T, self.k))
        c = np.zeros(T)
        a = self.pi * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        return alpha, c

    def filter(self, X: np.ndarray) -> np.ndarray:
        """Causal filtered state probabilities [T, k]."""
        return self._forward(self._emission(X))[0]


def regime_observations(close) -> np.ndarray:
    r = np.log(close).diff().mean(axis=1)
    rv = r.rolling(21).std()
    return np.column_stack([r.to_numpy(), np.log(rv.to_numpy() + 1e-8)])


def add_regime_features(fs, close, train_rows: slice, n_states: int = 2, seed: int = 0) -> GaussianHMM:
    """Fit on train rows, add filtered P(turbulent) to the global features."""
    X = regime_observations(close)
    ok = np.isfinite(X).all(1)
    tr_idx = np.arange(len(X))[train_rows]
    tr_idx = tr_idx[ok[tr_idx]]
    hmm = GaussianHMM(n_states, seed=seed).fit(X[tr_idx])
    first = int(np.argmax(ok))
    probs = np.full((len(X), n_states), np.nan)
    probs[first:] = hmm.filter(np.nan_to_num(X[first:]))
    fs.add_global_feature("p_turbulent", probs[:, -1])
    fs.extra["regime_probs"] = probs
    return hmm
