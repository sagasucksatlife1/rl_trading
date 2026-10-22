"""Chronological splits with purge and embargo (Lopez de Prado, AFML ch. 7).

Labels/rewards at row t depend on prices up to t + horizon, so the last `purge`
rows of a training block would leak into the next block; they are dropped.
`embargo` adds extra rows of separation after each evaluation block.
All ranges are half-open [lo, hi) row indices.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fold:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]

    def __repr__(self) -> str:
        return f"Fold(train={self.train}, val={self.val}, test={self.test})"


def chrono_split(first: int, n_rows: int, val_frac: float = 0.15, test_frac: float = 0.2, purge: int = 21,
                 embargo: int = 5) -> Fold:
    usable = n_rows - first
    n_test = int(usable * test_frac)
    n_val = int(usable * val_frac)
    test = (n_rows - n_test, n_rows)
    val = (test[0] - purge - embargo - n_val, test[0] - purge - embargo)
    train = (first, val[0] - purge - embargo)
    if train[1] - train[0] < 252:
        raise ValueError("not enough data for a train block of >= 1 year")
    return Fold(train, val, test)


def walk_forward(first: int, n_rows: int, min_train: int = 756, val_len: int = 252, test_len: int = 252,
                 purge: int = 21, embargo: int = 5, rolling: int | None = None) -> list[Fold]:
    """Expanding (or rolling, if `rolling` = train length) walk-forward folds.

    Test blocks tile the out-of-sample period without overlap, so stitching their
    returns gives one continuous OOS track record.
    """
    gap = purge + embargo
    folds = []
    test_lo = first + min_train + gap + val_len + gap
    while test_lo + 21 <= n_rows:
        test_hi = min(test_lo + test_len, n_rows)
        val = (test_lo - gap - val_len, test_lo - gap)
        tr_hi = val[0] - gap
        tr_lo = first if rolling is None else max(first, tr_hi - rolling)
        folds.append(Fold((tr_lo, tr_hi), val, (test_lo, test_hi)))
        test_lo = test_hi
    return folds
