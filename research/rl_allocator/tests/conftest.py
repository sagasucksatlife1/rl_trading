import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture(scope="session")
def prepared():
    from rltrade.data.synthetic import regime_garch_market
    from rltrade.eval.splits import chrono_split
    from rltrade.pipeline import build_dataset, fold_envdata

    cfg = {"signal": {"min_train": 504, "refit": 126}}
    market, fs = build_dataset(cfg, regime_garch_market(n_assets=6, n_days=1600, seed=1))
    fold = chrono_split(fs.first_valid(), len(market.dates))
    data, _ = fold_envdata(market, fs, fold.train, cfg)
    return cfg, market, fs, fold, data
