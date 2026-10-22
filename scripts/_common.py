"""Shared CLI plumbing: `python scripts/x.py --config configs/default.yaml key=value ...`"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse(desc: str, default_config: str = "configs/default.yaml", extra=None):
    from rltrade.config import load_config

    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default=default_config)
    if extra:
        extra(ap)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. agent.algo=sac")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    return args, load_config(args.config, args.overrides)
