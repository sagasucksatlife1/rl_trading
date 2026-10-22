"""Config loading: YAML files + dotted command-line overrides.

Every run is reproducible from (config, seed, data hash). `load_config` returns a
plain nested dict; `cfg_get` reads dotted keys.

    cfg = load_config("configs/default.yaml", ["agent.algo=sac", "env.reward.kind=dsr"])
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def cfg_get(cfg: dict, key: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. A top-level `defaults: [file, ...]` list is merged first."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    merged: dict = {}
    for parent in cfg.pop("defaults", []) or []:
        _deep_update(merged, load_config(path.parent / parent))
    _deep_update(merged, cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        set_dotted(merged, k.strip(), _parse_value(v.strip()))
    return merged


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
