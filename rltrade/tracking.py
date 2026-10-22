"""Run tracking. Every trial is logged, so the deflated-Sharpe trial count is honest.

Always writes local files:  runs/<run_id>/{config.yaml, metrics.json, ...}
and appends one line per trial to runs/trials.jsonl. If MLflow is installed and
`tracking.mlflow: true`, the same params/metrics also go to a local MLflow store.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import yaml

from rltrade.config import ROOT, cfg_get, config_hash


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


class Tracker:
    def __init__(self, cfg: dict, name: str = "run", data_hash: str = ""):
        self.cfg = cfg
        self.root = ROOT / cfg_get(cfg, "tracking.dir", "runs")
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_{config_hash(cfg)}"
        self.dir = self.root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data_hash = data_hash
        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(_jsonable(cfg) | {"_data_hash": data_hash}, f, sort_keys=False)
        self.mlflow = None
        if cfg_get(cfg, "tracking.mlflow", False):
            try:
                import mlflow

                mlflow.set_tracking_uri(f"sqlite:///{self.root / 'mlflow.db'}")
                mlflow.set_experiment(cfg_get(cfg, "tracking.experiment", "project24"))
                mlflow.start_run(run_name=self.run_id)
                mlflow.log_params({k: str(v)[:250] for k, v in _flatten(cfg).items()})
                mlflow.log_param("data_hash", data_hash)
                self.mlflow = mlflow
            except ImportError:
                pass

    def log_metrics(self, metrics: dict, name: str = "metrics") -> None:
        m = _jsonable(metrics)
        (self.dir / f"{name}.json").write_text(json.dumps(m, indent=2))
        if self.mlflow:
            flat = {f"{name}.{k}": v for k, v in _flatten(m).items() if isinstance(v, (int, float))}
            self.mlflow.log_metrics(flat)

    def log_trial(self, sharpe_val: float, sharpe_per_period: float, extra: dict | None = None) -> None:
        rec = {"run_id": self.run_id, "cfg_hash": config_hash(self.cfg), "data_hash": self.data_hash,
               "val_sharpe": sharpe_val, "val_sharpe_pp": sharpe_per_period, **(extra or {})}
        with open(self.root / "trials.jsonl", "a") as f:
            f.write(json.dumps(_jsonable(rec)) + "\n")

    def close(self) -> None:
        if self.mlflow:
            self.mlflow.end_run()


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def load_trials(root: Path | None = None) -> list[dict]:
    p = (root or ROOT / "runs") / "trials.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
