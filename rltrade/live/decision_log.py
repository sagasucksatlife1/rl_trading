"""Append-only decision log: every (state, action, reward, next_state) from shadow
and paper trading. It is the training set for offline RL, so the system can
improve from its own experience without paying for live exploration.

Format: JSON lines, one record per decision, under live_state/decisions.jsonl.
Rewards are filled in the next day, once the realised return is known.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class DecisionLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)) + "\n")

    def fill_last_reward(self, reward: float, realised: dict) -> None:
        recs = self.records()
        if not recs:
            return
        recs[-1]["reward"] = reward
        recs[-1]["realised"] = realised
        self.path.write_text("".join(json.dumps(r) + "\n" for r in recs))

    def transitions(self) -> dict[str, np.ndarray]:
        """(s, a, r, s', done) arrays for offline RL, from consecutive filled records."""
        recs = [r for r in self.records() if r.get("reward") is not None]
        if len(recs) < 2:
            return {}
        S = np.array([r["obs"] for r in recs[:-1]], dtype=np.float32)
        A = np.array([r["action"] for r in recs[:-1]], dtype=np.float32)
        R = np.array([r["reward"] for r in recs[:-1]], dtype=np.float32)
        S2 = np.array([r["obs"] for r in recs[1:]], dtype=np.float32)
        D = np.zeros(len(R), dtype=np.float32)
        D[-1] = 1.0
        return {"obs": S, "actions": A, "rewards": R, "next_obs": S2, "terminals": D}
