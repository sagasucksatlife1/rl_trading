"""Stable-Baselines3 agents: PPO, RecurrentPPO (main), SAC (challenger), TD3 (check).

Training discipline baked in:
  * VecNormalize on obs and rewards; stats are FROZEN and saved for evaluation
  * early stopping on the VALIDATION env (Sharpe of a deterministic rollout),
    never on the test env; the best checkpoint is kept, not the last
  * N seeds per config; final policy = ensemble average of the top seeds' weights
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from rltrade.config import cfg_get, seed_everything
from rltrade.eval.metrics import sharpe
from rltrade.eval.rollout import rollout

log = logging.getLogger(__name__)

ALGOS = ("ppo", "recurrent_ppo", "sac", "td3")


def _vec_env(make_fn, n_envs: int, subproc: bool):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [lambda i=i: make_fn(i) for i in range(n_envs)]
    return SubprocVecEnv(fns) if subproc and n_envs > 1 else DummyVecEnv(fns)


def make_model(algo: str, venv, cfg: dict, seed: int):
    import torch
    from stable_baselines3 import PPO, SAC, TD3

    torch.set_num_threads(cfg_get(cfg, "agent.torch_threads", 1))
    hp = dict(cfg_get(cfg, f"agent.{algo}", {}) or {})
    net = hp.pop("net_arch", [256, 256])
    extra_pk = hp.pop("policy_kwargs", {}) or {}
    common = dict(seed=seed, verbose=0, device="cpu")
    if algo == "ppo" and hp.pop("shared_asset_policy", False):
        from research.rl_allocator.agents.policies import AssetSharedPolicy

        get = lambda a: venv.get_attr(a)[0]
        c = get("c")
        layout = dict(k=get("k"), F=get("F"), G=get("G"), window=c["window"],
                      long_only=c["action_mode"] == "long_only", learned_band=bool(c["learned_band"]),
                      hidden=hp.pop("shared_hidden", 64))
        return PPO(AssetSharedPolicy, venv, policy_kwargs=dict(layout=layout, **extra_pk), **common,
                   **{"n_steps": 1024, "batch_size": 256, "learning_rate": 3e-4, "gamma": 0.99,
                      "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "n_epochs": 10, **hp})
    hp.pop("shared_asset_policy", None)
    hp.pop("shared_hidden", None)
    if algo == "ppo":
        return PPO("MlpPolicy", venv, policy_kwargs=dict(net_arch=dict(pi=net, vf=net), **extra_pk), **common,
                   **{"n_steps": 1024, "batch_size": 256, "learning_rate": 3e-4, "gamma": 0.99,
                      "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.005, "n_epochs": 10, **hp})
    if algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        lstm = hp.pop("lstm_hidden_size", 128)
        return RecurrentPPO("MlpLstmPolicy", venv,
                            policy_kwargs=dict(net_arch=dict(pi=net, vf=net), lstm_hidden_size=lstm, **extra_pk), **common,
                            **{"n_steps": 512, "batch_size": 256, "learning_rate": 3e-4, "gamma": 0.99,
                               "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.005, "n_epochs": 10, **hp})
    if algo == "sac":
        return SAC("MlpPolicy", venv, policy_kwargs=dict(net_arch=net, **extra_pk), **common,
                   **{"learning_rate": 3e-4, "buffer_size": 200_000, "batch_size": 256, "gamma": 0.99,
                      "tau": 0.005, "learning_starts": 5000, "ent_coef": "auto", "train_freq": 1, **hp})
    if algo == "td3":
        from stable_baselines3.common.noise import NormalActionNoise

        n = venv.action_space.shape[0]
        return TD3("MlpPolicy", venv, policy_kwargs=dict(net_arch=net, **extra_pk), **common,
                   action_noise=NormalActionNoise(np.zeros(n), 0.2 * np.ones(n)),
                   **{"learning_rate": 3e-4, "buffer_size": 200_000, "batch_size": 256, "gamma": 0.99,
                      "learning_starts": 5000, **hp})
    raise ValueError(f"unknown algo {algo!r}; choose from {ALGOS}")


class SB3Policy:
    """Deterministic weight-space policy from a trained SB3 model + frozen obs stats."""

    def __init__(self, model, obs_mean: np.ndarray, obs_var: np.ndarray, clip: float = 10.0):
        self.model, self.mean, self.var, self.clip = model, obs_mean, obs_var, clip
        self.recurrent = model.__class__.__name__ == "RecurrentPPO"
        self.reset()

    def reset(self):
        self.state = None
        self.start = np.ones((1,), dtype=bool)

    def norm(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)

    def action(self, obs: np.ndarray) -> np.ndarray:
        o = self.norm(obs)
        if self.recurrent:
            a, self.state = self.model.predict(o[None], state=self.state, episode_start=self.start, deterministic=True)
            self.start = np.zeros((1,), dtype=bool)
            return a[0]
        a, _ = self.model.predict(o, deterministic=True)
        return a

    def __call__(self, obs, env):
        return env.action_to_weights(self.action(obs))

    @classmethod
    def load(cls, path: str | Path):
        from stable_baselines3 import PPO, SAC, TD3

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        algo = meta["algo"]
        if algo == "recurrent_ppo":
            from sb3_contrib import RecurrentPPO as Cls
        else:
            Cls = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo]
        model = Cls.load(path / "model.zip", device="cpu")
        st = np.load(path / "obs_rms.npz")
        return cls(model, st["mean"], st["var"])


class Ensemble:
    """Combine several policies into one target.

    Three rules, all measured on the same Ticket 4 dry-run seeds (Sharpe at 2x costs,
    turnover/yr; the median single seed scored 0.54 at 21.7x):
      mean          (default) average raw targets, then band once         0.29   5.3x
      median_banded per-asset median of each member's banded target      -0.10  32.5x
      majority      trade an asset only if most members' bands say so,    0.09   2.4x
                    voting on overlay-projected (executable) targets
    No rule beats the median seed: the seeds disagree about WHICH assets to trade, so
    any combination either churns or freezes. Decisions therefore use the median seed
    (see eval/report.py); the ensemble is reported for completeness only.
    """

    # absorbs the open-vs-close drift between the book the members saw and the one traded
    DRIFT_BAND = 0.002

    def __init__(self, members: list, mode: str = "mean"):
        if mode not in ("majority", "median_banded", "mean"):
            raise ValueError(mode)
        self.members, self.mode = members, mode

    def reset(self):
        for m in self.members:
            if hasattr(m, "reset"):
                m.reset()

    def __call__(self, obs, env):
        ws, bs = zip(*[m(obs, env) for m in self.members])
        if self.mode == "mean":
            return np.mean(ws, axis=0), float(np.mean(bs))
        long_only = env.c["action_mode"] == "long_only"
        cur = env.w_close
        # vote on what each member could actually execute (overlay-projected), never on raw
        # targets: raw targets above a cap "want to trade" forever (the band-order bug again)
        W = np.asarray([env.project(w, cur) for w in ws])
        trade = np.abs(W - cur) >= np.asarray(bs)[:, None]  # [members, assets]: member wants to trade
        if self.mode == "median_banded":
            w = np.median(np.where(trade, W, cur), axis=0)
        else:
            maj = trade.sum(0) > len(self.members) / 2
            mean_tr = (W * trade).sum(0) / np.maximum(trade.sum(0), 1)
            w = np.where(maj, mean_tr, cur)
        if long_only and w.sum() > 1.0:
            w = w / w.sum()
        return w, self.DRIFT_BAND


def _save(model, rms, path: Path, algo: str, extra: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save(path / "model.zip")
    np.savez(path / "obs_rms.npz", mean=rms.mean, var=rms.var)
    (path / "meta.json").write_text(json.dumps({"algo": algo, **extra}, default=float))


def train_seed(make_train_env, make_val_env, cfg: dict, seed: int, out_dir: Path) -> dict:
    """Train one seed with validation early stopping. Returns metadata of the best checkpoint."""
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize

    algo = cfg_get(cfg, "agent.algo", "ppo")
    seed_everything(seed)
    n_envs = cfg_get(cfg, "agent.n_envs", 4)
    venv = _vec_env(lambda i: make_train_env(seed * 1000 + i), n_envs, cfg_get(cfg, "agent.subproc", False))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg_get(cfg, "agent.gamma", 0.99))
    model = make_model(algo, venv, cfg, seed)
    total = int(cfg_get(cfg, "agent.total_steps", 200_000))
    eval_every = int(cfg_get(cfg, "agent.eval_every", 20_000))
    patience = int(cfg_get(cfg, "agent.patience", 5))
    best = {"val_sharpe": -np.inf, "step": 0}
    history = []

    class ValCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.bad = 0
            self.next_eval = eval_every

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_eval:
                return True
            self.next_eval += eval_every
            rms = venv.obs_rms
            pol = SB3Policy(model, rms.mean.copy(), rms.var.copy())
            rec = rollout(make_val_env(), pol)
            s = sharpe(rec["r_net"].to_numpy())
            history.append({"step": self.num_timesteps, "val_sharpe": s,
                            "val_turnover": float(rec["turnover"].sum() / len(rec) * 252)})
            log.info("[%s seed %d] step %d  val Sharpe %.3f", algo, seed, self.num_timesteps, s)
            if s > best["val_sharpe"]:
                best.update(val_sharpe=s, step=self.num_timesteps,
                            val_sharpe_pp=float(rec["r_net"].mean() / (rec["r_net"].std() + 1e-12)))
                _save(model, rms, out_dir, algo, {"seed": seed, **best})
                self.bad = 0
            else:
                self.bad += 1
            return self.bad < patience

    model.learn(total_timesteps=total, callback=ValCallback(), progress_bar=False)
    if not (out_dir / "model.zip").exists():  # never evaluated (tiny budgets)
        _save(model, venv.obs_rms, out_dir, algo, {"seed": seed, **best})
    venv.close()
    (out_dir / "history.json").write_text(json.dumps(history))
    return {"seed": seed, "path": str(out_dir), **best, "history": history}


def load_ensemble(paths: list[str | Path]) -> Ensemble:
    return Ensemble([SB3Policy.load(p) for p in paths])
