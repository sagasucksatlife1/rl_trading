"""Permutation-equivariant actor-critic for the portfolio env (DeepSets-style).

A plain MLP over the flattened observation must learn the same "how good is this
asset" function separately for every asset slot, and breaks if the universe is
reordered. Here ONE shared network scores every asset from its own features plus
the shared context (global features, portfolio state, its current weight):

    e_i     = phi([x_i, w_i, g, s])                 per-asset embedding (shared phi)
    c       = mean_i e_i                            permutation-invariant pool
    logit_i = rho_pi([e_i, c])                      equivariant actor head
    cash    = psi_pi([c, g, s, w_cash])             cash logit (long-only)
    band    = psi_band([c, g, s])                   learned no-trade band (optional)
    V       = psi_v([c, g, s])                      invariant critic

The logits feed the env's softmax/tanh mapping unchanged. Use with PPO via
`agent.algo=ppo` and `agent.ppo.shared_asset_policy: true`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class AssetSharedExtractor(nn.Module):
    def __init__(self, k: int, F: int, G: int, window: int, long_only: bool, learned_band: bool, hidden: int = 64):
        super().__init__()
        self.k, self.F, self.G, self.W = k, F, G, window
        self.long_only, self.learned_band = long_only, learned_band
        self.d_single = k * F + G + (k + 1) + 3
        ctx = G + 3
        self.phi = _mlp(window * F + 1 + ctx, hidden, hidden)
        self.rho_pi = _mlp(2 * hidden, hidden, 1)
        self.psi_cash = _mlp(hidden + ctx + 1, hidden, 1) if long_only else None
        self.psi_band = _mlp(hidden + ctx, hidden, 1) if learned_band else None
        self.psi_v = _mlp(hidden + ctx, hidden, hidden)
        self.latent_dim_pi = k + int(long_only) + int(learned_band)
        self.latent_dim_vf = hidden

    def _split(self, obs: torch.Tensor):
        B = obs.shape[0]
        frames = obs.view(B, self.W, self.d_single)
        k, F, G = self.k, self.F, self.G
        x = frames[:, :, : k * F].view(B, self.W, k, F).permute(0, 2, 1, 3).reshape(B, k, self.W * F)
        last = frames[:, -1]
        g = last[:, k * F: k * F + G]
        port = last[:, k * F + G: k * F + G + k + 1]
        s = last[:, k * F + G + k + 1:]
        return x, port[:, :k], port[:, k:], torch.cat([g, s], dim=1)

    def _embed(self, obs):
        x, w, w_cash, ctx = self._split(obs)
        B, k = x.shape[0], self.k
        ctx_k = ctx.unsqueeze(1).expand(B, k, ctx.shape[1])
        e = self.phi(torch.cat([x, w.unsqueeze(-1), ctx_k], dim=-1))  # [B, k, h]
        c = e.mean(dim=1)
        return e, c, ctx, w_cash

    def forward_actor(self, obs):
        e, c, ctx, w_cash = self._embed(obs)
        B, k = e.shape[0], self.k
        logits = self.rho_pi(torch.cat([e, c.unsqueeze(1).expand(B, k, c.shape[1])], -1)).squeeze(-1)
        out = [logits]
        if self.long_only:
            out.append(self.psi_cash(torch.cat([c, ctx, w_cash], -1)))
        if self.learned_band:
            out.append(self.psi_band(torch.cat([c, ctx], -1)))
        return torch.cat(out, dim=1)

    def forward_critic(self, obs):
        _, c, ctx, _ = self._embed(obs)
        return self.psi_v(torch.cat([c, ctx], -1))

    def forward(self, obs):
        return self.forward_actor(obs), self.forward_critic(obs)


class AssetSharedPolicy(ActorCriticPolicy):
    """`layout` = dict(k, F, G, window, long_only, learned_band, hidden)."""

    def __init__(self, *args, layout: dict | None = None, **kwargs):
        self.layout = layout or {}
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AssetSharedExtractor(**self.layout)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # the actor head already outputs one logit per action dim: drop the extra linear map
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def layout_from_env(env) -> dict:
    return dict(k=env.k, F=env.F, G=env.G, window=env.c["window"], long_only=env.c["action_mode"] == "long_only",
                learned_band=bool(env.c["learned_band"]))
