"""The shared per-asset policy must be permutation-equivariant in the asset logits
and invariant in the cash / band outputs."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from research.rl_allocator.agents.policies import AssetSharedExtractor, layout_from_env  # noqa: E402
from rltrade.pipeline import make_env  # noqa: E402


def test_equivariance(prepared):
    cfg, _, _, fold, data = prepared
    env = make_env(data, fold.train, {**cfg, "env": {"learned_band": True}}, train=True, seed=0)
    o, _ = env.reset()
    ext = AssetSharedExtractor(**layout_from_env(env))
    k, F, G = env.k, env.F, env.G
    perm = np.random.default_rng(0).permutation(k)
    feats = o[: k * F].reshape(k, F)[perm].ravel()
    g, port, st = o[k * F: k * F + G], o[k * F + G: k * F + G + k + 1], o[k * F + G + k + 1:]
    o2 = np.concatenate([feats, g, np.concatenate([port[:k][perm], port[k:]]), st]).astype(np.float32)
    a = ext.forward_actor(torch.as_tensor(o[None]))[0]
    b = ext.forward_actor(torch.as_tensor(o2[None]))[0]
    assert a.shape[0] == env.action_space.shape[0]
    assert torch.allclose(a[:k][perm], b[:k], atol=1e-5)
    assert torch.allclose(a[k:], b[k:], atol=1e-5)
    v1 = ext.forward_critic(torch.as_tensor(o[None]))
    v2 = ext.forward_critic(torch.as_tensor(o2[None]))
    assert torch.allclose(v1, v2, atol=1e-5)
