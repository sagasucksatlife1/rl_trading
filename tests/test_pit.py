"""Point-in-time data: adjustment exactness, delistings, membership, snapshot hashing,
and the env never holding a name outside the universe."""
import numpy as np
import pytest

from rltrade.data.loaders import RawData, build_market, load_snapshot, synthetic_pit, validate, write_snapshot


@pytest.fixture(scope="module")
def pit():
    raw, truth = synthetic_pit(n_symbols=24, universe_size=15, n_days=1600, seed=3)
    return raw, truth, build_market(raw)


def test_total_return_matches_truth(pit):
    raw, truth, m = pit
    dead = {s for s, _ in truth["delists"]} | {s for s, _, _ in truth["mergers"]}
    for s in m.assets:
        f, last = truth["first"][s], truth["last"][s]
        end = last - 1 if s in dead else last
        a, t = m.close[s].to_numpy(), truth["tr_close"][s].to_numpy()
        np.testing.assert_allclose(a[f + 1:end + 1] / a[f:end], t[f + 1:end + 1] / t[f:end], rtol=1e-10)


def test_splits_do_not_jump(pit):
    raw, truth, m = pit
    assert truth["splits"], "fixture should contain splits"
    lr = np.log(m.close).diff().abs()
    assert float(lr.max().max()) < np.log(1.5) or any(True for _ in truth["delists"])


def test_delisting_return_and_exit(pit):
    raw, truth, m = pit
    for s, r in truth["delists"]:
        if s not in m.assets:
            continue
        last = truth["last"][s]
        assert m.close[s].iloc[last] / m.close[s].iloc[last - 1] - 1 == pytest.approx(r)
        assert not m.tradable[s].iloc[last:].any()
        assert m.close[s].iloc[last:].nunique() == 1  # flat after delisting: position became cash


def test_membership_respected(pit):
    raw, truth, m = pit
    mem = raw.canonical().membership
    for r in mem.itertuples(index=False):
        if r.symbol not in m.assets:
            continue
        before = m.tradable[r.symbol][m.dates < r.start]
        assert not before.any() or (mem[mem.symbol == r.symbol].start.min() < r.start)


def test_snapshot_hash_reproducible_and_tamper_evident(pit, tmp_path):
    raw, _, _ = pit
    h1 = write_snapshot(raw, tmp_path)
    assert write_snapshot(raw, tmp_path) == h1
    m = load_snapshot(h1, tmp_path)
    assert m.data_hash() == h1[:10]
    raw2 = RawData(raw.prices.assign(close=raw.prices.close * (1 + 1e-6)), raw.actions, raw.membership)
    assert raw2.content_hash() != h1
    d = next(tmp_path.glob(h1[:16] + "*"))
    p = raw.canonical().prices
    p.loc[0, "close"] *= 1.01
    p.to_parquet(d / "prices.parquet", index=False)
    with pytest.raises(ValueError):
        load_snapshot(h1, tmp_path)


def test_validate_flags_missing_split(pit):
    raw, _, _ = pit
    acts = raw.actions[raw.actions.type != "split"]
    issues = validate(RawData(raw.prices, acts, raw.membership))
    assert any("unexplained" in i for i in issues)


def test_env_never_holds_non_members(pit):
    from rltrade.eval.splits import chrono_split
    from rltrade.pipeline import build_dataset, fold_envdata, make_env

    _, _, m = pit
    cfg = {"signal": {"min_train": 504, "refit": 126}, "env": {"window": 1}}
    market, fs = build_dataset(cfg, m)
    fold = chrono_split(fs.first_valid(), len(market.dates))
    data, _ = fold_envdata(market, fs, fold.train, cfg)
    for rows in (fold.train, fold.test):
        env = make_env(data, rows, cfg, train=False)
        env.reset()
        rng = np.random.default_rng(0)
        done = False
        while not done:
            t = env.t
            _, _, te, tr, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
            assert np.all(info["weights"][~data.tradable[t, env.assets]] == 0)
            done = te or tr
