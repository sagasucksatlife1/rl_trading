"""Main experiment: data -> features -> GBM signal -> RL seeds -> ensemble -> OOS report.

    python scripts/train.py --config configs/quick.yaml
    python scripts/train.py --config configs/default.yaml agent.algo=sac split.mode=single
"""
from _common import parse

from rltrade.config import cfg_get
from rltrade.eval.splits import chrono_split, walk_forward
from rltrade.experiment import report, run_fold
from rltrade.pipeline import build_dataset
from rltrade.tracking import Tracker


def main():
    _, cfg = parse(__doc__)
    market, fs = build_dataset(cfg)
    tracker = Tracker(cfg, name=cfg_get(cfg, "agent.algo"), data_hash=market.data_hash())
    first, n = fs.first_valid(), len(market.dates)
    sp = cfg_get(cfg, "split", {})
    # the signal starts after its own min_train; RL rows start where the signal exists
    first = max(first, fs.first_valid())
    if sp.get("mode", "walk_forward") == "single":
        folds = [chrono_split(first, n, sp.get("val_frac", 0.15), sp.get("test_frac", 0.2), sp.get("purge", 21),
                              sp.get("embargo", 5))]
    else:
        folds = walk_forward(first, n, sp.get("min_train", 756), sp.get("val_len", 252), sp.get("test_len", 252),
                             sp.get("purge", 21), sp.get("embargo", 5), sp.get("rolling"))
    print(f"run {tracker.run_id}: {len(folds)} fold(s), data hash {market.data_hash()}, "
          f"GBM OOS IC {fs.extra.get('gbm_oos_ic', float('nan')):.4f}")
    outs = []
    for i, fold in enumerate(folds):
        print(f"--- fold {i+1}/{len(folds)}: {fold}")
        outs.append(run_fold(cfg, market, fs, fold, tracker.dir / f"fold_{i}", tracker))
    m = report(cfg, outs, tracker.dir, tracker)
    tracker.close()
    print("\n" + m["verdict"])
    print(f"DSR {m['dsr']['dsr']:.3f} | report: {tracker.dir / 'tearsheet.md'}")


if __name__ == "__main__":
    main()
