"""Build / list / verify hashed point-in-time data snapshots (data/snapshots/<hash>/).

    python scripts/snapshot.py build --source csv_pit --dir vendor/my_universe     # prices/actions/membership.csv
    python scripts/snapshot.py build --source sharadar --dir vendor/sharadar        # SEP.csv, ACTIONS.csv, SP500.csv
    python scripts/snapshot.py build --source synthetic_pit                          # test fixture
    python scripts/snapshot.py list
    python scripts/snapshot.py verify <hash>

Then train on it reproducibly:  python scripts/train.py data.source=snapshot data.snapshot=<hash>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rltrade.data.loaders import (from_csv_dir, from_sharadar, list_snapshots, load_snapshot,  # noqa: E402
                                  synthetic_pit, write_snapshot)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "list", "verify"])
    ap.add_argument("hash", nargs="?")
    ap.add_argument("--source", default="synthetic_pit", choices=["csv_pit", "sharadar", "synthetic_pit"])
    ap.add_argument("--dir")
    ap.add_argument("--membership-file", default="SP500.csv")
    ap.add_argument("--not-pit", action="store_true", help="mark a csv source as NOT survivorship-free")
    a = ap.parse_args()
    if a.cmd == "list":
        for m in list_snapshots():
            print(f"{m['hash'][:16]}  {m['source']:14s} pit={m['pit']!s:5s} {m['symbols']:4d} symbols "
                  f"{m['date_range'][0]}..{m['date_range'][1]}  issues={len(m['quality_issues'])}")
        return
    if a.cmd == "verify":
        m = load_snapshot(a.hash)
        print(f"OK {a.hash[:16]}: {len(m.assets)} tradable-ever symbols, {len(m.dates)} days; "
              f"issues={m.meta.get('quality_issues')}")
        return
    if a.source == "csv_pit":
        raw = from_csv_dir(Path(a.dir), pit=not a.not_pit)
    elif a.source == "sharadar":
        d = Path(a.dir)
        raw = from_sharadar(d / "SEP.csv", d / "ACTIONS.csv", d / a.membership_file)
    else:
        raw, _ = synthetic_pit()
    h = write_snapshot(raw)
    print(json.dumps({"hash": h, "short": h[:16], "use": f"data.source=snapshot data.snapshot={h[:16]}"}, indent=2))


if __name__ == "__main__":
    main()
