#!/usr/bin/env python3
"""Exercise the whole pipeline on synthetic sessions (no participant data needed).

    python tools/selftest.py [--keep DIR]

Builds three synthetic session folders with the real on-disk structure (see
synthetic_sessions.py), runs build_databases.py and verify_databases.py on them,
and checks that a second run is byte-identical. It shows the code is sound; it says
nothing about the real data, which is what verify_databases.py is for.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=Path, default=None, help="work here and keep the files")
    a = ap.parse_args()
    work = a.keep or Path(tempfile.mkdtemp(prefix="myoproband_selftest_"))
    work.mkdir(parents=True, exist_ok=True)
    os.environ["EMG8_DATASET"] = "s26-01"
    os.environ["EMG8_OUT"] = str(work / "_analysis_work")
    sys.path.insert(0, str(ROOT / "analysis_scripts"))
    sys.path.insert(0, str(HERE))
    import build_databases
    import synthetic_sessions as sy
    import verify_databases

    sessions = work / "sessions"
    print("generating synthetic sessions ...")
    sy.make_session(sessions, "sS01_n1_20260926_113055", "S01", all_labels=False, rng_seed=1, dropout=(100.0, 101.5))
    sy.make_session(sessions, "sS00_n1_20261001_173409", "S00", all_labels=True, seed=56056, rng_seed=2)
    sy.make_session(sessions, "sS05_n1_20260926_094549", "S05", n_trials=6, rng_seed=3)

    print("\nbuilding databases ...")
    assert build_databases.main(["--sessions", str(sessions), "--out", str(work / "db1")]) == 0
    print("\nverifying ...")
    if verify_databases.main(["--sessions", str(sessions), "--databases", str(work / "db1")]) != 0:
        print("SELFTEST FAILED: verification")
        return 1

    import pandas as pd
    idx = pd.read_csv(work / "db1" / "sessions_index.csv")
    assert set(idx[idx.included_in_databases].participant) == {"S01", "S07"}, idx
    assert list(idx[~idx.included_in_databases].session_dir) == ["sS05_n1_20260926_094549"], idx
    assert (idx[idx.included_in_databases].sweeps_in_databases == 49).all()
    print("\nindex: 2 complete sessions (S01, S07 via alias), 1 aborted start excluded, 49 sweeps each: OK")

    print("\nrebuilding to check determinism ...")
    assert build_databases.main(["--sessions", str(sessions), "--out", str(work / "db2")]) == 0
    h1, h2 = tree_hashes(work / "db1"), tree_hashes(work / "db2")
    if h1 != h2:
        print("SELFTEST FAILED: second run differs:", [k for k in h1 if h1.get(k) != h2.get(k)][:5])
        return 1
    print(f"{len(h1)} output files byte-identical across two runs")
    print("\nSELFTEST PASSED")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
