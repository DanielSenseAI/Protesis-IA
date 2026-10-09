#!/usr/bin/env python3
"""Compile the raw sEMG/IMU recordings (raw/data/*.xlsx) into one tidy CSV.

Usage:
    python build_dataset.py                      # raw/ -> processed/
    python build_dataset.py --raw DIR --out DIR  # custom locations
    python build_dataset.py --fs 1000            # also add time_s (see README)

Outputs (in --out):
    semg_synchronized.csv   one row per sample, all recordings stacked
    recordings_index.csv    one row per raw file: status, row counts, SHA-256

Every dropped or modified row is reported in recordings_index.csv; nothing is
filtered silently. The script is deterministic: same input -> identical bytes.
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

SIGNALS = ["ax", "ay", "az", "gx", "gy", "gz", "t",
           "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
RAW_COLUMNS = ["sample"] + SIGNALS
# Decimals as written by the acquisition script (grip_save_data.py)
FLOAT_FORMAT = {c: "%.6f" for c in ["ax", "ay", "az", "gx", "gy", "gz", "t"]}
FLOAT_FORMAT.update({f"s{i}": "%.3f" for i in range(1, 9)})

# <Grip>_<MM>_<DD>_<HH>_<MM>_<SS>.xlsx (some files lack the "_" after the grip)
NAME_RE = re.compile(r"^([A-Za-z]+?)_?(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.xlsx$")
# Canonical grip names (file names are inconsistent in case: "lateral"/"Lateral")
GRIPS = {g.lower(): g for g in
         ["BSphere", "Lateral", "MedWrap", "PointTripod", "Ring", "Tripod"]}
GRIPS.update({"reposo": "rest", "test": "test"})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_name(path: Path, year: int):
    m = NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"unexpected file name: {path.name}")
    label, mo, d, h, mi, s = m.groups()
    if label.lower() not in GRIPS:
        raise ValueError(f"unknown grip label {label!r} in {path.name}")
    grip = GRIPS[label.lower()]
    start = pd.Timestamp(year=year, month=int(mo), day=int(d),
                         hour=int(h), minute=int(mi), second=int(s))
    return grip, start


def load_recording(path: Path):
    """Return (clean DataFrame, notes). Rows that are not fully numeric
    (truncated serial lines) are dropped and counted."""
    raw = pd.read_excel(path, dtype=str)
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError(f"{path.name}: unexpected columns {list(raw.columns)}")
    n_raw = len(raw)
    num = raw.apply(pd.to_numeric, errors="coerce")
    bad = num.isna().any(axis=1)
    notes = []
    if bad.any():
        notes.append(f"dropped {int(bad.sum())} non-numeric/truncated row(s) "
                     f"at position(s) {list(map(int, bad[bad].index))}")
    num = num[~bad].reset_index(drop=True)
    num["sample"] = num["sample"].astype("int64")
    steps = num["sample"].diff().dropna()
    if len(num) and not (steps == 1).all():
        notes.append(f"sample counter not contiguous: {int((steps != 1).sum())} jump(s)")
    return num, n_raw, notes


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=here / "raw")
    ap.add_argument("--out", type=Path, default=here / "processed")
    ap.add_argument("--year", type=int, default=2024,
                    help="year of the recordings (not in file names; from sessions sheet)")
    ap.add_argument("--fs", type=float, default=None,
                    help="sampling rate in Hz; if given, adds time_s = sample_idx / fs")
    args = ap.parse_args(argv)

    files = sorted((args.raw / "data").glob("*.xlsx"))
    if not files:
        sys.exit(f"no .xlsx files found in {args.raw / 'data'}")

    index, frames = [], []
    for path in files:
        grip, start = parse_name(path, args.year)
        df, n_raw, notes = load_recording(path)
        index.append({"file": path.name, "grip": grip,
                      "start_time": start.isoformat(), "rows_raw": n_raw,
                      "rows_kept": len(df), "sha256": sha256(path),
                      "status": "included" if len(df) else "excluded_empty",
                      "notes": "; ".join(notes)})
        if len(df):
            frames.append((path.name, grip, start, df))

    # Stable order: chronological, then file name
    frames.sort(key=lambda x: (x[2], x[0]))
    out_frames = []
    for rec_id, (name, grip, start, df) in enumerate(frames, start=1):
        d = pd.DataFrame({
            "recording_id": rec_id,
            "file": name,
            "grip": grip,
            "start_time": start.isoformat(),
            "sample_idx": range(len(df)),
            "sample_counter": df["sample"],
        })
        if args.fs:
            d["time_s"] = d["sample_idx"] / args.fs
        out_frames.append(pd.concat([d, df[SIGNALS]], axis=1))
    data = pd.concat(out_frames, ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    fmt = dict(FLOAT_FORMAT)
    out = data.copy()
    for col, f in fmt.items():
        out[col] = out[col].map(lambda v, f=f: f % v)
    if args.fs:
        out["time_s"] = out["time_s"].map("{:.6f}".format)
    out.to_csv(args.out / "semg_synchronized.csv", index=False, lineterminator="\n")
    pd.DataFrame(index).sort_values(["start_time", "file"]).to_csv(
        args.out / "recordings_index.csv", index=False, lineterminator="\n")

    # Self-checks a reviewer can read at a glance
    inc = [i for i in index if i["status"] == "included"]
    assert len(data) == sum(i["rows_kept"] for i in inc), "row count mismatch"
    assert not data[SIGNALS].isna().any().any(), "NaN in output"
    print(f"{len(files)} raw files: {len(inc)} included, {len(files) - len(inc)} empty")
    print(f"{len(data)} rows -> {args.out / 'semg_synchronized.csv'}")
    print(f"sha256 csv: {sha256(args.out / 'semg_synchronized.csv')}")


if __name__ == "__main__":
    main()
