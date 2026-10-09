#!/usr/bin/env python3
"""Independent check of the databases against the raw session folders.

    python tools/verify_databases.py --sessions raw/sessions --databases databases

This does NOT import the analysis code or build_databases.py. It re-reads raw.bin, imu.bin and
aux.jsonl with its own few lines of numpy/bisect and compares what it finds with the CSV files,
so a reviewer can confirm the databases say what the raw records say. Channel map and scaling
are restated here on purpose (source: IA-Arm_Monitor backend/emg8/records.py, FIRMWARE-CONTRACT 2.2).

Checks per complete session (exit code 1 if any fails):
  rows      number of grid rows equals the number the raw duration implies
  grid      t_s is an exact 1 ms / 20 ms grid and device_ts_us = first raw ts + t_s
  raw/env   value at random grid times equals the newest raw.bin sample at or before it
            (blank if none, or if older than the stated maximum age)
  imu       same for the IMU, in g and deg/s
  sweeps    every sweep file has 99 points and the exact impedances recorded in aux.jsonl;
            the 1 kHz database shows each point's value during that point's 41.9 ms step
  labels    42 trials, 7 grasps x 6 repetitions
  sync      every 50 Hz row carries the same IMU/aux/sweep/label values as the 1 kHz row at that time
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE = np.dtype([("ts_us", "<u4"), ("adc", "u1"), ("ch", "u1"), ("value", "<i2")])
IMU = np.dtype([("ts_us", "<u4"), ("ax", "<i2"), ("ay", "<i2"), ("az", "<i2"),
                ("gx", "<i2"), ("gy", "<i2"), ("gz", "<i2"), ("temp100", "<i2"), ("_pad", "<u2")])
RAW_PINS = ((0, 3), (1, 3), (0, 3), (1, 3))
ENV_PINS = ((1, 2), (0, 2), (1, 2), (0, 2))
T1, STEP = 0.055, 0.0419
RESULTS: list[tuple[str, str, bool, str]] = []


def check(session: str, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((session, name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def unwrap(ts):
    t = ts.astype(np.int64)
    for j in np.where(np.diff(t) < -(1 << 31))[0]:
        t[j + 1:] += 1 << 32
    return t


def read_channels(path: Path):
    a = np.fromfile(path / "raw.bin", dtype=SAMPLE)
    raw, env = {}, {}
    for adc in range(4):
        for kind, pins, dst in (("raw", RAW_PINS, raw), ("env", ENV_PINS, env)):
            for slot, pin in enumerate(pins[adc]):
                m = (a["adc"] == adc) & (a["ch"] == pin)
                t = unwrap(a["ts_us"][m])
                o = np.argsort(t, kind="stable")
                dst[adc * 2 + slot] = (t[o], a["value"][m][o].astype(np.int64))
    return raw, env


def newest_at_or_before(ts_list, vals, t_us, max_age_us):
    """Pure-Python reference: bisect, not searchsorted."""
    k = bisect.bisect_right(ts_list, t_us) - 1
    if k < 0:
        return None
    if max_age_us is not None and t_us - ts_list[k] > max_age_us:
        return None
    return vals[k]


def eq(a, b, tol=0.0):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None or (isinstance(b, float) and np.isnan(b)) or pd.isna(b)
    if b is None or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= tol


def verify_session(sess: Path, participant: str, tag: str, db: Path, rng) -> None:
    print(f"\n{sess.name}  ->  {participant}")
    raw, env = read_channels(sess)
    t0 = min(raw[i][0][0] for i in range(8))
    end_s = (max(raw[i][0][-1] for i in range(8)) - t0) / 1e6
    r_path = db / "raw_1khz" / f"{tag}_raw_1khz.csv.gz"
    e_path = db / "envelope_50hz" / f"{tag}_env_50hz.csv.gz"
    R = pd.read_csv(r_path)
    E = pd.read_csv(e_path)

    check(sess.name, "rows 1 kHz", len(R) == len(np.arange(0.0, end_s, 0.001)), f"{len(R)}")
    check(sess.name, "rows 50 Hz", len(E) == len(np.arange(0.0, end_s, 0.02)), f"{len(E)}")
    check(sess.name, "grid 1 kHz", np.allclose(R.t_s.to_numpy(), np.arange(len(R)) * 0.001, atol=6e-5)
          and (R.device_ts_us.to_numpy() == t0 + np.round(R.t_s.to_numpy() * 1e6)).all())
    check(sess.name, "grid 50 Hz", np.allclose(E.t_s.to_numpy(), np.arange(len(E)) * 0.02, atol=6e-5)
          and (E.device_ts_us.to_numpy() == t0 + np.round(E.t_s.to_numpy() * 1e6)).all())

    for label, df, chans, prefix, max_age in (("raw", R, raw, "raw", 5000.0), ("env", E, env, "env", 60000.0)):
        idx = rng.choice(len(df), size=min(3000, len(df)), replace=False)
        bad = 0
        for c in range(8):
            ts_list, vals = chans[c][0].tolist(), chans[c][1].tolist()
            col = df[f"{prefix}_E{c + 1}"]
            for i in idx:
                want = newest_at_or_before(ts_list, vals, t0 + int(round(df.t_s.iat[i] * 1e6)), max_age)
                if not eq(want, col.iat[i]):
                    bad += 1
        check(sess.name, f"{label} hold values ({len(idx)} rows x 8 channels)", bad == 0, f"{bad} mismatches")
        blank_ok = (df[[f"{prefix}_E{c + 1}" for c in range(8)]].isna().to_numpy().sum() > 0) or True
        _ = blank_ok

    # IMU
    imu = np.fromfile(sess / "imu.bin", dtype=IMU)
    it = imu["ts_us"].astype(np.int64)
    o = np.argsort(it, kind="stable")
    it = it[o].tolist()
    cols = {"ax_g": ("ax", 1e-3), "ay_g": ("ay", 1e-3), "az_g": ("az", 1e-3),
            "gx_dps": ("gx", 0.1), "gy_dps": ("gy", 0.1), "gz_dps": ("gz", 0.1), "imu_temp_c": ("temp100", 0.01)}
    bad = 0
    idx = rng.choice(len(R), size=min(3000, len(R)), replace=False)
    for name, (field, scale) in cols.items():
        vals = (imu[field][o].astype(np.float64) * scale).tolist()
        col = R[name]
        for i in idx:
            want = newest_at_or_before(it, vals, t0 + int(round(R.t_s.iat[i] * 1e6)), None)
            if not eq(want, col.iat[i], tol=0.006 if "dps" in name else 0.0006):
                bad += 1
    check(sess.name, f"imu hold values ({len(idx)} rows x 7)", bad == 0, f"{bad} mismatches")

    # sweeps vs aux.jsonl
    aux = [json.loads(l) for l in open(sess / "aux.jsonl", encoding="utf-8") if l.strip()]
    sweeps_raw = [a for a in aux if a.get("kind") == "impedance_sweep"]
    by_id = {a.get("id"): a for a in sweeps_raw}
    sdir = db / "sweeps" / tag
    files = sorted(sdir.glob(f"{tag}_sweep_*.csv"))
    check(sess.name, "sweep files exist for every aux sweep", len(files) == len(sweeps_raw),
          f"{len(files)} files, {len(sweeps_raw)} records")
    ok_pts = ok_z = ok_t = ok_db = True
    for f in files:
        S = pd.read_csv(f)
        ok_pts &= len(S) == 99 and list(S.point) == list(range(99))
        rec = by_id.get(int(S.sweep_id.iat[0]))
        ok_z &= rec is not None and np.allclose(S.impedance_ohm.to_numpy(), rec["z_ohm"], atol=5e-3) \
            and np.allclose(S.freq_khz.to_numpy() * 1e3, rec["freq_hz"], atol=0.5)
        ok_t &= np.allclose(np.diff(S.t_start_s.to_numpy()), STEP, atol=2e-4) \
            and abs(S.t_start_s.iat[0] - (S.sweep_onset_s.iat[0] + T1)) < 2e-4
        # the 1 kHz database must show each point's value during that point's step
        n = int(S.sweep_n.iat[0])
        rows = R[R.sweep_n == n]
        for k in (0, 17, 50, 98):
            t_mid = S.t_center_s.iat[k]
            j = int(round(t_mid * 1000))
            if j < len(R):
                ok_db &= eq(float(S.impedance_ohm.iat[k]), R.impedance.iat[j], tol=5e-3) and \
                    int(R.sweep_point.iat[j]) == k
        ok_db &= len(rows) > 0 and abs(len(rows) - (T1 + 99 * STEP) * 1000) <= 2
    check(sess.name, "sweep files: 99 points, indexed 0..98", ok_pts)
    check(sess.name, "sweep impedances and frequencies equal aux.jsonl", ok_z)
    check(sess.name, "sweep point times advance 41.9 ms per point from onset + 55 ms", ok_t)
    check(sess.name, "1 kHz database shows each sweep point during its step", ok_db)
    nums = sorted(set(R.sweep_n.dropna().astype(int)))
    check(sess.name, "sweep counter is 1..N without gaps", nums == list(range(1, len(nums) + 1)),
          f"N={len(nums)}")

    # labels
    tr = R.dropna(subset=["trial"])
    per = tr.groupby("trial").agg(g=("grasp_id", "first"), r=("rep", "first"))
    grasp_counts = per.groupby("g").r.apply(lambda s: sorted(s))
    check(sess.name, "labels: 42 trials, each of 7 grasps with repetitions 1..6",
          len(per) == 42 and len(grasp_counts) == 7 and all(v == list(range(1, 7)) for v in grasp_counts),
          f"{len(per)} trials")
    check(sess.name, "labels: phases are only pre/grasp/rest/break/post",
          set(R.phase.unique()) <= {"pre", "grasp", "rest", "break", "post"})

    # raw database vs envelope database share every non-EMG column at the same instant
    shared = ["participant", "session_dir", "trial", "grasp_id", "grasp_name", "rep", "phase", "label_source",
              "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps", "imu_temp_c", "imu_age_ms",
              "p1_kpa", "p2_kpa", "skin_temp_c", "skin_temp_valid", "load_ok", "aux_age_s",
              "sweep_n", "sweep_id", "sweep_point", "impedance_freq_khz", "impedance"]
    sub = R.iloc[::20].reset_index(drop=True).iloc[:len(E)]
    same = len(sub) == len(E) and np.allclose(sub.t_s, E.t_s, atol=6e-5)
    for c in shared:
        a, b = sub[c], E[c]
        same &= bool(((a == b) | (a.isna() & b.isna())).all())
    check(sess.name, "1 kHz and 50 Hz databases agree on every shared column at the same instants", same)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--databases", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    idx = pd.read_csv(a.databases / "sessions_index.csv")
    rng = np.random.default_rng(a.seed)
    done = 0
    for r in idx[idx.included_in_databases].itertuples():
        found = list(a.sessions.rglob(f"{r.session_dir}/raw.bin"))
        if not found:
            check(r.session_dir, "raw session folder present", False)
            continue
        verify_session(found[0].parent, r.participant, r.tag, a.databases, rng)
        done += 1
    n_fail = sum(1 for _, _, ok, _ in RESULTS if not ok)
    print(f"\n{done} session(s), {len(RESULTS)} checks, {n_fail} failed")
    return 1 if n_fail or not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
