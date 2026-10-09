#!/usr/bin/env python3
"""Build the synchronized databases from the raw session folders.

    python tools/build_databases.py --sessions raw/sessions --out databases

Outputs, one file per complete session unless noted:

    databases/raw_1khz/<tag>_raw_1khz.csv.gz                8 raw sEMG channels on a 1 kHz grid
    databases/envelope_50hz/<tag>_env_50hz.csv.gz           8 hardware-envelope channels on a 50 Hz grid
    databases/sweeps/<tag>/<tag>_sweep_NN.csv               one file per impedance sweep (99 points)
    (<tag> = participant + folder date and time, e.g. S01_20260926_113055)
    databases/sessions_index.csv                            every folder found, complete or not
    databases/participants.csv                              folder code <-> participant label

Both grid databases carry the same auxiliary columns (IMU, contact load,
skin temperature, impedance, trial/phase labels), so either one can be used on
its own. Everything is placed on the DEVICE clock (seconds since the first raw
sample), using the article's own code: `s26_common.load26` (label/trial
segmentation, host->device clock fit), `sweeps.phase_commands` and
`s26_common.sweep_onsets` (sweep onset detected in the sEMG). Nothing here
re-implements those; this script only resamples what they produce.

Resampling rule (as requested): zero-order HOLD, no interpolation. The value at
grid time t is the most recent sample at or before t; its age is stored so a
reader can mask stale values. `--emg-resample linear` reproduces the
interpolation used by the article's figures instead.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Sweep geometry, from the article's scripts (s26_05_sweeps.py, s26_fig7_showcase.py):
# the AD5933 applies a 1 kHz point that is discarded, then 99 points of 1 kHz steps
# (2..100 kHz). Point k is excited during [onset + T1 + k*STEP, + STEP).
T1, STEP, ONSET = 0.055, 0.0419, 0.15
N_POINTS = 99
N_TRIALS = 42                    # a session is complete if it has exactly this many trials
SENTINEL_TEMP_C = -126.0         # skin-temperature sensor absent reads -126.79 (see report)
MAX_PRED_LAG = (0.08, 0.30)      # accepted Prest -> sweep-onset latency (s), as in s26_fig7_showcase.py


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", type=Path, default=ROOT / "raw" / "sessions",
                    help="folder that contains the session folders (searched recursively)")
    ap.add_argument("--out", type=Path, default=ROOT / "databases")
    ap.add_argument("--dataset", default="s26-01",
                    help="analysis dataset: s26 (26 Sep, six participants) or s26-01 (adds 1 Oct)")
    ap.add_argument("--emg-resample", choices=("hold", "linear"), default="hold")
    ap.add_argument("--max-age-ms-raw", type=float, default=5.0,
                    help="raw sEMG is blank where the newest sample is older than this (article: 5 ms)")
    ap.add_argument("--max-age-ms-env", type=float, default=60.0,
                    help="envelope is blank where the newest sample is older than this (3 periods at 50 Hz)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these session folder names")
    return ap.parse_args(argv)


# ----------------------------------------------------------------- resampling
def hold_at(t_src: np.ndarray, v_src: np.ndarray, tg: np.ndarray, max_age: float):
    """Most recent sample at or before each grid time. Returns (values, age_s).

    values is NaN before the first sample and wherever age > max_age (None = never blank).
    age is NaN only before the first sample."""
    order = np.argsort(t_src, kind="stable")
    t, v = t_src[order], v_src[order].astype(np.float64)
    idx = np.searchsorted(t, tg, side="right") - 1
    before = idx < 0
    idx = np.clip(idx, 0, len(t) - 1)
    age = tg - t[idx]
    age[before] = np.nan
    val = v[idx].copy()
    val[before] = np.nan
    if max_age is not None:
        val[age > max_age] = np.nan
    return val, age


def linear_at(t_src, v_src, tg, max_age):
    """The article's grid_raw rule: linear interpolation, blank where the nearest sample is > max_age away."""
    order = np.argsort(t_src, kind="stable")
    t, v = t_src[order], v_src[order].astype(np.float64)
    y = np.interp(tg, t, v)
    j = np.clip(np.searchsorted(t, tg), 1, len(t) - 1)
    near = np.minimum(np.abs(t[j] - tg), np.abs(t[j - 1] - tg))
    y[near > max_age] = np.nan
    age = tg - t[np.clip(np.searchsorted(t, tg, side="right") - 1, 0, len(t) - 1)]
    return y, age


def stream_columns(chans: dict, t0_us: int, tg: np.ndarray, prefix: str, max_age: float, mode: str):
    """Eight channels on grid tg. Returns (columns dict, max-age-over-channels in µs)."""
    cols, ages = {}, []
    fn = linear_at if mode == "linear" else hold_at
    for i in range(8):
        c = chans[i]
        t = (c.ts_us - t0_us) / 1e6
        val, age = fn(t, c.counts, tg, max_age)
        # counts are integers when held; interpolated values (--emg-resample linear) are not
        cols[f"{prefix}_E{i + 1}"] = (pd.array(val, dtype="Int16") if mode != "linear" else np.round(val, 3))
        ages.append(age)
    worst = np.fmax.reduce(np.vstack(ages), axis=0)       # NaN only where no channel has a sample yet
    return cols, worst


def imu_columns(x, tg: np.ndarray, S):
    """IMU in physical units on grid tg, held. Same scaling as s26_common.imu_arrays."""
    imu = x.s.imu
    out = {k: np.full(len(tg), np.nan) for k in
           ("ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps", "imu_temp_c", "imu_age_ms")}
    if imu is None or len(imu) == 0:
        return out
    t = (imu["ts_us"].astype(np.int64) - x.s.t0_us) / 1e6
    order = np.argsort(t, kind="stable")
    t = t[order]
    spec = (("ax_g", "ax", S.ACC_G_PER_LSB), ("ay_g", "ay", S.ACC_G_PER_LSB), ("az_g", "az", S.ACC_G_PER_LSB),
            ("gx_dps", "gx", S.GYR_DPS_PER_LSB), ("gy_dps", "gy", S.GYR_DPS_PER_LSB),
            ("gz_dps", "gz", S.GYR_DPS_PER_LSB), ("imu_temp_c", "temp100", 0.01))
    for name, field, scale in spec:
        v = imu[field][order].astype(np.float64) * scale
        out[name], age = hold_at(t, v, tg, None)
    out["imu_age_ms"] = age * 1e3
    return out


def aux_columns(x, tg: np.ndarray):
    """Contact load and skin temperature, held from the estimated time of each reading (t_est)."""
    sen = x.sensors
    out = {k: np.full(len(tg), np.nan) for k in ("p1_kpa", "p2_kpa", "skin_temp_c", "load_ok", "aux_age_s")}
    if len(sen["t"]) == 0:
        out["skin_temp_valid"] = np.full(len(tg), np.nan)
        return out
    t = sen["t_est"]
    for name, key in (("p1_kpa", "p1"), ("p2_kpa", "p2"), ("skin_temp_c", "temp")):
        out[name], age = hold_at(t, sen[key], tg, None)
    out["load_ok"], _ = hold_at(t, sen["load_ok"].astype(float), tg, None)
    out["aux_age_s"] = age
    out["skin_temp_valid"] = np.where(np.isnan(out["skin_temp_c"]), np.nan,
                                      (out["skin_temp_c"] > SENTINEL_TEMP_C).astype(float))
    return out


# --------------------------------------------------------------------- labels
def label_columns(x, tg: np.ndarray, S):
    """trial / grasp / rep / phase from the article's segmentation (s26_common._segment)."""
    n = len(tg)
    trial = np.full(n, np.nan)
    grasp = np.full(n, np.nan)
    rep = np.full(n, np.nan)
    phase = np.full(n, "pre", dtype=object)
    seen: dict = {}
    for tr in x.trials:
        # Repetition = occurrence order of this grasp among the session's trials. It is derived
        # here rather than taken from Trial.rep: when only some trials have a logged label,
        # s26_common._segment numbers the others from 1 and can repeat a number
        # (1, 1, 2, 3, 4, 5). rep_mismatches() reports how often the two differ.
        seen[tr.grasp] = seen.get(tr.grasp, 0) + 1
        m = (tg >= tr.t_grasp) & (tg < tr.t_next)
        trial[m], grasp[m], rep[m] = tr.index + 1, tr.grasp, seen[tr.grasp]
        phase[m & (tg < tr.t_rest)] = "grasp"
        phase[m & (tg >= tr.t_rest)] = "rest"
    for b0, b1 in x.breaks:
        phase[(tg >= b0) & (tg < b1)] = "break"
    if x.trials:
        phase[tg >= x.trials[-1].t_next] = "post"
    gname = np.full(n, "", dtype=object)
    for g in np.unique(grasp[np.isfinite(grasp)]):
        if g >= 0:
            gname[grasp == g] = S.GRASP_NAMES.get(int(g), "")
    grasp[grasp < 0] = np.nan
    return dict(trial=trial, grasp_id=grasp, grasp_name=gname, rep=rep, phase=phase)


def rep_mismatches(x) -> int:
    """Trials whose repetition number in the article's segmentation differs from the occurrence order."""
    seen, bad = {}, 0
    for tr in x.trials:
        seen[tr.grasp] = seen.get(tr.grasp, 0) + 1
        bad += int(tr.rep != seen[tr.grasp])
    return bad


# --------------------------------------------------------------------- sweeps
def assign_sweeps(x, S):
    """Pair every impedance record with its onset and return a chronological list.

    Same rule as s26_fig7_showcase.sweep_for: the onset is the sEMG-detected one if it falls
    0.08-0.30 s after the Prest, else Prest + 0.15 s (the article's median); the record is the
    sweep that arrived 3.5-7.5 s after the Prest. Sweeps no Prest explains (the start-of-test
    sweep) have no onset in the sEMG: it is estimated as arrival - 4.9 s, the same window the
    article uses for them (s26_01_integrity / s26_05_sweeps)."""
    prest = S.prest_times(x)
    onsets = S.sweep_onsets(x)
    used, out = set(), []
    for p in prest:
        hit = [o for o in onsets if MAX_PRED_LAG[0] <= o - p <= MAX_PRED_LAG[1]]
        t_on, src = (hit[0], "semg") if hit else (p + ONSET, "nominal_prest+0.15s")
        k = next((i for i, w in enumerate(x.sweeps) if i not in used and 3.5 < w["t_arrival"] - p < 7.5), None)
        if k is None:
            continue
        used.add(k)
        out.append(dict(w=x.sweeps[k], t_on=float(t_on), source=src, prest=float(p)))
    for k, w in enumerate(x.sweeps):
        if k not in used:
            out.append(dict(w=w, t_on=float(w["t_arrival"] - 4.9), source="start_of_test_arrival-4.9s", prest=np.nan))
    out.sort(key=lambda r: r["t_on"])
    for n, r in enumerate(out, start=1):
        r["n"] = n
    return out


def sweep_columns(sweeps, tg: np.ndarray):
    n = len(tg)
    sw_n = np.full(n, np.nan)
    sw_id = np.full(n, np.nan)
    pt = np.full(n, np.nan)
    fq = np.full(n, np.nan)
    z = np.full(n, np.nan)
    for r in sweeps:
        w, a = r["w"], r["t_on"]
        npts = len(w["z"])
        end = a + T1 + npts * STEP
        m = (tg >= a) & (tg < end)
        if not m.any():
            continue
        sw_n[m] = r["n"]
        sw_id[m] = w.get("id") if w.get("id") is not None else np.nan
        ins = m & (tg >= a + T1)
        k = np.clip(np.floor((tg[ins] - (a + T1)) / STEP).astype(int), 0, npts - 1)
        pt[ins] = k
        fq[ins] = np.asarray(w["freq"])[k] / 1e3
        z[ins] = np.asarray(w["z"])[k]
    return dict(sweep_n=sw_n, sweep_id=sw_id, sweep_point=pt, impedance_freq_khz=fq, impedance=z)


def sweep_files(x, sweeps, t0_us: int, out_dir: Path, participant: str, tag: str):
    d = out_dir / tag
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in sweeps:
        w, a = r["w"], r["t_on"]
        k = np.arange(len(w["z"]))
        start = a + T1 + k * STEP
        df = pd.DataFrame({
            "participant": participant, "session_dir": x.name, "sweep_n": r["n"], "sweep_id": w.get("id"),
            "point": k, "freq_khz": np.asarray(w["freq"]) / 1e3, "impedance_ohm": np.asarray(w["z"]),
            "t_start_s": np.round(start, 4), "t_center_s": np.round(start + STEP / 2, 4),
            "t_end_s": np.round(start + STEP, 4),
            "device_ts_us_center": (t0_us + np.round((start + STEP / 2) * 1e6)).astype(np.int64),
            "sweep_onset_s": round(a, 4), "onset_source": r["source"],
            "prest_s": None if np.isnan(r["prest"]) else round(r["prest"], 4),
            "arrival_s": round(w["t_arrival"], 4),
        })
        p = d / f"{tag}_sweep_{r['n']:02d}.csv"
        write_csv(df, p, gz=False)
        paths.append(p)
    return paths


# ------------------------------------------------------------------------ io
def write_csv(df: pd.DataFrame, path: Path, gz: bool) -> None:
    """Deterministic output: fixed line endings, and a gzip header with no name or timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not gz:
        df.to_csv(path, index=False, lineterminator="\n")
        return
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as g:
            with io.TextIOWrapper(g, encoding="utf-8", newline="") as t:
                df.to_csv(t, index=False, lineterminator="\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt(df: pd.DataFrame, decimals: dict) -> pd.DataFrame:
    for c, d in decimals.items():
        if c in df:
            df[c] = df[c].round(d)
    return df


def session_tag(participant: str, session_dir: str) -> str:
    """S01_20260926_113055: participant plus the folder's start date and time (unique per folder)."""
    parts = session_dir.split("_")
    return f"{participant}_{parts[2]}_{parts[3]}"


def build_session(x, S, args, out: Path, participant: str):
    tag = session_tag(participant, x.name)
    t0 = x.s.t0_us
    sweeps = assign_sweeps(x, S)
    common_meta = {"participant": participant, "session_dir": x.name}
    lab = {"label_source": x.order_source}
    imu_dec = {"ax_g": 3, "ay_g": 3, "az_g": 3, "gx_dps": 1, "gy_dps": 1, "gz_dps": 1,
               "imu_temp_c": 2, "imu_age_ms": 3}

    def one(tg, prefix, max_age_ms, mode, label):
        cols, worst = stream_columns(getattr(x.s, "raw" if prefix == "raw" else "env"), t0, tg, prefix,
                                     max_age_ms / 1e3, mode)
        d = {**common_meta,
             "t_s": np.round(tg, 4),
             "device_ts_us": (t0 + np.round(tg * 1e6)).astype(np.int64),
             **{k: v for k, v in label_columns(x, tg, S).items()}, "label_source": lab["label_source"],
             **cols,
             ("emg_age_us" if prefix == "raw" else "env_age_us"): np.round(worst * 1e6),
             **imu_columns(x, tg, S), **aux_columns(x, tg), **sweep_columns(sweeps, tg)}
        df = pd.DataFrame(d)
        for c in ("trial", "grasp_id", "rep", "sweep_n", "sweep_id", "sweep_point"):
            df[c] = df[c].astype("Int64")
        age_col = "emg_age_us" if prefix == "raw" else "env_age_us"
        df[age_col] = df[age_col].astype("Int64")
        df["load_ok"] = df["load_ok"].astype("Int64")
        df["skin_temp_valid"] = df["skin_temp_valid"].astype("Int64")
        df = fmt(df, {**imu_dec, "aux_age_s": 3, "impedance": 2, "impedance_freq_khz": 1})
        sub = out / label
        write_csv(df, sub / f"{tag}_{'raw_1khz' if prefix == 'raw' else 'env_50hz'}.csv.gz", gz=True)
        return len(df)

    n_raw = one(np.arange(0.0, x.end_s, 1.0 / 1000.0), "raw", args.max_age_ms_raw, args.emg_resample, "raw_1khz")
    n_env = one(np.arange(0.0, x.end_s, 1.0 / 50.0), "env", args.max_age_ms_env, "hold", "envelope_50hz")
    files = sweep_files(x, sweeps, t0, out / "sweeps", participant, tag)
    return n_raw, n_env, sweeps, files


def main(argv=None) -> int:
    args = _args(argv)
    # The analysis modules read their configuration from the environment at import time.
    os.environ["EMG8_DATASET"] = args.dataset
    os.environ.setdefault("EMG8_OUT", str(args.out.parent / "_analysis_work"))
    sys.path.insert(0, str(ROOT / "analysis_scripts"))
    import s26_common as S  # noqa: E402

    paths = S.find_sessions(args.sessions)
    if args.only:
        paths = [p for p in paths if p.name in args.only]
    if not paths:
        sys.exit(f"no session folders (raw.bin) for dataset {args.dataset} under {args.sessions}")
    out = args.out
    index, parts = [], []
    for path in paths:
        x = S.load26(path)
        participant = x.subject
        complete = len(x.trials) == N_TRIALS
        row = dict(participant=participant, session_dir=path.name, folder_code=path.name.split("_")[0],
                   metadata_subject=x.s.meta.get("subject", ""),
                   start=pd.to_datetime(path.name.split("_")[2] + path.name.split("_")[3],
                                        format="%Y%m%d%H%M%S").isoformat(),
                   duration_s=round(x.end_s, 3), trials=len(x.trials), complete=complete,
                   included_in_databases=complete, label_source=x.order_source,
                   clock_fit_resid_ms=round(x.resid_ms, 2), sweep_records=len(x.sweeps),
                   rep_mismatches_vs_analysis=rep_mismatches(x),
                   raw_bin_sha256=sha256(path / "raw.bin"))
        row["tag"] = session_tag(participant, path.name)
        parts.append(dict(participant=participant, folder_code=row["folder_code"],
                          metadata_subject=row["metadata_subject"], session_dir=path.name,
                          label_origin="analysis alias (s26_common._DATASETS)" if path.name in S.CFG["alias"]
                          else "folder name"))
        if not complete:
            row.update(rows_raw_1khz=0, rows_env_50hz=0,
                       note=f"excluded: {len(x.trials)} trials, a complete session has {N_TRIALS}")
            index.append(row)
            print(f"{path.name}: {len(x.trials)} trials -> excluded (aborted start)")
            continue
        n_raw, n_env, sweeps, _ = build_session(x, S, args, out, participant)
        src = pd.Series([r["source"] for r in sweeps]).value_counts().to_dict()
        row.update(rows_raw_1khz=n_raw, rows_env_50hz=n_env, sweeps_in_databases=len(sweeps),
                   sweeps_onset_semg=src.get("semg", 0), sweeps_onset_nominal=src.get("nominal_prest+0.15s", 0),
                   sweeps_onset_start_of_test=src.get("start_of_test_arrival-4.9s", 0), note="")
        index.append(row)
        print(f"{path.name} -> {participant}: {n_raw} rows @1 kHz, {n_env} @50 Hz, "
              f"{len(sweeps)} sweeps {src}")
    done = [r for r in index if r["included_in_databases"]]
    for lab in sorted({r["participant"] for r in done}):
        n = sum(r["participant"] == lab for r in done)
        if n > 1:
            print(f"WARNING: participant {lab} has {n} complete sessions "
                  f"({[r['session_dir'] for r in done if r['participant'] == lab]}); files are kept apart by date/time.")
    for r in done:
        if r["participant"] == "S00":
            print(f"WARNING: {r['session_dir']} is complete and labelled S00 (folder code, no alias in "
                  f"s26_common._DATASETS). S00 was the pre-rename label of {'sS01'}; check participants.csv.")
    pd.DataFrame(index).sort_values(["start", "session_dir"]).to_csv(out / "sessions_index.csv", index=False,
                                                                       lineterminator="\n")
    pd.DataFrame(parts).sort_values("session_dir").to_csv(out / "participants.csv", index=False,
                                                          lineterminator="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
