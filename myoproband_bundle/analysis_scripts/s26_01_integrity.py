"""Sabado 26: integridad y desempeno de las ocho grabaciones del portatil 2.

Por sesion: tasa efectiva y continuidad de cada flujo (mismo criterio que
09_performance.py), dispersion de la trama de 8 canales, jitter del
anfitrion, protocolo completado (ensayos, sostenimientos, pausas), barridos
de impedancia y muestras de presion/temperatura recibidas, y cuadre de la
copia del PC: lo que metadata.json declara contra lo que hay en los archivos.

Lo que no se puede medir desde aqui: la SD (solo quedo la ultima sesion en la
tarjeta, ver WORKLOG 2026-09-27) y la bateria (no va en el WAL).

Escribe data/s26/{sessions,channels,trials,sweeps,sensors}.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import SAMPLE_DTYPE, IMU_DTYPE
from s26_common import (DATA26, GRASP_NAMES, channel_stats, find_sessions, load26,
                        sweep_onsets)

DATA26.mkdir(parents=True, exist_ok=True)
N_TRIALS = 42

def full_rate_start(ts):
    """Como en 09_performance.py: el flujo arranca con la vista previa de 1
    muestra/s y pasa a tasa completa despues del ultimo intervalo de >0.5 s
    dentro de los primeros 40 s. Ese tramo no es perdida."""
    dt = np.diff(ts)
    t = (ts - ts[0]) / 1e6
    k = np.where((dt > 500_000) & (t[:-1] < 40))[0]
    return int(k[-1] + 1) if len(k) else 0


rows_s, rows_c, rows_t, rows_w, rows_a = [], [], [], [], []
for path in find_sessions():
    x = load26(path)
    s = x.s
    k0 = full_rate_start(s.raw[0].ts_us)
    t_start = s.raw[0].ts_us[k0]
    preview_s = (t_start - s.raw[0].ts_us[0]) / 1e6
    per = {}
    for kind, chans in (("raw", s.raw), ("env", s.env)):
        for i, c in chans.items():
            st = channel_stats(c.ts_us[c.ts_us >= t_start])
            per[(kind, i)] = st
            rows_c.append(dict(session=x.name, subject=x.subject, kind=kind, channel=f"E{i+1}",
                               adc=c.adc, pin=c.pin, **st))
    imu = None
    if s.imu is not None and len(s.imu) > 10:
        ti = np.sort(s.imu["ts_us"].astype(np.int64))
        imu = channel_stats(ti[ti >= t_start])
        rows_c.append(dict(session=x.name, subject=x.subject, kind="imu", channel="IMU",
                           adc=-1, pin=-1, **imu))
    # Trama: desfase de cada crudo respecto a E1, muestra mas cercana
    ref = s.raw[0].ts_us[s.raw[0].ts_us >= t_start].astype(np.int64)
    ref = ref[1000:-1000] if len(ref) > 5000 else ref
    offs = []
    for i in range(8):
        tc = s.raw[i].ts_us.astype(np.int64)
        j = np.clip(np.searchsorted(tc, ref), 1, len(tc) - 1)
        near = np.where(np.abs(tc[j] - ref) < np.abs(tc[j - 1] - ref), tc[j], tc[j - 1])
        offs.append(near - ref)
    offs = np.array(offs)
    spread = offs.max(axis=0) - offs.min(axis=0)

    # Protocolo
    holds = np.array([t.t_rest - t.t_grasp for t in x.trials])
    tr0 = (s.raw[0].ts_us - s.t0_us) / 1e6
    for t in x.trials:
        n_in = np.count_nonzero((tr0 >= t.t_grasp) & (tr0 < t.t_rest))
        t.coverage = n_in / max(1.0, (t.t_rest - t.t_grasp) * 1000.0)
        rows_t.append(dict(session=x.name, subject=x.subject, trial=t.index, grasp=t.grasp,
                           coverage=round(t.coverage, 4),
                           grasp_name=GRASP_NAMES.get(t.grasp, str(t.grasp)), rep=t.rep,
                           t_grasp=round(t.t_grasp, 3), t_rest=round(t.t_rest, 3),
                           t_next=round(t.t_next, 3), hold_s=round(t.t_rest - t.t_grasp, 3)))

    # Barridos: llegada al PC y arranque visto en el sEMG
    ons = sweep_onsets(x)
    for k, w in enumerate(x.sweeps):
        z = w["z"]
        fr = w["freq"]
        near = [o for o in ons if -6.0 < w["t_arrival"] - o < 12.0]
        rows_w.append(dict(session=x.name, subject=x.subject, k=k, id=w["id"],
                           t_arrival=round(w["t_arrival"], 3), points=len(z),
                           finite=int(np.isfinite(z).sum()), f_min=fr.min(), f_max=fr.max(),
                           z10k=float(np.interp(10e3, fr, z)), z50k=float(np.interp(50e3, fr, z)),
                           z100k=float(z[-1]),
                           onset=round(max(near), 3) if near else np.nan))
    sen = x.sensors
    for k in range(len(sen["t"])):
        rows_a.append(dict(session=x.name, subject=x.subject, t=round(sen["t"][k], 3),
                           p1_kpa=sen["p1"][k], p2_kpa=sen["p2"][k], temp_c=sen["temp"][k]))
    dts = np.diff(sen["t"]) if len(sen["t"]) > 1 else np.array([np.nan])

    # Copia del PC: lo declarado en metadata.json contra los archivos
    wal = s.meta.get("wal", {})
    n_raw_file = (path / "raw.bin").stat().st_size / SAMPLE_DTYPE.itemsize
    n_imu_file = (path / "imu.bin").stat().st_size / IMU_DTYPE.itemsize
    n_ev_file = len(s.events)
    wal_ok = (wal.get("samples") == n_raw_file and wal.get("imu_samples") == n_imu_file
              and wal.get("events") == n_ev_file)

    temp_imu = s.imu["temp100"] / 100.0 if s.imu is not None else np.array([np.nan])
    raw_loss = [per[("raw", i)]["loss_pct"] for i in range(8)]
    complete = len(x.trials) == N_TRIALS
    rows_s.append(dict(
        session=x.name, subject=x.subject, start=x.name[-15:], complete=complete,
        span_s=round(s.duration_s, 1), preview_s=round(preview_s, 1),
        full_rate_s=round(per[("raw", 0)]["dur_s"], 1),
        raw_hz=round(np.median([per[("raw", i)]["rate_hz"] for i in range(8)]), 1),
        raw_hz_min=round(min(per[("raw", i)]["rate_hz"] for i in range(8)), 1),
        env_hz=round(np.median([per[("env", i)]["rate_hz"] for i in range(8)]), 2),
        imu_hz=round(imu["rate_hz"], 1) if imu else np.nan,
        raw_med_interval_us=round(np.median([per[("raw", i)]["med_us"] for i in range(8)]), 0),
        raw_p99_interval_us=round(np.median([per[("raw", i)]["p99_us"] for i in range(8)]), 0),
        raw_gaps=int(np.median([per[("raw", i)]["n_gaps"] for i in range(8)])),
        raw_gap_ms_max=round(max(per[("raw", i)]["max_gap_ms"] for i in range(8)), 1),
        raw_loss_pct=round(float(np.mean(raw_loss)), 4),
        raw_missing=round(float(np.sum([per[("raw", i)]["missing"] for i in range(8)])), 0),
        raw_samples=int(np.sum([per[("raw", i)]["n"] for i in range(8)])),
        env_loss_pct=round(float(np.mean([per[("env", i)]["loss_pct"] for i in range(8)])), 4),
        imu_loss_pct=round(imu["loss_pct"], 4) if imu else np.nan,
        nonmono=int(sum(per[k]["nonmono"] for k in per)),
        frame_spread_med_us=round(float(np.median(spread)), 0),
        frame_spread_p95_us=round(float(np.percentile(spread, 95)), 0),
        # parte de las tramas dentro del objetivo de la tabla de requisitos (<= 1 ms)
        frame_spread_le1ms_pct=round(100 * float(np.mean(spread <= 1000)), 1),
        chan_off_abs_p95_us=round(float(np.percentile(np.abs(offs[1:]), 95)), 0),
        host_jitter_ms=round(x.resid_ms, 1),
        trials=len(x.trials), order_from=x.order_source,
        trials_full_rate=int(sum(getattr(t, "coverage", 0) >= 0.95 for t in x.trials)),
        hold_med_s=round(float(np.median(holds)), 3) if len(holds) else np.nan,
        hold_min_s=round(float(holds.min()), 3) if len(holds) else np.nan,
        breaks=len(x.breaks),
        sweeps=len(x.sweeps), sweeps_99pts=int(sum(len(w["z"]) == 99 for w in x.sweeps)),
        sweep_onsets=len(ons),
        sensors=len(sen["t"]), sensor_dt_med_s=round(float(np.nanmedian(dts)), 2),
        sensor_dt_max_s=round(float(np.nanmax(dts)), 2),
        imu_temp_start=round(float(np.median(temp_imu[:500])), 2),
        imu_temp_end=round(float(np.median(temp_imu[-500:])), 2),
        wal_samples=wal.get("samples"), wal_imu=wal.get("imu_samples"), wal_events=wal.get("events"),
        file_samples=int(n_raw_file), file_imu=int(n_imu_file), file_events=n_ev_file,
        wal_consistent=wal_ok))
    r = rows_s[-1]
    print(f"{x.name}: {r['span_s']:6.1f} s (vista previa {r['preview_s']} s)  crudo {r['raw_hz']:6.1f} Hz (min {r['raw_hz_min']})  "
          f"env {r['env_hz']} Hz  IMU {r['imu_hz']} Hz  perdida {r['raw_loss_pct']:.4f} % "
          f"({r['raw_missing']:.0f} muestras, huecos {r['raw_gaps']}, max {r['raw_gap_ms_max']} ms)  "
          f"trama {r['frame_spread_med_us']:.0f}/{r['frame_spread_p95_us']:.0f} us  jitter {r['host_jitter_ms']} ms")
    print(f"    ensayos {r['trials']}/{N_TRIALS}, {r['trials_full_rate']} a tasa completa ({r['order_from']})  sostener {r['hold_med_s']} s  pausas {r['breaks']}  "
          f"barridos {r['sweeps']} ({r['sweeps_99pts']} con 99 puntos, {r['sweep_onsets']} arranques en el sEMG)  "
          f"sensores {r['sensors']} cada {r['sensor_dt_med_s']} s (max {r['sensor_dt_max_s']})  "
          f"IMU {r['imu_temp_start']}->{r['imu_temp_end']} C  WAL cuadra: {r['wal_consistent']}")

S = pd.DataFrame(rows_s)
S.to_csv(DATA26 / "sessions.csv", index=False)
pd.DataFrame(rows_c).to_csv(DATA26 / "channels.csv", index=False)
pd.DataFrame(rows_t).to_csv(DATA26 / "trials.csv", index=False)
pd.DataFrame(rows_w).to_csv(DATA26 / "sweeps.csv", index=False)
pd.DataFrame(rows_a).to_csv(DATA26 / "sensors.csv", index=False)
full = S[S.complete]
print(f"\ncompletas {len(full)}/{len(S)}; tiempo registrado {S.span_s.sum()/60:.1f} min "
      f"({full.full_rate_s.sum()/60:.1f} a tasa completa en las completas); muestras crudas {S.raw_samples.sum():,d}; "
      f"perdida cruda (completas) media ponderada {np.average(full.raw_loss_pct, weights=full.full_rate_s):.4f} %, "
      f"{full.raw_missing.sum():.0f} muestras en {int(full.raw_gaps.sum())} huecos; "
      f"ensayos a tasa completa {int(full.trials_full_rate.sum())}/{int(full.trials.sum())}")
