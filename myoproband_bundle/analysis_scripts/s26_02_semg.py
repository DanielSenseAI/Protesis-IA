"""Sabado 26: sEMG de los seis participantes, ensayo a ensayo.

Ventanas (reloj del equipo, s):
- sostener: [Pgrasp + 1.0, Prest - 0.5]  (3.5 s estables; sin reaccion ni suelta)
- reposo del ensayo: [Prest + 4.5, Prest + 5.3]  (despues del barrido, que acaba
  a +4.35 s, y antes de que el aviso del siguiente agarre, a +5.0 s, pueda moverlo)
- linea base: las pausas de 30 s, [inicio + 4.6, siguiente Pgrasp - 2.5]  (~23 s
  en reposo, sin barrido ni aviso). Es la referencia de las relaciones en dB.
- barrido en reposo: el Prest que abre cada pausa dispara un barrido con el
  sujeto ya relajado; [inicio + 0.3, inicio + 4.3] contra la linea base de esa
  misma pausa mide lo que el barrido deja en el sEMG sin actividad de por medio.

Crudo: rejilla de 1 kHz y pasa-banda 20-450 Hz de fase cero. Envolvente: la del
hardware (50 Hz), en mV del ADC, sin filtrar.

Escribe data/s26/semg_trials.csv (ensayo x canal), semg_breaks.csv (pausa x
canal) y semg_channels.csv (resumen por sujeto y canal).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s26_common import (DATA26, GRASP_NAMES, bandpass_fft, find_sessions, grid_raw, load26,
                        prest_times, win)

N_TRIALS = 42
REPS = 6


def rms(y):
    y = y[np.isfinite(y)]
    return float(np.sqrt(np.mean(y ** 2))) if len(y) > 50 else np.nan


def welch_mdf(y, fs=1000.0, n=256):
    y = y[np.isfinite(y)]
    if len(y) < 2 * n:
        return np.nan, np.nan
    segs = [y[j:j + n] for j in range(0, len(y) - n + 1, n // 2)]
    P = np.mean([np.abs(np.fft.rfft((s - s.mean()) * np.hanning(n))) ** 2 for s in segs], 0)
    f = np.fft.rfftfreq(n, 1 / fs)
    k = (f >= 20) & (f <= 450)
    c = np.cumsum(P[k])
    mdf = float(f[k][np.searchsorted(c, c[-1] / 2)])
    mnf = float(np.sum(f[k] * P[k]) / np.sum(P[k]))
    return mdf, mnf


rows_t, rows_b, rows_c = [], [], []
for path in find_sessions():
    x = load26(path)
    if len(x.trials) != N_TRIALS:
        continue
    tg, Y = grid_raw(x)
    B = bandpass_fft(Y)
    pr = prest_times(x)
    env_t = {i: (x.s.env[i].ts_us - x.s.t0_us) / 1e6 for i in range(8)}
    env_v = {i: x.s.env[i].mv for i in range(8)}

    def env_mean(i, a, b):
        m = (env_t[i] >= a) & (env_t[i] < b)
        return float(np.mean(env_v[i][m])) if m.sum() >= 5 else np.nan

    # Pausas: tras el ultimo ensayo de cada agarre (menos el final), el
    # corredor manda Prest a +7 s. Se toma el Prest detectado si esta cerca.
    breaks = []
    for tr in x.trials:
        if tr.index % REPS == REPS - 1 and tr.index + 1 < N_TRIALS:
            nominal = tr.t_rest + 7.0
            near = [p for p in pr if abs(p - nominal) < 0.6]
            b0 = near[0] if near else nominal
            b1 = x.trials[tr.index + 1].t_grasp
            breaks.append((tr.index // REPS, b0, b1))
    base = {i: [] for i in range(8)}
    ebase = {i: [] for i in range(8)}
    for blk, b0, b1 in breaks:
        for i in range(8):
            r_clean = rms(B[i, win(tg, b0 + 4.6, b1 - 2.5)])
            r_sweep = rms(B[i, win(tg, b0 + 0.3, b0 + 4.3)])
            seg = B[i, win(tg, b0 + 0.05, b0 + 0.45)]
            peak = float(np.nanmax(np.abs(seg))) if np.isfinite(seg).any() else np.nan
            base[i].append(r_clean)
            e_clean = env_mean(i, b0 + 4.6, b1 - 2.5)
            ebase[i].append(e_clean)
            rows_b.append(dict(session=x.name, subject=x.subject, block=blk, channel=f"E{i+1}",
                               t_break=round(b0, 3), rms_clean=r_clean, rms_sweep=r_sweep,
                               sweep_db=20 * np.log10(r_sweep / r_clean),
                               onset_peak=peak, onset_crest_db=20 * np.log10(peak / r_clean),
                               env_clean=e_clean, env_sweep=env_mean(i, b0 + 0.3, b0 + 4.3)))
    rbase = {i: float(np.nanmedian(base[i])) for i in range(8)}
    eb = {i: float(np.nanmedian(ebase[i])) for i in range(8)}
    for tr in x.trials:
        a, b = tr.t_grasp + 1.0, tr.t_rest - 0.5
        for i in range(8):
            h = B[i, win(tg, a, b)]
            cov = float(np.mean(np.isfinite(h))) if len(h) else 0.0
            r_h = rms(h) if cov > 0.95 else np.nan
            r_r = rms(B[i, win(tg, tr.t_rest + 4.5, tr.t_rest + 5.3)])
            mdf, mnf = welch_mdf(h) if cov > 0.95 else (np.nan, np.nan)
            e_h = env_mean(i, a, b)
            rows_t.append(dict(session=x.name, subject=x.subject, trial=tr.index, grasp=tr.grasp,
                               grasp_name=GRASP_NAMES.get(tr.grasp, str(tr.grasp)), rep=tr.rep,
                               block=tr.index // REPS, t_grasp=round(tr.t_grasp, 3), channel=f"E{i+1}",
                               coverage=round(cov, 3), rms_hold=r_h, rms_rest=r_r, rms_base=rbase[i],
                               act_db=20 * np.log10(r_h / rbase[i]) if np.isfinite(r_h) else np.nan,
                               rest_db=20 * np.log10(r_r / rbase[i]),
                               mdf_hz=mdf, mnf_hz=mnf,
                               env_hold=e_h, env_base=eb[i], env_delta=e_h - eb[i]))
    T = pd.DataFrame([r for r in rows_t if r["session"] == x.name])
    Bk = pd.DataFrame([r for r in rows_b if r["session"] == x.name])
    for i in range(8):
        ch = f"E{i+1}"
        t = T[T.channel == ch]
        bk = Bk[Bk.channel == ch]
        rows_c.append(dict(session=x.name, subject=x.subject, channel=ch,
                           base_rms_mv=rbase[i], hold_rms_med_mv=float(t.rms_hold.median()),
                           act_db_med=float(t.act_db.median()), act_db_p90=float(t.act_db.quantile(0.9)),
                           rest_db_med=float(t.rest_db.median()),
                           mdf_med=float(t.mdf_hz.median()),
                           sweep_db_med=float(bk.sweep_db.median()), sweep_db_max=float(bk.sweep_db.max()),
                           onset_crest_db_med=float(bk.onset_crest_db.median()),
                           env_base_mv=eb[i], env_delta_med_mv=float(t.env_delta.median())))
    c = pd.DataFrame([r for r in rows_c if r["session"] == x.name])
    print(f"{x.subject} ({x.name}): base {np.round(c.base_rms_mv.values, 1)} mV")
    print(f"    act dB mediana {np.round(c.act_db_med.values, 1)}   p90 {np.round(c.act_db_p90.values, 1)}")
    print(f"    reposo del ensayo dB {np.round(c.rest_db_med.values, 1)}   MDF {np.round(c.mdf_med.values, 0)} Hz")
    print(f"    barrido en pausa dB {np.round(c.sweep_db_med.values, 2)} (max {np.round(c.sweep_db_max.values, 1)})  "
          f"pico de arranque dB {np.round(c.onset_crest_db_med.values, 1)}")
    print(f"    envolvente base {np.round(c.env_base_mv.values, 0)} mV, delta sostener {np.round(c.env_delta_med_mv.values, 0)} mV")

pd.DataFrame(rows_t).to_csv(DATA26 / "semg_trials.csv", index=False)
pd.DataFrame(rows_b).to_csv(DATA26 / "semg_breaks.csv", index=False)
pd.DataFrame(rows_c).to_csv(DATA26 / "semg_channels.csv", index=False)
