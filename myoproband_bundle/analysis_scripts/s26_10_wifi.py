"""Sabado 26: quitar la interferencia de la baliza Wi-Fi (ver wifi_clean.py).

Por sesion: fase y ventana del hundimiento (pliegue de 102.4 ms de los ocho
canales en las pausas relajadas), y para cada canal cuatro versiones del
crudo — original, notch, plantilla restada y reparada — con su espectro en
reposo (pausas, Welch 4096 = 0.24 Hz), su espectro en los sostenes (Welch
1024) y su RMS de 20-450 Hz en reposo y sosteniendo. Que el RMS del sostener
casi no cambie es lo que muestra que la limpieza no se come el sEMG.

Escribe data/s26/wifi_clean.csv y wifi_clean.npz (y un tramo de ejemplo).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s26_common import DATA26, SUBJECT_ORDER, bandpass_fft, find_sessions, load26, win
import wifi_clean as wc

REPS, N_TRIALS = 6, 42
METHODS = ["original", "notch", "template", "repair"]
EXAMPLE = ("S01", 0)                 # sesion y pausa del tramo de ejemplo
S = pd.read_csv(DATA26 / "sessions.csv")
full = S[S.complete].set_index("subject")
paths = {p.name: p for p in find_sessions()}


def grid(t_s, y, tg):
    """Como grid_raw: lineal sobre una rejilla de 1 kHz, NaN lejos de toda muestra real."""
    g = np.interp(tg, t_s, y)
    j = np.clip(np.searchsorted(t_s, tg), 1, len(t_s) - 1)
    near = np.minimum(np.abs(t_s[j] - tg), np.abs(t_s[j - 1] - tg))
    g[near > 0.005] = np.nan
    return g


def welch(seg, n):
    out, k = None, 0
    for j in range(0, seg.shape[-1] - n + 1, n // 2):
        w = seg[..., j:j + n]
        if not np.isfinite(w).all():
            continue
        w = w - w.mean(-1, keepdims=True)
        p = np.abs(np.fft.rfft(w * np.hanning(n), axis=-1)) ** 2
        out = p if out is None else out + p
        k += 1
    return out, k


def rms(B, sls):
    v = np.concatenate([B[..., s_] for s_ in sls], -1)
    return np.sqrt(np.nanmean(v ** 2, -1))


rows, arrays = [], {}
Pr_all = np.zeros((len(SUBJECT_ORDER), 8, len(METHODS), 2049))
Ph_all = np.zeros((len(SUBJECT_ORDER), 8, len(METHODS), 513))
for si, subj in enumerate(SUBJECT_ORDER):
    x = load26(paths[full.loc[subj, "session"]])
    t0 = x.s.t0_us
    rest = []
    for tr in x.trials:
        if tr.index % REPS == REPS - 1 and tr.index + 1 < N_TRIALS:
            rest.append((tr.t_rest + 7.0 + 4.6, x.trials[tr.index + 1].t_grasp - 2.5))
    holds = [(tr.t_grasp + 1.0, tr.t_rest - 0.5) for tr in x.trials]
    # fase del hundimiento con los ocho canales en reposo
    tpls = []
    chans = []
    for i in range(8):
        c = x.s.raw[i]
        t_us = c.ts_us.astype(np.int64)
        t_s = (t_us - t0) / 1e6
        y = c.mv.astype(float)
        m = np.zeros(len(t_s), bool)
        for a, b in rest:
            m |= (t_s >= a) & (t_s < b)
        tpl, _ = wc.fold(t_us[m].astype(float), wc.detrend(y[m]))
        tpls.append(tpl - np.median(tpl))
        chans.append((t_us, t_s, y, m))
    T8 = np.array(tpls)
    phase, lo, hi, depth = wc.dip_window(T8.mean(0))
    arrays[f"tpl_{subj}"] = T8
    arrays[f"win_{subj}"] = np.array([phase, lo, hi, depth])
    tg = np.arange(0.0, x.end_s, 1e-3)
    G = np.full((len(METHODS), 8, len(tg)), np.nan)
    frac_rep = np.zeros(8)
    for i, (t_us, t_s, y, m) in enumerate(chans):
        y_rep, bad = wc.repair(t_us.astype(float), y, lo, hi)
        y_tpl = wc.subtract_template(t_us.astype(float), y, T8[i])
        frac_rep[i] = bad.mean()
        G[0, i], G[2, i], G[3, i] = grid(t_s, y, tg), grid(t_s, y_tpl, tg), grid(t_s, y_rep, tg)
        if (subj, i) == (EXAMPLE[0], EXAMPLE[1]):
            a, b = rest[1][0] + 5.0, rest[1][0] + 5.0 + 0.6
            k = (t_s >= a) & (t_s < b)
            arrays.update(ex_t=t_s[k] - a, ex_y=y[k], ex_rep=y_rep[k], ex_bad=bad[k],
                          ex_info=np.array([si, i, a]))
    G[1] = wc.notch_grid(G[0])
    rest_sl = [win(tg, a, b) for a, b in rest]
    hold_sl = [win(tg, a, b) for a, b in holds]
    for mi, meth in enumerate(METHODS):
        B = bandpass_fft(G[mi])
        r_rms, h_rms = rms(B, rest_sl), rms(B, hold_sl)
        acc4, n4 = np.zeros((8, 2049)), 0
        for s_ in rest_sl:
            p, k = welch(G[mi][:, s_], 4096)
            if k:
                acc4 += p; n4 += k
        acc1, n1 = np.zeros((8, 513)), 0
        for s_ in hold_sl:
            p, k = welch(G[mi][:, s_], 1024)
            if k:
                acc1 += p; n1 += k
        Pr_all[si, :, mi], Ph_all[si, :, mi] = acc4 / n4, acc1 / n1
        f4 = np.fft.rfftfreq(4096, 1e-3)
        for i in range(8):
            rows.append(dict(subject=subj, channel=f"E{i+1}", method=meth, rest_rms_mv=float(r_rms[i]),
                             hold_rms_mv=float(h_rms[i]), lines_share_rest=float(wc.lines_share(f4, Pr_all[si, i, mi])),
                             frac_repaired=float(frac_rep[i]) if meth == "repair" else 0.0,
                             dip_phase_ms=phase, dip_lo_ms=lo, dip_hi_ms=hi, dip_depth_mv=depth))
    D = pd.DataFrame([r for r in rows if r["subject"] == subj])
    piv = D.pivot(index="channel", columns="method", values="lines_share_rest").median()
    print(f"{subj}: hundimiento {depth:+.1f} mV en {phase:.1f} ms (ventana {lo:.1f}-{hi:.1f} ms, "
          f"{100 * frac_rep.mean():.1f} % de muestras); lineas en reposo "
          + ", ".join(f"{m_} {100 * piv[m_]:.1f} %" for m_ in METHODS))

D = pd.DataFrame(rows)
D.to_csv(DATA26 / "wifi_clean.csv", index=False)
np.savez(DATA26 / "wifi_clean.npz", f4=np.fft.rfftfreq(4096, 1e-3), f1=np.fft.rfftfreq(1024, 1e-3),
         Pr=Pr_all, Ph=Ph_all, methods=np.array(METHODS), subjects=np.array(SUBJECT_ORDER), **arrays)
g = D.groupby("method", sort=False)
ref = D[D.method == "original"].set_index(["subject", "channel"])
for meth in METHODS[1:]:
    d = D[D.method == meth].set_index(["subject", "channel"])
    dh = 20 * np.log10(d.hold_rms_mv / ref.hold_rms_mv)
    dr = 20 * np.log10(d.rest_rms_mv / ref.rest_rms_mv)
    print(f"{meth:9s}: lineas en reposo {100 * d.lines_share_rest.median():.1f} % (original "
          f"{100 * ref.lines_share_rest.median():.1f} %); RMS en reposo {dr.median():+.2f} dB, "
          f"sosteniendo {dh.median():+.2f} dB (p10-p90 {dh.quantile(0.1):+.2f} a {dh.quantile(0.9):+.2f})")
