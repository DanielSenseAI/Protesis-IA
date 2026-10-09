"""Figura 13 (sabado 26): como se mide el retardo de la envolvente del hardware.

Un sostener de ejemplo con el mismo codigo que produce la tabla (envlag.py):
(a) crudo filtrado 20-450 Hz y su RMS movil centrado de 50 ms, con las tres
    ventanas de nivel (reposo, meseta, reposo tras soltar);
(b) la envolvente del hardware (50 Hz) con las mismas ventanas;
(c, d) ambas normalizadas a su propio reposo (0) y meseta (1) alrededor del
    agarre y de soltar: retardo = distancia entre los cruces sostenidos del 50 %;
(e) correlacion cruzada de todo el sostener: el desplazamiento que mejor alinea
    la envolvente con el RMS;
(f) los tres retardos en todos los sostenes y canales que califican.
Uso: python s26_fig13_envlag.py [sujeto ensayo canal]   (indice de ensayo desde 0;
por omision S05 25 E8: activacion alta y los tres retardos en su mediana)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
import envlag
from s26_common import DATA26, FIG26, bandpass_fft, find_sessions, grid_raw, load26, win

fs.apply()
SUBJ, TRIAL, CH = (sys.argv[1], int(sys.argv[2]), sys.argv[3]) if len(sys.argv) > 3 else ("S05", 25, "E8")
i = int(CH[1:]) - 1
S = pd.read_csv(DATA26 / "sessions.csv")
full = S[S.complete].set_index("subject")
LG = pd.read_csv(DATA26 / "envelope_lag.csv")
x = load26({p.name: p for p in find_sessions()}[full.loc[SUBJ, "session"]])
tr = x.trials[TRIAL]
tg, Y = grid_raw(x)
B = bandpass_fft(Y)
rr_all = envlag.moving_rms(B[i:i + 1])[0]
sl = win(tg, tr.t_grasp - 1.0, tr.t_rest + 3.0)
tt, rr, raw = tg[sl], rr_all[sl], B[i, sl]
te = (x.s.env[i].ts_us - x.s.t0_us) / 1e6
m = (te >= tr.t_grasp - 1.0) & (te < tr.t_rest + 3.0)
ee_t, ee = te[m], x.s.env[i].mv[m]
L = envlag.trial_lag(tt, rr, ee_t, ee, tr.t_grasp, tr.t_rest)
assert L is not None, "el ensayo de ejemplo no califica"
row = LG[(LG.subject == SUBJ) & (LG.trial == TRIAL) & (LG.channel == CH)]
assert len(row) == 1 and abs(row.lag_xcorr_ms.iloc[0] - 1000 * L["lag_xcorr"]) < 0.01, "la figura no coincide con la tabla"

g0, r0 = tr.t_grasp, tr.t_rest
C_RMS, C_ENV = fs.INK, fs.C_ENV
WIN_C = "#eef1f4"


def windows(ax):
    for a, b in ((g0 + envlag.W_REST[0], g0 + envlag.W_REST[1]), (g0 + envlag.W_PLATEAU[0], r0 + envlag.W_PLATEAU[1]),
                 (r0 + envlag.W_AFTER[0], r0 + envlag.W_AFTER[1])):
        ax.axvspan(a - g0, b - g0, color=WIN_C, lw=0, zorder=0)
    for v in (0, r0 - g0):
        ax.axvline(v, color=fs.INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)


def levels(ax, lo, hi, lo2):
    for lv, a, b in ((lo, envlag.W_REST[0], envlag.W_REST[1]),
                     (hi, envlag.W_PLATEAU[0], r0 - g0 + envlag.W_PLATEAU[1]),
                     (lo2, r0 - g0 + envlag.W_AFTER[0], r0 - g0 + envlag.W_AFTER[1])):
        ax.plot([a, b], [lv, lv], color=fs.C_SWEEP, lw=1.4, zorder=4)


fig = plt.figure(figsize=(7.2, 8.6))
gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.75, 1.0, 1.0], hspace=0.7, wspace=0.3, left=0.09, right=0.98,
                      top=0.96, bottom=0.06)
panels = {}

# (a) crudo y RMS movil
ax = fig.add_subplot(gs[0, :])
windows(ax)
ax.plot(tt - g0, raw, lw=0.35, color="#b8bec5", zorder=2)
ax.plot(tt - g0, rr, lw=1.1, color=C_RMS, zorder=3)
levels(ax, L["r_lo"], L["r_hi"], L["r_lo2"])
lim = 1.15 * np.nanpercentile(np.abs(raw), 99.8)
ax.set_ylim(-lim, lim)
ax.set_xlim(-1.0, r0 - g0 + 3.0)
ax.set_ylabel("mV at ADC input")
ax.set_title(f"(1) Raw {CH} band-passed 20–450 Hz (grey) and its centred {envlag.RMS_MS} ms moving RMS (black); "
             "orange: median level in each shaded window", loc="left", fontsize=7.2)
for xc, s_ in ((np.mean(envlag.W_REST), "rest"),
               ((envlag.W_PLATEAU[0] + r0 - g0 + envlag.W_PLATEAU[1]) / 2, "plateau"),
               (r0 - g0 + np.mean(envlag.W_AFTER), "rest after release")):
    ax.text(xc, lim * 0.92, s_, ha="center", va="top", fontsize=6.3, color=fs.INK2)
plt.setp(ax.get_xticklabels(), visible=False)
panels["a"] = ax

# (b) envolvente del hardware
ax = fig.add_subplot(gs[1, :], sharex=panels["a"])
windows(ax)
ax.plot(ee_t - g0, ee, lw=0.9, color=C_ENV, zorder=3)
ax.plot(ee_t - g0, ee, "o", ms=1.8, color=C_ENV, mew=0, zorder=3)
levels(ax, L["e_lo"], L["e_hi"], L["e_lo2"])
ax.set_ylabel("Envelope (mV)")
ax.set_xlabel("Time from grasp cue (s)   ·   dashed: grasp and release cues")
ax.set_title("(2) Hardware envelope of the same channel, 50 Hz (dots = samples), same windows and levels",
             loc="left", fontsize=7.2)
panels["b"] = ax


def norm(y, lo, hi):
    return (y - lo) / (hi - lo)


def zoom(ax, t_ref, a, b, lo_r, lo_e, cr, ce, rising, title):
    k = (tt - t_ref >= a) & (tt - t_ref <= b)
    ke = (ee_t - t_ref >= a) & (ee_t - t_ref <= b)
    ax.axhline(0.5, color=fs.GRID, lw=0.8, zorder=0)
    ax.plot(tt[k] - t_ref, norm(rr[k], lo_r, L["r_hi"]), color=C_RMS, lw=1.1, label="raw RMS")
    ax.plot(ee_t[ke] - t_ref, norm(ee[ke], lo_e, L["e_hi"]), color=C_ENV, lw=1.0, marker="o", ms=2.0, mew=0,
            label="hardware envelope")
    for tc, c in ((cr, C_RMS), (ce, C_ENV)):
        # el tramo que tiene que sostenerse del otro lado del 50 %
        ax.plot([tc - t_ref, tc - t_ref + envlag.SUSTAIN_S], [0.5, 0.5], color=c, lw=3.0, alpha=0.25,
                solid_capstyle="butt", zorder=4)
        ax.plot(tc - t_ref, 0.5, "o", ms=5, mfc="white", mec=c, mew=1.3, zorder=5)
    y_ar = -0.12 if rising else 1.12
    ax.annotate("", xy=(ce - t_ref, y_ar), xytext=(cr - t_ref, y_ar),
                arrowprops=dict(arrowstyle="<->", color=fs.INK, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(max(cr, ce) - t_ref + 0.04, y_ar, f"{1000 * (ce - cr):.0f} ms", ha="left", va="center", fontsize=6.5,
            color=fs.INK)
    ax.set_xlim(a, b)
    # el rango y sale de los datos de la ventana: con un tope fijo (1.45) la subida del agarre
    # quedaba cortada arriba
    vals = np.concatenate([norm(rr[k], lo_r, L["r_hi"]), norm(ee[ke], lo_e, L["e_hi"])])
    ax.set_ylim(min(-0.3, np.nanmin(vals) - 0.08), max(1.45, np.nanmax(vals) + 0.08))
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["rest", "50 %", "plateau"], fontsize=6.5)
    ax.set_title(title, loc="left", fontsize=7.2)


ax = fig.add_subplot(gs[2, 0])
zoom(ax, g0, min(-0.3, L["r_on"] - g0 - 0.5), max(0.9, L["e_on"] - g0 + 0.5), L["r_lo"], L["e_lo"], L["r_on"],
     L["e_on"], True, f"(3) At the grasp: sustained 50 % crossings")
ax.set_xlabel("Time from grasp cue (s)")
ax.legend(loc="upper left", fontsize=6.0, handlelength=1.2, borderaxespad=0.2)
panels["c"] = ax
ax = fig.add_subplot(gs[2, 1])
zoom(ax, r0, min(-0.4, L["r_off"] - r0 - 0.5), max(0.8, L["e_off"] - r0 + 0.5), L["r_lo2"], L["e_lo2"], L["r_off"],
     L["e_off"], False, "(4) At the release")
ax.set_xlabel("Time from release cue (s)")
panels["d"] = ax

# (e) correlacion cruzada del ejemplo
ax = fig.add_subplot(gs[3, 0])
lags_ms, c = 1000 * L["xcorr_lags"], L["xcorr_c"]
ax.plot(lags_ms, c, color=fs.INK, lw=1.0)
ax.axvline(0, color=fs.GRID, lw=0.8, zorder=0)
ax.plot(1000 * L["lag_xcorr"], L["xcorr_r"], "o", ms=5, mfc="white", mec=C_ENV, mew=1.3)
ax.text(1000 * L["lag_xcorr"] + 15, L["xcorr_r"] + 0.07, f"peak at {1000 * L['lag_xcorr']:.0f} ms, r = {L['xcorr_r']:.2f}",
        va="bottom", fontsize=6.5, color=fs.INK)
ax.set_xlim(lags_ms[0], lags_ms[-1])
ax.set_ylim(min(c.min(), 0) - 0.05, 1.2)
ax.set_xlabel("Envelope shifted back by (ms)")
ax.set_ylabel("Correlation with raw RMS")
ax.set_title(f"(5) Whole hold, grasp − 1 s to release + 2 s", loc="left", fontsize=7.2)
panels["e"] = ax

# (f) todos los sostenes
ax = fig.add_subplot(gs[3, 1])
rng = np.random.default_rng(2)
cols = (("lag_on_ms", "grasp\n50 %"), ("lag_off_ms", "release\n50 %"), ("lag_xcorr_ms", "cross-\ncorrelation"))
meds = []
for j, (col, _) in enumerate(cols):
    v = LG[col].dropna().values
    ax.plot(j + rng.uniform(-0.22, 0.22, len(v)), v, "o", ms=1.5, color=C_ENV, alpha=0.4, mew=0)
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([j + 0.33, j + 0.33], [q1, q3], color=fs.INK, lw=1.6)
    ax.plot([j + 0.27, j + 0.39], [q2, q2], color=fs.INK, lw=1.6)
    meds.append((q2, len(v)))
ax.axhline(0, color=fs.GRID, lw=0.8, zorder=0)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([c_[1] for c_ in cols], fontsize=6.5)
ax.set_xlim(-0.5, 2.6)
ax.set_ylim(-250, 450)
ax.set_ylabel("Envelope lag (ms)")
out = sum(int((LG[c_].dropna() > 450).sum() + (LG[c_].dropna() < -250).sum()) for c_, _ in cols)
ax.set_title("(6) All qualifying holds: medians " + ", ".join(f"{q:.0f}" for q, _ in meds) + " ms\n"
             f"(n = {meds[0][1]}, {meds[1][1]}, {meds[2][1]}; {out} outside the axis)", loc="left", fontsize=7.2)
panels["f"] = ax

fs.save(fig, FIG26 / f"{FIGP}_13_envlag", panels=panels)
print(f"ok {SUBJ} ensayo {TRIAL + 1} {CH}: {1000 * L['lag_on']:.0f} ms al agarrar, {1000 * L['lag_off']:.0f} ms al soltar, "
      f"{1000 * L['lag_xcorr']:.0f} ms por correlacion (r = {L['xcorr_r']:.2f})")
