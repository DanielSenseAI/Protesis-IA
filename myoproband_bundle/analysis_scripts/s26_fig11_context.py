"""Figura 11 (sabado 26): las variables de contexto a lo largo de la sesion.

(a) carga de contacto por ensayo (la 3a lectura del sostener, la primera
    libre del filtro de mediana), p1 (lado flexor, E1) y p2 (lado extensor,
    E5), coloreada por el agarre del bloque; lecturas en 0 kPa marcadas.
    Cambia entre bloques, no dentro. En revision (--review) los seis paneles
    comparten la escala y, para comparar participantes.
(b) carga por agarre relativa a la mediana del participante (p2), para ver si
    el agarre ordena la carga igual en todos.
(c) temperatura de piel (tres sesiones con sensor) y del chip del IMU (seis),
    en funcion del tiempo, sin interpolar entre sostenes.
(d) quietud del antebrazo: RMS del giroscopio al agarrar, sostener, soltar y
    en reposo, por participante.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
from s26_common import DATA26, FIG26, GRASP_COLOR, GRASP_ORDER, GRASP_SHORT, SUBJECT_ORDER

fs.apply()
P = pd.read_csv(DATA26 / "sensors_trials.csv")
I = pd.read_csv(DATA26 / "imu_trials.csv")
S = pd.read_csv(DATA26 / "sessions.csv")
full = S[S.complete].set_index("subject")

# sin IMU (--no-imu) no hay panel (d) de quietud ni la temperatura del IMU en (c)
fig = plt.figure(figsize=(7.2, 5.9 if fs.NO_IMU else 8.2))
gs = fig.add_gridspec(2 if fs.NO_IMU else 3, 1, height_ratios=[1.35, 1.0] + ([] if fs.NO_IMU else [1.0]), hspace=0.45,
                      left=0.09, right=0.97, top=0.95, bottom=0.07 if not fs.NO_IMU else 0.09)
NC = 3 if len(SUBJECT_ORDER) <= 6 else 4          # columnas de la rejilla por sujeto
ga = gs[0].subgridspec(2, NC, hspace=0.35, wspace=0.22)
for j, s_ in enumerate(SUBJECT_ORDER):
    ax = fig.add_subplot(ga[j // NC, j % NC])
    p = P[P.subject == s_].sort_values("trial")
    for col, marker in (("p1", "o"), ("p2", "s")):
        # la 3a lectura de cada agarre: las dos primeras arrastran el filtro de
        # mediana de la placa auxiliar desde el agarre anterior
        v = p[f"{col}_2"]
        zero = (v <= 0.05) & v.notna()
        m = v.where(v > 0.05)
        # ensayos numerados desde 1, como en el texto (1-42)
        n_tr = p.trial + 1
        for g in GRASP_ORDER:
            k = (p.grasp == g).values & ~zero.values
            ax.plot(n_tr[k], m[k], marker, ms=2.4 if marker == "o" else 2.2,
                    mfc=GRASP_COLOR[g] if col == "p1" else "white", mec=GRASP_COLOR[g], mew=0.7, ls="none")
        if zero.any():
            ax.plot(n_tr[zero], np.zeros(zero.sum()), "x", ms=2.6, color="#e34948", mew=0.7)
    for b in range(1, 7):
        ax.axvline(b * 6 + 0.5, color=fs.GRID, lw=0.6, zorder=0)
    ax.set_title(s_, loc="left", fontsize=7, pad=2)
    ax.set_xlim(0, 43)
    ax.set_xticks([1, 7, 13, 19, 25, 31, 37, 42])
    ax.tick_params(labelsize=6)
    if j + NC >= len(SUBJECT_ORDER):     # sin panel debajo
        ax.set_xlabel("Trial", fontsize=6.5)
    if j % NC == 0:
        ax.set_ylabel("Contact load (kPa)", fontsize=6.5)
# la clave de los marcadores va sobre el titulo del primer panel (antes, como
# texto de la figura, pisaba los titulos S05 y S06)
fig.axes[0].text(0, 1.2, "last reading of each hold; filled: p1 (flexor side, E1); open: p2 (extensor side, E5, "
                 "provisional calibration); colour: grasp of the block; ×: 0 kPa", transform=fig.axes[0].transAxes,
                 fontsize=6.2, color=fs.INK2, va="bottom", ha="left")
panels = {"a": fig.axes[:len(SUBJECT_ORDER)]}
if fs.REVIEW:
    # una sola escala y para los seis participantes (desde 0: la carga es absoluta)
    top = np.nanmax(P[["p1_2", "p2_2"]].where(P[["p1_2", "p2_2"]] > 0.05).values)
    for j, ax in enumerate(fig.axes[:len(SUBJECT_ORDER)]):
        ax.set_ylim(-4, top * 1.06)
        if j % NC:
            plt.setp(ax.get_yticklabels(), visible=False)

gb = gs[1].subgridspec(1, 2, wspace=0.3, width_ratios=[1.1, 1.0])
ax = fig.add_subplot(gb[0])
# p1: el sensor con calibracion completa (p2 es un ajuste provisional que el
# firmware de la placa auxiliar marca como solo de referencia); 3a lectura
for j, s_ in enumerate(SUBJECT_ORDER):
    p = P[P.subject == s_]
    v = p["p1_2"].where(p["p1_2"] > 0.05)
    med = np.nanmedian(v)
    by = pd.Series(v.values, index=p.grasp.values).groupby(level=0).median()
    for gi, g in enumerate(GRASP_ORDER):
        if g in by.index and np.isfinite(by[g]):
            ax.plot(gi + (j - 2.5) * 0.08, 100 * (by[g] / med - 1), "o", ms=3, color=GRASP_COLOR[g], mew=0)
ax.axhline(0, color=fs.GRID, lw=0.8, zorder=0)
ax.set_xticks(range(len(GRASP_ORDER)))
ax.set_xticklabels([GRASP_SHORT[g] for g in GRASP_ORDER], fontsize=6, rotation=30, ha="right")
ax.set_ylabel("Difference from own\nmedian (%)")
ax.set_title("Contact load by grasp (p1, flexor side; one dot per participant)", loc="left", fontsize=7.2)
panels["b"] = ax

ax = fig.add_subplot(gb[1])
tones = dict(zip(SUBJECT_ORDER, fs.tones(len(SUBJECT_ORDER))))
for s_ in SUBJECT_ORDER:
    p = P[P.subject == s_].set_index("trial")
    i = I[(I.subject == s_) & (I.imu_temp > 5)]
    tt = p.loc[i.trial.values, "t_grasp"].values / 60.0
    if not fs.NO_IMU:
        ax.plot(tt, i.imu_temp, lw=0.9, color=tones[s_])
    if p[["temp_0", "temp_1", "temp_2"]].notna().any().any():
        # temperatura de piel: las lecturas de cada sostener en su instante, unidas solo dentro del sostener
        for trial, r in p.iterrows():
            pts = [(r.t_grasp + r[f"dt{k}"], r[f"temp_{k}"]) for k in range(3)
                   if np.isfinite(r.get(f"temp_{k}", np.nan)) and np.isfinite(r.get(f"dt{k}", np.nan))]
            if pts:
                xs_, ys_ = zip(*pts)
                ax.plot(np.array(xs_) / 60.0, ys_, "-", lw=0.6, color=tones[s_])
                ax.plot(np.array(xs_) / 60.0, ys_, "s", ms=1.6, color=tones[s_], mew=0)
n_skin = sum(bool(P[P.subject == s_][["temp_0", "temp_1", "temp_2"]].notna().any().any()) for s_ in SUBJECT_ORDER)
# sin IMU solo queda la piel: el eje se ajusta a ella y el rotulo de dentro
# quedaba sobre el marco, asi que la cuenta pasa al titulo
if not fs.NO_IMU:
    ax.text(0.2, 36.3, f"skin ({n_skin} sessions with a sensor)", fontsize=6, color=fs.INK)
    ax.text(0.2, 29.4, f"device: IMU die ({len(SUBJECT_ORDER)} sessions)", fontsize=6, color=fs.INK)
ax.set_xlabel("Time in session (min)")
ax.set_ylabel("Temperature (°C)")
ax.set_title(f"Skin temperature ({n_skin} sessions with a sensor)" if fs.NO_IMU else "Skin and device temperature",
             loc="left", fontsize=7.2)
panels["c"] = ax

if not fs.NO_IMU:
    ax = fig.add_subplot(gs[2])
    cols = [("gyr_onset_rms", "grasp onset"), ("gyr_hold_rms", "hold"), ("gyr_release_rms", "release"),
            ("gyr_rest_rms", "rest")]
    xs = np.arange(len(SUBJECT_ORDER))
    for k, (c, lbl) in enumerate(cols):
        med = I.groupby("subject")[c].median().reindex(SUBJECT_ORDER)
        q1 = I.groupby("subject")[c].quantile(0.25).reindex(SUBJECT_ORDER)
        q3 = I.groupby("subject")[c].quantile(0.75).reindex(SUBJECT_ORDER)
        xx = xs + (k - 1.5) * 0.17
        ax.vlines(xx, q1, q3, color=["#1f2328", "#57606a", "#8b949e", "#c5cbd1"][k], lw=1.4)
        ax.plot(xx, med, "o", ms=3.4, color=["#1f2328", "#57606a", "#8b949e", "#c5cbd1"][k], label=lbl)
    ax.set_xticks(xs)
    ax.set_xticklabels(SUBJECT_ORDER)
    ax.set_ylabel("Gyroscope RMS (°/s)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=6, ncol=4)
    # cambio de inclinacion de la banda entre sostenes (max - min): el valor absoluto
    # depende de como va montado el IMU
    tilt = (I.groupby("subject").incl_deg.max() - I.groupby("subject").incl_deg.min()).reindex(SUBJECT_ORDER)
    ax.set_title("Forearm stillness (median and IQR, 42 trials); band inclination change between holds: " +
                 ", ".join(f"{s_} {tilt[s_]:.0f}°" for s_ in SUBJECT_ORDER), loc="left", fontsize=7.2)
    panels["d"] = ax
fs.save(fig, FIG26 / f"{FIGP}_11_context", panels=panels)
print("ok")
