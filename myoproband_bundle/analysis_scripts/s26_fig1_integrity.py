"""Figura 1 (sabado 26): que se registro, de que modalidad y cuando.

Una banda por participante, en el orden en que se grabaron, con el
protocolo arriba y cada flujo debajo: sEMG (8 crudos + 8 envolventes), IMU,
barridos de impedancia, carga de contacto y temperatura de piel. A la derecha
las cifras de integridad de esa sesion.
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import DATASET, FIGP
from s26_common import (C_SWEEP, DATA26, FIG26, GRASP_COLOR, GRASP_ORDER, GRASP_SHORT, SUBJECT_ORDER,
                        find_sessions, load26, prest_times)

fs.apply()
S = pd.read_csv(DATA26 / "sessions.csv")
P = pd.read_csv(DATA26 / "sensors_trials.csv")
full = S[S.complete].set_index("subject")

# sin IMU (--no-imu en el corredor): una fila menos por participante
ROWS = ["Task", "sEMG"] + ([] if fs.NO_IMU else ["IMU"]) + ["Impedance", "Contact load", "Skin temp."]
FIG_H = (6.6 if fs.NO_IMU else 7.4) * (0.1 + 0.9 * len(SUBJECT_ORDER) / 6)
fig = plt.figure(figsize=(7.2, FIG_H))
# margenes en pulgadas, no en fraccion: la version sin IMU es mas baja y con
# fracciones fijas la etiqueta del eje x tocaba la leyenda de abajo
gs = fig.add_gridspec(len(SUBJECT_ORDER), 1, hspace=0.62, left=0.12, right=0.80,
                      top=1 - 0.70 / FIG_H, bottom=0.70 / FIG_H)
paths = {p.name: p for p in find_sessions()}

for r, subj in enumerate(SUBJECT_ORDER):
    name = full.loc[subj, "session"]
    x = load26(paths[name])
    s = x.s
    ax = fig.add_subplot(gs[r])
    H = len(ROWS)
    y = {k: H - 1 - i for i, k in enumerate(ROWS)}
    T = x.end_s / 60.0
    # --- protocolo
    for tr in x.trials:
        ax.add_patch(mpatches.Rectangle((tr.t_grasp / 60, y["Task"] - 0.32), (tr.t_rest - tr.t_grasp) / 60, 0.64,
                                        color=GRASP_COLOR[tr.grasp], lw=0))
    # --- sEMG: tasa completa, vista previa y huecos
    t0 = s.raw[0].ts_us
    tt = (t0 - s.t0_us) / 1e6
    dt = np.diff(tt)
    pv = float(full.loc[subj, "preview_s"])
    ax.add_patch(mpatches.Rectangle((pv / 60, y["sEMG"] - 0.28), (x.end_s - pv) / 60, 0.56, color=fs.INK, lw=0))
    if pv > 0:
        ax.add_patch(mpatches.Rectangle((0, y["sEMG"] - 0.28), pv / 60, 0.56, color=fs.GRID, lw=0))
    gaps = np.where((dt > 0.005) & (tt[:-1] > pv + 0.5))[0]
    for g in gaps:
        ax.add_patch(mpatches.Rectangle((tt[g] / 60, y["sEMG"] - 0.42), max(dt[g], 1.5) / 60, 0.84,
                                        color="#e34948", lw=0, zorder=3))
    lost = sum(dt[k] for k in gaps)
    # --- IMU: cobertura (huecos > 0.25 s en blanco)
    if "IMU" in y:
        ti = np.sort((s.imu["ts_us"].astype(np.int64) - s.t0_us) / 1e6)
        ti = ti[ti >= pv]
        cuts = np.where(np.diff(ti) > 0.25)[0]
        starts = np.r_[ti[0], ti[cuts + 1]]
        ends = np.r_[ti[cuts], ti[-1]]
        for a, b in zip(starts, ends):
            ax.add_patch(mpatches.Rectangle((a / 60, y["IMU"] - 0.28), (b - a) / 60, 0.56, color=fs.INK2, lw=0))
    # --- barridos: ventana [Prest + 0.12, + 4.35], rayada
    for p in prest_times(x):
        ax.add_patch(mpatches.Rectangle((p / 60 + 0.12 / 60, y["Impedance"] - 0.3), 4.23 / 60, 0.6,
                                        facecolor="none", edgecolor=C_SWEEP, hatch="////", lw=0.0))
        ax.add_patch(mpatches.Rectangle((p / 60 + 0.12 / 60, y["Impedance"] - 0.3), 4.23 / 60, 0.6,
                                        facecolor=C_SWEEP, alpha=0.35, lw=0))
    # el barrido de arranque (START de la placa auxiliar, sin Prest): ventana
    # estimada desde su llegada, que trae el latido de 1 s del reenvio
    prs = prest_times(x)
    for w in x.sweeps:
        if not any(3.5 < w["t_arrival"] - p_ < 7.5 for p_ in prs):
            a_, b_ = max(0.0, w["t_arrival"] - 4.9), w["t_arrival"] - 0.7
            for kw in (dict(facecolor="none", edgecolor=C_SWEEP, hatch="////", lw=0.0),
                       dict(facecolor=C_SWEEP, alpha=0.35, lw=0)):
                ax.add_patch(mpatches.Rectangle((a_ / 60, y["Impedance"] - 0.3), (b_ - a_) / 60, 0.6, **kw))
    # --- carga de contacto y temperatura
    sen = x.sensors
    ok = (sen["p1"] > 0.05) & (sen["p2"] > 0.05)
    ax.plot(sen["t_est"][ok] / 60, np.full(ok.sum(), y["Contact load"]), "|", ms=5, mew=0.8, color=fs.INK)
    ax.plot(sen["t_est"][~ok] / 60, np.full((~ok).sum(), y["Contact load"]), "|", ms=5, mew=0.8, color="#e34948")
    tv = sen["temp"] > -126
    if tv.any():
        ax.plot(sen["t_est"][tv] / 60, np.full(tv.sum(), y["Skin temp."]), "|", ms=5, mew=0.8, color=fs.INK)
    else:
        ax.add_patch(mpatches.Rectangle((0, y["Skin temp."] - 0.28), T, 0.56, facecolor="#f3f4f6",
                                        edgecolor=fs.GRID, hatch="////", lw=0.0))
        ax.text(T / 2, y["Skin temp."], "sensor not detected (reads \u2212126.8 \u00b0C)", fontsize=5.8, color=fs.INK2,
                ha="center", va="center", bbox=dict(facecolor="#f3f4f6", edgecolor="none", pad=0.4))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(-0.6, H - 0.4)
    ax.set_yticks([y[k] for k in ROWS])
    ax.set_yticklabels(ROWS, fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{subj}  \u00b7  started {name[-6:-4]}:{name[-4:-2]}", loc="left", fontsize=7.5, pad=2,
                 fontweight="bold")
    if r < len(SUBJECT_ORDER) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time in session (min)")
    row = full.loc[subj]
    unl = int((~ok).sum())
    txt = (f"sEMG {row.raw_hz:.1f} Hz/ch\n"
           + (f"loss {row.raw_loss_pct:.2f} % ({lost:.1f} s gap)\n" if len(gaps) else f"loss {row.raw_loss_pct:.2f} %\n")
           + ("" if fs.NO_IMU else f"IMU {row.imu_hz:.0f} Hz\n")
           + f"{int(row.trials)}/42 trials\n"
           f"{int(row.sweeps_99pts)}/{int(row.sweeps)} sweeps\n"
           + f"{int(row.sensors)} load" + (f" ({unl} at 0 kPa)" if unl else "")
           + ("" if tv.any() else ", no temp."))
    ax.text(1.015, 0.5, txt, transform=ax.transAxes, fontsize=6.3, color=fs.INK, va="center", ha="left",
            linespacing=1.35)

# Leyenda de agarres (orden de presentacion) y de marcas
handles = [mpatches.Patch(color=GRASP_COLOR[g], label=GRASP_SHORT[g]) for g in GRASP_ORDER]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.11, 1 - 0.037 / FIG_H), ncol=7, fontsize=6.3,
           handlelength=1.2, columnspacing=1.0, title="Hold, by grasp (presentation order)" if DATASET == "s26" else "Hold, by grasp", title_fontsize=6.5)
extra = [mpatches.Patch(facecolor=C_SWEEP, alpha=0.5, hatch="////", edgecolor=C_SWEEP, label="impedance sweep"),
         mpatches.Patch(color=fs.GRID, label="1 Hz start-up preview"),
         mpatches.Patch(color="#e34948", label="sEMG dropout / contact sensor at 0 kPa")]
if not fs.NO_IMU:
    extra.append(mpatches.Patch(color=fs.INK2, label="IMU (breaks = no sample for > 0.25 s)"))
fig.legend(handles=extra, loc="lower left", bbox_to_anchor=(0.11, -0.089 / FIG_H), ncol=2, fontsize=6.3,
           handlelength=1.4)
FIG26.mkdir(parents=True, exist_ok=True)
fs.save(fig, FIG26 / f"{FIGP}_1_integrity")
print("ok")
