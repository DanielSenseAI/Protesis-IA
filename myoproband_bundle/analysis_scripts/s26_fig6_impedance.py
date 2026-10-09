"""Figura 6 (sabado 26): impedancia piel-electrodo en los seis participantes.

(a) los 49 barridos de cada sesion, coloreados por el momento de la sesion
    (claro = inicio, oscuro = final);
(b) |Z| a 5 kHz relativo al primer barrido tras el inicio: el asentamiento de
    la interfaz;
(c) |Z| a 100 kHz en los 294 barridos: el mismo valor en todos los
    participantes. Pasado el codo (10-20 kHz; hasta ~80 kHz en los primeros
    barridos de S02) se mide la cadena de medida, no la piel.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
from s26_common import C_SWEEP, DATA26, FIG26, SUBJECT_ORDER

fs.apply()
Z = np.load(DATA26 / "impedance.npz")
S = pd.read_csv(DATA26 / "sessions.csv")
full = S[S.complete].set_index("subject")
f = Z["freq"] / 1e3
cmap = cm.get_cmap("Oranges")
norm = Normalize(vmin=-4, vmax=12)       # que el primer barrido no quede blanco

fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.38, left=0.08, right=0.97, top=0.95, bottom=0.08)
NC = 3 if len(SUBJECT_ORDER) <= 6 else 4          # columnas de la rejilla por sujeto
top = gs[0].subgridspec(2, NC, wspace=0.12, hspace=0.3)
ax0 = None
panels = {"a": []}
floor = np.median(np.concatenate([Z[s_][:, f >= 95].ravel() for s_ in SUBJECT_ORDER])) / 1e3
for j, subj in enumerate(SUBJECT_ORDER):
    ax = fig.add_subplot(top[j // NC, j % NC], sharex=ax0, sharey=ax0)
    ax0 = ax0 or ax
    panels["a"].append(ax)
    zz = Z[subj] / 1e3
    tt = Z[subj + "_t"] / 60.0
    for k in np.argsort(tt):
        ax.plot(f, zz[k], lw=0.6, color=cmap(norm(tt[k])))
    ax.axhline(floor, color=fs.INK2, lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2, 100)
    ax.set_ylim(5, 200)
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "200"])
    nm = full.loc[subj, "session"]
    ax.text(0.97, 0.95, f"{subj} · {nm[-6:-4]}:{nm[-4:-2]}", transform=ax.transAxes, ha="right", va="top",
            fontsize=7, fontweight="bold")
    if j % NC:
        plt.setp(ax.get_yticklabels(), visible=False)
    else:
        ax.set_ylabel("|Z| (kΩ)")
    if j + NC < len(SUBJECT_ORDER):      # hay otro panel debajo
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel("Frequency (kHz)")
    if j == 0:
        ax.text(40, floor * 1.08, f"{floor:.2f} kΩ floor", fontsize=5.8, color=fs.INK2, va="bottom",
                ha="center")
sm = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=11.4), cmap=cmap)
if len(SUBJECT_ORDER) < 2 * NC:
    # rejilla con un hueco (siete sujetos en 2 x 4): la barra va en el hueco, no a la derecha,
    # donde chocaba con la escala del panel de arriba
    pos = top[1, NC - 1].get_position(fig)
    cax = fig.add_axes([pos.x0 + 0.03, pos.y0 + 0.1 * pos.height, 0.010, 0.8 * pos.height])
else:
    cax = fig.add_axes([0.975, 0.62, 0.008, 0.28])
cb = fig.colorbar(sm, cax=cax)
cb.set_label("time in session (min)", fontsize=6)
cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_visible(False)

bot = gs[1].subgridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.3)
ax = fig.add_subplot(bot[0])
j5 = int(np.argmin(np.abs(f - 5)))
tones = dict(zip(SUBJECT_ORDER, fs.tones(len(SUBJECT_ORDER))))
ends = []
for subj in SUBJECT_ORDER:
    tt = Z[subj + "_t"]
    zz = Z[subj][:, j5]
    k = np.isfinite(tt) & (tt > 1.0)
    o = np.argsort(tt[k])
    t_, z_ = tt[k][o] / 60.0, zz[k][o]
    rel = 100 * (z_ / z_[0] - 1)
    ax.plot(t_, rel, lw=1.0, color=tones[subj], marker="o", ms=1.6)
    ends.append((rel[-1], subj, t_[-1]))
# etiquetas directas al final, separadas
ends.sort()
ys = np.array([e[0] for e in ends], float)
# separacion minima en unidades del eje: 7 % bastaba con seis curvas; con mas, el eje crece y
# hace falta la que corresponde a la altura del texto
lo_y, hi_y = ax.get_ylim()
gap = max(7.0, 0.065 * (hi_y - lo_y)) if len(ends) > 6 else 7.0
for i in range(1, len(ys)):
    ys[i] = max(ys[i], ys[i - 1] + gap)
for (v, subj, tl), yl in zip(ends, ys):
    ax.text(tl + 0.15, yl, f"{subj} {fs.num(v, '+.0f')} %", fontsize=6, va="center", color=fs.INK)
ax.axhline(0, color=fs.GRID, lw=0.8, zorder=0)
# sitio a la derecha para los rotulos finales de la sesion mas larga
ax.set_xlim(0, max(e[2] for e in ends) + 2.1)
ax.set_xlabel("Time in session (min)")
ax.set_ylabel("|Z| at 5 kHz, change from\nfirst sweep (%)")
ax.set_title("Skin–electrode settling during the session", loc="left", fontsize=7.5)
panels["b"] = ax

ax = fig.add_subplot(bot[1])
j50 = len(f) - 1                      # 100 kHz
for r, subj in enumerate(SUBJECT_ORDER):
    v = Z[subj][:, j50] / 1e3
    jit = (np.random.default_rng(r).random(len(v)) - 0.5) * 0.4
    ax.plot(v, r + jit, "o", ms=1.8, color=C_SWEEP, alpha=0.6, mew=0)
ax.set_yticks(range(len(SUBJECT_ORDER)))
ax.set_yticklabels(SUBJECT_ORDER, fontsize=6.8)
ax.invert_yaxis()
allv = np.concatenate([Z[s_][:, j50] for s_ in SUBJECT_ORDER]) / 1e3
ax.set_xlim(allv.min() - 0.05, allv.max() + 0.05)
ax.set_xlabel("|Z| at 100 kHz (kΩ)")
ax.set_title(f"Same floor in every participant\n({allv.min():.2f}–{allv.max():.2f} kΩ, {len(allv)} sweeps)",
             loc="left", fontsize=7.5)
panels["c"] = ax
fs.save(fig, FIG26 / f"{FIGP}_6_impedance", panels=panels, extras={"a": [cax]})
for subj in SUBJECT_ORDER:
    tt = Z[subj + "_t"]; k = np.isfinite(tt) & (tt > 1.0)
    z2 = Z[subj][k][:, 0]; z5 = Z[subj][k][:, j5]
    print(f"{subj}: |Z| 2 kHz {z2[0]/1e3:.1f} -> {z2[-1]/1e3:.1f} kOhm; 5 kHz {z5[0]/1e3:.1f} -> {z5[-1]/1e3:.1f}")
print("ok")
