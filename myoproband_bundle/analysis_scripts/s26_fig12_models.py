"""Figura 12 (sabado 26): separabilidad de agarres, como metrica secundaria.

Resultados de s26_03_classify.py: LDA con encogimiento sobre el conjunto de
Hudgins por canal (log-MAV, log-WL, ZC, SSC; 32 rasgos) en ventanas de 250 ms
con paso de 125 ms dentro del sostener estable, validacion dejando fuera una
repeticion (6 pliegues) por participante; por ensayo, voto de sus ventanas.

(a) precision por participante: rasgos crudos (ventana y ensayo), normalizados
    al reposo previo de cada ensayo, el control de reposo (ventanas de reposo
    etiquetadas con el bloque: lo que el bloque comparte sin agarre) y las
    etiquetas de la hipotesis de orden equivocada (debe dar azar);
(b) matriz de confusion por ensayo, sumada sobre los seis participantes, en el
    orden de presentacion y en % por fila.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
from s26_common import DATA26, FIG26, GRASP_ORDER, GRASP_SHORT, SUBJECT_ORDER

fs.apply()
CL = pd.read_csv(DATA26 / "classify.csv").set_index("subject").reindex(SUBJECT_ORDER)
CM = np.load(DATA26 / "confusion.npz")
classes = list(CM["classes"])

fig = plt.figure(figsize=(7.2, 4.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.42, left=0.08, right=0.98, top=0.92, bottom=0.3)

ax = fig.add_subplot(gs[0])
y = np.arange(len(SUBJECT_ORDER) + 1)
labels = SUBJECT_ORDER + ["mean"]
series = [
    ("acc_trial", "grasp, per trial (vote)", fs.INK, "D", 4.2),
    ("acc_window", "grasp, per 250 ms window", fs.INK, "o", 4.2),
    ("acc_norm_window", "grasp, normalised to own pre-cue rest", fs.INK2, "o", 4.2),
    ("acc_rest_window", "control: rest windows labelled by block", "#a3aab2", "^", 4.4),
    ("acc_blockorder_window", "control: labels from the wrong order", "#c5cbd1", "x", 4.6),
]
for col, lbl, c, m, ms in series:
    v = 100 * np.r_[CL[col].values, CL[col].mean()]
    face = "none" if col == "acc_norm_window" else c
    ax.plot(v, y, m, ms=ms, color=c, mfc=face, mec=c, mew=1.0 if m != "x" else 1.3, ls="none", label=lbl)
# linea de azar en tinta neutra: el naranja queda reservado para el barrido en todas las figuras
ax.axvline(100 / 7, color=fs.INK2, lw=0.9, ls="--")
ax.text(100 / 7 + 1.2, -0.75, "chance 14.3 %", fontsize=6, color=fs.INK, va="center")
ax.axhline(len(SUBJECT_ORDER) - 0.5, color=fs.GRID, lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=6.8)
ax.invert_yaxis()
ax.set_xlim(0, 102)
ax.set_ylim(len(labels) - 0.4, -1.2)
ax.set_xlabel("Accuracy, 7 grasps, leave-one-repetition-out (%)")
# la clave va con su panel (debajo del eje x), para que el panel suelto la lleve
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=6, handletextpad=0.3, columnspacing=1.4)
ax.set_title("Accuracy per participant and pooled mean", loc="left", fontsize=7.2)
panels = {"a": ax}

ax = fig.add_subplot(gs[1])
tot = sum(CM[s_] for s_ in SUBJECT_ORDER if s_ in CM.files)
idx = [classes.index(g) for g in GRASP_ORDER]
M = tot[np.ix_(idx, idx)].astype(float)
P = 100 * M / M.sum(1, keepdims=True)
im = ax.imshow(P, cmap="Greys", vmin=0, vmax=100)
for (r, c), v in np.ndenumerate(P):
    if v > 0:
        ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=5.8, color="white" if v > 55 else fs.INK)
names = [GRASP_SHORT[g] for g in GRASP_ORDER]
ax.set_xticks(range(7))
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6)
ax.set_yticks(range(7))
ax.set_yticklabels(names, fontsize=6)
for t, g in zip(ax.get_yticklabels(), GRASP_ORDER):
    t.set_color(fs.INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("Predicted", labelpad=2)
ax.set_ylabel("Performed", labelpad=2)
n_trials = int(M.sum())
ax.set_title(f"Per-trial confusion, 6 participants ({n_trials} trials, % of row)", loc="left", fontsize=7.2)
panels["b"] = ax
fs.save(fig, FIG26 / f"{FIGP}_12_models", panels=panels)
print("ok", n_trials, "ensayos;", "diagonal media", round(float(np.mean(np.diag(P))), 1), "%")
