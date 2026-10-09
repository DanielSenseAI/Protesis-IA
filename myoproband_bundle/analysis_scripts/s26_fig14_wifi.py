"""Figura 14 (sabado 26): la interferencia de la baliza Wi-Fi y como quitarla.

(a) el crudo en reposo plegado sobre el periodo de la baliza (102.4 ms, reloj
    del equipo): un solo hundimiento de ~3 ms, igual en los ocho canales;
(b) un tramo de reposo de un canal, original y reparado (las muestras dentro
    de la ventana del hundimiento, interpoladas);
(c) espectro en reposo (mediana de los 48 canales): original, reparado y con
    notch en n x 9.766 Hz;
(d) por metodo y canal: parte de la potencia en reposo que queda en las lineas
    y cambio del RMS sosteniendo (lo que la limpieza le quita al sEMG).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
from s26_common import DATA26, FIG26, SUBJECT_ORDER
import wifi_clean as wc

fs.apply()
Z = np.load(DATA26 / "wifi_clean.npz")
D = pd.read_csv(DATA26 / "wifi_clean.csv")
METHODS = list(Z["methods"])
# color por metodo, fijo en toda la figura
COL = {"original": "#a3aab2", "notch": "#4a3aa7", "template": "#1baf7a", "repair": "#2a78d6"}
LBL = {"original": "original", "notch": "comb notch", "template": "mean template subtracted",
       "repair": "hit samples repaired"}
si, ch, _ = Z["ex_info"]
si, ch = int(si), int(ch)
subj = SUBJECT_ORDER[si]

fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.55, wspace=0.3, left=0.09, right=0.97,
                      top=0.94, bottom=0.09)
panels = {}

# (a) pliegue sobre el periodo de la baliza
ax = fig.add_subplot(gs[0, 0])
T8 = Z[f"tpl_{subj}"]
ph = (np.arange(T8.shape[1]) + 0.5) * 102.4 / T8.shape[1]
for i in range(8):
    ax.plot(ph, T8[i], lw=0.7, color=fs.INK, alpha=0.55)
phase, lo, hi, depth = Z[f"win_{subj}"]
ax.axvspan(lo, hi, color=COL["repair"], alpha=0.15, lw=0)
depths = [Z[f"win_{s_}"][3] for s_ in SUBJECT_ORDER]
ax.text(hi + 3, depth * 0.8, f"{fs.num(depth)} mV dip,\nall 8 channels at once\n"
        f"({len(SUBJECT_ORDER)} sessions: {fs.num(min(depths))}\nto {fs.num(max(depths))} mV)", fontsize=6.2, color=fs.INK, va="center")
ax.set_xlim(0, 102.4)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Phase within the 102.4 ms beacon period (ms)")
ax.set_ylabel("Mean raw sEMG at rest (mV)")
ax.set_title(f"Rest samples folded on the beacon period ({subj})", loc="left", fontsize=7.5)
panels["a"] = ax

# (b) tramo de reposo, original y reparado
ax = fig.add_subplot(gs[0, 1])
t, y, yr, bad = Z["ex_t"] * 1000, Z["ex_y"], Z["ex_rep"], Z["ex_bad"].astype(bool)
off = 1.2 * (np.percentile(y, 99.5) - np.percentile(y, 0.5))
ax.plot(t, y, lw=0.6, color=COL["original"])
ax.plot(t[bad], y[bad], "o", ms=2.2, color=fs.INK, mew=0)
ax.plot(t, yr - off, lw=0.6, color=COL["repair"])
# cada trazo se nombra en el eje (como las filas de la fig. 2); barra de escala a la derecha
ax.set_yticks([np.median(y), np.median(yr) - off])
ax.set_yticklabels(["original", "repaired"], fontsize=6.5)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, t[-1] + 95)
ax.spines["left"].set_visible(False)
bar = 50.0
x0 = t[-1] + 15
yb = np.percentile(y, 99.5)
ax.plot([x0, x0], [yb - bar, yb], color=fs.INK, lw=1.2)
ax.text(x0 + 5, yb - bar / 2, f"{bar:.0f} mV", fontsize=6.0, va="center", ha="left", color=fs.INK)
ax.set_xlabel("Time (ms)")
ax.set_title(f"Raw E{ch + 1} at rest: dots = samples inside the dip window ({100 * bad.mean():.1f} %)",
             loc="left", fontsize=7.5)
panels["b"] = ax

# (c) espectro en reposo
ax = fig.add_subplot(gs[1, 0])
f4, Pr = Z["f4"], Z["Pr"]
k = (f4 >= 15) & (f4 <= 125)
base = np.median(10 * np.log10(Pr[:, :, 0, :].reshape(-1, len(f4))[:, k]), 0)
ref = np.median(base)
for meth in ("original", "notch", "repair"):
    mi = METHODS.index(meth)
    q = np.median(10 * np.log10(Pr[:, :, mi, :].reshape(-1, len(f4))[:, k]), 0)
    ax.plot(f4[k], q - ref, lw=0.8 if meth != "original" else 0.7, color=COL[meth], label=LBL[meth],
            zorder=3 if meth == "repair" else 2)
# la linea que queda tras reparar: la tasa de paquetes UDP del crudo (8000 registros/s / 174 por paquete)
pk = 8000 / 174
kk = np.argmin(np.abs(f4[k] - pk))
qrep = np.median(10 * np.log10(Pr[:, :, METHODS.index("repair"), :].reshape(-1, len(f4))[:, k]), 0) - ref
ax.annotate(f"{pk:.0f} Hz: UDP data packets", xy=(f4[k][kk], qrep[kk] + 0.3), xytext=(22, 11.0),
            fontsize=6.0, color=fs.INK, arrowprops=dict(arrowstyle="-", color=fs.INK2, lw=0.6))
ax.set_xlim(15, 125)
ax.set_ylim(-4, 13)
ax.set_xlabel("Frequency (Hz), 0.24 Hz resolution")
ax.set_ylabel("dB relative to the median")
ax.set_title(f"Resting power spectral density, {8 * len(SUBJECT_ORDER)} channels", loc="left", fontsize=7.5)
ax.legend(loc="upper right", fontsize=6.2, handlelength=1.4)
panels["c"] = ax

# (d) resumen por metodo
sub = gs[1, 1].subgridspec(1, 2, wspace=0.55)
orig = D[D.method == "original"].set_index(["subject", "channel"])
order = ["original", "notch", "template", "repair"]
rng = np.random.default_rng(3)
axl = fig.add_subplot(sub[0])
for j, meth in enumerate(order):
    v = 100 * D[D.method == meth].lines_share_rest.values
    axl.plot(j + rng.uniform(-0.18, 0.18, len(v)), v, "o", ms=2.0, color=COL[meth], alpha=0.8, mew=0)
    axl.plot([j - 0.28, j + 0.28], [np.median(v)] * 2, color=fs.INK, lw=1.2)
axl.set_xticks(range(len(order)))
axl.set_xticklabels(["original", "notch", "template", "repaired"], rotation=35, ha="right", fontsize=6.5)
axl.set_ylabel("Resting power in the lines (%)")
axl.set_title("Lines left at rest", loc="left", fontsize=7.2)
axr = fig.add_subplot(sub[1])
for j, meth in enumerate(order[1:]):
    d = D[D.method == meth].set_index(["subject", "channel"])
    v = (20 * np.log10(d.hold_rms_mv / orig.hold_rms_mv)).values
    axr.plot(j + rng.uniform(-0.18, 0.18, len(v)), v, "o", ms=2.0, color=COL[meth], alpha=0.8, mew=0)
    axr.plot([j - 0.28, j + 0.28], [np.median(v)] * 2, color=fs.INK, lw=1.2)
axr.axhline(0, color=fs.GRID, lw=0.8, zorder=0)
axr.set_xticks(range(3))
axr.set_xticklabels(["notch", "template", "repaired"], rotation=35, ha="right", fontsize=6.5)
axr.set_ylabel("Change of hold RMS (dB)")
axr.set_title("sEMG kept", loc="left", fontsize=7.2)
panels["d"] = [axl, axr]

fs.save(fig, FIG26 / f"{FIGP}_14_wifi", panels=panels)
print("ok")
