"""Figura 8 (sabado 26): anatomia del muestreo del brazalete.

(a) cada conversion de los 16 canales durante ~26 ms (S01), agrupadas por ADC:
    crudo en negro, envolvente en azul. Dentro de un ADC el orden es fijo
    (segundo crudo +0.46 ms; cada 20 rondas, las dos envolventes); entre ADC
    la fase corre libre.
(b) intervalos de E1: 19 cortos (~0.91 ms) y 1 largo (~2.5 ms) cada 20.
(c) error de cada marca respecto a una rejilla uniforme de 1 kHz.
(d) un tono de 100 Hz muestreado en las marcas reales, analizado como si el
    muestreo fuera uniforme (gris) y remuestreado por marcas (negro).
(e) el mayor espurio segun la frecuencia del tono (mediana de 6 sesiones).
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

C_RAW, C_ENV = fs.INK, "#2a78d6"
fs.apply()
ex = np.load(DATA26 / "timing_example.npz")
SP = pd.read_csv(DATA26 / "timing_spurs.csv")
D = pd.read_csv(DATA26 / "timing_deep.csv")
O = pd.read_csv(DATA26 / "timing_offsets.csv")

e1 = ex["e1_ts"].astype(np.int64)
d = np.diff(e1)
k_long = np.where(d > 1800)[0]
k0 = max(0, k_long[1] - 12)             # ventana alrededor de una ronda larga de ADC0
t0 = e1[k0]
t1 = t0 + 26000

# la fila del medio alta como para su etiqueta de dos lineas (si no, subia hasta la letra)
fig = plt.figure(figsize=(7.2, 8.1))
gs = fig.add_gridspec(3, 2, height_ratios=[1.55, 0.98, 1.0], hspace=0.55, wspace=0.36,
                      left=0.1, right=0.97, top=0.96, bottom=0.07)
panels = {}

# (a) diagrama de tiempos
ax = fig.add_subplot(gs[0, :])
order = []
for adc in range(4):
    for kind in ("raw", "env"):
        chs = O[(O.subject == "S01") & (O.kind == kind) & (O.adc == adc)].sort_values("pin")
        for _, r in chs.iterrows():
            order.append((kind, int(r.channel[1:]) - 1, adc, int(r.pin)))
ylab = []
for row, (kind, i, adc, pin) in enumerate(order):
    tc = ex[f"{kind}{i}"].astype(np.int64)
    tc = tc[(tc >= t0) & (tc <= t1)]
    yv = len(order) - 1 - row
    ax.plot((tc - t0) / 1000.0, np.full(len(tc), yv), "|", ms=7, mew=1.1, color=C_RAW if kind == "raw" else C_ENV)
    ylab.append((yv, f"{'raw' if kind == 'raw' else 'env'} E{i+1}  (AIN{pin})"))
for adc in range(1, 4):
    ax.axhline(len(order) - adc * 4 - 0.5, color=fs.GRID, lw=0.8)
for adc in range(4):
    ax.text(26.3, len(order) - 1.5 - adc * 4, f"ADC{adc}\n(I2C bus {adc // 2})", fontsize=6, color=fs.INK2,
            va="center", ha="left")
# ronda larga de ADC0
lo_ = (e1[k_long[1]] - t0) / 1000.0
hi_ = (e1[k_long[1] + 1] - t0) / 1000.0
ax.axvspan(lo_, hi_, color=C_ENV, alpha=0.08, lw=0)
ax.annotate(f"every 20th round of an ADC also converts\nits two envelope channels: {D.long_us_med.median()/1000:.2f} ms "
            f"instead of {D.short_us_med.median()/1000:.2f} ms",
            ((lo_ + hi_) / 2, len(order) - 0.6), xytext=((lo_ + hi_) / 2 + 3.2, len(order) + 0.9),
            fontsize=6.3, ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.6, color=fs.INK2))
# el +0.46 ms entre los dos crudos de un ADC va en el pie: entre filas tan
# juntas el rotulo caia siempre sobre alguna marca
ax.set_yticks([y for y, _ in ylab])
ax.set_yticklabels([l for _, l in ylab], fontsize=5.8)
ax.set_ylim(-0.7, len(order) + 1.6)
ax.set_xlim(0, 26)
ax.set_xlabel("Time (ms)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_title("Every conversion of the 16 channels (8 raw sEMG, 8 hardware envelope), one participant session",
             loc="left", fontsize=7.5)
panels["a"] = ax

# (b) intervalos de E1
ax = fig.add_subplot(gs[1, 0])
ax.vlines(np.arange(len(d)), 0, d / 1000.0, color=C_RAW, lw=0.8)
ax.plot(np.arange(len(d)), d / 1000.0, "o", ms=1.6, color=C_RAW)
ax.axhline(1.0, color=fs.INK2, lw=0.6, ls="--")
ax.set_xlim(-1, len(d))
ax.set_ylim(0, 2.8)
ax.set_xlabel("Sample index (E1)")
ax.set_ylabel("Interval (ms)")
# el "1 ms nominal" va en el titulo: dentro del panel lo cruzaba un tallo
ax.set_title(f"E1 intervals: {D.short_us_med.median()/1000:.2f} ms × 19, {D.long_us_med.median()/1000:.2f} ms × 1 "
             "(dashed: 1 ms)", loc="left", fontsize=7.5)
panels["b"] = ax

# (c) error frente a la rejilla uniforme
ax = fig.add_subplot(gs[1, 1])
te, em = ex["err_t"], ex["err_ms"]
k = te < 0.2
ax.plot(te[k] * 1000, em[k], lw=0.8, color=C_RAW)
ax.axhline(0, color=fs.GRID, lw=0.8)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Timestamp − uniform\n1 kHz grid (ms)")
ax.set_title(f"If assumed uniform: 50 Hz sawtooth, {D.err_p2p_ms.min():.1f}–{D.err_p2p_ms.max():.1f} ms p–p",
             loc="left", fontsize=7.5)
panels["c"] = ax

# (d) tono de 100 Hz: uniforme supuesto vs remuestreo por marcas
ax = fig.add_subplot(gs[2, 0])
f = ex["tone_f"]
k = (f > 1) & (f < 500)
ax.plot(f[k], ex["tone_naive"][k], lw=0.8, color="#a3aab2", label="treated as uniform")
ax.plot(f[k], ex["tone_resampled"][k], lw=0.8, color=fs.INK2, label="linear resampling on timestamps")
ax.plot(f[k], ex["tone_cubic"][k], lw=0.9, color=C_RAW, label="cubic resampling on timestamps")
ax.set_ylim(-120, 8)
ax.set_xlim(0, 500)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power relative to the\n100 Hz tone (dB)")
ax.legend(loc="upper right", fontsize=6)
r100 = SP[SP.tone_hz == 100].median(numeric_only=True)
ax.set_title(f"100 Hz tone: largest spur {fs.num(r100.spur_dbc)} / {fs.num(r100.resampled_spur_dbc)} / "
             f"{fs.num(r100.cubic_spur_dbc)} dB", loc="left", fontsize=7.5)
panels["d"] = ax

# (e) espurio segun frecuencia del tono
ax = fig.add_subplot(gs[2, 1])
cols3 = ["spur_dbc", "resampled_spur_dbc", "cubic_spur_dbc"]
g = SP.groupby("tone_hz")[cols3].median()
lo = SP.groupby("tone_hz")[cols3].min()
hi = SP.groupby("tone_hz")[cols3].max()
for col, c, lbl in (("spur_dbc", "#a3aab2", "treated as uniform"), ("resampled_spur_dbc", fs.INK2, "linear resampling"),
                   ("cubic_spur_dbc", C_RAW, "cubic resampling")):
    ax.plot(g.index, g[col], "o-", ms=3, lw=1.0, color=c, label=lbl)
    ax.vlines(g.index, lo[col], hi[col], color=c, lw=0.8)
ax.set_xticks(list(g.index))
ax.set_xlabel("Tone frequency (Hz)")
ax.set_ylabel("Largest spur relative\nto the tone (dB)")
ax.legend(loc="lower right", fontsize=6)
ax.set_title(f"Largest spur vs tone frequency ({len(SUBJECT_ORDER)} sessions)", loc="left", fontsize=7.5)
panels["e"] = ax
fs.save(fig, FIG26 / f"{FIGP}_8_timing", panels=panels)
print("ok")
