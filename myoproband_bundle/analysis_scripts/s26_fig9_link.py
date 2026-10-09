"""Figura 9 (sabado 26): enlace, latencias y presupuesto de datos.

(a) tasa efectiva en ventanas de 10 s a lo largo de cada sesion: sEMG crudo
    (arriba) e IMU (abajo); la envolvente queda en 50.0 Hz en todas.
(b) deriva del reloj del equipo frente al del PC (ppm).
(c) variacion del retardo de entrega al PC, medida contra la entrega mas
    rapida de cada sesion (no es latencia absoluta: no hay reloj comun).
(d) latencia comando -> efecto: Prest registrado en el PC hasta el arranque
    del barrido visto en el sEMG (incluye el retardo de entrega).
(e) cuanto ocupa un minuto de registro, por flujo (archivos del PC; la SD
    guarda los mismos registros de 8 bytes).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
from s26_common import C_SWEEP, DATA26, FIG26, SUBJECT_ORDER

fs.apply()
L = np.load(DATA26 / "link_arrays.npz")
D = pd.read_csv(DATA26 / "timing_deep.csv").set_index("subject").reindex(SUBJECT_ORDER)
BU = pd.read_csv(DATA26 / "budget.csv").set_index("subject")
tones = dict(zip(SUBJECT_ORDER, fs.tones(len(SUBJECT_ORDER))))

fig = plt.figure(figsize=(7.2, 7.0))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.8], hspace=0.62, wspace=0.55,
                      left=0.09, right=0.97, top=0.95, bottom=0.08)

# (a) tasas a lo largo de la sesion
# sin IMU (--no-imu) la fila (a) es solo la tasa del sEMG
ga = gs[0, :].subgridspec(1, 1 if fs.NO_IMU else 2, wspace=0.28)
ax = fig.add_subplot(ga[0])
for s_ in SUBJECT_ORDER:
    ax.plot(L[f"rate_t_{s_}"], L[f"rate_raw_{s_}"], lw=0.9, color=tones[s_])
ax.set_ylim(998.5, 1001.5)
dip = L["rate_raw_S02"].min()
kd = int(np.argmin(L["rate_raw_S02"]))
# el corte cae en mas de una ventana de 10 s: se nombran todas las que quedan fuera de escala
off = np.sort(L["rate_raw_S02"][L["rate_raw_S02"] < 998.5])
ax.annotate(f"S02 dropout: {len(off)} window{'s' if len(off) != 1 else ''}\n"
            f"off scale ({', '.join(f'{v:.0f}' for v in off)} Hz)", (L["rate_t_S02"][kd], 998.5),
            xytext=(L["rate_t_S02"][kd] - 0.5, 999.0), fontsize=6, ha="right",
            arrowprops=dict(arrowstyle="->", lw=0.6, color=fs.INK2))
ax.set_xlabel("Time in session (min)")
ax.set_ylabel("sEMG rate (Hz/ch)")
ax.set_title(f"sEMG, 10 s windows ({len(SUBJECT_ORDER)} sessions overlaid)", loc="left", fontsize=7.5)
if not fs.NO_IMU:
    ax = fig.add_subplot(ga[1])
    for s_ in SUBJECT_ORDER:
        ax.plot(L[f"rate_t_{s_}"], L[f"rate_imu_{s_}"], lw=0.9, color=tones[s_])
    ax.axhline(200, color=fs.INK2, lw=0.8, ls="--")
    ax.text(0.2, 202, "nominal 200 Hz polling", fontsize=6, color=fs.INK, va="bottom")
    ax.set_ylim(100, 215)
    ax.set_xlabel("Time in session (min)")
    ax.set_ylabel("IMU rate (Hz)")
    imu_med = [np.median(L[f"rate_imu_{s_}"]) for s_ in SUBJECT_ORDER]
    ax.set_title(f"IMU, 10 s windows (session medians {min(imu_med):.0f}–{max(imu_med):.0f} Hz)", loc="left",
                 fontsize=7.5)
panels = {"a": fig.axes[:1 if fs.NO_IMU else 2]}

# (b) deriva del reloj
ax = fig.add_subplot(gs[1, 0])
y = np.arange(len(SUBJECT_ORDER))
ax.plot(D.clock_ppm, y, "o", ms=4, color=fs.INK)
ax.set_yticks(y)
ax.set_yticklabels(SUBJECT_ORDER, fontsize=6.8)
ax.invert_yaxis()
ax.set_xlim(0, 30)
ax.set_xlabel("Device clock vs PC (ppm)")
ax.set_title(f"Clock drift\n(+{D.clock_ppm.min():.0f} to +{D.clock_ppm.max():.0f} ppm = "
             f"{D.clock_ppm.min()*3.6:.0f}–{D.clock_ppm.max()*3.6:.0f} ms/h)", loc="left", fontsize=7.5)
panels["b"] = ax

# (c) variacion del retardo de entrega
ax = fig.add_subplot(gs[1, 1])
for s_ in SUBJECT_ORDER:
    v = np.sort(L[f"delay_{s_}"])
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=0.9, color=tones[s_])
ax.set_xscale("log")
ax.set_xlim(0.5, 250)
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(["1", "10", "100"])
ax.set_xlabel("Delivery delay above fastest (ms)")
ax.set_ylabel("Cumulative fraction")
ax.set_title(f"Delivery to the PC\n(p50 {D.delay_p50_ms.min():.0f}–{D.delay_p50_ms.max():.0f}, "
             f"p95 {D.delay_p95_ms.min():.0f}–{D.delay_p95_ms.max():.0f} ms)", loc="left", fontsize=7.5)
panels["c"] = ax

# (d) comando -> efecto
ax = fig.add_subplot(gs[1, 2])
for j, s_ in enumerate(SUBJECT_ORDER):
    v = L[f"lat_{s_}"]
    if not len(v):
        ax.text(60, j, "no onsets\ndetected", fontsize=5.6, va="center", color=fs.INK2)
        continue
    jit = (np.random.default_rng(j).random(len(v)) - 0.5) * 0.4
    ax.plot(v, j + jit, "o", ms=2, color=C_SWEEP, alpha=0.7, mew=0)
    ax.plot([np.median(v)] * 2, [j - 0.3, j + 0.3], color=fs.INK, lw=1.2)
ax.set_yticks(y)
ax.set_yticklabels(SUBJECT_ORDER, fontsize=6.8)
# limites explicitos (invertidos): sin puntos en S02 la fila quedaba sobre el eje
ax.set_ylim(len(SUBJECT_ORDER) - 0.5, -0.5)
ax.set_xlim(50, 250)
ax.set_xlabel("Rest cue → sweep onset (ms)")
allv = np.concatenate([L[f"lat_{s_}"] for s_ in SUBJECT_ORDER])
ax.set_title(f"Command to effect\n(median {np.median(allv):.0f} ms, n = {len(allv)})", loc="left", fontsize=7.5)
panels["d"] = ax

# (e) presupuesto por flujo
ax = fig.add_subplot(gs[2, :])
b = BU.mean(numeric_only=True)
parts = [("raw sEMG, 8 × 1 kHz", b.raw_mb_min, fs.INK), ("envelope, 8 × 50 Hz", b.env_mb_min, "#2a78d6"),
         ("IMU", b.imu_mb_min, fs.INK2), ("auxiliary + events", b.aux_mb_min + b.events_mb_min, "#a3aab2")]
# sin IMU: el presupuesto de los demas flujos (el IMU se graba, pero no se muestra)
if fs.NO_IMU:
    parts = [q for q in parts if q[0] != "IMU"]
total = sum(q[1] for q in parts)
for j, (lbl, v, c) in enumerate(parts):
    ax.barh(j, v, color=c, height=0.62)
    txt = f"{v:.2f} MB/min ({100 * v / total:.0f} %)" if v >= 0.1 else f"{v * 1000:.0f} kB/min"
    ax.text(v + 0.04, j, txt, va="center", fontsize=6.3, color=fs.INK)
ax.set_yticks(range(len(parts)))
ax.set_yticklabels([p[0] for p in parts], fontsize=6.8)
ax.invert_yaxis()
ax.set_xlim(0, b.raw_mb_min * 1.35)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel("MB per minute of recording")
ax.set_title(f"Data budget{' without the IMU stream' if fs.NO_IMU else ''}: {total:.2f} MB/min = {total*60/1000:.2f} GB/h "
             f"on the PC and on the SD (same 8-byte records); {total * 8 / 60:.2f} Mbit/s of payload over UDP",
             loc="left", fontsize=7.5)
panels["e"] = ax
fs.save(fig, FIG26 / f"{FIGP}_9_link", panels=panels)
print("ok")
