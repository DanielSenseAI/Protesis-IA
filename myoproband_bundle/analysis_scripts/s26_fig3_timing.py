"""Figura 3 (sabado 26): tiempos de muestreo y alineacion entre modalidades.

(a) intervalos entre muestras del crudo, por sesion (tope R1000: la mayoria a
    ~0.9 ms y un 5 % a ~2.5 ms, promedio 1.000 ms);
(b) dispersion de la trama de 8 canales (desfase entre el primero y el
    ultimo canal en muestrear para cada trama) frente al objetivo de <= 1 ms;
(c) intervalos del IMU (sondeo a 200 Hz con vTaskDelayUntil: las lecturas atrasadas
    se ponen al dia seguidas; el firmware no usa la FIFO del ICM-42605);
(d) cuando llegan al PC los datos de la placa auxiliar respecto al evento de
    la tarea: presion/temperatura respecto al Pgrasp y el barrido completo
    respecto al Prest. Esas modalidades se fechan en el PC al llegar, asi que
    su alineacion queda limitada por esta latencia y su variacion.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
from s26_common import C_SWEEP, DATA26, FIG26, SUBJECT_ORDER, find_sessions, load26, prest_times

fs.apply()
S = pd.read_csv(DATA26 / "sessions.csv")
full = S[S.complete].set_index("subject")
paths = {p.name: p for p in find_sessions()}
dts, spreads, imu_dt, sw_lat = {}, {}, {}, {}
for subj in SUBJECT_ORDER:
    x = load26(paths[full.loc[subj, "session"]])
    s = x.s
    pv = full.loc[subj, "preview_s"]
    d = []
    for i in range(8):
        t = s.raw[i].ts_us.astype(np.int64)
        t = t[(t - s.t0_us) / 1e6 > pv + 0.5]
        dd = np.diff(t)
        d.append(dd[dd < 5000])
    dts[subj] = np.concatenate(d) / 1000.0
    ref = s.raw[0].ts_us.astype(np.int64)
    ref = ref[(ref - s.t0_us) / 1e6 > pv + 0.5][1000:-1000:7]
    offs = []
    for i in range(8):
        tc = s.raw[i].ts_us.astype(np.int64)
        j = np.clip(np.searchsorted(tc, ref), 1, len(tc) - 1)
        near = np.where(np.abs(tc[j] - ref) < np.abs(tc[j - 1] - ref), tc[j], tc[j - 1])
        offs.append(near - ref)
    offs = np.array(offs)
    spreads[subj] = (offs.max(0) - offs.min(0)) / 1000.0
    ti = np.sort(s.imu["ts_us"].astype(np.int64))
    ti = ti[(ti - s.t0_us) / 1e6 > pv + 0.5]
    imu_dt[subj] = np.diff(ti) / 1000.0
    pr = prest_times(x)
    lat = []
    for w in x.sweeps:
        c = [p for p in pr if 3.5 < w["t_arrival"] - p < 7.5]
        if c:
            lat.append(w["t_arrival"] - max(c))
    sw_lat[subj] = np.array(lat)
    print(subj, "ok")

P = pd.read_csv(DATA26 / "sensors_trials.csv")
fig = plt.figure(figsize=(7.2, 5.4))
gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.5, left=0.08, right=0.98, top=0.94, bottom=0.1)
panels = {}
tone = {s: c for s, c in zip(SUBJECT_ORDER, fs.tones(len(SUBJECT_ORDER)))}

# (a) intervalos del crudo: histograma comun, una linea por sesion
ax = fig.add_subplot(gs[0, 0])
bins = np.arange(0.0, 4.01, 0.05)
for subj in SUBJECT_ORDER:
    h, e = np.histogram(dts[subj], bins=bins)
    ax.step(e[:-1], np.maximum(h / h.sum(), 1e-7), where="post", lw=0.8, color=tone[subj])
allv = np.concatenate(list(dts.values()))
if fs.REVIEW:
    # revision: escala lineal, para que el 5 % de intervalos largos se vea en su proporcion real
    ax.set_ylim(0, 0.8)
    ax.annotate("", xy=(2.45, 0.015), xytext=(2.45, 0.2),
                arrowprops=dict(arrowstyle="->", color=fs.INK2, lw=0.7))
else:
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.5)
ax.set_xlim(0, 4)
ax.set_xlabel("sEMG inter-sample interval (ms)")
ax.set_ylabel("Fraction of intervals")
m1 = np.mean(allv)
# entre los dos modos (1.35-1.75 ms) no hay intervalos: 1.5 ms los separa
frac_long = np.mean(allv > 1.5)
ax.text(0.98, 0.95, f"mean {m1:.3f} ms  →  {1/m1*1000:.1f} Hz/ch\n{100*frac_long:.1f} % of intervals near "
        f"{np.median(allv[allv > 1.5]):.1f} ms", transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
        color=fs.INK)
ax.set_title(f"sEMG sampling (8 ch, {len(SUBJECT_ORDER)} sessions)", loc="left", fontsize=7.5)
panels["a"] = ax

# (b) dispersion de la trama: ECDF por sesion y objetivo de <= 1 ms
ax = fig.add_subplot(gs[0, 1])
for subj in SUBJECT_ORDER:
    v = np.sort(spreads[subj])
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=0.9, color=tone[subj])
# objetivo de la tabla de requisitos del 29-09 (antes < 100 us por trama)
ax.axvline(1.0, color=fs.INK2, lw=0.9, ls="--")
ax.text(1.04, 0.08, "target\n≤ 1 ms", fontsize=6.2, color=fs.INK, va="bottom")
allsp = np.concatenate(list(spreads.values()))
ax.set_xlim(0, 2.5)
ax.set_xlabel("Spread across the 8 channels of one frame (ms)")
ax.set_ylabel("Cumulative fraction")
ax.text(0.98, 0.3, f"median {np.median(allsp):.2f} ms\np95 {np.percentile(allsp, 95):.2f} ms\n"
        "(each sample carries its\nown device timestamp)", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.3)
ax.set_title(f"Inter-channel sampling skew ({len(SUBJECT_ORDER)} sessions)", loc="left", fontsize=7.5)
panels["b"] = ax

# (c) IMU: intervalos (no sale con --no-imu)
if not fs.NO_IMU:
    ax = fig.add_subplot(gs[1, 0])
    bins = np.geomspace(0.1, 1000, 80)
    for subj in SUBJECT_ORDER:
        h, e = np.histogram(imu_dt[subj], bins=bins)
        ax.step(e[:-1], np.maximum(h / h.sum(), 1e-7), where="post", lw=0.8, color=tone[subj])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("IMU inter-sample interval (ms)")
    ax.set_ylabel("Fraction of intervals")
    rates = [full.loc[s_, "imu_hz"] for s_ in SUBJECT_ORDER]
    ai = np.concatenate(list(imu_dt.values()))
    ax.text(0.98, 0.95, f"{min(rates):.0f}–{max(rates):.0f} Hz by session\n"
            f"{100*np.mean(ai > 50):.2f} % of intervals > 50 ms", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.3)
    ax.set_title(f"IMU sampling, {len(SUBJECT_ORDER)} sessions (polled, nominal 200 Hz)", loc="left", fontsize=7.5)
    panels["c"] = ax

# (d) llegada de los datos auxiliares respecto al evento de la tarea (sin IMU, a todo el ancho)
ax = fig.add_subplot(gs[1, :] if fs.NO_IMU else gs[1, 1])
rows = []
for k, lbl in enumerate(["load/temp #1", "load/temp #2", "load/temp #3"]):
    v = P[f"dt{k}"].dropna().values
    rows.append((lbl, v, fs.INK))
rows.append(("full sweep result", np.concatenate(list(sw_lat.values())), C_SWEEP))
for j, (lbl, v, col) in enumerate(rows):
    yj = len(rows) - 1 - j
    jit = (np.random.default_rng(j).random(len(v)) - 0.5) * 0.35
    ax.plot(v, yj + jit, "o", ms=1.4, color=col, alpha=0.5, mew=0, rasterized=True)
    lo_, hi_ = np.percentile(v, [2, 98])
    ax.text(hi_ + 0.1 if hi_ < 4.3 else lo_ - 0.1, yj,
            f"median {np.median(v):.2f} s\nIQR {np.percentile(v, 75)-np.percentile(v, 25):.2f} s",
            fontsize=5.8, ha="left" if hi_ < 4.3 else "right", va="center")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.5)
ax.set_ylim(-0.6, len(rows) - 0.2)
ax.set_xlim(0, 6)
ax.set_xlabel("Arrival at the PC after the task event (s)\n(load/temp: after grasp cue; sweep: after rest cue)")
ax.set_title("Auxiliary data are timestamped on arrival", loc="left", fontsize=7.5)
panels["d"] = ax

# Las seis sesiones se superponen a proposito: lo que se muestra es que no
# difieren. Sin leyenda por sesion; lo dicen los titulos (la nota dentro de
# los paneles caia sobre las curvas).
fs.save(fig, FIG26 / f"{FIGP}_3_timing", panels=panels)
pd.DataFrame([dict(subject=s_, raw_dt_mean_ms=np.mean(dts[s_]), raw_dt_p95_ms=np.percentile(dts[s_], 95),
                   spread_med_ms=np.median(spreads[s_]), spread_p95_ms=np.percentile(spreads[s_], 95),
                   imu_dt_med_ms=np.median(imu_dt[s_]), imu_gt50ms_pct=100 * np.mean(imu_dt[s_] > 50),
                   sweep_arrival_med_s=np.median(sw_lat[s_]), sweep_arrival_iqr_s=np.subtract(*np.percentile(sw_lat[s_], [75, 25])))
              for s_ in SUBJECT_ORDER]).to_csv(DATA26 / "timing.csv", index=False)
print("ok")
