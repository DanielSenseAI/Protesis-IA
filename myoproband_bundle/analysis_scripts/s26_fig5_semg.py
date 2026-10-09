"""Figura 5 (sabado 26): calidad del sEMG en los participantes.

(a) activacion al sostener, mediana por sujeto y canal (dB sobre la linea base
    de las pausas); (b) piso de ruido en reposo relajado (RMS 20-450 Hz en la
    entrada del ADC; 1 LSB = 2 mV); (c) espectro de sostener y de reposo;
    (d) detalle del reposo a 0.24 Hz: lineas en multiplos de 9.766 Hz, el
    intervalo de baliza del punto de acceso Wi-Fi del brazalete (100 TU =
    102.4 ms), que esta encendido porque la tasa completa solo sale por UDP;
    (e) la envolvente del hardware frente a la activacion del crudo, ensayo a
    ensayo: la zona muerta.

Escribe data/s26/spectra.npz, data/s26/wifi_lines.csv y fig_s26_5_semg.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
from s26_common import DATA26, FIG26, SUBJECT_ORDER, find_sessions, grid_raw, load26, win

BEACON_HZ = 1000.0 / 102.4
REPS, N_TRIALS = 6, 42


def welch(seg, n):
    out, k = None, 0
    for j in range(0, seg.shape[1] - n + 1, n // 2):
        w = seg[:, j:j + n]
        if not np.isfinite(w).all():
            continue
        w = w - w.mean(1, keepdims=True)
        p = np.abs(np.fft.rfft(w * np.hanning(n), axis=1)) ** 2
        out = p if out is None else out + p
        k += 1
    return out, k


C = pd.read_csv(DATA26 / "semg_channels.csv")
T = pd.read_csv(DATA26 / "semg_trials.csv")
cache = DATA26 / "spectra.npz"
if cache.exists() and "--recompute" not in sys.argv:
    z = np.load(cache)
    f1, f4, Ph, Pr, Pr4 = z["f1"], z["f4"], z["Ph"], z["Pr"], z["Pr4"]
else:
    paths = {p.name: p for p in find_sessions()}
    Ph, Pr, Pr4 = [], [], []
    for subj in SUBJECT_ORDER:
        name = C[C.subject == subj].session.iloc[0]
        x = load26(paths[name])
        tg, Y = grid_raw(x)
        acc_h, nh = np.zeros((8, 513)), 0
        acc_r, nr = np.zeros((8, 513)), 0
        acc_r4, nr4 = np.zeros((8, 2049)), 0
        for tr in x.trials:
            p, k = welch(Y[:, win(tg, tr.t_grasp + 1.0, tr.t_rest - 0.5)], 1024)
            if k:
                acc_h += p; nh += k
            if tr.index % REPS == REPS - 1 and tr.index + 1 < N_TRIALS:
                b0, b1 = tr.t_rest + 7.0, x.trials[tr.index + 1].t_grasp
                seg = Y[:, win(tg, b0 + 4.6, b1 - 2.5)]
                p, k = welch(seg, 1024)
                if k:
                    acc_r += p; nr += k
                p, k = welch(seg, 4096)
                if k:
                    acc_r4 += p; nr4 += k
        Ph.append(acc_h / nh); Pr.append(acc_r / nr); Pr4.append(acc_r4 / nr4)
        print(subj, "espectros ok")
    Ph, Pr, Pr4 = np.array(Ph), np.array(Pr), np.array(Pr4)
    f1, f4 = np.fft.rfftfreq(1024, 1e-3), np.fft.rfftfreq(4096, 1e-3)
    np.savez(cache, f1=f1, f4=f4, Ph=Ph, Pr=Pr, Pr4=Pr4)

# Aporte de las lineas de baliza a la potencia en reposo (20-450 Hz)
rows = []
band = (f4 >= 20) & (f4 <= 450)
near = np.min(np.abs(f4[:, None] - BEACON_HZ * np.arange(1, 47)[None, :]), axis=1) <= 0.37
for si, subj in enumerate(SUBJECT_ORDER):
    for ch in range(8):
        p = Pr4[si, ch]
        # piso sin lineas: mediana movil de 10 Hz, integrada sobre los mismos bins
        k = int(round(5.0 / (f4[1] - f4[0])))
        floor = np.array([np.median(p[max(0, i - k):i + k]) for i in range(len(p))])
        excess = np.sum(np.clip(p[band & near] - floor[band & near], 0, None))
        total = np.sum(p[band])
        peak_db = 10 * np.log10(np.max((p / floor)[band & near]))
        rows.append(dict(subject=subj, channel=f"E{ch+1}", comb_power_pct=100 * excess / total,
                         comb_rms_db=10 * np.log10(total / (total - excess)), peak_line_db=peak_db))
W = pd.DataFrame(rows)
W.to_csv(DATA26 / "wifi_lines.csv", index=False)
print(f"lineas de baliza: {W.comb_power_pct.median():.1f} % de la potencia en reposo (p10-p90 "
      f"{W.comb_power_pct.quantile(0.1):.1f}-{W.comb_power_pct.quantile(0.9):.1f} %), "
      f"+{W.comb_rms_db.median():.2f} dB sobre el RMS de reposo; linea mas alta {W.peak_line_db.median():.1f} dB")

# ------------------------------------------------------------------- figura
fs.apply()
fig = plt.figure(figsize=(7.2, 7.0))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.05], wspace=0.42, hspace=0.62,
                      left=0.08, right=0.97, top=0.95, bottom=0.08)
chs = [f"E{i}" for i in range(1, 9)]
panels = {}

ax = fig.add_subplot(gs[0, 0])
A = C.pivot(index="subject", columns="channel", values="act_db_med").reindex(SUBJECT_ORDER)[chs]
im = ax.imshow(A.values, cmap="Greys", vmin=0, vmax=15, aspect="auto")
for (r, c), v in np.ndenumerate(A.values):
    ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=6.2, color="white" if v > 9 else fs.INK)
ax.set_xticks(range(8))
ax.set_xticklabels([c + ("\nflexor" if c == "E1" else "\nextensor" if c == "E5" else "") for c in chs],
                   fontsize=6.8)
ax.set_yticks(range(len(A))); ax.set_yticklabels(A.index, fontsize=6.8)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_visible(False)
cb.set_label("dB relative to rest", fontsize=6)
ax.set_title("Hold activation, median over 42 trials", loc="left", fontsize=7.5)
panels["a"] = ax

ax = fig.add_subplot(gs[0, 1])
Bm = C.pivot(index="subject", columns="channel", values="base_rms_mv").reindex(SUBJECT_ORDER)[chs]
for r, subj in enumerate(SUBJECT_ORDER):
    v = Bm.loc[subj].values
    ax.plot(v, np.full(8, r) + np.linspace(-0.18, 0.18, 8), "o", ms=3, color=fs.INK, mew=0)
    hi = sorted((vv, chs[c]) for c, vv in enumerate(v) if vv > 24)
    if len(hi) <= 2:
        # uno o dos rotulos: como siempre, repartidos alrededor de la fila
        for j, (vv, lbl) in enumerate(hi):
            ax.text(vv * 1.06, r + (j - (len(hi) - 1) / 2) * 0.45, lbl, fontsize=5.8, va="center", color=fs.INK)
    else:
        # tres o mas (S07): en la fila, y solo se escalona el que quedaria pegado al anterior;
        # repartirlos todos invadia los rotulos de la fila vecina
        prev, off = None, 0.0
        step = -0.42 if r == len(SUBJECT_ORDER) - 1 else 0.42      # en la ultima fila, hacia adentro
        for vv, lbl in hi:
            off = (step if off == 0 else -off) if prev is not None and np.log10(vv / prev) < 0.09 else 0.0
            ax.text(vv * 1.06, r + off, lbl, fontsize=5.8, va="center", color=fs.INK)
            prev = vv
ax.axvline(2.0 / np.sqrt(12), color=fs.GRID, lw=0.8)
ax.text(2.0 / np.sqrt(12) * 1.08, -0.45, "quantisation\nnoise 0.58 mV", fontsize=5.6, color=fs.INK2, va="top")
ax.set_xscale("log")
ax.set_xlim(0.4, 80)
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
ax.set_yticks(range(len(SUBJECT_ORDER))); ax.set_yticklabels(SUBJECT_ORDER, fontsize=6.8)
ax.invert_yaxis()
ax.set_xlabel("Resting RMS, 20–450 Hz (mV at ADC input)")
med = np.median(Bm.values)
ax.set_title(f"Resting noise floor (median {med:.1f} mV = {med/2:.1f} LSB)", loc="left", fontsize=7.5)
panels["b"] = ax

ax = fig.add_subplot(gs[1, 0])
k = (f1 >= 2) & (f1 <= 500)
for P_, col, lbl in ((Ph, fs.INK, "hold (3.5 s steady grasp)"), (Pr, fs.INK2, "relaxed rest (breaks)")):
    q = np.percentile(10 * np.log10(P_.reshape(-1, P_.shape[-1])[:, k]), [25, 50, 75], axis=0)
    ref = np.median(10 * np.log10(Pr.reshape(-1, Pr.shape[-1])[:, (f1 >= 300) & (f1 <= 400)]))
    ax.fill_between(f1[k], q[0] - ref, q[2] - ref, color=col, alpha=0.15, lw=0)
    ax.plot(f1[k], q[1] - ref, color=col, lw=0.9, label=lbl)
if fs.REVIEW:
    # revision: el reposo sin la interferencia de la baliza (muestras reparadas, s26_10_wifi.py),
    # llevado a la resolucion de 1 Hz (promedio de 4 bins de 0.24 Hz) y a su propia referencia de 300-400 Hz
    WZ = np.load(DATA26 / "wifi_clean.npz")
    Prep = WZ["Pr"][:, :, list(WZ["methods"]).index("repair"), :].reshape(-1, len(WZ["f4"]))
    Prep1 = np.apply_along_axis(lambda v: np.convolve(v, [0.125, 0.25, 0.25, 0.25, 0.125], "same"), 1, Prep)[:, ::4]
    qr = np.median(10 * np.log10(Prep1[:, k]), axis=0)
    refr = np.median(10 * np.log10(Prep1[:, (f1 >= 300) & (f1 <= 400)]))
    ax.plot(f1[k], qr - refr, color=fs.C_ENV, lw=0.8, label="rest, Wi-Fi dips repaired")
    ax.legend(loc="upper right", fontsize=6)
ax.axvline(60, color=fs.GRID, lw=0.8, zorder=0)
ax.text(64, -5, "60 Hz\nanalog notch", fontsize=5.6, color=fs.INK2, va="bottom")
ax.set_xlim(0, 500)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("dB relative to rest\nat 300–400 Hz")
ax.legend(loc="upper right", fontsize=6)
ax.set_title(f"Power spectral density, {8 * len(SUBJECT_ORDER)} channels (median and IQR)", loc="left", fontsize=7.5)
panels["c"] = ax

ax = fig.add_subplot(gs[1, 1])
k4 = (f4 >= 15) & (f4 <= 125)
q = np.median(10 * np.log10(Pr4.reshape(-1, Pr4.shape[-1])[:, k4]), axis=0)
ax.plot(f4[k4], q - np.median(q), color=fs.INK2, lw=0.7)
# marcas de n x 9.766 Hz solo hasta la altura de los picos: la franja de
# arriba queda libre para el rotulo
ax.vlines([n * BEACON_HZ for n in range(2, 13)], -3, 10.5, color="#a3aab2", lw=0.5, alpha=0.8, zorder=0)
ax.set_ylim(-3, 16.5 if fs.REVIEW else 14.5)
ax.set_xlim(15, 125)
ax.set_xlabel("Frequency (Hz), 0.24 Hz resolution")
ax.set_ylabel("dB relative to\nthe median")
ax.set_title("Resting power spectral density: lines at n × 9.766 Hz", loc="left", fontsize=7.5)
# el periodo coincide con el intervalo de beacon por omision (100 TU); que
# esa sea la fuente es lo mas probable, no esta probado (ver el reporte).
# En revision una tercera linea explica la curva azul (sin otro rotulo encima de las marcas)
ax.text(0.98, 0.95, f"period 102.4 ms, the default Wi-Fi beacon interval;\nthe lines carry "
        f"{W.comb_power_pct.median():.0f} % of resting "
        f"power (p10–p90 {W.comb_power_pct.quantile(0.1):.0f}–{W.comb_power_pct.quantile(0.9):.0f} %)"
        + ("\nblue: beacon-hit samples repaired (Figure 14)" if fs.REVIEW else ""),
        transform=ax.transAxes, ha="right", va="top", fontsize=6.2)
if fs.REVIEW:
    # revision: el mismo reposo con las muestras de cada baliza reparadas (figura 14)
    WZ = np.load(DATA26 / "wifi_clean.npz")
    Prep = WZ["Pr"][:, :, list(WZ["methods"]).index("repair"), :].reshape(-1, len(WZ["f4"]))
    qr = np.median(10 * np.log10(Prep[:, k4]), axis=0)
    ax.plot(f4[k4], qr - np.median(q), color=fs.C_ENV, lw=0.8)
panels["d"] = ax

ax = fig.add_subplot(gs[2, :])
T2 = T.dropna(subset=["act_db", "env_delta"])
ax.plot(T2.act_db, T2.env_delta, "o", ms=1.3, color=fs.INK2, alpha=0.35, mew=0, rasterized=True)
bins = np.arange(-3, 25, 1.0)
cen = (bins[:-1] + bins[1:]) / 2
qq = [T2.env_delta[(T2.act_db >= a) & (T2.act_db < b)].quantile([0.25, 0.5, 0.75]).values
      if ((T2.act_db >= a) & (T2.act_db < b)).sum() > 15 else [np.nan] * 3 for a, b in zip(bins[:-1], bins[1:])]
qq = np.array(qq)
ax.fill_between(cen, qq[:, 0], qq[:, 2], color=fs.INK, alpha=0.18, lw=0)
ax.plot(cen, qq[:, 1], color=fs.INK, lw=1.2, label="median and IQR per 1 dB bin")
thr = cen[np.nanargmax(qq[:, 1] > 20)] if np.any(qq[:, 1] > 20) else np.nan
ax.axvline(thr, color=fs.INK2, lw=0.8, ls="--")
# a la izquierda de la linea: a la derecha caia sobre la nube de puntos
ax.text(thr - 0.3, np.nanmax(qq[:, 2]) * 0.95, f"envelope rises > 20 mV only\nabove ~{thr:.0f} dB of raw activation",
        fontsize=6.2, va="top", ha="right")
ax.set_xlim(-3, 24)
ax.set_ylim(-100, np.nanpercentile(T2.env_delta, 99.5))
ax.set_xlabel("Raw sEMG activation during hold (dB relative to rest)")
ax.set_ylabel("Hardware envelope\nrise (mV)")
ax.legend(loc="upper left", fontsize=6)
ax.set_title(f"Hardware envelope vs. raw activation ({len(T2)} trial × channel pairs)", loc="left",
             fontsize=7.5)
panels["e"] = ax
fs.save(fig, FIG26 / f"{FIGP}_5_semg", panels=panels)
print("umbral de la envolvente ~", thr, "dB")
