"""Figura 15 (sabado 26): ganancia de cada canal y como mostrar los ocho juntos.

(a) RMS en reposo de cada canal (20-450 Hz, sin la baliza: muestras
    reparadas, s26_10_wifi.py) relativo a la mediana de los ocho de su sesion.
    Si el ruido de entrada fuera igual en todos, esto seria la ganancia
    relativa; es un limite inferior: parte del ruido entra despues de la
    ganancia de cada modulo, como el hundimiento de la baliza (b).
(b) profundidad del hundimiento de la baliza por canal, relativa a la de su
    sesion: ~1 en todos los canales bien apoyados, o sea que entra despues de
    la ganancia; los que se salen son canales con mal contacto.
(c, d, e) un mismo sostener en mV, en veces el RMS en reposo de cada canal, y
    con escala propia por canal (percentil 99.5 del tramo).
Escribe data/s26/channel_gain.csv.
Uso: python s26_fig15_channels.py [sujeto ensayo]   (por omision S01 27)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
from s26_common import DATA26, FIG26, SUBJECT_ORDER, bandpass_fft, find_sessions, grid_raw, load26, win

fs.apply()
SUBJ = sys.argv[1] if len(sys.argv) > 1 else "S01"
TRIAL = int(sys.argv[2]) if len(sys.argv) > 2 else 27
WC = pd.read_csv(DATA26 / "wifi_clean.csv")
WZ = np.load(DATA26 / "wifi_clean.npz")
CH = [f"E{i}" for i in range(1, 9)]

# (a) suelo en reposo sin baliza, relativo a la sesion; (b) hundimiento relativo
R = WC[WC.method == "repair"].pivot(index="subject", columns="channel", values="rest_rms_mv").reindex(SUBJECT_ORDER)[CH]
rel = R.div(R.median(axis=1), axis=0)
dips = pd.DataFrame({s_: np.min(WZ[f"tpl_{s_}"], axis=1) for s_ in SUBJECT_ORDER}, index=CH).T
drel = dips.div(dips.median(axis=1), axis=0)
G = pd.DataFrame({"rest_floor_rel_median": rel.median(), "rest_floor_rel_p25": rel.quantile(0.25),
                  "rest_floor_rel_p75": rel.quantile(0.75), "beacon_dip_rel_median": drel.median()})
G.index.name = "channel"
G.to_csv(DATA26 / "channel_gain.csv")

fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.25], hspace=0.5, wspace=0.28, left=0.08, right=0.98,
                      top=0.94, bottom=0.07)
panels = {}
rng = np.random.default_rng(4)
for col, (D, title, ylab) in enumerate(((rel, "Resting floor without the Wi-Fi dips", "Relative to the session's\nmedian channel"),
                                        (drel, "Wi-Fi beacon dip depth (same scale)", None))):
    ax = fig.add_subplot(gs[0, col])
    for j, c in enumerate(CH):
        v = D[c].values
        ax.plot(j + rng.uniform(-0.15, 0.15, len(v)), v, "o", ms=2.6, color=fs.INK2, alpha=0.7, mew=0)
        ax.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color=fs.INK, lw=1.4)
    ax.axhline(1, color=fs.GRID, lw=0.8, zorder=0)
    ax.set_xticks(range(8))
    ax.set_xticklabels(CH, fontsize=6.5)
    ax.set_ylim(0.5, 2.0)
    if ylab:
        ax.set_ylabel(ylab, fontsize=7)
    ax.set_title(title, loc="left", fontsize=7.2)
    out = int((D.values > 2.0).sum())
    if out:
        ax.text(0.98, 0.97, f"{out} above the axis", transform=ax.transAxes, ha="right", va="top", fontsize=5.8,
                color=fs.INK2)
    panels["a" if col == 0 else "b"] = ax
e3, e4 = rel["E3"].median(), rel["E4"].median()
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
ax.text(0.0, 0.95, f"E3 / E4 at rest: {e3 / e4:.2f}×\n(median over {len(SUBJECT_ORDER)} sessions)\n\n"
        "If the input noise is similar on\nall channels, this estimates the\ngain ratio. Noise that enters\n"
        "after the gains (as the dip,\nnearly equal on all channels,\n"
        f"E3/E4 {drel['E3'].median() / drel['E4'].median():.2f}×) pulls it toward 1,\n"
        "so the true ratio is likely\nhigher.",
        transform=ax.transAxes, va="top", fontsize=6.6, color=fs.INK, linespacing=1.35)
panels["a"] = [panels["a"], ax]

# (c, d, e) un sostener, tres escalas
S = pd.read_csv(DATA26 / "sessions.csv")
name = S[(S.subject == SUBJ) & S.complete].session.iloc[0]
x = load26([p for p in find_sessions() if p.name == name][0])
tg, Y = grid_raw(x)
B = bandpass_fft(Y)
tr = x.trials[TRIAL]
a, b = tr.t_grasp - 1.0, tr.t_rest + 1.5
sl = win(tg, a, b)
tz = tg[sl] - tr.t_grasp
rr = R.loc[SUBJ].values
# el valor de la barra de escala va en el titulo: como rotulo junto a la barra se salia del marco
views = (("In mV (bar: 100 mV)", B[:, sl], "mV", 50.0, ""),
         ("× resting RMS of each channel (bar: 10)", B[:, sl] / rr[:, None], "rest", 5.0, ""),
         ("Own scale per channel (bar: its p99.5)",
          B[:, sl] / np.nanpercentile(np.abs(B[:, sl]), 99.5, axis=1)[:, None], "auto", 0.5, ""))
for col, (title, V, kind, half, lbl) in enumerate(views):
    ax = fig.add_subplot(gs[1, col])
    sp = 2.2 * np.nanpercentile(np.abs(V), 99.5)
    for i in range(8):
        ax.plot(tz, np.clip(V[i], -0.6 * sp, 0.6 * sp) - i * sp, lw=0.2, color=fs.INK, rasterized=True)
        if col == 0:
            ax.text(-0.02, -i * sp, CH[i], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.3)
    ax.axvspan(0, tr.t_rest - tr.t_grasp, color="#e8f3ec", lw=0, zorder=0)
    x_bar = tz[-1] + 0.15
    ax.plot([x_bar, x_bar], [-7 * sp - half, -7 * sp + half], color=fs.INK, lw=1.1, clip_on=False)
    ax.set_ylim(-7.8 * sp, 0.7 * sp)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time from grasp cue (s)")
    ax.set_title(title, loc="left", fontsize=7.2)
    panels["cde"[col]] = ax
fs.save(fig, FIG26 / f"{FIGP}_15_channels", panels=panels)
print(f"ok: E3/E4 en reposo {e3 / e4:.2f}; hundimiento E3/E4 {drel['E3'].median() / drel['E4'].median():.2f}; "
      f"ejemplo {SUBJ} ensayo {TRIAL + 1}")
