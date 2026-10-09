"""Figura 10 (sabado 26): caracteristicas del instrumento.

(a) correlacion a retardo cero entre los 8 crudos durante el sostener y (b) en
    el reposo relajado (media de 6 participantes, transformada de Fisher);
(c) correlacion segun la distancia alrededor del antebrazo: sostener, reposo,
    y reposo sin las lineas de baliza Wi-Fi;
(d) retardo de la envolvente del hardware frente al RMS del crudo al agarrar;
(e) monitoreo de la interfaz: |Z| a 5 kHz frente al RMS en reposo, dentro de
    cada sesion (rho de Spearman por participante).

La separabilidad de agarres salio de aqui: es secundaria y tiene su propia
figura, la ultima (s26_fig12_models.py), para no mostrar lo mismo dos veces.
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

fs.apply()
Cz = np.load(DATA26 / "corr.npz")
LG = pd.read_csv(DATA26 / "envelope_lag.csv")
IF = pd.read_csv(DATA26 / "interface.csv")


def fisher_mean(key):
    M = np.array([np.arctanh(np.clip(Cz[f"{key}_{s_}"], -0.999, 0.999)) for s_ in SUBJECT_ORDER])
    return np.tanh(M.mean(0))


def by_distance(key):
    out = []
    for s_ in SUBJECT_ORDER:
        C = Cz[f"{key}_{s_}"]
        out.append([np.mean([C[i, (i + d) % 8] for i in range(8)]) for d in range(1, 5)])
    return np.array(out)


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


W_IN, H_IN = 7.2, 5.0
fig = plt.figure(figsize=(W_IN, H_IN))
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.5, left=0.08, right=0.97, top=0.93, bottom=0.11)
panels = {}
chs = [f"E{i}" for i in range(1, 9)]
for j, (key, title) in enumerate((("hold", "Hold"), ("rest", "Relaxed rest"))):
    ax = fig.add_subplot(gs[0, j])
    M = fisher_mean(key)
    np.fill_diagonal(M, np.nan)
    im = ax.imshow(M, cmap="Greys", vmin=0, vmax=0.6)
    for (r, c), v in np.ndenumerate(M):
        if np.isfinite(v):
            # mismo formato para positivos y negativos (.49, -.08) y sin "-0.00"
            s_v = fs.num(v, ".2f").replace("0.", ".", 1)
            ax.text(c, r, s_v, ha="center", va="center", fontsize=4.8, color="white" if v > 0.38 else fs.INK)
    ax.set_xticks(range(8)); ax.set_xticklabels(chs, fontsize=6)
    ax.set_yticks(range(8)); ax.set_yticklabels(chs, fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{title}: r between channels", loc="left", fontsize=7.2)
    panels["ab"[j]] = ax

ax = fig.add_subplot(gs[0, 2])
for key, c, lbl in (("hold", fs.INK, "hold"), ("rest", "#a3aab2", "rest"), ("restnb", fs.INK2, "rest, beacon lines removed")):
    D = by_distance(key)
    ax.plot(range(1, 5), D.mean(0), "o-", ms=3, color=c, lw=1.0, label=lbl)
    ax.fill_between(range(1, 5), D.min(0), D.max(0), color=c, alpha=0.12, lw=0)
ax.set_xticks(range(1, 5))
ax.set_xticklabels(["1\nneighbour", "2", "3", "4\nopposite"], fontsize=6)
ax.set_ylim(0, 0.55)
ax.set_ylabel("Mean r")
ax.legend(loc="upper right", fontsize=5.8)
ax.set_title("By distance around the forearm", loc="left", fontsize=7.2)
panels["c"] = ax

# (d) retardo de la envolvente
ax = fig.add_subplot(gs[1, 0])
v = LG.lag_on_ms.dropna()
v = v[(v > -100) & (v < 500)]
ax.hist(v, bins=np.arange(-100, 505, 20), color=fs.INK2, edgecolor="white", lw=0.5)
ax.axvline(np.median(v), color=fs.INK, lw=1.0)
ax.text(np.median(v) + 12, ax.get_ylim()[1] * 0.9, f"median {np.median(v):.0f} ms\n(p10–p90 "
        f"{np.percentile(v, 10):.0f}–{np.percentile(v, 90):.0f})", fontsize=6, va="top")
ax.set_xlabel("Envelope − raw RMS onset (ms)")
ax.set_ylabel("Trials × channels")
ax.set_title(f"Envelope lag at onset (n = {len(v)})", loc="left", fontsize=7.2)
panels["d"] = ax

# (e) interfaz: |Z| frente a RMS en reposo, dentro de cada sesion
ax = fig.add_subplot(gs[1, 1:])
tones = dict(zip(SUBJECT_ORDER, fs.tones(len(SUBJECT_ORDER))))
txt = []
for s_ in SUBJECT_ORDER:
    g = IF[IF.subject == s_]
    zrel = 100 * (g.z5k / g.z5k.median() - 1)
    ax.plot(zrel, g.rest_db, "o", ms=2.4, color=tones[s_], mew=0, alpha=0.9)
    txt.append(f"{s_} {fs.num(spearman(g.z5k, g.rest_db), '+.2f')}")
ax.axhline(0, color=fs.GRID, lw=0.8, zorder=0)
ax.axvline(0, color=fs.GRID, lw=0.8, zorder=0)
ax.set_xlim(-60, 150)
ax.set_xlabel("|Z| at 5 kHz, change from session median (%)")
ax.set_ylabel("Resting RMS (dB relative\nto session median)")
ax.text(1.0, 1.0, "Spearman ρ within session:\n" + "\n".join(txt), transform=ax.transAxes, fontsize=5.8,
        ha="right", va="top", color=fs.INK)
ax.set_title("Interface monitoring: |Z| (impedance electrodes) vs resting sEMG, per trial",
             loc="left", fontsize=7.2)
panels["e"] = ax
fs.save(fig, FIG26 / f"{FIGP}_10_instrument", panels=panels)
print("ok")
