"""Figura 7 (sabado 26): la MyoProBand midiendo cada variable cuando toca.

Sesion S01: la mas completa (0 % de perdida, 42/42 ensayos a tasa completa,
temperatura y los dos sensores de carga validos, ningun canal capta el
barrido). El mensaje no es que todo salga perfecto, sino que el brazalete deja
decidir cuando se mide cada variable auxiliar: aqui la carga de contacto y la
temperatura durante el agarre, y el barrido de impedancia en el reposo, con el
sEMG y el IMU continuos de fondo.

(a) un bloque completo (6 repeticiones de un agarre) en un solo eje: sEMG de
    8 canales (20-450 Hz), giroscopio, carga de contacto y temperatura (puntos
    solo durante el agarre, unidos dentro de cada agarre y nunca a traves del
    reposo) y cada barrido dibujado en el instante en que se excito cada
    frecuencia (arranque + 55 ms + k x 41.9 ms; 02_sweep_timing.py);
(b) la misma repeticion (la tercera) de cada uno de los siete agarres, con el
    patron de activacion de los 8 electrodos alrededor del antebrazo (mediana
    de las 6 repeticiones, dB sobre el reposo).

Uso: python s26_fig7_showcase.py [sujeto] [bloque] [repeticion]
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
from s26_common import (C_SWEEP, DATA26, FIG26, GRASP_COLOR, GRASP_ORDER, GRASP_SHORT, bandpass_fft,
                        find_sessions, grid_raw, imu_arrays, load26, prest_times, sweep_onsets, win)

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "S01"
BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 4
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 3
# --swap-load-sides: los sensores de carga se pusieron al reves (el 1-10, segun el usuario): p1 quedo del
# lado extensor y p2 del flexor. Solo cambia el rotulo de lado; los datos y la marca de cada sensor no.
SWAP_SIDES = "--swap-load-sides" in sys.argv
T1, STEP, ONSET = 0.055, 0.0419, 0.15
REPS = 6
TRACE = "#2d333b"
OMEGA = "Ω"

fs.apply()
S = pd.read_csv(DATA26 / "sessions.csv")
T = pd.read_csv(DATA26 / "semg_trials.csv")
name = S[(S.subject == SUBJ) & S.complete].session.iloc[0]
x = load26([p for p in find_sessions() if p.name == name][0])
tg, Y = grid_raw(x)
B = bandpass_fft(Y)
pr = prest_times(x)
ons = sweep_onsets(x)
t_imu, acc, gyr = imu_arrays(x)
gm = np.linalg.norm(gyr - np.median(gyr, 0), axis=1)
sen = x.sensors
tv = sen["temp"] > -126
# revision (--review): cada canal en veces su RMS en reposo (sin la baliza, s26_10_wifi.py), para que
# la ganancia distinta de cada modulo (E3 alta, E4 baja) no decida que canal se ve grande
if fs.REVIEW:
    WC = pd.read_csv(DATA26 / "wifi_clean.csv")
    rr_ = WC[(WC.subject == SUBJ) & (WC.method == "repair")].set_index("channel").rest_rms_mv
    NORM = np.array([rr_[f"E{i+1}"] for i in range(8)])
    B = B / NORM[:, None]


def sweep_for(p):
    """(arranque visto en el sEMG, registro) del barrido que disparo el Prest p."""
    on = [o for o in ons if 0.08 <= o - p <= 0.30]
    t_on = on[0] if on else p + ONSET
    w = [w for w in x.sweeps if 3.5 < w["t_arrival"] - p < 7.5]
    return t_on, (w[0] if w else None)


def sweeps_in(a, b):
    out = []
    for p in pr:
        if a - 4.5 < p < b:
            t_on, w = sweep_for(p)
            if w is not None:
                out.append((p, t_on, w))
    return out


def shade(ax, a, b, trials, sw, alpha=0.10):
    for tr in trials:
        if tr.t_rest > a and tr.t_grasp < b:
            ax.axvspan(tr.t_grasp - a, tr.t_rest - a, color=GRASP_COLOR[tr.grasp], alpha=alpha, lw=0, zorder=0)
    for p, t_on, w in sw:
        ax.axvspan(t_on - a, t_on + T1 + 99 * STEP - a, color=C_SWEEP, alpha=alpha * 0.8, lw=0, zorder=0)


def draw_semg(ax, a, b, sp, lw=0.18, labels=True):
    sl = win(tg, a, b)
    tz = tg[sl] - a
    for i in range(8):
        yv = B[i, sl]
        ax.plot(tz, np.clip(yv, -0.6 * sp, 0.6 * sp) - i * sp, lw=lw, color=TRACE, rasterized=True)
        if labels:
            lab = f"E{i+1}" + (" flexor" if i == 0 else " extensor" if i == 4 else "")
            ax.text(-0.012, -i * sp, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=6.2, color=fs.INK)
    ax.set_ylim(-7.65 * sp, 0.65 * sp)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


def draw_aux(ax_p, ax_t, a, b, trials, ms=2.6):
    """Temperatura: sus tres lecturas por agarre, unidas solo dentro del agarre.
    Carga: solo la lectura que ya no arrastra el filtro de mediana de la placa
    auxiliar (la 3a de cada agarre, s26_common: load_ok)."""
    for tr in trials:
        if not (tr.t_rest > a and tr.t_grasp < b):
            continue
        k = np.where((sen["t_est"] >= tr.t_grasp - 0.3) & (sen["t_est"] < tr.t_rest + 0.8))[0]
        if not len(k):
            continue
        kp = k[sen["load_ok"][k]]
        if fs.REVIEW:
            # revision: tambien las lecturas 1-2, en gris (arrastran el filtro de mediana del sostener anterior)
            k0 = k[~sen["load_ok"][k]]
            ax_p.plot(sen["t_est"][k0] - a, sen["p1"][k0], "o", ms=ms * 0.8, color="#b8bec5", mew=0, zorder=2)
            ax_p.plot(sen["t_est"][k0] - a, sen["p2"][k0], "o", ms=ms * 0.8, mfc="white", mec="#b8bec5", mew=0.6,
                      zorder=2)
        ax_p.plot(sen["t_est"][kp] - a, sen["p1"][kp], "o", ms=ms, color=fs.INK, mew=0, zorder=3)
        ax_p.plot(sen["t_est"][kp] - a, sen["p2"][kp], "o", ms=ms, mfc="white", mec=fs.INK2, mew=0.7, zorder=3)
        kt = k[tv[k]]
        if len(kt):
            ax_t.plot(sen["t_est"][kt] - a, sen["temp"][kt], lw=0.7, color=fs.INK)
            ax_t.plot(sen["t_est"][kt] - a, sen["temp"][kt], "s", ms=ms * 0.85, color=fs.INK, mew=0, zorder=3)


def draw_sweeps(ax, a, sw, label_first=False, ms=1.5):
    for n, (p, t_on, w) in enumerate(sw):
        tk = t_on + T1 + (np.arange(len(w["z"])) + 0.5) * STEP - a
        ax.plot(tk, w["z"] / 1e3, lw=0.6, color=C_SWEEP)
        ax.plot(tk, w["z"] / 1e3, "o", ms=ms, color=C_SWEEP, mew=0)
        if label_first and n == 0:
            # 10 kHz a la izquierda de su punto: a la derecha quedaba pegado al de 100 kHz
            for fk, dx, dy, ha in ((2, 3, 1, "left"), (10, -4, 2, "right"), (100, 3, 4, "left")):
                j = int(np.argmin(np.abs(w["freq"] / 1e3 - fk)))
                ax.plot([tk[j]], [w["z"][j] / 1e3], "o", ms=3.0, mfc="none", mec=fs.INK, mew=0.6)
                ax.annotate(f"{fk} kHz", (tk[j], w["z"][j] / 1e3), xytext=(dx, dy), textcoords="offset points",
                            fontsize=5.6, color=fs.INK2, ha=ha, va="bottom")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([10, 30, 100]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["10", "30", "100"])


def row_label(ax, text, sub=None):
    ax.set_ylabel(text, rotation=0, ha="right", va="center", fontsize=7, labelpad=6)
    if sub:
        ax.text(1.008, 0.5, sub, transform=ax.transAxes, fontsize=5.8, color=fs.INK2, ha="left", va="center",
                linespacing=1.25)


# ---------------------------------------------------------------- ventanas
blk = [t for t in x.trials if t.index // REPS == BLOCK]
a0 = blk[0].t_grasp - 2.0
b0 = blk[-1].t_rest + 5.0
sw_a = sweeps_in(a0, b0)
sl = win(tg, a0, b0)
sp = 2.3 * np.nanpercentile(np.abs(B[:, sl]), 99.5)
sp = float(np.ceil(sp / 5) * 5) if fs.REVIEW else float(np.ceil(sp / 20) * 20)
reps = {g: [t for t in x.trials if t.grasp == g and t.rep == REP][0] for g in GRASP_ORDER}
z_lo = 0.85 * min(w["z"].min() for w in x.sweeps) / 1e3
z_hi = 1.25 * max(w["z"][0] for w in x.sweeps) / 1e3
p_all = (np.r_[sen["p1"], sen["p2"]] if fs.REVIEW else np.r_[sen["p1"][sen["load_ok"]], sen["p2"][sen["load_ok"]]])
p_lo, p_hi = np.nanmin(p_all), np.nanmax(p_all)
t_all = sen["temp"][tv]

fig = plt.figure(figsize=(7.2, 10.0))
outer = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.2, left=0.115, right=0.885,
                         top=0.955, bottom=0.04)
# filas de (a) con nombre: sin IMU (--no-imu) no hay fila de giroscopio
ROWS_A = ["sched", "semg"] + ([] if fs.NO_IMU else ["gyro"]) + ["load", "temp", "z"]
H_A = dict(sched=0.34, semg=3.0, gyro=0.5, load=0.85, temp=0.6, z=1.15)
ga = outer[0].subgridspec(len(ROWS_A), 1, height_ratios=[H_A[k] for k in ROWS_A], hspace=0.14)
gb = outer[1].subgridspec(5, 7, height_ratios=[1.0, 2.3, 0.6, 0.48, 0.9], hspace=0.12, wspace=0.08)

# ------------------------------------------------------------------ (a)
A = [fig.add_subplot(ga[i]) for i in range(len(ROWS_A))]
RA = dict(zip(ROWS_A, A))
for ax_ in A[1:]:
    ax_.sharex(A[0])
ax = RA["sched"]
for tr in blk:
    ax.add_patch(mpatches.Rectangle((tr.t_grasp - a0, 0.1), tr.t_rest - tr.t_grasp, 0.8,
                                    color=GRASP_COLOR[tr.grasp], lw=0))
for p, t_on, w in sw_a:
    ax.add_patch(mpatches.Rectangle((t_on - a0, 0.1), T1 + 99 * STEP, 0.8, facecolor=C_SWEEP, alpha=0.35, lw=0))
    ax.add_patch(mpatches.Rectangle((t_on - a0, 0.1), T1 + 99 * STEP, 0.8, facecolor="none", edgecolor=C_SWEEP,
                                    hatch="////", lw=0))
ax.set_ylim(0, 1)
ax.set_yticks([])
for s0 in ("left", "bottom"):
    ax.spines[s0].set_visible(False)
ax.tick_params(axis="x", length=0)
row_label(ax, "Schedule")
g0 = blk[0].grasp
handles = [mpatches.Patch(color=GRASP_COLOR[g0], label=f"hold, 5 s ({GRASP_SHORT[g0]}): contact load + temperature"),
           mpatches.Patch(facecolor=C_SWEEP, alpha=0.5, edgecolor=C_SWEEP, hatch="////",
                          label="rest: bioimpedance sweep 2→100 kHz, 4.2 s"),
           mpatches.Patch(facecolor="white", edgecolor=fs.GRID, label="rest, no auxiliary measurement")]
leg_a = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.105, 0.997), ncol=3, fontsize=6.2,
           handlelength=1.3, columnspacing=1.2, borderaxespad=0.2)

ax = RA["semg"]
shade(ax, a0, b0, blk, sw_a, alpha=0.08)
draw_semg(ax, a0, b0, sp)
bar, bar_lbl = (5.0, "10 × rest\nRMS") if fs.REVIEW else (50.0, "100 mV")
ax.plot([1.006, 1.006], [-7 * sp - bar, -7 * sp + bar], transform=ax.get_yaxis_transform(), color=fs.INK,
        lw=1.1, clip_on=False)
ax.text(1.014, -7 * sp, bar_lbl, transform=ax.get_yaxis_transform(), fontsize=5.8, va="center")
ax.text(1.008, -2.2 * sp, "sEMG\n8 × 1 kHz,\ncontinuous", transform=ax.get_yaxis_transform(), fontsize=5.8,
        color=fs.INK2, va="center", linespacing=1.25)

if "gyro" in RA:
    ax = RA["gyro"]
    shade(ax, a0, b0, blk, sw_a, alpha=0.08)
    k = (t_imu >= a0) & (t_imu < b0)
    ax.plot(t_imu[k] - a0, gm[k], lw=0.5, color=fs.INK2)
    ax.set_ylim(0, max(1.0, np.percentile(gm[k], 99.9) * 1.15))
    row_label(ax, "Gyro\n(°/s)", f"IMU ~{int(np.round(k.sum() / (b0 - a0), -1))} Hz,\ncontinuous")

ax_p, ax_t = RA["load"], RA["temp"]
for ax_ in (ax_p, ax_t):
    shade(ax_, a0, b0, blk, sw_a, alpha=0.08)
draw_aux(ax_p, ax_t, a0, b0, blk)
pk = (sen["t_est"] >= a0) & (sen["t_est"] < b0)
pl = pk if fs.REVIEW else pk & sen["load_ok"]
lo_, hi_ = np.nanmin(np.r_[sen["p1"][pl], sen["p2"][pl]]), np.nanmax(np.r_[sen["p1"][pl], sen["p2"][pl]])
ax_p.set_ylim(lo_ - 0.25 * (hi_ - lo_) - 1, hi_ + 0.45 * (hi_ - lo_) + 1)
# lecturas por sostener en esta sesion: 3 el sabado (cada 2 s); ~15 con la placa auxiliar del 1-10
n_hold = int(np.median([np.sum((sen["t_est"] >= tr.t_grasp - 0.3) & (sen["t_est"] < tr.t_rest + 0.6))
                        for tr in x.trials]))
if n_hold <= 3:
    load_sub = ("read every 2 s,\nholds only; grey:\nreadings 1–2 (filter\ncarry-over)" if fs.REVIEW
                else "read every 2 s,\nholds only; last\nreading shown")
    temp_sub = "every 2 s,\nholds only"
else:
    load_sub = (f"{n_hold} readings per\nhold, holds only;" + ("\ngrey: readings 1–2\n(filter carry-over)" if fs.REVIEW
                                                           else "\nfrom the 3rd on"))
    temp_sub = f"~{n_hold / 5:.0f} per s,\nholds only"
row_label(ax_p, "Contact\nload (kPa)", load_sub)
side1, side2 = ("extensor side (E5)", "flexor side (E1)") if SWAP_SIDES else ("flexor side (E1)", "extensor side (E5)")
ax_p.plot([], [], "o", ms=2.6, color=fs.INK, label=f"p1 · {side1}")
ax_p.plot([], [], "o", ms=2.6, mfc="white", mec=fs.INK2, mew=0.7, label=f"p2 · {side2}, provisional")
ax_p.legend(loc="upper left", fontsize=5.8, ncol=2, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
tk_ = pk & tv
if tk_.any():
    lo_, hi_ = sen["temp"][tk_].min(), sen["temp"][tk_].max()
    ax_t.set_ylim(lo_ - 0.35 * max(hi_ - lo_, 0.1), hi_ + 0.35 * max(hi_ - lo_, 0.1))
row_label(ax_t, "Skin\ntemp. (°C)", temp_sub)
if fs.REVIEW:
    # revision: carga y temperatura con la misma escala en (a) y en (b), la de toda la sesion
    ax_p.set_ylim(p_lo - 0.1 * (p_hi - p_lo), p_hi + 0.45 * (p_hi - p_lo))
    if len(t_all):
        ax_t.set_ylim(t_all.min() - 0.15, t_all.max() + 0.15)

ax = RA["z"]
shade(ax, a0, b0, blk, sw_a, alpha=0.08)
draw_sweeps(ax, a0, sw_a, label_first=True)
ax.set_ylim(z_lo, z_hi)
row_label(ax, f"|Z| (k{OMEGA})", "99 points in 4.2 s,\nafter each release")
ax.set_xlim(0, b0 - a0)
ax.set_xlabel("Time (s)", labelpad=1)
for ax_ in A[:-1]:
    plt.setp(ax_.get_xticklabels(), visible=False)

# ------------------------------------------------------------------ (b)
act = (T[(T.subject == SUBJ)].groupby(["grasp", "channel"]).act_db.median().unstack()
       [[f"E{i}" for i in range(1, 9)]])
anorm = Normalize(vmin=0, vmax=15)
acmap = cm.get_cmap("Greys")
cols = []
for c, g in enumerate(GRASP_ORDER):
    tr = reps[g]
    a, b = tr.t_grasp - 1.0, tr.t_grasp + 11.0
    sw = sweeps_in(a, b)
    col = [fig.add_subplot(gb[0, c], projection="polar")] + [fig.add_subplot(gb[r, c]) for r in range(1, 5)]
    cols.append(col)
    # anillo: los 8 electrodos alrededor del antebrazo, E1 arriba, en sentido horario
    ax = col[0]
    th = np.pi / 2 - np.arange(8) * 2 * np.pi / 8
    ax.bar(th, np.ones(8), width=2 * np.pi / 8 * 0.9, bottom=0.6, color=acmap(anorm(act.loc[g].values)),
           edgecolor="white", lw=0.6)
    ax.set_ylim(0, 1.6)
    ax.set_axis_off()
    for r in range(1, 5):
        shade(col[r], a, b, [tr], sw, alpha=0.10)
    draw_semg(col[1], a, b, sp, lw=0.16, labels=(c == 0))
    draw_aux(col[2], col[3], a, b, [tr], ms=2.2)
    draw_sweeps(col[4], a, sw, ms=1.1)
    col[2].set_ylim(p_lo - 0.1 * (p_hi - p_lo), p_hi + 0.1 * (p_hi - p_lo))
    if len(t_all):
        col[3].set_ylim(t_all.min() - 0.15, t_all.max() + 0.15)
    col[4].set_ylim(z_lo, z_hi)
    for r in range(1, 5):
        col[r].set_xlim(0, b - a)
        col[r].set_xticks([1, 6, 11])
        col[r].set_xticklabels(["0", "5", "10"])
        col[r].tick_params(axis="both", labelsize=6)
        if r < 4:
            plt.setp(col[r].get_xticklabels(), visible=False)
        if c > 0:
            plt.setp(col[r].get_yticklabels(), visible=False)
for r, lbl in zip(range(2, 5), ["Contact\nload (kPa)", "Skin\ntemp. (°C)", f"|Z| (k{OMEGA})"]):
    cols[0][r].set_ylabel(lbl, rotation=0, ha="right", va="center", fontsize=6.6, labelpad=4)

# cabeceras: nombre y barra de color por columna, a partir de la geometria final
fig.canvas.draw()
extra_b = []
for c, g in enumerate(GRASP_ORDER):
    p1_ = cols[c][1].get_position()
    p0_ = cols[c][0].get_position()
    xm = (p1_.x0 + p1_.x1) / 2
    extra_b.append(fig.add_artist(mpatches.Rectangle((p1_.x0 + 0.004, p0_.y1 + 0.004), p1_.width - 0.008, 0.004,
                                      transform=fig.transFigure, color=GRASP_COLOR[g], lw=0)))
    extra_b.append(fig.text(xm, p0_.y1 + 0.011, GRASP_SHORT[g], ha="center", va="bottom", fontsize=6.6, color=fs.INK))
p_last = cols[-1][4].get_position()
p_first = cols[0][4].get_position()
extra_b.append(fig.text((p_first.x0 + p_last.x1) / 2, p_last.y0 - 0.028, "Time from grasp cue (s)", ha="center",
                        va="top", fontsize=7))
sm = cm.ScalarMappable(norm=anorm, cmap=acmap)
p0_ = cols[-1][0].get_position()
cax = fig.add_axes([0.9, p0_.y0 + 0.1 * p0_.height, 0.008, 0.8 * p0_.height])
cb = fig.colorbar(sm, cax=cax)
cb.set_label("activation (dB\nrelative to rest)", fontsize=5.6)
cb.ax.tick_params(labelsize=5.6, length=2)
cb.outline.set_visible(False)
extra_b += [cax, fig.text(0.9, p0_.y0 - 0.004, "ring: E1 (flexor)\nat top, E2–E8 in\nnumbering order;\nE5 (extensor)\nat bottom", fontsize=5.4,
         color=fs.INK2, va="top", ha="left", linespacing=1.2)]
out = FIG26 / (f"{FIGP}_7_showcase" if (SUBJ, BLOCK, REP) == ("S01", 4, 3) else f"{FIGP}_7_showcase_{SUBJ}_{BLOCK}_{REP}")
if SWAP_SIDES:
    out = out.with_name(out.name + "_load-sides-fixed")        # nombre propio: no pisa la version anterior
fs.save(fig, out, panels={"a": A, "b": [a_ for col in cols for a_ in col]},
        extras={"a": [leg_a], "b": extra_b})
print("ok", name, "bloque", BLOCK, GRASP_SHORT[blk[0].grasp], "ventana", round(a0, 1), round(b0, 1),
      "sp", sp, "barridos en (a)", len(sw_a))
