"""Figura 2 (sabado 26): una sesion completa, todas las modalidades.

Arriba, la sesion entera en un eje de minutos: protocolo, activacion de los
8 canales (RMS en 250 ms, dB sobre la linea base de las pausas), los 49
barridos como columnas de |Z|(f), carga de contacto, temperatura de piel y
giroscopio. Abajo, ~20 s alrededor de un barrido: crudos, envolvente del
hardware, IMU, carga de contacto y los 99 puntos del barrido puestos en el
instante en que se excito cada frecuencia (arranque + 55 ms + k x 41.9 ms,
medido en 02_sweep_timing.py).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
from s26_common import (C_SWEEP, DATA26, FIG26, GRASP_COLOR, GRASP_SHORT, bandpass_fft, find_sessions,
                        grid_raw, imu_arrays, load26, prest_times, win)

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "S02"
ZOOM_TRIAL = int(sys.argv[2]) if len(sys.argv) > 2 else 26
T1, STEP = 0.055, 0.0419          # 02_sweep_timing.py
ONSET = 0.15                      # Prest -> primer punto (mediana de 03_find_sweeps)

fs.apply()
S = pd.read_csv(DATA26 / "sessions.csv")
name = S[(S.subject == SUBJ) & S.complete].session.iloc[0]
x = load26([p for p in find_sessions() if p.name == name][0])
tg, Y = grid_raw(x)
B = bandpass_fft(Y)
Bk = pd.read_csv(DATA26 / "semg_channels.csv")
base = Bk[Bk.session == name].set_index("channel").base_rms_mv
pr = prest_times(x)
t_imu, acc, gyr = imu_arrays(x)
gm = np.linalg.norm(gyr - np.median(gyr, 0), axis=1)
sen = x.sensors

# filas con nombre: sin IMU (--no-imu) desaparece la del giroscopio en los dos bloques.
# En revision (--review) el detalle no lleva fila de envolvente (va encima de cada crudo) y si
# lleva temperatura; los crudos van en veces su RMS en reposo (sin la baliza, s26_10_wifi.py).
TOP = ["task", "semg", "imp", "load", "temp"] + ([] if fs.NO_IMU else ["gyro"])
BOT = (["task", "raw"] + ([] if fs.REVIEW else ["env"]) + ["load"] + (["temp"] if fs.REVIEW else [])
       + ([] if fs.NO_IMU else ["gyro"]) + ["z"])
H_TOP = dict(task=0.35, semg=1.25, imp=1.1, load=0.7, temp=0.6, gyro=0.55)
H_BOT = dict(task=0.3, raw=3.0 if fs.REVIEW else 2.6, env=0.9, load=0.5, temp=0.45, gyro=0.45, z=1.0)
if fs.REVIEW:
    WC = pd.read_csv(DATA26 / "wifi_clean.csv")
    rest_rms = (WC[(WC.subject == SUBJ) & (WC.method == "repair")].set_index("channel").rest_rms_mv)
fig = plt.figure(figsize=(7.2, 8.1 if fs.NO_IMU else 8.6))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.16, left=0.1, right=0.9, top=0.965, bottom=0.05)
top = outer[0].subgridspec(len(TOP), 1, height_ratios=[H_TOP[k] for k in TOP], hspace=0.12)
bot = outer[1].subgridspec(len(BOT), 1, height_ratios=[H_BOT[k] for k in BOT], hspace=0.14)

# ------------------------------------------------------------- sesion entera
axs = [fig.add_subplot(top[i]) for i in range(len(TOP))]
A = dict(zip(TOP, axs))
for a in axs[1:]:
    a.sharex(axs[0])
m = lambda t: np.asarray(t) / 60.0
ax = A["task"]
blocks = {}
for tr in x.trials:
    ax.axvspan(m(tr.t_grasp), m(tr.t_rest), color=GRASP_COLOR[tr.grasp], lw=0)
    blocks.setdefault(tr.grasp, []).append(tr.t_grasp)
for g, ts in blocks.items():
    ax.text(m(np.mean(ts)) + 0.05, 1.25, GRASP_SHORT[g], ha="center", va="bottom", fontsize=6.2, color=fs.INK)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_ylabel("Task", rotation=0, ha="right", va="center", fontsize=7)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x", length=0)
zb = x.trials[ZOOM_TRIAL]
za, zb_ = zb.t_grasp - 1.0, x.trials[ZOOM_TRIAL + 1].t_rest + 4.7


# activacion: RMS en 250 ms, dB sobre la linea base
ax = A["semg"]
nb = 250
nbin = B.shape[1] // nb
R = np.sqrt(np.nanmean(B[:, :nbin * nb].reshape(8, nbin, nb) ** 2, axis=2))
D = 20 * np.log10(R / base.values[:, None])
te = (np.arange(nbin + 1) * nb) / 1000.0
im = ax.pcolormesh(m(te), np.arange(9) + 0.5, np.clip(D, 0, 20), cmap="Greys", vmin=0, vmax=20,
                   shading="flat", rasterized=True)
ax.set_ylim(8.5, 0.5)
sweep_ch = Bk[Bk.session == name].set_index("channel").sweep_db_med.idxmax()
ax.set_yticks(range(1, 9))
ax.set_yticklabels([f"E{i}" for i in range(1, 9)], fontsize=5.6)
for lbl in ax.get_yticklabels():
    if lbl.get_text() == sweep_ch:
        lbl.set_color(C_SWEEP)
        lbl.set_fontweight("bold")
ax.tick_params(axis="y", length=1.5, pad=1)
ax.add_patch(mpatches.Rectangle((m(za), 0.5), m(zb_) - m(za), 8.0, facecolor="none", edgecolor=C_SWEEP,
                                lw=0.9, zorder=5))
# Fondo blanco: en S07 la fila E1 trae barras negras (captacion del barrido) justo detras.
ax.text(m(zb_) + 0.05, 0.6, "detail", ha="left", va="top", fontsize=5.8, color=fs.INK2, zorder=6,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
ax.set_ylabel("sEMG", rotation=0, ha="right", va="center", fontsize=7, labelpad=14)
cax = ax.inset_axes([1.015, 0.08, 0.012, 0.84])
cb = fig.colorbar(im, cax=cax)
cb.set_label("dB relative\nto rest", fontsize=6)
cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_visible(False)

# impedancia: una columna por barrido, del Prest al siguiente
ax = A["imp"]
f = x.sweeps[0]["freq"] / 1e3
Z = np.array([w["z"] for w in x.sweeps]) / 1e3
tp = []
for w in x.sweeps:
    c = [p for p in pr if 3.5 < w["t_arrival"] - p < 7.5]
    tp.append(max(c) if c else w["t_arrival"] - 5.2)
tp = np.array(tp)
edges = np.r_[tp, tp[-1] + 7.0]
fe = np.r_[f - 0.5, f[-1] + 0.5]
im = ax.pcolormesh(m(edges), fe, Z.T, cmap="Oranges", norm=LogNorm(vmin=7, vmax=np.nanmax(Z)),
                   shading="flat", rasterized=True)
ax.set_yscale("log")
ax.set_ylim(2, 100)
ax.set_yticks([2, 10, 100])
ax.set_yticklabels(["2", "10", "100"], fontsize=6.5)
ax.set_ylabel("Impedance\nf (kHz)", rotation=0, ha="right", va="center", fontsize=7, labelpad=4)
cax = ax.inset_axes([1.015, 0.08, 0.012, 0.84])
cb = fig.colorbar(im, cax=cax)
cb.set_label("|Z| (kΩ)", fontsize=6)
cb.ax.tick_params(labelsize=6, length=2, which="both")
cb.outline.set_visible(False)

# carga de contacto: solo la lectura que ya no arrastra el filtro de mediana de
# la placa auxiliar (la 3a de cada agarre; s26_common: load_ok)
ax = A["load"]
lk = sen["load_ok"]
if fs.REVIEW:
    # revision: tambien las lecturas 1-2 de cada sostener, en gris: el filtro de mediana de la placa
    # auxiliar todavia arrastra ahi valores del sostener anterior (72 % de las 1as repiten la 3a previa)
    ax.plot(m(sen["t_est"][~lk]), sen["p1"][~lk], "o", ms=1.5, color="#b8bec5", mew=0, zorder=1,
            label="readings 1–2 of\neach hold (carry-over)")
    ax.plot(m(sen["t_est"][~lk]), sen["p2"][~lk], "o", ms=1.5, mfc="white", mec="#b8bec5", mew=0.5, zorder=1)
ax.plot(m(sen["t_est"][lk]), sen["p1"][lk], "o", ms=1.8, color=fs.INK, label="p1 · flexor (E1)")
ax.plot(m(sen["t_est"][lk]), sen["p2"][lk], "o", ms=1.8, mfc="white", mec=fs.INK2, mew=0.6,
        label="p2 · extensor (E5),\nprovisional")
ax.set_ylabel("Contact\nload (kPa)", rotation=0, ha="right", va="center", fontsize=7)
# sin marca en el 15 % de arriba: chocaba con el "2" de la fila de impedancia
lo_, hi_ = ax.get_ylim()
ax.set_yticks([v for v in MaxNLocator(3).tick_values(lo_, hi_) if lo_ <= v <= hi_ - 0.15 * (hi_ - lo_)])
ax.set_ylim(lo_, hi_)
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.1), fontsize=6, handletextpad=0.2, borderaxespad=0)
# temperatura de piel
ax = A["temp"]
tv = sen["temp"] > -126
ax.plot(m(sen["t_est"][tv]), sen["temp"][tv], "o", ms=1.8, color=fs.INK)
ax.set_ylabel("Skin\ntemp. (°C)", rotation=0, ha="right", va="center", fontsize=7)
# giroscopio
if "gyro" in A:
    ax = A["gyro"]
    k = t_imu >= 0
    ax.plot(m(t_imu[k]), gm[k], lw=0.35, color=fs.INK2, rasterized=True)
    ax.set_ylim(0, max(2.0, np.percentile(gm, 99.9)))
    ax.set_ylabel("Gyro\n(°/s)", rotation=0, ha="right", va="center", fontsize=7)
axs[-1].set_xlabel("Time in session (min)", labelpad=1)
axs[-1].set_xlim(0, m(x.end_s))
for a in axs[:-1]:
    plt.setp(a.get_xticklabels(), visible=False)
for a in axs[1:]:
    for s0 in ("top", "right"):
        a.spines[s0].set_visible(False)

# ------------------------------------------------------------------- detalle
bx = [fig.add_subplot(bot[i]) for i in range(len(BOT))]
X = dict(zip(BOT, bx))
for a in bx[1:]:
    a.sharex(bx[0])
zp = [p for p in pr if za < p < zb_]
ax = X["task"]
for tr in x.trials:
    if tr.t_rest > za and tr.t_grasp < zb_:
        ax.axvspan(tr.t_grasp - za, tr.t_rest - za, color=GRASP_COLOR[tr.grasp], lw=0)
        ax.text((tr.t_grasp + tr.t_rest) / 2 - za, 1.2, f"hold · {GRASP_SHORT[tr.grasp]}", ha="center",
                va="bottom", fontsize=6.3)
for p in zp:
    ax.add_patch(mpatches.Rectangle((p + ONSET - za, 0), 4.2, 1, facecolor=C_SWEEP, alpha=0.35, lw=0))
    ax.add_patch(mpatches.Rectangle((p + ONSET - za, 0), 4.2, 1, facecolor="none", edgecolor=C_SWEEP,
                                    hatch="////", lw=0))
    ax.text(p + ONSET + 2.1 - za, 1.2, "impedance sweep 2→100 kHz", ha="center", va="bottom", fontsize=6.3)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_ylabel("Task", rotation=0, ha="right", va="center", fontsize=7)
for s0 in ("left", "bottom"):
    ax.spines[s0].set_visible(False)
ax.tick_params(axis="x", length=0)

# crudos apilados (solo sin desplazamiento, sin filtrar: lo que entrega el ADC)
ax = X["raw"]
sl = win(tg, za, zb_)
tz = tg[sl] - za
sp = 26.0 if fs.REVIEW else 420.0
sweep_ch = Bk[Bk.session == name].set_index("channel").sweep_db_med.idxmax()
if fs.REVIEW:
    # envolvente de cada canal llevada a mV de RMS del crudo: desfase = mediana en las pausas y
    # ganancia por minimos cuadrados contra el RMS movil de 50 ms de toda la sesion
    import envlag
    RM = envlag.moving_rms(B)
    rest_k = np.zeros(len(tg), bool)
    for a_, b_ in x.breaks:
        rest_k |= (tg >= a_) & (tg < b_)
for i in range(8):
    yv = Y[i, sl]
    yv = yv - np.nanmedian(yv)
    if fs.REVIEW:
        yv = yv / rest_rms[f"E{i+1}"]
        c = x.s.env[i]
        te_ = (c.ts_us - x.s.t0_us) / 1e6
        r_at = np.interp(te_, tg, RM[i])
        in_rest = np.interp(te_, tg, rest_k.astype(float)) > 0.5
        off = np.median(c.mv[in_rest]) if in_rest.any() else np.percentile(c.mv, 5)
        e_ = c.mv - off
        ok = np.isfinite(r_at) & np.isfinite(e_)
        gain = np.sum(e_[ok] * r_at[ok]) / np.sum(e_[ok] ** 2)
        k = (te_ >= za) & (te_ < zb_)
        band = np.clip(2.0 * gain * e_[k] / rest_rms[f"E{i+1}"], 0, sp * 0.62)
        ax.fill_between(te_[k] - za, -band - i * sp, band - i * sp, color=fs.C_ENV, alpha=0.18, lw=0, zorder=1)
        ax.plot(te_[k] - za, band - i * sp, color=fs.C_ENV, lw=0.6, zorder=3)
        ax.plot(te_[k] - za, -band - i * sp, color=fs.C_ENV, lw=0.6, zorder=3)
    ax.plot(tz, np.clip(yv, -sp * 0.62, sp * 0.62) - i * sp, lw=0.22, color=fs.INK, rasterized=True, zorder=2)
    ax.text(-0.25, -i * sp, f"E{i+1}", ha="right", va="center", fontsize=6.5,
            color=C_SWEEP if f"E{i+1}" == sweep_ch else fs.INK,
            fontweight="bold" if f"E{i+1}" == sweep_ch else "normal")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
if fs.REVIEW:
    ax.plot([tz[-1] + 0.3] * 2, [-7 * sp - 5, -7 * sp + 5], color=fs.INK, lw=1, clip_on=False)
    ax.text(tz[-1] + 0.45, -7 * sp, "10 × rest\nRMS", fontsize=6, va="center", clip_on=False)
    ax.set_ylabel("Raw sEMG, × its resting RMS\n(blue: hardware envelope, ±2 RMS)", fontsize=7, labelpad=16)
else:
    ax.plot([tz[-1] + 0.3] * 2, [-7 * sp - 50, -7 * sp + 50], color=fs.INK, lw=1, clip_on=False)
    ax.text(tz[-1] + 0.45, -7 * sp, "100 mV", fontsize=6, va="center", clip_on=False)
    ax.set_ylabel("Raw sEMG (ADC input)", fontsize=7, labelpad=16)

# envolvente del hardware (en revision va encima de cada crudo, no en su fila)
if "env" in X:
    ax = X["env"]
    act = Bk[Bk.session == name].set_index("channel").act_db_med.sort_values(ascending=False)
    top3 = [int(c[1:]) - 1 for c in act.index[:3]]
    ymax = 0.0
    for i in range(8):
        c = x.s.env[i]
        te_ = (c.ts_us - x.s.t0_us) / 1e6
        k = (te_ >= za) & (te_ < zb_)
        is_sw = f"E{i+1}" == sweep_ch
        ax.plot(te_[k] - za, c.mv[k], lw=0.9 if (i in top3 or is_sw) else 0.5,
                color=C_SWEEP if is_sw else (fs.INK if i in top3 else fs.GRID), zorder=3 if i in top3 else 2)
        if k.any():
            ymax = max(ymax, np.nanmax(c.mv[k]))
        if is_sw:
            kk = np.argmax(c.mv[k]) if k.any() else 0
            ax.text(te_[k][kk] - za + 0.2, c.mv[k][kk] * 1.02, f"E{i+1}: sweep pickup", fontsize=6, va="bottom",
                    ha="left", color=fs.INK)
    # Margen arriba para que los rotulos queden sobre las curvas: en S07 la captacion
    # del barrido satura justo donde iban.
    ax.set_ylim(top=ymax * 1.35)
    top_lbl = ", ".join(f"E{i+1}" for i in top3)
    ax.text(0.01, 0.95, f"most active: {top_lbl} (black)", transform=ax.transAxes, fontsize=6, va="top",
            color=fs.INK2)
    ax.set_ylabel("Hardware\nenvelope (mV)", rotation=0, ha="right", va="center", fontsize=7)

# carga de contacto: solo se lee durante el agarre; se muestra la lectura
# libre del filtro de mediana (la 3a de cada agarre); en revision, las tres
ax = X["load"]
kw = (sen["t_est"] >= za) & (sen["t_est"] < zb_)
kk = kw & sen["load_ok"]
if fs.REVIEW:
    k0 = kw & ~sen["load_ok"]
    ax.plot(sen["t_est"][k0] - za, sen["p1"][k0], "o", ms=2.2, color="#b8bec5", mew=0, label="readings 1–2\n(carry-over)")
    ax.plot(sen["t_est"][k0] - za, sen["p2"][k0], "o", ms=2.2, mfc="white", mec="#b8bec5", mew=0.6)
    kw_all = kw
else:
    kw_all = kk
ax.plot(sen["t_est"][kk] - za, sen["p1"][kk], "o", ms=2.6, color=fs.INK, label="p1 flexor")
ax.plot(sen["t_est"][kk] - za, sen["p2"][kk], "o", ms=2.6, mfc="white", mec=fs.INK2, mew=0.7,
        label="p2 extensor,\nprovisional")
lo_, hi_ = np.nanmin(np.r_[sen["p1"][kw_all], sen["p2"][kw_all]]), np.nanmax(np.r_[sen["p1"][kw_all], sen["p2"][kw_all]])
ax.set_ylim(lo_ - 0.25 * (hi_ - lo_) - 1, hi_ + 0.25 * (hi_ - lo_) + 1)
ax.set_ylabel("Contact\nload (kPa)", rotation=0, ha="right", va="center", fontsize=7)
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.15 if not fs.REVIEW else 1.5), fontsize=6, handletextpad=0.2,
          borderaxespad=0)
# temperatura de piel en el detalle (revision): las tres lecturas de cada sostener, sin filtro
if "temp" in X:
    ax = X["temp"]
    kt = kw & (sen["temp"] > -126)
    # unidas solo dentro de cada sostener: entre sostenes no hay lectura
    for tr in x.trials:
        kh = kt & (sen["t_est"] >= tr.t_grasp) & (sen["t_est"] < tr.t_next)
        if kh.any():
            ax.plot(sen["t_est"][kh] - za, sen["temp"][kh], "o-", ms=2.4, lw=0.6, color=fs.INK)
    if kt.any():
        lo_, hi_ = np.nanmin(sen["temp"][kt]), np.nanmax(sen["temp"][kt])
        ax.set_ylim(lo_ - 0.3 * max(hi_ - lo_, 0.2), hi_ + 0.3 * max(hi_ - lo_, 0.2))
    ax.set_ylabel("Skin\ntemp. (°C)", rotation=0, ha="right", va="center", fontsize=7)
# IMU
if "gyro" in X:
    ax = X["gyro"]
    k = (t_imu >= za) & (t_imu < zb_)
    ax.plot(t_imu[k] - za, gm[k], lw=0.6, color=fs.INK2)
    ax.set_ylabel("Gyro\n(°/s)", rotation=0, ha="right", va="center", fontsize=7)

# |Z| del barrido en el tiempo de excitacion
ax = X["z"]
for p in zp:
    w = [w for w in x.sweeps if abs((w["t_arrival"] - 5.2) - p) < 2.0]
    if not w:
        continue
    w = w[0]
    tk = p + ONSET + T1 + (np.arange(len(w["z"])) + 0.5) * STEP - za
    ax.plot(tk, w["z"] / 1e3, "o", ms=1.6, color=C_SWEEP)
    last = None                    # ultimo rotulo puesto: (instante, |Z|, altura del rotulo)
    for fk in (2, 10, 20, 50, 100):
        j = int(np.argmin(np.abs(w["freq"] / 1e3 - fk)))
        if tk[j] > zb_ - za:
            continue
        ax.plot([tk[j]], [w["z"][j] / 1e3], "o", ms=3.2, mfc="none", mec=fs.INK, mew=0.6)
        # las frecuencias se rotulan solo en el primer barrido: en el segundo
        # repetian lo mismo y el de 100 kHz se salia del eje. Si la curva ya
        # toco el piso, dos puntos quedan juntos y a la misma altura (S07: 10 y
        # 20 kHz): el segundo no se rotula (subirlo lo hacia chocar con el de 2 kHz)
        if p == zp[0]:
            if last and tk[j] - last[0] < 0.8 and abs(np.log10(w["z"][j] / last[1])) < 0.15:
                continue
            ax.annotate(f"{fk} kHz", (tk[j], w["z"][j] / 1e3), xytext=(3, 5), textcoords="offset points",
                        fontsize=5.8, ha="left", va="bottom", color=fs.INK2)
            last = (tk[j], w["z"][j], 5)
ax.set_yscale("log")
ax.set_ylim(5, 250)
ax.set_ylabel("|Z| (kΩ)", rotation=0, ha="right", va="center", fontsize=7)
ax.set_xlim(0, zb_ - za)
ax.set_xlabel("Time (s)", labelpad=1)
for a in bx[:-1]:
    plt.setp(a.get_xticklabels(), visible=False)
if fs.REVIEW:
    # revision: carga y temperatura del detalle con la escala de la sesion entera
    X["load"].set_ylim(A["load"].get_ylim())
    X["temp"].set_ylim(A["temp"].get_ylim())
    # la fila de temperatura del detalle es baja: dos marcas, o se pisan
    X["temp"].yaxis.set_major_locator(MaxNLocator(2))
# la de siempre (S02, ensayo 26) lleva el nombre corto; otra sesion o ensayo, su propio archivo
stem = (f"{FIGP}_2_multimodal" if (SUBJ, ZOOM_TRIAL) == ("S02", 26)
        else f"{FIGP}_2_multimodal_{SUBJ}_{ZOOM_TRIAL}")
fs.save(fig, FIG26 / stem, panels={"a": axs, "b": bx})
print("ok", name, "zoom", round(za, 1), round(zb_, 1))
