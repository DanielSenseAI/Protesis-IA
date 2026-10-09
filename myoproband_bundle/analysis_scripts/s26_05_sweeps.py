"""Sabado 26: lo que el barrido de impedancia deja en el sEMG de los participantes.

Solo los barridos que abren cada pausa de 30 s: el sujeto lleva 7 s relajado
y el sEMG no arrastra actividad del agarre, asi que lo que cambie durante el
barrido es del barrido. Referencia: el reposo limpio de esa misma pausa
[inicio + 4.6, siguiente Pgrasp - 2.5].

- dB por paso de frecuencia: RMS (20-450 Hz) del sEMG mientras se excita cada
  frecuencia (arranque = Prest + 0.15 s; primer punto de 1 kHz 55 ms; pasos
  de 41.9 ms) sobre el RMS de referencia; y por bandas 2-5, 5-10, 10-20,
  20-30 y 30-100 kHz como en el analisis de banco (04_band_impact.py).
- Envolvente del hardware: p95 durante el barrido contra p95 durante los
  sostenes, ambos sobre la linea base. Un canal cuya envolvente sube mas con
  el barrido que con un agarre es un falso positivo para quien la use.
- Holgura: los barridos acaban a +4.35 s del Prest y el siguiente agarre
  llega a +7.0 s; se cuenta cuantos sostenes pisa algun barrido.

Escribe data/s26/sweep_steps.csv, sweep_bands.csv, sweep_env.csv,
hold_overlap.csv (cada sostener: cuanto barrido tuvo encima y de donde vino)
y la figura fig_s26_4_sweeps.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from s26_common import FIGP
from s26_common import (C_SWEEP, DATA26, FIG26, SUBJECT_ORDER, bandpass_fft, find_sessions, grid_raw,
                        load26, prest_times, win)

T1, STEP, ONSET = 0.055, 0.0419, 0.15
BANDS = [(2, 5), (5, 10), (10, 20), (20, 30), (30, 100)]
REPS, N_TRIALS = 6, 42

rows_k, rows_e, overlap = [], [], []
for path in find_sessions():
    x = load26(path)
    if len(x.trials) != N_TRIALS:
        continue
    tg, Y = grid_raw(x)
    B = bandpass_fft(Y)
    pr = prest_times(x)
    # Sostenes pisados por algun barrido: los que dispara un Prest y el de
    # arranque (START de la placa auxiliar al empezar a grabar, sin Prest): de
    # ese solo se sabe cuando llego (~0.1-1.1 s despues de terminar, por el
    # latido de 1 s), asi que su ventana se estima desde la llegada.
    wins = [(p + ONSET, p + ONSET + 4.2, "rest command") for p in pr]
    for w in x.sweeps:
        if not any(3.5 < w["t_arrival"] - p < 7.5 for p in pr):
            wins.append((w["t_arrival"] - 4.9, w["t_arrival"] - 0.7, "start of test"))
    for tr in x.trials:
        ov = [(min(b, tr.t_rest) - max(a, tr.t_grasp), src) for a, b, src in wins
              if (a < tr.t_rest) and (b > tr.t_grasp)]
        best = max(ov) if ov else (0.0, "")
        overlap.append(dict(session=x.name, subject=x.subject, trial=tr.index, t_grasp=round(tr.t_grasp, 3),
                            t_rest=round(tr.t_rest, 3), overlap_s=round(best[0], 3), source=best[1]))
        if ov:
            print(f"    {x.subject}: sostener {tr.index} ({tr.t_grasp:.2f}-{tr.t_rest:.2f} s) con "
                  f"{best[0]:.1f} s de barrido encima ({best[1]})")
    env_t = {i: (x.s.env[i].ts_us - x.s.t0_us) / 1e6 for i in range(8)}
    for tr in x.trials:
        if not (tr.index % REPS == REPS - 1 and tr.index + 1 < N_TRIALS):
            continue
        nominal = tr.t_rest + 7.0
        near = [p for p in pr if abs(p - nominal) < 0.6]
        b0 = near[0] if near else nominal
        b1 = x.trials[tr.index + 1].t_grasp
        ref = B[:, win(tg, b0 + 4.6, b1 - 2.5)]
        rref = np.sqrt(np.nanmean(ref ** 2, 1))
        on = b0 + ONSET
        for k in range(99):
            a = on + T1 + k * STEP
            seg = B[:, win(tg, a, a + STEP)]
            r = np.sqrt(np.nanmean(seg ** 2, 1))
            for i in range(8):
                rows_k.append(dict(session=x.name, subject=x.subject, block=tr.index // REPS, channel=f"E{i+1}",
                                   f_khz=k + 2, db=20 * np.log10(r[i] / rref[i])))
    # envolvente: sostenes contra barridos de las pausas
    Hh = [(t.t_grasp + 1.0, t.t_rest - 0.5) for t in x.trials]
    Sw = []
    for tr in x.trials:
        if tr.index % REPS == REPS - 1 and tr.index + 1 < N_TRIALS:
            near = [p for p in pr if abs(p - (tr.t_rest + 7.0)) < 0.6]
            b0 = near[0] if near else tr.t_rest + 7.0
            Sw.append((b0, b0 + ONSET + 4.3, b0 + 4.6, x.trials[tr.index + 1].t_grasp - 2.5))
    for i in range(8):
        c = x.s.env[i]
        te, v = env_t[i], c.mv

        def pick(spans):
            m = np.zeros(len(te), bool)
            for a, b in spans:
                m |= (te >= a) & (te < b)
            return v[m]
        base = np.median(pick([(s[2], s[3]) for s in Sw]))
        hold = pick(Hh)
        sw = pick([(s[0], s[1]) for s in Sw])
        rows_e.append(dict(session=x.name, subject=x.subject, channel=f"E{i+1}", env_base=base,
                           hold_p95=float(np.percentile(hold, 95) - base),
                           sweep_p95=float(np.percentile(sw, 95) - base),
                           sweep_max=float(sw.max() - base)))
    print(x.subject, "ok")

K = pd.DataFrame(rows_k)
E = pd.DataFrame(rows_e)
K.to_csv(DATA26 / "sweep_steps.csv", index=False)
E.to_csv(DATA26 / "sweep_env.csv", index=False)
# bandas: mediana sobre los barridos de pausa de la RMS en dB (potencia media por banda)
rows_b = []
for (sj, ch), g in K.groupby(["subject", "channel"]):
    for lo, hi in BANDS:
        gg = g[(g.f_khz >= lo) & (g.f_khz < hi if hi < 100 else g.f_khz <= hi)]
        per_block = gg.groupby("block").db.apply(lambda d: 10 * np.log10(np.mean(10 ** (d / 10))))
        rows_b.append(dict(subject=sj, channel=ch, band=f"{lo}-{hi}", db=float(per_block.median())))
Bd = pd.DataFrame(rows_b)
Bd.to_csv(DATA26 / "sweep_bands.csv", index=False)
OV = pd.DataFrame(overlap)
OV.to_csv(DATA26 / "hold_overlap.csv", index=False)
print(f"sostenes pisados por un barrido: {int((OV.overlap_s > 0).sum())}/{len(OV)}")
piv = Bd.pivot_table(index=["subject", "channel"], columns="band", values="db")
print(piv.round(1).to_string())

# ------------------------------------------------------------------- figura
fs.apply()
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1, 1], wspace=0.34, hspace=0.42,
                      left=0.08, right=0.97, top=0.94, bottom=0.09)
panels = {}

# (a) matriz sujeto x canal: dB de toda la ventana de barrido (potencia media de 2-100 kHz)
ax = fig.add_subplot(gs[0, 0])
whole = (K.groupby(["subject", "channel", "block"]).db.apply(lambda d: 10 * np.log10(np.mean(10 ** (d / 10))))
         .groupby(["subject", "channel"]).median().unstack())
whole = whole.reindex(SUBJECT_ORDER)[[f"E{i}" for i in range(1, 9)]]
im = ax.imshow(whole.values, cmap="Oranges", vmin=0, vmax=30, aspect="auto")
for (r, c), v in np.ndenumerate(whole.values):
    ax.text(c, r, fs.num(v), ha="center", va="center", fontsize=6.2, color="white" if v > 18 else fs.INK)
ax.set_xticks(range(8))
ax.set_xticklabels(whole.columns, fontsize=6.8)
ax.set_yticks(range(len(whole)))
ax.set_yticklabels(whole.index, fontsize=6.8)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Whole sweep, relaxed forearm", loc="left", fontsize=7.5)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_visible(False)
cb.set_label("dB relative to rest", fontsize=6)
panels["a"] = ax

# (b) dB por frecuencia de excitacion: todos los pares en gris, los afectados en naranja
ax = fig.add_subplot(gs[0, 1])
med = K.groupby(["subject", "channel", "f_khz"]).db.median().reset_index()
hot = whole.stack()
hot = hot[hot > 8].index.tolist()
for (sj, ch), g in med.groupby(["subject", "channel"]):
    if (sj, ch) in hot:
        continue
    ax.plot(g.f_khz, g.db, lw=0.5, color=fs.GRID, zorder=1)
for sj, ch in hot:
    g = med[(med.subject == sj) & (med.channel == ch)]
    ax.plot(g.f_khz, g.db, lw=1.0, color=C_SWEEP, zorder=3)
ax.axhline(0, color=fs.INK2, lw=0.6)
ax.set_xscale("log")
ax.set_xlim(2, 100)
ax.set_xticks([2, 5, 10, 20, 30, 50, 100])
ax.set_xticklabels(["2", "5", "10", "20", "30", "50", "100"])
ax.set_xlabel("Excitation frequency (kHz)")
ax.set_ylabel("sEMG RMS change (dB)")
ax.set_title(f"By excitation frequency (orange: the {len(hot)} cells > 8 dB in a; grey: other {48-len(hot)})",
             loc="left", fontsize=7.5)
panels["b"] = ax

# (c) envolvente: barrido contra sostener, por canal
ax = fig.add_subplot(gs[1, 0])
Ee = E.copy()
ax.plot([1, 6000], [1, 6000], color=fs.INK2, lw=0.6, ls="--")
flag = (Ee.sweep_p95 > Ee.hold_p95) & (Ee.sweep_p95 > 100)
normal = Ee[~flag]
bad = Ee[flag].sort_values("sweep_p95", ascending=False)
ax.scatter(normal.hold_p95.clip(lower=1), normal.sweep_p95.clip(lower=1), s=9, color=fs.INK2, lw=0)
ax.scatter(bad.hold_p95.clip(lower=1), bad.sweep_p95.clip(lower=1), s=14, color=C_SWEEP, lw=0, zorder=3)
print("envolvente, barrido > agarre:", ", ".join(f"{r.subject} {r.channel} ({r.hold_p95:.0f} -> {r.sweep_p95:.0f} mV)"
                                                 for _, r in bad.iterrows()))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 6000)
ax.set_ylim(1, 6000)
ax.set_xlabel("Hardware envelope during holds, p95 (mV)")
ax.set_ylabel("During sweeps, p95 (mV)")
ax.set_title("Envelope during sweeps vs. holds", loc="left", fontsize=7.5)
ax.text(1.3, 60, f"above the line ({len(bad)} of 48):\nsweep moves the\nenvelope more\nthan a grasp", fontsize=5.8,
        color=fs.INK2, va="bottom")
panels["c"] = ax

# (d) bandas, solo pares afectados, y la mediana de los demas
ax = fig.add_subplot(gs[1, 1])
labels = [f"{lo}–{hi}" for lo, hi in BANDS]
xs = np.arange(len(BANDS))
rest = Bd[~Bd.set_index(["subject", "channel"]).index.isin(hot)]
q = rest.groupby("band").db.quantile([0.1, 0.5, 0.9]).unstack().reindex([f"{lo}-{hi}" for lo, hi in BANDS])
ax.fill_between(xs, q[0.1], q[0.9], color=fs.GRID, lw=0, label="other channels, p10–p90")
ax.plot(xs, q[0.5], color=fs.INK2, lw=1.0, marker="o", ms=3, label="other channels, median")
for sj, ch in hot:
    g = Bd[(Bd.subject == sj) & (Bd.channel == ch)].set_index("band").reindex([f"{lo}-{hi}" for lo, hi in BANDS])
    ax.plot(xs, g.db, color=C_SWEEP, lw=0.9, marker="o", ms=2.6)
ax.plot([], [], color=C_SWEEP, lw=0.9, marker="o", ms=2.6, label="affected channels")
ax.axhline(0, color=fs.INK2, lw=0.6)
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_xlabel("Excitation band (kHz)")
ax.set_ylabel("sEMG RMS change (dB)")
ax.legend(loc="upper left", fontsize=6)
ax.set_ylim(-3, 44)
ax.set_title("By band, as in the bench analysis", loc="left", fontsize=7.5)
panels["d"] = ax
fs.save(fig, FIG26 / f"{FIGP}_4_sweeps", panels=panels)
print("figura ok")
