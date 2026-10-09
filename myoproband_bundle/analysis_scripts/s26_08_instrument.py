"""Sabado 26: caracteristicas del instrumento que el articulo aun no tiene.

1. Correlacion entre canales (retardo cero, crudo 20-450 Hz) durante el
   sostener y en el reposo relajado de las pausas; en reposo, tambien sin las
   lineas de baliza Wi-Fi (bins a +-0.2 Hz de n x 9.765625 Hz a cero). Perfil
   segun la distancia alrededor del antebrazo (vecinos = 1, opuestos = 4).
2. Retardo de la envolvente del hardware frente al RMS movil del crudo (50 ms)
   al agarrar y al soltar: cruce del 50 % entre reposo y meseta, en ensayos y
   canales con activacion >= 8 dB y sin captacion del barrido.
3. Monitoreo de la interfaz (descriptivo): dentro de cada sesion, |Z| a 5 kHz
   del barrido que sigue a cada agarre frente al RMS en reposo de ese mismo
   reposo (media de los 8 canales, dB sobre la mediana de la sesion).

Escribe data/s26/corr.npz, envelope_lag.csv, interface.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s26_common import DATA26, SUBJECT_ORDER, bandpass_fft, find_sessions, grid_raw, load26, win
import envlag

BEACON = 1000.0 / 102.4
REPS, N_TRIALS = 6, 42
S = pd.read_csv(DATA26 / "sessions.csv")
full = S[S.complete].set_index("subject")
T = pd.read_csv(DATA26 / "semg_trials.csv")
Z = pd.read_csv(DATA26 / "impedance.csv")
K = pd.read_csv(DATA26 / "sweep_steps.csv")
whole = (K.groupby(["subject", "channel", "block"]).db.apply(lambda d: 10 * np.log10(np.mean(10 ** (d / 10))))
         .groupby(["subject", "channel"]).median())
paths = {p.name: p for p in find_sessions()}


def notch_beacon(Bm, fs=1000.0, half=0.2):
    out = np.empty_like(Bm)
    n = Bm.shape[1]
    f = np.fft.rfftfreq(n, 1 / fs)
    near = np.abs(f / BEACON - np.round(f / BEACON)) * BEACON < half
    near &= f > 5
    for i in range(Bm.shape[0]):
        y = Bm[i]
        bad = ~np.isfinite(y)
        Yf = np.fft.rfft(np.where(bad, 0.0, y))
        Yf[near] = 0
        out[i] = np.fft.irfft(Yf, n)
        out[i][bad] = np.nan
    return out


def corr(M):
    M = M[:, np.isfinite(M).all(0)]
    return np.corrcoef(M)


C_hold, C_rest, C_rest_nb = {}, {}, {}
lag_rows, if_rows = [], []
for subj in SUBJECT_ORDER:
    x = load26(paths[full.loc[subj, "session"]])
    tg, Y = grid_raw(x)
    B = bandpass_fft(Y)
    hold = np.concatenate([B[:, win(tg, t.t_grasp + 1.0, t.t_rest - 0.5)] for t in x.trials], 1)
    rest_sl = []
    for tr in x.trials:
        if tr.index % REPS == REPS - 1 and tr.index + 1 < N_TRIALS:
            b0, b1 = tr.t_rest + 7.0, x.trials[tr.index + 1].t_grasp
            rest_sl.append(win(tg, b0 + 4.6, b1 - 2.5))
    rest = np.concatenate([B[:, s_] for s_ in rest_sl], 1)
    Bn = notch_beacon(B)
    rest_nb = np.concatenate([Bn[:, s_] for s_ in rest_sl], 1)
    C_hold[subj], C_rest[subj], C_rest_nb[subj] = corr(hold), corr(rest), corr(rest_nb)
    # envolvente frente a RMS movil del crudo (metodo en envlag.py, el mismo que dibuja la figura 13)
    rms_mov = envlag.moving_rms(B)
    t_sub = T[T.subject == subj]
    for tr in x.trials:
        for i in range(8):
            ch = f"E{i+1}"
            row = t_sub[(t_sub.trial == tr.index) & (t_sub.channel == ch)]
            if not len(row) or not (row.act_db.iloc[0] >= 8) or whole.get((subj, ch), 0) > 3:
                continue
            te = (x.s.env[i].ts_us - x.s.t0_us) / 1e6
            ev = x.s.env[i].mv
            sl = win(tg, tr.t_grasp - 1.0, tr.t_rest + 3.0)
            tt, rr = tg[sl], rms_mov[i, sl]
            m = (te >= tr.t_grasp - 1.0) & (te < tr.t_rest + 3.0)
            ee_t, ee = te[m], ev[m]
            if len(ee) < 50:
                continue
            L = envlag.trial_lag(tt, rr, ee_t, ee, tr.t_grasp, tr.t_rest)
            if L is None:
                continue
            lag_rows.append(dict(subject=subj, trial=tr.index, channel=ch, act_db=float(row.act_db.iloc[0]),
                                 raw_onset_ms=1000 * (L["r_on"] - tr.t_grasp),
                                 env_onset_ms=1000 * (L["e_on"] - tr.t_grasp),
                                 lag_on_ms=1000 * L["lag_on"], raw_off_ms=1000 * (L["r_off"] - tr.t_rest),
                                 lag_off_ms=1000 * L["lag_off"], lag_xcorr_ms=1000 * L["lag_xcorr"],
                                 xcorr_r=L["xcorr_r"]))
    # interfaz: |Z| a 5 kHz tras cada agarre frente al RMS en reposo de ese reposo
    zz = Z[(Z.subject == subj) & (Z.context == "post")]
    rr_ = t_sub.groupby("trial").apply(lambda g: np.nanmean(20 * np.log10(g.rms_rest / g.rms_rest.median())))
    for tr in x.trials:
        zk = zz[np.abs(zz.t_prest - tr.t_rest) < 0.3]
        if not len(zk):
            continue
        if_rows.append(dict(subject=subj, trial=tr.index, t_min=tr.t_rest / 60, z5k=float(zk.z5k.iloc[0]),
                            rest_db=float(rr_.loc[tr.index])))
    L_ = pd.DataFrame([r for r in lag_rows if r["subject"] == subj])
    # una sesion puede no tener ningun sostener que califique (S07: sus canales activos captan el barrido)
    lag_txt = (f"retardo envolvente al agarrar {L_.lag_on_ms.median():.0f} ms, al soltar {L_.lag_off_ms.median():.0f} ms "
               f"(n={len(L_)})" if len(L_) else "ningun sostener califica para el retardo de la envolvente")
    print(f"{subj}: corr sostener vecinos {np.mean([C_hold[subj][i, (i+1) % 8] for i in range(8)]):.2f}, "
          f"reposo {np.mean([C_rest[subj][i, (i+1) % 8] for i in range(8)]):.2f} "
          f"(sin baliza {np.mean([C_rest_nb[subj][i, (i+1) % 8] for i in range(8)]):.2f}); " + lag_txt)

np.savez(DATA26 / "corr.npz", **{f"hold_{k}": v for k, v in C_hold.items()}, **{f"rest_{k}": v for k, v in C_rest.items()},
         **{f"restnb_{k}": v for k, v in C_rest_nb.items()})
LG = pd.DataFrame(lag_rows)
LG.to_csv(DATA26 / "envelope_lag.csv", index=False)
IF = pd.DataFrame(if_rows)
IF.to_csv(DATA26 / "interface.csv", index=False)


def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


print(f"\nretardo de la envolvente: al agarrar {LG.lag_on_ms.median():.0f} ms (p10-p90 {LG.lag_on_ms.quantile(0.1):.0f}-"
      f"{LG.lag_on_ms.quantile(0.9):.0f}), al soltar {LG.lag_off_ms.median():.0f} ms (p10-p90 "
      f"{LG.lag_off_ms.quantile(0.1):.0f}-{LG.lag_off_ms.quantile(0.9):.0f}); n={len(LG)}; "
      f"inicio del crudo tras la senal {LG.raw_onset_ms.median():.0f} ms")
for s_ in SUBJECT_ORDER:
    g = IF[IF.subject == s_]
    print(f"{s_}: |Z|5k vs RMS en reposo rho={spearman(g.z5k, g.rest_db):+.2f}; |Z| vs tiempo {spearman(g.z5k, g.t_min):+.2f}; "
          f"RMS vs tiempo {spearman(g.rest_db, g.t_min):+.2f}")
