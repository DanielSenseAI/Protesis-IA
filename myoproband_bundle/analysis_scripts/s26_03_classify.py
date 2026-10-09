"""Sabado 26: cuanto de los siete agarres hay en los ocho canales.

LDA con encogimiento sobre el conjunto de Hudgins por canal (log-MAV, log-WL,
ZC, SSC; 32 rasgos) en ventanas de 250 ms con paso de 125 ms dentro del
sostener estable [Pgrasp + 1.0, Prest - 0.5]. Validacion dejando fuera una
repeticion (6 pliegues): se entrena con cinco repeticiones de cada agarre y
se prueba con la sexta. Precision por ventana y por ensayo (voto).

Controles, porque el orden fue agrupado (las seis repeticiones de un agarre
seguidas) y una deriva lenta tambien separaria bloques:
- deriva: el mismo clasificador sobre el reposo que sigue a cada ensayo
  [Prest + 4.4, Prest + 5.0] (acabado el barrido, antes del aviso del
  siguiente agarre), con la etiqueta del bloque. Lo que se clasifique ahi es
  del bloque, no del agarre: actividad residual, contacto, deriva.
- normalizado: cada ventana de sostener menos la media de su propio reposo
  previo [Pgrasp - 2.5, Pgrasp - 2.0]. Quita lo que el bloque comparte con su
  reposo y deja el patron de activacion.
- orden: las etiquetas de la hipotesis 'block' (mismo semilla) en vez de
  'grouped'. Con la hipotesis equivocada la precision cae al azar.

Escribe data/s26/classify.csv y data/s26/confusion.npz.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s26_common import (DATA26, bandpass_fft, find_sessions, grasp_sequence, grid_raw, load26,
                        win)

N_TRIALS = 42
W, STEP = 250, 125          # muestras a 1 kHz
LAMBDA = 0.2


def features(seg, thr):
    """seg: (8, n). Hudgins por canal."""
    d = np.diff(seg, axis=1)
    mav = np.mean(np.abs(seg), 1)
    wl = np.mean(np.abs(d), 1)
    zc = np.sum((seg[:, :-1] * seg[:, 1:] < 0) & (np.abs(d) >= thr[:, None]), 1) / seg.shape[1]
    ssc = np.sum(((d[:, :-1] * d[:, 1:]) < 0) &
                 ((np.abs(d[:, :-1]) >= thr[:, None]) | (np.abs(d[:, 1:]) >= thr[:, None])), 1) / seg.shape[1]
    return np.concatenate([np.log(mav + 1e-3), np.log(wl + 1e-3), zc, ssc])


def windows(B, tg, a, b, thr):
    sl = win(tg, a, b)
    seg = B[:, sl]
    out = []
    for j in range(0, seg.shape[1] - W + 1, STEP):
        s = seg[:, j:j + W]
        if np.isfinite(s).all():
            out.append(features(s, thr))
    return out


def lda_fit(X, y, lam=LAMBDA):
    cls = np.unique(y)
    mu = np.array([X[y == c].mean(0) for c in cls])
    Xc = np.concatenate([X[y == c] - mu[k] for k, c in enumerate(cls)])
    S = Xc.T @ Xc / (len(X) - len(cls))
    d = S.shape[0]
    S = (1 - lam) * S + lam * np.trace(S) / d * np.eye(d)
    Si = np.linalg.pinv(S)
    W_ = mu @ Si
    b_ = -0.5 * np.sum(W_ * mu, 1) + np.log(np.array([np.mean(y == c) for c in cls]))
    return cls, W_, b_


def lda_predict(model, X):
    cls, W_, b_ = model
    return cls[np.argmax(X @ W_.T + b_, 1)]


def cv(feats, labels, reps, trial_ids):
    """Deja fuera una repeticion. Devuelve (acc ventana, acc ensayo, pares y/yhat por ensayo)."""
    X = np.array(feats); y = np.array(labels); r = np.array(reps); tid = np.array(trial_ids)
    hits_w, n_w, pairs = 0, 0, []
    for rep in np.unique(r):
        tr, te = r != rep, r == rep
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        model = lda_fit((X[tr] - mu) / sd, y[tr])
        yh = lda_predict(model, (X[te] - mu) / sd)
        hits_w += np.sum(yh == y[te]); n_w += te.sum()
        for t in np.unique(tid[te]):
            k = tid[te] == t
            vals, cnt = np.unique(yh[k], return_counts=True)
            pairs.append((y[te][k][0], vals[np.argmax(cnt)]))
    acc_t = np.mean([a == b for a, b in pairs])
    return hits_w / n_w, acc_t, pairs


rows, conf = [], {}
for path in find_sessions():
    x = load26(path)
    if len(x.trials) != N_TRIALS:
        continue
    tg, Y = grid_raw(x)
    B = bandpass_fft(Y)
    # umbral de ZC/SSC: el RMS en reposo de cada canal (las pausas, sin barrido)
    rest_idx = np.concatenate([np.arange(len(tg))[win(tg, t.t_rest + 4.5, t.t_rest + 5.3)] for t in x.trials])
    thr = np.sqrt(np.nanmean(B[:, rest_idx] ** 2, 1))
    meta = x.s.meta
    seq_block = grasp_sequence(meta["grasps"], int(meta["reps"]), int(meta["seed"]), "block")
    F, L, R, T, Lb = [], [], [], [], []
    Fr, Lr, Rr, Tr = [], [], [], []
    Fn, Ln, Rn, Tn = [], [], [], []
    for tr in x.trials:
        pre = windows(B, tg, tr.t_grasp - 2.5 - (W - STEP) / 1000.0, tr.t_grasp - 2.0, thr)
        pre_mu = np.mean(pre, 0) if pre else None
        for f in windows(B, tg, tr.t_grasp + 1.0, tr.t_rest - 0.5, thr):
            F.append(f); L.append(tr.grasp); R.append(tr.rep); T.append(tr.index); Lb.append(seq_block[tr.index])
            if pre_mu is not None:
                Fn.append(f - pre_mu); Ln.append(tr.grasp); Rn.append(tr.rep); Tn.append(tr.index)
        for f in windows(B, tg, tr.t_rest + 4.4 - (W - STEP) / 1000.0, tr.t_rest + 5.0, thr):
            Fr.append(f); Lr.append(tr.grasp); Rr.append(tr.rep); Tr.append(tr.index)
    acc_w, acc_t, pairs = cv(F, L, R, T)
    # repeticion bajo la hipotesis 'block': el orden dentro del agarre
    cnt, rb_trial = {}, {}
    for i, g in enumerate(seq_block):
        cnt[g] = cnt.get(g, 0) + 1
        rb_trial[i] = cnt[g]
    Rb = [rb_trial[t] for t in T]
    accb_w, accb_t, _ = cv(F, Lb, Rb, T)
    accr_w, accr_t, _ = cv(Fr, Lr, Rr, Tr)
    accn_w, accn_t, _ = cv(Fn, Ln, Rn, Tn)
    cls = sorted(set(L))
    M = np.zeros((len(cls), len(cls)), int)
    for a, b in pairs:
        M[cls.index(a), cls.index(b)] += 1
    conf[x.subject] = M
    rows.append(dict(session=x.name, subject=x.subject, n_windows=len(F),
                     acc_window=acc_w, acc_trial=acc_t,
                     acc_rest_window=accr_w, acc_rest_trial=accr_t,
                     acc_norm_window=accn_w, acc_norm_trial=accn_t, n_rest_windows=len(Fr),
                     acc_blockorder_window=accb_w, acc_blockorder_trial=accb_t))
    print(f"{x.subject}: agarre {100*acc_w:5.1f} % por ventana, {100*acc_t:5.1f} % por ensayo | "
          f"normalizado {100*accn_w:5.1f} / {100*accn_t:5.1f} % | "
          f"control reposo {100*accr_w:5.1f} / {100*accr_t:5.1f} % ({len(Fr)} ventanas) | "
          f"etiquetas 'block' {100*accb_w:5.1f} / {100*accb_t:5.1f} %  (azar 14.3 %)")

D = pd.DataFrame(rows)
D.to_csv(DATA26 / "classify.csv", index=False)
np.savez(DATA26 / "confusion.npz", classes=np.array(sorted(set(L))), **conf)
print(f"\nmedia: agarre {100*D.acc_window.mean():.1f} % (ventana) / {100*D.acc_trial.mean():.1f} % (ensayo); "
      f"normalizado {100*D.acc_norm_window.mean():.1f} / {100*D.acc_norm_trial.mean():.1f} %; "
      f"reposo {100*D.acc_rest_window.mean():.1f} / {100*D.acc_rest_trial.mean():.1f} %; "
      f"'block' {100*D.acc_blockorder_window.mean():.1f} / {100*D.acc_blockorder_trial.mean():.1f} %")
