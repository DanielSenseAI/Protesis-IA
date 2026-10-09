"""Interferencia de la baliza Wi-Fi del propio brazalete en el sEMG crudo.

La radio del brazalete (punto de acceso propio) emite una baliza cada 100 TU
= 102.4 ms, contada con el mismo cristal que marca las muestras: en el reloj
del equipo el periodo es exacto (la plantilla pierde la mitad de su potencia a
+-8 ppm) y cada baliza hunde a la vez los ocho canales durante ~1-3 ms, con la
misma amplitud en todos (entra despues de la ganancia de cada modulo). En las
placas de banco sin modulos no aparece, ni con la radio encendida.

Tres maneras de quitarla, siempre sobre las marcas de tiempo reales:
- notch: ceros del espectro a n x 9.765625 Hz +-0.2 Hz (rejilla de 1 kHz);
- plantilla: restar la forma media en fase (pliegue de 102.4 ms);
- reparar: las muestras que caen dentro de la ventana del hundimiento se
  sustituyen por interpolacion lineal entre las vecinas de fuera de ella.
La plantilla y el notch solo quitan la parte media (periodica); el
hundimiento cambia mucho de una baliza a otra, y eso solo lo quita reparar.
"""
import numpy as np
import pandas as pd

PERIOD_US = 102400.0
BEACON_HZ = 1e6 / PERIOD_US
NB = 512                       # casillas de 0.2 ms en el pliegue


def detrend(y, n=51):
    """Quita la deriva lenta (mediana movil de n muestras) antes de plegar."""
    return y - pd.Series(y).rolling(n, center=True, min_periods=1).median().values


def fold(t_us, y, nb=NB):
    k = np.minimum(((t_us % PERIOD_US) / PERIOD_US * nb).astype(int), nb - 1)
    n = np.bincount(k, None, nb)
    return np.bincount(k, y, nb) / np.maximum(n, 1), n


def dip_window(tpl, frac=0.1, margin_bins=1):
    """Fase (ms) del fondo del hundimiento y ventana [lo, hi) donde la plantilla
    media queda por debajo de frac x la profundidad, con un margen de casillas.
    Las fases van en [0, 102.4) y la ventana puede cruzar el cero."""
    nb = len(tpl)
    t0 = tpl - np.median(tpl)
    j0 = int(np.argmin(t0))
    depth = t0[j0]
    lo = j0
    while t0[(lo - 1) % nb] < frac * depth and (j0 - lo) < nb // 4:
        lo -= 1
    hi = j0
    while t0[(hi + 1) % nb] < frac * depth and (hi - j0) < nb // 4:
        hi += 1
    lo, hi = lo - margin_bins, hi + 1 + margin_bins
    step = PERIOD_US / 1000 / nb
    return j0 * step, lo * step, hi * step, float(depth)


def in_window(t_us, lo_ms, hi_ms):
    ph = (t_us % PERIOD_US) / 1000.0
    lo, hi = lo_ms % 102.4, hi_ms % 102.4
    return (ph >= lo) & (ph < hi) if lo < hi else (ph >= lo) | (ph < hi)


def repair(t_us, y, lo_ms, hi_ms):
    """Muestras dentro de la ventana -> interpolacion lineal entre las de fuera."""
    bad = in_window(t_us, lo_ms, hi_ms)
    out = y.astype(float).copy()
    if bad.any() and (~bad).sum() > 2:
        out[bad] = np.interp(t_us[bad], t_us[~bad], y[~bad])
    return out, bad


def subtract_template(t_us, y, tpl):
    """Resta la plantilla media evaluada en la fase de cada muestra (lineal entre casillas)."""
    nb = len(tpl)
    x = (t_us % PERIOD_US) / PERIOD_US * nb - 0.5
    xp = np.arange(-1, nb + 1)
    fp = np.concatenate([[tpl[-1]], tpl, [tpl[0]]])
    return y - np.interp(x, xp, fp)


def notch_grid(Y, frame=4096, spread=1):
    """Notch en peine local: FFT corta (Hann, 75 % de solape, suma-solapada) con
    ceros en n x 9.766 Hz. A 1 kHz, 4096 muestras son justo 40 periodos de la
    baliza, asi que cada armonico cae en el bin 40n exacto; se anulan ese bin y
    `spread` vecinos por lado (fuga de la ventana). Cada trama solo afecta sus
    4 s: un notch sobre toda la sesion restaba una componente media que en
    sostenes con barridos o rafagas agregaba energia en vez de quitarla.
    Los huecos (NaN) se rellenan por interpolacion y vuelven a NaN."""
    out = np.empty_like(Y)
    n = Y.shape[1]
    hop = frame // 4
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)       # Hann periodica: suma 2 a 75 %
    kill = np.zeros(frame // 2 + 1, bool)
    per = int(round(frame / (1000.0 / BEACON_HZ)))                    # 40 bins por armonico
    for c in range(per, frame // 2 + 1, per):
        kill[max(c - spread, 0):c + spread + 1] = True
    pad = frame
    idx = np.arange(n)
    for i in range(Y.shape[0]):
        y = Y[i]
        bad = ~np.isfinite(y)
        mu = np.nanmean(y)
        yf = (np.interp(idx, idx[~bad], y[~bad]) if bad.any() else y) - mu
        z = np.concatenate([np.zeros(pad), yf, np.zeros(pad + frame)])
        acc = np.zeros(len(z))
        starts = np.arange(0, len(z) - frame + 1, hop)
        for j in range(0, len(starts), 256):                             # por bloques: memoria acotada
            st = starts[j:j + 256]
            fr = np.stack([z[s:s + frame] for s in st]) * w
            F = np.fft.rfft(fr, axis=1)
            F[:, kill] = 0
            back = np.fft.irfft(F, frame, axis=1)
            for s, b in zip(st, back):
                acc[s:s + frame] += b
        out[i] = acc[pad:pad + n] / 2.0 + mu
        out[i][bad] = np.nan
    return out


def lines_share(f, P, lo=20.0, hi=450.0, half=0.37):
    """Parte de la potencia de lo-hi Hz que esta en las lineas (exceso sobre el suelo local)."""
    band = (f >= lo) & (f <= hi)
    near = np.abs(f / BEACON_HZ - np.round(f / BEACON_HZ)) * BEACON_HZ <= half
    excess = 0.0
    for m in range(int(np.ceil(lo / BEACON_HZ)), int(hi / BEACON_HZ) + 1):
        fc = m * BEACON_HZ
        k = np.abs(f - fc) <= half
        loc = (np.abs(f - fc) <= 3) & ~near
        if loc.any():
            excess += (P[k] - np.median(P[loc])).clip(0).sum()
    return excess / P[band].sum()
