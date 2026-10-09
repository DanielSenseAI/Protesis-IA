"""Retardo de la envolvente del hardware frente al crudo, ensayo por ensayo.

Referencia: RMS movil centrado de 50 ms del crudo filtrado 20-450 Hz (no
agrega retardo). Para cada senal (RMS del crudo y envolvente del hardware):
- nivel de reposo = mediana en [agarre - 1.0, agarre - 0.1] s;
- nivel de meseta = mediana en [agarre + 1.5, soltar - 0.5] s;
- nivel de reposo tras soltar = mediana en [soltar + 1.5, soltar + 3.0] s;
- inicio = primer cruce del 50 % entre reposo y meseta despues de
  agarre - 0.2 s que se SOSTIENE (>= 80 % de los siguientes 0.25 s del lado
  de arriba), con interpolacion lineal entre muestras; fin = igual hacia
  abajo, entre meseta y reposo tras soltar, despues de soltar - 0.5 s.
retardo al agarrar = inicio(envolvente) - inicio(RMS); al soltar, igual con
los fines. Solo cuenta si la envolvente sube > 30 mV y el RMS mas del doble.

La primera version tomaba el primer cruce sin pedir que se sostuviera: con
una meseta ruidosa (RMS de 50 ms, ~14 dB de activacion) el 72 % de los
"fines" caian antes de la senal de soltar, en un bajon de la meseta. Por eso
el cruce sostenido, y ademas un estimado independiente de todo el sostener:
- correlacion cruzada: el desplazamiento (-0.2..+0.4 s) que mejor alinea la
  envolvente con el RMS en [agarre - 1, soltar + 2] s, con refinamiento
  parabolico del pico.
Lo usan s26_08_instrument.py (todos los ensayos) y la figura 13 (el metodo).
"""
import numpy as np

W_REST = (-1.0, -0.1)          # respecto al agarre
W_PLATEAU = (1.5, -0.5)        # agarre + 1.5 hasta soltar - 0.5
W_AFTER = (1.5, 3.0)           # respecto a soltar
W_XCORR = (-1.0, 2.0)          # agarre - 1 hasta soltar + 2
RMS_MS = 50
SUSTAIN_S = 0.25               # un cruce cuenta si se sostiene este tiempo...
SUSTAIN_FRAC = 0.8             # ...en esta parte de las muestras
XCORR_LAGS = (-0.2, 0.4)


def moving_rms(B, n=RMS_MS):
    k = np.ones(n) / n
    return np.sqrt(np.stack([np.convolve(np.nan_to_num(B[i] ** 2), k, "same") for i in range(B.shape[0])]))


def cross50(t, y, t_from, lo, hi, rising=True, sustain=SUSTAIN_S, frac=SUSTAIN_FRAC):
    """Primer cruce del punto medio entre lo y hi despues de t_from que se
    sostiene: en los `sustain` s siguientes, al menos `frac` de las muestras
    quedan del lado de destino. NaN si al empezar ya estaba de ese lado."""
    mid = lo + 0.5 * (hi - lo)
    k = np.where(t >= t_from)[0]
    if len(k) < 2:
        return np.nan
    side = (y[k] >= mid) if rising else (y[k] <= mid)
    if side[0]:
        return np.nan
    for s in np.where(~side[:-1] & side[1:])[0] + 1:
        j = k[s]
        end = np.searchsorted(t, t[j] + sustain, side="right")
        if t[end - 1] < t[j] + 0.9 * sustain:         # sin datos para comprobarlo
            return np.nan
        seg = y[j:end]
        if np.mean(seg >= mid if rising else seg <= mid) >= frac:
            t0, t1, y0, y1 = t[j - 1], t[j], y[j - 1], y[j]
            return float(t0 + (mid - y0) * (t1 - t0) / (y1 - y0)) if y1 != y0 else float(t1)
    return np.nan


def xcorr_lag(tt, rr, ee_t, ee, a, b, lags=XCORR_LAGS):
    """Desplazamiento (s) de la envolvente que maximiza su correlacion con el
    RMS en [a, b) (positivo = la envolvente llega tarde), su r y la curva."""
    k = (tt >= a) & (tt < b)
    t, r = tt[k], rr[k]
    if len(t) < 500 or len(ee_t) < 20:
        return np.nan, np.nan, None, None
    e = np.interp(t, ee_t, ee)
    r = (r - r.mean()) / r.std()
    e = (e - e.mean()) / e.std()
    dt = float(np.median(np.diff(t)))
    L = np.arange(int(round(lags[0] / dt)), int(round(lags[1] / dt)) + 1)
    n = len(r)
    c = np.array([np.mean(r[max(0, -l):n - max(0, l)] * e[max(0, l):n - max(0, -l)]) for l in L])
    j = int(np.argmax(c))
    shift = 0.0
    if 0 < j < len(c) - 1:
        den = c[j - 1] - 2 * c[j] + c[j + 1]
        shift = 0.5 * (c[j - 1] - c[j + 1]) / den if den != 0 else 0.0
    return float((L[j] + shift) * dt), float(c[j]), L * dt, c


def level(tv, yv, a, b):
    k = (tv >= a) & (tv < b)
    return float(np.median(yv[k])) if k.any() else np.nan


def trial_lag(tt, rr, ee_t, ee, t_grasp, t_rest):
    """Todo lo de un ensayo y canal: niveles, cruces y retardos (s), o None si no califica."""
    r_lo = level(tt, rr, t_grasp + W_REST[0], t_grasp + W_REST[1])
    r_hi = level(tt, rr, t_grasp + W_PLATEAU[0], t_rest + W_PLATEAU[1])
    e_lo = level(ee_t, ee, t_grasp + W_REST[0], t_grasp + W_REST[1])
    e_hi = level(ee_t, ee, t_grasp + W_PLATEAU[0], t_rest + W_PLATEAU[1])
    if not (e_hi - e_lo > 30 and r_hi > 2 * r_lo):
        return None
    r_on = cross50(tt, rr, t_grasp - 0.2, r_lo, r_hi, True)
    e_on = cross50(ee_t, ee, t_grasp - 0.2, e_lo, e_hi, True)
    r_lo2 = level(tt, rr, t_rest + W_AFTER[0], t_rest + W_AFTER[1])
    e_lo2 = level(ee_t, ee, t_rest + W_AFTER[0], t_rest + W_AFTER[1])
    r_off = cross50(tt, rr, t_rest - 0.5, r_lo2, r_hi, False)
    e_off = cross50(ee_t, ee, t_rest - 0.5, e_lo2, e_hi, False)
    xl, xr, xlags, xc = xcorr_lag(tt, rr, ee_t, ee, t_grasp + W_XCORR[0], t_rest + W_XCORR[1])
    return dict(r_lo=r_lo, r_hi=r_hi, e_lo=e_lo, e_hi=e_hi, r_lo2=r_lo2, e_lo2=e_lo2,
                r_on=r_on, e_on=e_on, r_off=r_off, e_off=e_off,
                lag_on=e_on - r_on, lag_off=e_off - r_off,
                lag_xcorr=xl, xcorr_r=xr, xcorr_lags=xlags, xcorr_c=xc)
