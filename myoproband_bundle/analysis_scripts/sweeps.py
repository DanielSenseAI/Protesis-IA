"""Localizacion de los barridos de bioimpedancia dentro del sEMG.

Lo que se sabe del firmware (leido, no supuesto):

- Placa auxiliar, Electromyographic-interaction-variables @ 973534b
  (src/main_medicion.cpp + include/config_comun.h): START_FREQ 1000,
  FREQ_INCR 1000, NUM_INCR 100. measureUnknown() mide 1..100 kHz y descarta
  el punto de 1 kHz (transitorio de reset), asi que imp: trae 99 puntos
  2..100 kHz. Cada barrido empieza con reset() y startSweep()
  (STANDBY -> RESET -> INIT_START_FREQ -> 30 ms -> START_SWEEP) y cada punto
  se escribe en la SD con fflush+fsync antes de incrementar. Nunca se manda
  el AD5933 a standby/power-down al terminar.
- Lunes (sesiones con fases): un barrido por cada byte 05, que el brazalete
  manda con cada `Prest`. Sabado (@ 6291247): barridos seguidos, uno tras
  otro, desde el START hasta el STOP.
- El anfitrion pregunta `?` cada 10 s y la respuesta trae `#PHASE:`; esas
  respuestas quedan en events.jsonl como eventos de fase aunque nadie haya
  mandado nada. Solo los cambios de fase y los `Prest` fuera de esa rejilla
  son comandos.
"""
from __future__ import annotations

import numpy as np

from common import Session, host_to_device

POLL_S = 10.0              # link.py: self._next_status = now + 10.0
SPIKE_CH = (0, 1, 6, 7)    # E1, E2, E7, E8: donde el arranque se ve limpio


def phase_commands(s: Session):
    """Separa los comandos de fase reales de los ecos del sondeo de 10 s.

    Devuelve (comandos, ecos) como listas de (t_dev_s, fase). Un evento que
    repite la fase anterior es candidato a eco; los ecos caen en una rejilla
    de ~10 s. Un `Prest` repetido fuera de la rejilla es un comando real (la
    pausa entre bloques del corredor manda `Prest` estando ya en reposo).
    """
    f, _ = host_to_device(s)
    t0 = s.t0_us
    ev = [((f(e["wall"]) - t0) / 1e6, e["event"][1]) for e in s.events
          if e.get("event", [None])[0] == "phase"]
    if not ev:
        return [], []
    rep = [t for k, (t, p) in enumerate(ev) if k and p == ev[k - 1][1]]
    grid = None
    if len(rep) >= 2:
        # Periodo efectivo del sondeo y fase de la rejilla, por minimos
        # cuadrados sobre las repeticiones que encajan
        per = POLL_S
        base = rep[0]
        for _ in range(3):
            n = np.round((np.array(rep) - base) / per)
            ok = np.abs(np.array(rep) - base - n * per) < 0.3
            if ok.sum() >= 2:
                A = np.vstack([n[ok], np.ones(ok.sum())]).T
                (per, base), *_ = np.linalg.lstsq(A, np.array(rep)[ok], rcond=None)
        grid = (base, per)
    cmds, echoes = [], []
    slot_best = {}   # casilla de la rejilla -> indice del candidato a eco mas cercano
    cand = []
    for k, (t, p) in enumerate(ev):
        same = k and p == ev[k - 1][1]
        off = ((t - grid[0]) + grid[1] / 2) % grid[1] - grid[1] / 2 if grid is not None else np.inf
        if same and abs(off) < 0.3:
            n = int(np.round((t - grid[0]) / grid[1]))
            cand.append(k)
            if n not in slot_best or abs(off) < slot_best[n][1]:
                slot_best[n] = (k, abs(off))
    # Cada sondeo deja un solo eco: si dos eventos iguales caen en la misma
    # casilla, el mas lejano del punto de la rejilla es un comando real (la
    # pausa entre bloques de S01 y S02 cayo a 0.13-0.21 s de un sondeo y se
    # perdia como eco).
    echo_idx = {k for k, _ in slot_best.values()}
    for k, (t, p) in enumerate(ev):
        (echoes if k in echo_idx else cmds).append((t, p))
    return cmds, echoes


def detect_onset(s: Session, t_cmd: float, lo=0.03, hi=0.7, thr_mv=100.0, k_mad=4.0,
                 channels=SPIKE_CH, min_ch=2, win=0.03):
    """Arranque del AD5933 tras un comando: primer cruce simultaneo.

    Busca en [t_cmd+lo, t_cmd+hi] el primer instante en que >= min_ch de los
    canales dados se apartan mas de max(thr, k*MAD) de su mediana previa, a
    menos de `win` entre si. La ventana llega a 0.7 s porque un Prest que
    cae durante un barrido queda en cola y el siguiente arranca al terminar
    el anterior (se vio a +0.53 s). k=4 y no 6: justo antes del Prest el
    sujeto todavia aprieta y la MAD previa sale inflada por el EMG.
    Los canales que muestran el arranque cambian con la colocacion del
    brazalete (23:13: E1/E2/E7/E8; 18:13: E5/E6/E7/E8): con los ocho hay
    que pedir mas coincidencias (3 en 25 ms) para no confundirlo con el EMG.
    Devuelve (t_onset, {canal: pico_mV}) o None.
    """
    t0 = s.t0_us
    starts, peaks = [], {}
    for i in channels:
        c = s.raw[i]
        t = (c.ts_us - t0) / 1e6
        pre = (t >= t_cmd - 0.4) & (t < t_cmd + lo)
        look = (t >= t_cmd + lo) & (t <= t_cmd + hi)
        if pre.sum() < 50 or look.sum() < 50:
            continue
        y = c.mv
        med = np.median(y[pre])
        mad = np.median(np.abs(y[pre] - med)) * 1.4826
        k = np.where(np.abs(y[look] - med) > max(thr_mv, k_mad * mad))[0]
        if len(k):
            tk = t[look][k]
            # cada racha de cruces cuenta por su comienzo: un canal con EMG
            # cruza antes, pero su racha del arranque tambien se registra
            starts += [(x, i) for x in tk[np.r_[True, np.diff(tk) > 0.05]]]
            peaks[i] = float(np.max(np.abs(y[look] - med)))
    starts.sort()
    # el primer instante con >= min_ch canales distintos arrancando en `win`
    for a, (ta, _) in enumerate(starts):
        chans = {ch for tb, ch in starts[a:] if tb - ta < win}
        if len(chans) >= min_ch:
            return ta, {ch: peaks[ch] for ch in chans}
    return None


def detect_onset_peak(s: Session, t_cmd: float, channels, lo=-0.05, hi=0.7, cap=20.0, back=0.3):
    """Arranque por el evento mas grande y no por el primer cruce.

    Suma entre los canales del arranque la desviacion normalizada por la MAD
    previa (tope `cap` para que un canal saturado no mande solo), en bins de
    1 ms; toma el maximo en [t_cmd+lo, t_cmd+hi] y retrocede hasta donde la
    suma baja de `back` veces el pico. A las 16:07 hay caidas periodicas de
    ~-100 mV y el ajuste de reloj tiene +-68 ms: el primer cruce no sirve
    ahi, el pico si. Devuelve (t_onset, puntaje_pico) o None."""
    t0 = s.t0_us
    grid = np.arange(t_cmd + lo, t_cmd + hi, 0.001)
    score = np.zeros_like(grid)
    used = 0
    for i in channels:
        c = s.raw[i]
        t = (c.ts_us - t0) / 1e6
        pre = (t >= t_cmd - 0.5) & (t < t_cmd - 0.05)
        look = (t >= grid[0] - 0.01) & (t <= grid[-1] + 0.01)
        if pre.sum() < 50 or look.sum() < 50:
            continue
        med = np.median(c.mv[pre])
        mad = np.median(np.abs(c.mv[pre] - med)) * 1.4826 + 1e-6
        dev = np.minimum(np.abs(c.mv[look] - med) / mad, cap)
        # cada muestra vale hasta la siguiente (muestreo ~1 ms)
        k = np.clip(np.searchsorted(t[look], grid, side="right") - 1, 0, look.sum() - 1)
        score += dev[k]
        used += 1
    if used < 2:
        return None
    p = int(np.argmax(score))
    if score[p] < 8 * used:          # nada destacable: no hubo barrido
        return None
    j = p
    while j > 0 and score[j - 1] > back * score[p] and p - j < 80:
        j -= 1
    # el primer bin del evento: el ultimo por debajo del umbral, mas uno
    return float(grid[j]), float(score[p])


def spike_channels(s: Session, t_cmds, lo=0.05, hi=0.45, ratio=6.0):
    """Que canales llevan el transitorio de arranque en ESTA colocacion.

    Para cada canal, la mediana entre comandos del pico |y - mediana previa|
    en [lo, hi] dividido por la MAD previa. El EMG de soltar el agarre sube
    en unos Prest y en otros no; el transitorio sube en todos, asi que la
    mediana lo separa. Devuelve los canales con cociente >= ratio."""
    t0 = s.t0_us
    out = {}
    for i in range(8):
        c = s.raw[i]
        t = (c.ts_us - t0) / 1e6
        r = []
        for tc in t_cmds:
            pre = (t >= tc - 0.4) & (t < tc)
            look = (t >= tc + lo) & (t <= tc + hi)
            if pre.sum() < 50 or look.sum() < 50:
                continue
            med = np.median(c.mv[pre])
            mad = np.median(np.abs(c.mv[pre] - med)) * 1.4826 + 1e-6
            r.append(np.max(np.abs(c.mv[look] - med)) / mad)
        out[i] = float(np.median(r)) if r else 0.0
    return tuple(i for i, v in out.items() if v >= ratio), out


def common_mode(s: Session, fs: float = 500.0, max_gap_s: float = 0.02, channels=tuple(range(8))):
    """Modo comun de los crudos en una rejilla uniforme (mediana entre
    canales, mV). Donde un canal no tiene muestra a menos de max_gap_s queda
    NaN: el sabado los primeros segundos son la vista previa a 1 Hz. Con
    tierra (lunes) el arranque solo se ve en 2-3 canales: se pasan esos."""
    t0 = s.t0_us
    a = max(c.ts_us[0] for c in s.raw.values())
    b = min(c.ts_us[-1] for c in s.raw.values())
    tg = np.arange((a - t0) / 1e6, (b - t0) / 1e6, 1 / fs)
    Y = np.full((len(channels), len(tg)), np.nan)
    for row, i in enumerate(channels):
        c = s.raw[i]
        t = (c.ts_us - t0) / 1e6
        y = np.interp(tg, t, c.mv)
        k = np.clip(np.searchsorted(t, tg), 1, len(t) - 1)
        near = np.minimum(np.abs(t[k] - tg), np.abs(t[k - 1] - tg))
        y[near > max_gap_s] = np.nan
        Y[row] = y
    return tg, np.nanmedian(Y, axis=0), Y


def track_cycles(s: Session, period_guess=4.3, fs=500.0, channels=tuple(range(8))):
    """Arranques de barridos encadenados (firmware del sabado, 6291247: el
    barrido se repite solo hasta el STOP; el lunes a las 16:23 tambien). El
    arranque es un escalon del modo comun de `channels`; se buscan
    candidatos fuertes, se estima el ciclo y se sigue ciclo a ciclo buscando
    el primer cruce en +-0.25 s de lo esperado.
    Devuelve (onsets_s, periodo_s, detectados_bool)."""
    tg, cm, _ = common_mode(s, fs, channels=channels)
    ok = ~np.isnan(cm)
    # desvio respecto a la mediana de los 10-50 ms previos
    past = np.full_like(cm, np.nan)
    L0, L1 = int(0.010 * fs), int(0.050 * fs)
    from numpy.lib.stride_tricks import sliding_window_view
    if len(cm) > L1:
        w = sliding_window_view(cm, L1 - L0)
        med = np.nanmedian(w, axis=1)          # ventana [n-L1, n-L0)
        past[L1:] = med[:len(cm) - L1]
    d = np.abs(cm - past)
    dv = d[ok & ~np.isnan(d)]
    mad = np.median(np.abs(dv - np.median(dv))) * 1.4826
    base = np.median(dv)
    strong = base + 12 * mad
    weak = base + 5 * mad
    cand = []
    for n in np.where(d > strong)[0]:
        if not cand or tg[n] - cand[-1] > 1.0:
            cand.append(tg[n])
    cand = np.array(cand)
    iv = np.diff(cand)
    iv = iv[(iv > 0.85 * period_guess) & (iv < 1.15 * period_guess)]
    P = float(np.median(iv)) if len(iv) else period_guess
    # seguimiento desde el candidato mas fuerte hacia los dos lados
    def first_cross(t_exp, half=0.25):
        m = np.where((tg >= t_exp - half) & (tg <= t_exp + half) & (d > weak))[0]
        return tg[m[0]] if len(m) else None
    if not len(cand):
        return np.array([]), P, np.array([], bool)
    k0 = cand[np.argmax([d[np.searchsorted(tg, c)] for c in cand])]
    fwd, det_f = [k0], [True]
    t = k0
    while t + P < tg[-1]:
        hit = first_cross(t + P)
        t = hit if hit is not None else t + P
        fwd.append(t); det_f.append(hit is not None)
    bwd, det_b = [], []
    t = k0
    while t - P > tg[0]:
        hit = first_cross(t - P)
        t = hit if hit is not None else t - P
        bwd.append(t); det_b.append(hit is not None)
    on = np.array(bwd[::-1] + fwd)
    det = np.array(det_b[::-1] + det_f)
    # fuera del tramo con datos a tasa completa no hay nada que detectar
    valid = np.array([ok[min(np.searchsorted(tg, x), len(tg) - 1)] for x in on])
    return on[valid], P, det[valid]


def direct_transients(s: Session, channels, thr_mv=200.0, win=0.03, refractory=1.0):
    """Transitorios grandes y simultaneos en TODOS los canales dados, sin
    ancla de comando. Sirve para el lunes 16:23, cuando la placa auxiliar
    barria sola cada 4.27 s: el seguidor de ciclos se engancha ahi a las
    caidas periodicas de ~-100 mV, y un umbral absoluto de 200 mV no."""
    t0 = s.t0_us
    hits = []
    for i in channels:
        c = s.raw[i]
        t = (c.ts_us - t0) / 1e6
        y = c.mv - np.median(c.mv)
        k = np.where(np.abs(y) > thr_mv)[0]
        if len(k):
            tk = t[k]
            hits += [(x, i) for x in tk[np.r_[True, np.diff(tk) > 0.1]]]
    hits.sort()
    out = []
    for a, (ta, _) in enumerate(hits):
        chans = {ch for tb, ch in hits[a:] if tb - ta < win}
        if len(chans) >= len(channels) and (not out or ta - out[-1] > refractory):
            out.append(ta)
    return np.array(out)


def heartbeat_grid(times_s):
    """Rejilla del latido de reenvio (~1 s) a partir de las horas de llegada.

    Se trabaja en el reloj del anfitrion (wall): las llegadas y los acuses de
    fase viajan por el mismo camino serie, y asi no se suma el error del
    ajuste anfitrion->equipo. El periodo se busca por coherencia de fase
    (un ajuste por minimos cuadrados desde 1.0 s cae en el periodo vecino
    equivocado) y luego se refina. Devuelve (h0, P, resid_ms)."""
    ta = np.sort(np.asarray(times_s, dtype=float))
    if len(ta) < 3:
        return None
    ref = ta[0]
    periods = np.arange(0.980, 1.030, 0.00005)
    R = [abs(np.mean(np.exp(2j * np.pi * (ta - ref) / p))) for p in periods]
    P = periods[int(np.argmax(R))]
    h0 = ref
    for _ in range(3):
        n = np.round((ta - h0) / P)
        A = np.vstack([n, np.ones_like(n)]).T
        (P, h0), *_ = np.linalg.lstsq(A, ta, rcond=None)
    resid = ta - (h0 + np.round((ta - h0) / P) * P)
    return float(h0), float(P), float(np.std(resid) * 1000)


def imp_line_bytes(sweep: dict) -> int:
    """Largo exacto de la linea imp: que la placa auxiliar manda al terminar
    ("%.1f,%.2f;" por punto, sin el ultimo ';', entre 'imp:[' y ']\\n')."""
    body = ";".join(f"{fq:.1f},{z:.2f}" for fq, z in zip(sweep["freq_hz"], sweep["z_ohm"]))
    return len("imp:[") + len(body) + len("]\n")


def freq_at(dt_s, t_first_step: float, step_s: float):
    """Frecuencia (kHz) que el AD5933 esta aplicando dt_s despues del
    arranque, con pasos iguales de 1 kHz. Antes de t_first_step: 1 kHz (el
    punto que se descarta); despues del punto de 100 kHz se queda en 100."""
    dt = np.asarray(dt_s, dtype=float)
    k = 2 + np.floor((dt - t_first_step) / step_s)
    k = np.where(dt < t_first_step, 1, k)
    return np.clip(k, 1, 100)


def band_windows(t_first_step: float, step_s: float, edges_khz=(1, 5, 10, 20, 30, 100)):
    """Tramos de tiempo (desde el arranque) en que la excitacion esta en cada
    banda. Bandas cerradas a la derecha: 1-5 = {1..5 kHz}, 5-10 = {6..10}, ..."""
    def start_of(k):     # instante en que empieza el punto de k kHz
        return 0.0 if k <= 1 else t_first_step + (k - 2) * step_s
    out = []
    lo = edges_khz[0]
    for hi in edges_khz[1:]:
        k0 = lo if lo == edges_khz[0] else lo + 1
        out.append((f"{lo}-{hi} kHz", start_of(k0), start_of(hi + 1)))
        lo = hi
    return out
