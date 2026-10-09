"""Sesiones del sabado 26 de septiembre (portatil 2): carga, protocolo y
modalidades auxiliares en el reloj del equipo.

Privacidad: aqui solo entran los codigos de sujeto (S01, S02...) que ya
llevan las carpetas y metadata.json. La hoja de sujetos no se lee nunca, y
ningun nombre sale en figuras, tablas ni textos.

Protocolo (articulo, 2.8.3): siete agarres x cuatro repeticiones, 3 s de
agarre y 6 s de reposo, y 60 s de pausa entre agarres ('grouped'). El
corredor manda `Pgrasp` al agarrar y `Prest` al soltar y en cada pausa; el
anfitrion ademas registra como fase la respuesta a su `?` de cada 10 s, que
no es un comando (sweeps.phase_commands las separa).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import RAW26, OUT, load_session, host_to_device
from sweeps import phase_commands, detect_onset, detect_onset_peak, spike_channels

# Conjunto de datos (EMG8_DATASET, por omision s26, que deja todo lo ya hecho igual):
#   s26     el sabado 26-09, los seis participantes del articulo (S01-S06);
#   s26-01  s26 + la sesion del 1-10, grabada como S00 y que aqui es S07 (numeracion continua;
#           la carpeta no se renombra, el alias va en la tabla).
# Cada uno escribe sus tablas en data/<nombre>/ y sus figuras en figures/<nombre>[_review][_no_imu]/,
# con el prefijo fig_<nombre>_ en el nombre de archivo.
DATASET = os.environ.get("EMG8_DATASET", "s26")
_DATASETS = {
    "s26": dict(days=("20260926",), order=["S05", "S06", "S03", "S01", "S04", "S02"], alias={}),
    "s26-01": dict(days=("20260926", "20261001"), order=["S05", "S06", "S03", "S01", "S04", "S02", "S07"],
                   alias={"sS00_n1_20261001_173409": "S07"}),
}
CFG = _DATASETS[DATASET]
# Con --no-imu en el corredor (EMG8_NO_IMU=1) las figuras salen sin IMU y a
# otra carpeta, para no pisar las completas
# Con --review (EMG8_REVIEW=1), las variantes pedidas tras la reunion del 29-09 van a su propia carpeta;
# las dos marcas se combinan: s26, s26_review, s26_no_imu, s26_review_no_imu.
FIG26 = OUT / "figures" / (DATASET + ("_review" if os.environ.get("EMG8_REVIEW") == "1" else "")
                            + ("_no_imu" if os.environ.get("EMG8_NO_IMU") == "1" else ""))
FIG26.mkdir(parents=True, exist_ok=True)
DATA26 = OUT / "data" / DATASET
DATA26.mkdir(parents=True, exist_ok=True)
FIGP = "fig_" + DATASET              # prefijo de los archivos de figura (fig_s26_..., fig_s26-01_...)


def bench_long_runs() -> list[dict]:
    """Grabaciones largas de banco (bench_long_run/stop), de la mas vieja a la mas nueva.

    Una cortada a mano antes de los 120 min (0 por UART, como la del 29-09) no prueba
    nada y no entra. Son del equipo, no de un conjunto de datos: valen para todos.
    """
    runs = []
    for f in sorted((OUT / "data_raw").glob("bench_*/long-*/summary.json")):
        s = json.loads(f.read_text())
        cause = next((l.split(":", 1)[1] for l in s.get("stop_lines", []) if l.startswith("#STOPCAUSE")), None)
        minutes = s.get("minutes_recorded")
        if minutes is None or (cause == "UART0" and minutes < 120):
            continue
        sdio = next((l for l in s.get("status_after", []) if l.startswith("#SDIO")), "")
        st = next((l for l in s.get("status_after", []) if l.startswith("#STATUS:")), "")
        runs.append(dict(day=f.parent.parent.name.replace("bench_", ""), minutes=minutes, cause=cause,
                         sync_s=int(sdio.split("SYNC_MAX_US=")[1]) / 1e6 if "SYNC_MAX_US=" in sdio else np.nan,
                         # #STATUS: modo, rec, sd, imu, mV, %, perdidas raw, env, imu
                         raw_drops=int(st.split(":")[1].split(",")[6]) if st else None,
                         new_mb=sum(s.get("new_files", {}).values()) / 1e6,
                         previous_intact=s.get("existing_unchanged")))
    return runs

GRASP_NAMES = {
    0: "Rest", 18: "Large diameter", 19: "Small diameter", 20: "Fixed hook",
    21: "Index extension", 22: "Medium wrap", 23: "Ring", 24: "Prismatic 4-finger",
    25: "Stick", 26: "Writing tripod", 27: "Power sphere", 28: "Three-finger sphere",
    29: "Precision sphere", 30: "Tripod", 31: "Prismatic pinch", 32: "Tip pinch",
    33: "Quadpod", 34: "Lateral", 35: "Parallel extension", 36: "Extension type",
    37: "Power disk", 38: "Open a bottle", 39: "Turn a screwdriver", 40: "Cut with a knife",
}

# Orden de presentacion del sabado (semilla 712, agrupado) y su color: la
# paleta validada del skill en ese orden, sin el naranja, que en todas las
# figuras significa barrido de impedancia. Tres tonos quedan por debajo de
# 3:1 contra el blanco: los agarres siempre llevan su nombre al lado.
GRASP_ORDER = [25, 37, 22, 31, 27, 30, 40]
GRASP_COLOR = {25: "#2a78d6", 37: "#1baf7a", 22: "#eda100", 31: "#e87ba4",
               27: "#008300", 30: "#4a3aa7", 40: "#e34948"}
GRASP_SHORT = {25: "Stick", 37: "Power disk", 22: "Medium wrap", 31: "Prism. pinch",
               27: "Power sphere", 30: "Tripod", 40: "Knife cut"}
C_SWEEP = "#eb6834"
# Orden de grabacion (hora de inicio de la sesion completa), por conjunto de datos
SUBJECT_ORDER = list(CFG["order"])

# imu.bin guarda lo que manda el firmware, ya escalado: acelerometro en
# mili-g y giroscopio en decimas de grado/s (records.py: ACC_SCALE, GYRO_SCALE);
# temp100 es la temperatura del chip del IMU en centesimas de grado.
ACC_G_PER_LSB = 1e-3
GYR_DPS_PER_LSB = 0.1


def find_sessions(root: Path = RAW26) -> list[Path]:
    """Carpetas de sesion bajo `root`, esten al nivel que esten (un zip de
    Drive agrega una carpeta intermedia), solo de los dias del conjunto de
    datos: la sesion del 1-10 quedo dentro de la carpeta del sabado y no debe
    entrar en s26."""
    def day(p):
        parts = p.name.split("_")
        return parts[2] if len(parts) > 2 else ""
    out = sorted({p.parent for p in root.rglob("raw.bin") if day(p.parent) in CFG["days"]}, key=lambda p: p.name)
    return out


def subject_of(name: str) -> str:
    # sS04_n1_20260926_114838 -> S04; las sesiones con alias (S00 del 1-10 -> S07) van por la tabla
    if name in CFG["alias"]:
        return CFG["alias"][name]
    return name.split("_")[0][1:] if name.startswith("s") else name.split("_")[0]


# ------------------------------------------------------------ orden de agarres
def _rng(seed: int):
    """mulberry32, identico a U.rng del monitor (web/js/core/util.js)."""
    M = 0xFFFFFFFF
    st = [seed & M]

    def rand():
        st[0] = (st[0] + 0x6D2B79F5) & M
        a = st[0]
        t = ((a ^ (a >> 15)) * (1 | a)) & M
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M)) & M) ^ t
        return ((t ^ (t >> 14)) & M) / 4294967296
    return rand


def _shuffle(arr, rand):
    a = list(arr)
    for i in range(len(a) - 1, 0, -1):
        j = int(rand() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


def grasp_sequence(grasps: list[int], reps: int, seed: int, order: str) -> list[int]:
    """La secuencia de agarres que presento el corredor (Protocol.sequence)."""
    rand = _rng(seed or 1)
    if order == "sequential":
        return [g for g in grasps for _ in range(reps)]
    if order == "grouped":
        return [g for g in _shuffle(grasps, rand) for _ in range(reps)]
    if order == "block":
        out = []
        for _ in range(reps):
            out += _shuffle(grasps, rand)
        return out
    pool = [g for g in grasps for _ in range(reps)]
    return _shuffle(pool, rand)


# ------------------------------------------------------------------ ensayos
@dataclass
class Trial:
    index: int
    grasp: int
    rep: int
    t_grasp: float        # s, reloj del equipo desde la primera muestra
    t_rest: float         # fin del agarre (Prest)
    t_next: float         # siguiente Pgrasp o fin de la sesion


@dataclass
class S26:
    name: str
    subject: str
    s: object                     # common.Session
    f: object                     # anfitrion -> equipo
    resid_ms: float
    trials: list = field(default_factory=list)
    breaks: list = field(default_factory=list)     # (t_ini, t_fin) de las pausas
    order_source: str = ""
    sweeps: list = field(default_factory=list)     # dicts con t_arrival, freq, z
    sensors: dict = field(default_factory=dict)    # arrays t, p1, p2, temp
    onsets: list = field(default_factory=list)     # arranques de barrido en el sEMG
    end_s: float = 0.0

    def dev(self, wall):
        return (self.f(wall) - self.s.t0_us) / 1e6


def load26(path: Path) -> S26:
    s = load_session(path.name, root=path.parent)
    f, resid = host_to_device(s)
    x = S26(name=path.name, subject=subject_of(path.name), s=s, f=f, resid_ms=resid or float("nan"))
    x.end_s = s.duration_s
    _segment(x)
    _aux(x)
    return x


def _segment(x: S26) -> None:
    cmds, _ = phase_commands(x.s)
    # El primer Pgrasp puede salir a los 30 ms de `recording True` y caer, con
    # la latencia del enlace, unas decimas antes de la primera muestra (S05).
    cmds = [(max(t, 0.0), p) for t, p in cmds if t >= -1.0]
    labels = [((x.f(e["wall"]) - x.s.t0_us) / 1e6, e["event"][1], e["event"][2])
              for e in x.s.events if e.get("event", [None])[0] == "label"]
    grasp_t = [t for t, p in cmds if p == "grasp"]
    rest_t = [t for t, p in cmds if p == "rest"]
    trials = []
    for i, tg in enumerate(grasp_t):
        tr = min([t for t in rest_t if t > tg] + [x.end_s])
        tn = min([t for t in grasp_t if t > tg] + [x.end_s])
        trials.append(Trial(i, -1, 0, tg, tr, tn))
    # Pausas: un Prest que sigue a otro Prest (la pausa entre bloques)
    for (t0, p0), (t1, p1) in zip(cmds, cmds[1:]):
        if p0 == "rest" and p1 == "rest":
            nxt = min([t for t in grasp_t if t > t1] + [x.end_s])
            x.breaks.append((t1, nxt))
    # Identidad del agarre: la etiqueta registrada, si llego antes del agarre;
    # si no, la secuencia reconstruida con la semilla del protocolo.
    meta = x.s.meta
    seq = None
    if meta.get("grasps") and meta.get("reps"):
        order = meta.get("order") or "grouped"
        seq = grasp_sequence(meta["grasps"], int(meta["reps"]), int(meta.get("seed") or 1), order)
    used_labels = 0
    for tr in trials:
        prev = [l for l in labels if l[0] <= tr.t_grasp + 0.05]
        if prev and tr.t_grasp - prev[-1][0] < 12.0:
            tr.grasp, tr.rep = int(prev[-1][1]), int(prev[-1][2])
            used_labels += 1
        elif seq and tr.index < len(seq):
            tr.grasp = seq[tr.index]
    if seq:
        # repeticion = orden dentro del mismo agarre
        count = {}
        for tr in trials:
            if tr.rep == 0:
                count[tr.grasp] = count.get(tr.grasp, 0) + 1
                tr.rep = count[tr.grasp]
    x.order_source = ("labels" if used_labels == len(trials) and trials else
                      f"labels {used_labels}/{len(trials)} + seed" if used_labels else "seed")
    x.trials = trials


def _aux(x: S26) -> None:
    sw, ts, p1, p2, tc = [], [], [], [], []
    for a in x.s.aux:
        if a.get("kind") == "impedance_sweep":
            z = np.array(a["z_ohm"], dtype=float)
            sw.append(dict(id=a.get("id"), t_arrival=x.dev(a["wall"]),
                           freq=np.array(a["freq_hz"], dtype=float), z=z))
        elif a.get("kind") == "aux_sensors":
            ts.append(x.dev(a["wall"]))
            p1.append(a.get("p1_kpa", np.nan)); p2.append(a.get("p2_kpa", np.nan))
            tc.append(a.get("temp_c", np.nan))
    x.sweeps = sw
    x.sensors = dict(t=np.array(ts), p1=np.array(p1, dtype=float), p2=np.array(p2, dtype=float),
                     temp=np.array(tc, dtype=float))
    # La placa auxiliar (main_medicion.cpp, filtrarMediana3) da de cada FSR la
    # mediana de sus 3 ultimas lecturas crudas, y ese bufer pasa de un sostener
    # al siguiente: la 1a lectura tras cada Pgrasp suele repetir la ultima del
    # sostener anterior, y la 2a a veces. Solo desde la 3a todas las lecturas
    # del bufer son del sostener en curso. `load_ok` marca esas lecturas de
    # carga; la temperatura no pasa por ese filtro.
    t = x.sensors["t"]
    # El brazalete reenvia las lineas auxiliares en un latido de 1 s sin marca propia. El sabado
    # llegaba una por latido; desde el 1-10 la placa auxiliar lee ~3 por segundo y llegan de a tres,
    # con 1-5 ms entre ellas. Para dibujarlas, `t_est` reparte cada tanda por igual en el segundo
    # anterior a su llegada (inferido: el instante real de cada lectura no viaja). Con una lectura
    # por tanda, t_est = t. Los analisis de llegada usan `t`.
    t_est = t.copy()
    if len(t):
        cut = np.where(np.diff(t) > 0.1)[0] + 1
        for g in np.split(np.arange(len(t)), cut):
            n = len(g)
            if n > 1:
                t_est[g] = t[g[-1]] - (n - 1 - np.arange(n)) / n
    x.sensors["t_est"] = t_est
    nth = np.zeros(len(t), dtype=int)
    for tr in x.trials:
        k = np.where((t >= tr.t_grasp) & (t < tr.t_next))[0]
        nth[k] = np.arange(1, len(k) + 1)
    x.sensors["load_ok"] = nth >= 3


def sweep_onsets(x: S26) -> list[float]:
    """Arranque de cada barrido en el sEMG, anclado a los Prest (03_find_sweeps)."""
    cmds, _ = phase_commands(x.s)
    prest = [t for t, p in cmds if p == "rest" and t >= 0.2]
    if len(prest) < 3:
        return []
    chs, _ = spike_channels(x.s, prest)
    out, last = [], -1e9
    for t in prest:
        r1 = detect_onset(x.s, t, channels=chs, min_ch=2, win=0.03, thr_mv=200, k_mad=5) if len(chs) >= 2 else None
        r2 = detect_onset_peak(x.s, t, chs) if len(chs) >= 2 else None
        L1 = None if r1 is None else r1[0] - t
        L2 = None if r2 is None else r2[0] - t
        on = None
        if L1 is not None and 0.08 <= L1 <= 0.26:
            on = r1[0]
        elif L2 is not None and 0.08 <= L2 <= 0.26:
            on = r2[0]
        elif L1 is not None and 0.40 <= L1 <= 0.70 and t - last < 4.2:
            on = r1[0]
        if on is not None:
            out.append(on)
            last = on
    x.onsets = out
    return out


def imu_arrays(x: S26):
    """IMU en unidades fisicas y reloj del equipo: t (s), acc (g) y gyr (dps)."""
    imu = x.s.imu
    if imu is None or len(imu) == 0:
        return None
    t = (imu["ts_us"].astype(np.int64) - x.s.t0_us) / 1e6
    order = np.argsort(t, kind="stable")
    acc = np.stack([imu[k].astype(float) for k in ("ax", "ay", "az")], 1) * ACC_G_PER_LSB
    gyr = np.stack([imu[k].astype(float) for k in ("gx", "gy", "gz")], 1) * GYR_DPS_PER_LSB
    return t[order], acc[order], gyr[order]


# ------------------------------------------------------------ sEMG en rejilla
FS = 1000.0


def grid_raw(x: S26, fs: float = FS):
    """Los 8 crudos en una rejilla comun de 1 kHz desde la primera muestra.

    Interpolacion lineal (el tope R1000 deja intervalos de 0.9 y 2.7 ms que
    promedian 1 ms). Donde la muestra real mas cercana queda a mas de 5 ms
    —la vista previa de 1/s del arranque y el hueco de S02— va NaN: ahi no
    hay senal, y un tramo interpolado pasaria por sEMG plano.
    """
    tg = np.arange(0.0, x.end_s, 1.0 / fs)
    Y = np.full((8, len(tg)), np.nan)
    for i in range(8):
        c = x.s.raw[i]
        t = (c.ts_us - x.s.t0_us) / 1e6
        y = np.interp(tg, t, c.mv)
        j = np.clip(np.searchsorted(t, tg), 1, len(t) - 1)
        near = np.minimum(np.abs(t[j] - tg), np.abs(t[j - 1] - tg))
        y[near > 0.005] = np.nan
        Y[i] = y
    return tg, Y


def bandpass_fft(Y, fs=FS, lo=20.0, hi=450.0, w_lo=5.0, w_hi=25.0, notch=None):
    """Pasa-banda de fase cero por FFT con transiciones de coseno alzado.

    Los NaN se llenan con cero (tras quitar la media) para filtrar y se
    devuelven como NaN ensanchados 0.3 s, que es lo que tarda en apagarse el
    borde. Sin scipy: el instalado esta roto.
    """
    Y = np.atleast_2d(Y)
    out = np.empty_like(Y)
    n = Y.shape[1]
    f = np.fft.rfftfreq(n, 1.0 / fs)
    g = np.ones_like(f)
    g[f < lo - w_lo / 2] = 0.0
    k = (f >= lo - w_lo / 2) & (f < lo + w_lo / 2)
    g[k] = 0.5 - 0.5 * np.cos(np.pi * (f[k] - (lo - w_lo / 2)) / w_lo)
    g[f > hi + w_hi / 2] = 0.0
    k = (f > hi - w_hi / 2) & (f <= hi + w_hi / 2)
    g[k] = 0.5 + 0.5 * np.cos(np.pi * (f[k] - (hi - w_hi / 2)) / w_hi)
    for fn in (notch or []):
        g[np.abs(f - fn) < 1.0] = 0.0
    pad = int(0.3 * fs)
    for i in range(Y.shape[0]):
        y = Y[i]
        bad = ~np.isfinite(y)
        yy = np.where(bad, 0.0, y - np.nanmean(y))
        out[i] = np.fft.irfft(np.fft.rfft(yy) * g, n)
        if bad.any():
            b = np.convolve(bad.astype(float), np.ones(2 * pad + 1), "same") > 0
            out[i][b] = np.nan
    return out


def win(tg, a, b):
    """Indices de la rejilla en [a, b)."""
    return slice(int(np.searchsorted(tg, a)), int(np.searchsorted(tg, b)))


def prest_times(x: S26) -> list[float]:
    cmds, _ = phase_commands(x.s)
    return [max(t, 0.0) for t, p in cmds if p == "rest" and t >= -1.0]


# ------------------------------------------------------ continuidad (09_performance)
def channel_stats(ts):
    """Mismo criterio que 09_performance.py: hueco = intervalo > max(5 ms,
    2.2 x p99); muestras perdidas por hueco = hueco / intervalo medio - 1."""
    dt = np.diff(ts).astype(float)
    p99 = np.percentile(dt, 99)
    thr = max(5000.0, 2.2 * p99)
    gaps = dt[dt > thr]
    ok = dt[dt <= thr]
    mean_ok = ok.mean()
    missing = float(np.sum(gaps / mean_ok - 1)) if len(gaps) else 0.0
    dur = (ts[-1] - ts[0]) / 1e6
    return dict(n=len(ts), dur_s=dur, rate_hz=len(ts) / dur, med_us=np.median(dt),
                p1_us=np.percentile(dt, 1), p99_us=p99, max_us=dt.max(), mean_us=mean_ok,
                cv=ok.std() / mean_ok, n_gaps=len(gaps), gap_s=gaps.sum() / 1e6,
                max_gap_ms=(gaps.max() / 1e3 if len(gaps) else 0.0),
                missing=missing, loss_pct=100 * missing / (len(ts) + missing),
                nonmono=int(np.sum(dt <= 0)))
