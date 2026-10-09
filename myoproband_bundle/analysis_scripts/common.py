"""Cargadores compartidos para el analisis del articulo.

Autocontenido a proposito: los formatos de registro y el mapa de canales se
copian aqui en vez de importarse de IA-Arm_Monitor/backend, para que el
analisis no cambie si el backend cambia. Fuente de cada definicion:

  SAMPLE_DTYPE, IMU_DTYPE, RAW_CH_BY_ADC, ENV_CH_BY_ADC, emg_index
      <- IA-Arm_Monitor/backend/emg8/records.py  (dev @ 38a3a3b, 2026-09-22)

Todo lo que se lee aqui es de solo lectura: nada escribe en las carpetas de
los proyectos.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- Cambio para la publicacion de datos (unico cambio a este archivo) -------
# Las rutas eran fijas (D:/PhD/...). Ahora salen de variables de entorno y, si
# no estan, quedan como estaban:
#   EMG8_OUT       carpeta de trabajo: aqui se escriben data/, figures/, report_*.html
#   EMG8_SESSIONS  carpeta que contiene las carpetas de sesion (sS01_n1_...)
#   EMG8_MONITOR   solo para los scripts que leen sessions/ del monitor (no los s26_*)
MONITOR = Path(os.environ.get("EMG8_MONITOR", r"D:/PhD/Code/IA-Arm_Monitor"))
SESSIONS = MONITOR / "sessions"
OUT = Path(os.environ.get("EMG8_OUT", r"D:/PhD/Code/EMG8_article_analysis"))
FIG = OUT / "figures"
DATA = OUT / "data"
# Sesiones del sabado 26 (portatil 2), bajadas de Drive a mano. Sin la hoja
# de sujetos: los nombres no entran en ningun archivo de este analisis.
RAW26 = (Path(os.environ["EMG8_SESSIONS"]) if "EMG8_SESSIONS" in os.environ
         else OUT / "data_raw" / "2026-09-26")

SAMPLE_DTYPE = np.dtype([("ts_us", "<u4"), ("adc", "u1"), ("ch", "u1"), ("value", "<i2")])
IMU_DTYPE = np.dtype([("ts_us", "<u4"),
                      ("ax", "<i2"), ("ay", "<i2"), ("az", "<i2"),
                      ("gx", "<i2"), ("gy", "<i2"), ("gz", "<i2"),
                      ("temp100", "<i2"), ("_pad", "<u2")])

# Medido contra el hardware (FIRMWARE-CONTRACT.md §2.2): las patillas crudas
# no son las mismas en los cuatro ADC.
RAW_CH_BY_ADC = ((0, 3), (1, 3), (0, 3), (1, 3))
ENV_CH_BY_ADC = ((1, 2), (0, 2), (1, 2), (0, 2))
LSB_MV = 2.0          # ADS1015 con ConfigPGA::One = +-4.096 V, 12 bits con signo


def emg_index(adc: int, ch: int):
    if not 0 <= adc < 4:
        return None
    raw, env = RAW_CH_BY_ADC[adc], ENV_CH_BY_ADC[adc]
    if ch in raw:
        return "raw", adc * 2 + raw.index(ch)
    if ch in env:
        return "env", adc * 2 + env.index(ch)
    return None


def _unwrap_u32(ts: np.ndarray) -> np.ndarray:
    """ts_us es u32 en el firmware: da la vuelta cada ~71.6 min."""
    t = ts.astype(np.int64)
    jumps = np.where(np.diff(t) < -(1 << 31))[0]
    for j in jumps:
        t[j + 1:] += 1 << 32
    return t


@dataclass
class Channel:
    kind: str                 # raw | env
    index: int                # 0-7
    adc: int
    pin: int
    ts_us: np.ndarray         # int64, orden de llegada
    counts: np.ndarray        # int16 del ADC

    @property
    def mv(self) -> np.ndarray:
        return self.counts.astype(np.float64) * LSB_MV


@dataclass
class Session:
    name: str
    path: Path
    meta: dict
    raw: dict = field(default_factory=dict)     # indice -> Channel
    env: dict = field(default_factory=dict)
    imu: np.ndarray | None = None
    events: list = field(default_factory=list)
    aux: list = field(default_factory=list)
    n_records: int = 0

    @property
    def t0_us(self) -> int:
        return int(min(c.ts_us[0] for c in self.raw.values()))

    @property
    def duration_s(self) -> float:
        a = min(c.ts_us[0] for c in self.raw.values())
        b = max(c.ts_us[-1] for c in self.raw.values())
        return (b - a) / 1e6


def _jsonl(p: Path) -> list:
    out = []
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def load_session(name: str, root: Path | None = None) -> Session:
    path = (root or SESSIONS) / name
    meta = {}
    try:
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    s = Session(name=name, path=path, meta=meta)
    a = np.fromfile(path / "raw.bin", dtype=SAMPLE_DTYPE)
    s.n_records = len(a)
    for adc in range(4):
        for pin in range(4):
            k = emg_index(adc, pin)
            if k is None:
                continue
            kind, idx = k
            m = (a["adc"] == adc) & (a["ch"] == pin)
            if not m.any():
                continue
            ch = Channel(kind, idx, adc, pin, _unwrap_u32(a["ts_us"][m]), a["value"][m].copy())
            (s.raw if kind == "raw" else s.env)[idx] = ch
    ip = path / "imu.bin"
    if ip.exists() and ip.stat().st_size >= IMU_DTYPE.itemsize:
        s.imu = np.fromfile(ip, dtype=IMU_DTYPE)
    s.events = _jsonl(path / "events.jsonl")
    s.aux = _jsonl(path / "aux.jsonl")
    return s


def host_to_device(session: Session):
    """Ajuste lineal reloj del anfitrion (wall, s) -> reloj del equipo (ts_us).

    events.jsonl guarda, en cada evento, la hora del anfitrion y la ultima
    marca de tiempo del equipo vista. Es la unica cosa que ata los dos relojes
    en una grabacion del portatil. Devuelve (f, residuo_ms) con f(wall)->ts_us.
    """
    pts = [(e["wall"], e["last_ts_us"]) for e in session.events
           if e.get("last_ts_us") is not None]
    if len(pts) < 3:
        return None, None
    w = np.array([p[0] for p in pts]); d = np.array([p[1] for p in pts], dtype=np.float64)
    # Antes de `recording True` (cuenta regresiva) los eventos arrastran la
    # ultima marca de la grabacion ANTERIOR, y con el firmware del lunes por
    # la tarde la marca vuelve a cero al grabar: esos puntos no son de esta
    # sesion. Se descartan por consenso: con pendiente fija de 1e6 us/s los
    # buenos comparten desfase y los rancios quedan lejos.
    off = d - 1e6 * (w - w[0])
    med = np.median(off)
    keep = np.abs(off - med) < max(5 * np.median(np.abs(off - med)) * 1.4826, 5e4)
    w, d = w[keep], d[keep]
    # Centrado obligatorio: wall ~1.8e9 s con pendiente ~1e6 deja la matriz
    # tan mal condicionada que el ajuste se aplasta a una constante (se vio:
    # residuo de 33 s y los doce barridos en el mismo instante).
    w0, d0 = w[0], d[0]
    A = np.vstack([w - w0, np.ones_like(w)]).T
    (slope, icpt), *_ = np.linalg.lstsq(A, d - d0, rcond=None)
    resid = (d - d0) - (slope * (w - w0) + icpt)
    f = lambda wall: slope * (np.asarray(wall) - w0) + icpt + d0
    f.slope = slope
    f.inv = lambda dev_us: (np.asarray(dev_us) - icpt - d0) / slope + w0
    return f, float(np.std(resid) / 1000.0)


def real_sessions() -> list[str]:
    """Las grabaciones del equipo real (no la fuente sintetica).

    La huella que separa: la sintetica entrega intervalos perfectamente
    regulares (1805 us a 554 Hz); el hardware no. Ver 01_inventory.py.
    """
    out = []
    for d in sorted(os.listdir(SESSIONS)):
        p = SESSIONS / d / "raw.bin"
        if d.startswith("_") or not p.exists() or p.stat().st_size < 8000:
            continue
        a = np.fromfile(p, dtype=SAMPLE_DTYPE, count=200000)
        m = (a["adc"] == 0) & (a["ch"] == 0)
        dt = np.diff(np.sort(a["ts_us"][m].astype(np.int64)))
        dt = dt[dt > 0]
        if len(dt) > 50 and np.percentile(dt, 90) - np.percentile(dt, 10) > 20:
            out.append(d)
    return out


def stft_db(y: np.ndarray, fs: float, nperseg: int = 256, noverlap: int = 192):
    """Espectrograma en dB con numpy solo (el scipy instalado esta roto).

    Ventana de Hann, sin relleno. Devuelve (f_Hz, t_s, S_dB) con t en el
    centro de cada ventana, relativo al inicio de `y`.
    """
    step = nperseg - noverlap
    n = (len(y) - nperseg) // step + 1
    if n < 1:
        return np.array([]), np.array([]), np.zeros((0, 0))
    win = np.hanning(nperseg)
    scale = 1.0 / (fs * (win ** 2).sum())
    frames = np.lib.stride_tricks.sliding_window_view(y, nperseg)[::step][:n]
    frames = (frames - frames.mean(axis=1, keepdims=True)) * win
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2 * scale
    spec[:, 1:-1] *= 2
    f = np.fft.rfftfreq(nperseg, 1 / fs)
    t = (np.arange(n) * step + nperseg / 2) / fs
    return f, t, 10 * np.log10(spec.T + 1e-12)


def uniform(ch_ts_us: np.ndarray, values: np.ndarray, fs: float, t0_us: int):
    """Remuestreo lineal a una rejilla uniforme, en segundos desde t0."""
    t = (ch_ts_us - t0_us) / 1e6
    tg = np.arange(t[0], t[-1], 1.0 / fs)
    return tg, np.interp(tg, t, values)
