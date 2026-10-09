"""Grabaciones de banco en SD (del repositorio del firmware, solo lectura):
radio apagada frente a encendida, y tope de 1000 Hz frente a velocidad maxima.

Para cada captura y cada canal crudo:
- intervalos entre muestras: mediana de los cortos, parte y valor de los
  largos (> 1.5 ms), tasa media;
- la interferencia de la baliza Wi-Fi como plantilla en fase con 102.4 ms
  (pliegue sobre el reloj del equipo, sin rejilla) y su parte de la varianza;
- lineas a n x 9.765625 Hz en el espectro (20-450 Hz, rejilla de 1 kHz).

Las capturas de banco pueden no tener electrodos: miden la placa, no el
musculo. Escribe data/bench/radio_sampling.csv y radio_sampling.npz.
Lo corre s26_run_all.py si las capturas existen; si no, avisa y termina.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import LSB_MV, OUT

FW = Path(r"D:\PhD\Code\SenseAI_emg8\benchmarks")
# (carpeta de captura, placa) — la placa sale del perfil Wi-Fi usado ese dia
CAPTURES = [
    ("sync-baseline/sd-off-35", "8790"), ("sync-baseline/sd-udp-35", "8790"),
    ("sync-task/storage-off-max-35", "8790"), ("sync-task/storage-udp-max-35", "8790"),
    ("participant-review-2026-09-28/sd-A-30s", "8770"),
    ("participant-review-2026-09-28/udp-B-60s-retry", "8770"),
    ("robustness-2026-09-28/sd-max-30s", "8770"),
    ("robustness-2026-09-28/sd-udp-60s-repeat", "8770"),
    ("robustness-2026-09-28/sd-udp-60s-tether", "8770"),
]
RAW = ((0, 3), (1, 3), (0, 3), (1, 3))
REC = np.dtype([("ts_us", "<u4"), ("adc", "u1"), ("ch", "u1"), ("value", "<i2")])
PERIOD_US = 102400.0
BEACON_HZ = 1e6 / PERIOD_US
NB = 128
OUTD = OUT / "data" / "bench"


def unwrap(ts):
    t = ts.astype(np.int64)
    t += np.concatenate([[0], np.cumsum(np.diff(t) < -2 ** 31)]) * 2 ** 32
    return t


def fold(t_us, y, period=PERIOD_US, nb=NB):
    k = np.minimum(((t_us % period) / period * nb).astype(int), nb - 1)
    n = np.bincount(k, None, nb)
    return np.bincount(k, y, nb) / np.maximum(n, 1), n


def lines_fraction(t_us, y):
    """Parte de la potencia de 20-450 Hz en las lineas n x 9.766 Hz (rejilla de 1 kHz, Welch 4096)."""
    tg = np.arange(t_us[0], t_us[-1], 1000.0)
    yg = np.interp(tg, t_us, y)
    n = 4096
    segs = [yg[k:k + n] for k in range(0, len(yg) - n + 1, n // 2)]
    if not segs:
        return np.nan, None, None
    w = np.hanning(n)
    P = np.mean([np.abs(np.fft.rfft((s - s.mean()) * w)) ** 2 for s in segs], 0)
    f = np.fft.rfftfreq(n, 1e-3)
    band = (f >= 20) & (f <= 450)
    near = np.abs(f / BEACON_HZ - np.round(f / BEACON_HZ)) * BEACON_HZ <= 0.37
    excess = 0.0
    for m in range(3, 47):
        fc = m * BEACON_HZ
        k = np.abs(f - fc) <= 0.37
        loc = (np.abs(f - fc) <= 3) & ~near
        excess += (P[k] - np.median(P[loc])).clip(0).sum()
    return excess / P[band].sum(), f, P


def main():
    if not FW.exists():
        print("sin capturas de banco en", FW)
        return
    OUTD.mkdir(parents=True, exist_ok=True)
    rows, arrays = [], {}
    for rel, board in CAPTURES:
        d = FW / rel
        meta = json.loads((d / "capture" / "metadata.json").read_text())
        rfiles = sorted((d / "sd").glob("R*.bin"))
        if not rfiles:
            continue
        a = np.fromfile(rfiles[0], dtype=REC)
        tag = rel.replace("/", "__")
        for adc in range(4):
            for j, ch in enumerate(RAW[adc]):
                m = (a["adc"] == adc) & (a["ch"] == ch)
                t = unwrap(a["ts_us"][m]).astype(float)
                y = a["value"][m].astype(float) * LSB_MV
                if len(t) < 5000:
                    continue
                dt = np.diff(t)
                ok = dt < 5000
                longm = ok & (dt > 1500)
                yd = y - pd.Series(y).rolling(51, center=True, min_periods=1).median().values
                tpl, cnt = fold(t, yd)
                tpl = tpl - tpl.mean()
                # varianza de la plantilla corregida por el ruido que queda en cada casilla
                noise_var = np.var(yd) / max(cnt.mean(), 1)
                tvar = max(np.mean(tpl ** 2) - noise_var, 0.0)
                frac_l, f, P = lines_fraction(t, yd)
                e = adc * 2 + j + 1
                rows.append(dict(capture=rel, board=board, condition=meta["condition"], mode=meta["mode"],
                                 rate=meta.get("rate") or "default", seconds=meta["seconds"], channel=f"E{e}",
                                 n=len(t), rate_hz=(len(t) - 1) / ((t[-1] - t[0]) / 1e6),
                                 short_med_us=float(np.median(dt[ok & ~longm])),
                                 long_frac=float(longm.sum() / ok.sum()),
                                 long_med_us=float(np.median(dt[longm])) if longm.any() else np.nan,
                                 dt_max_us=float(dt[ok].max()), sd_mv=float(np.std(yd)),
                                 template_rms_mv=float(np.sqrt(tvar)), template_var_frac=tvar / np.var(yd),
                                 template_peak_mv=float(tpl[np.argmax(np.abs(tpl))]),
                                 template_phase_ms=float(np.argmax(np.abs(tpl)) / NB * PERIOD_US / 1000),
                                 lines_frac=float(frac_l)))
                arrays[f"{tag}__E{e}__tpl"] = tpl
                if e == 1 and P is not None:
                    arrays[f"{tag}__f"], arrays[f"{tag}__P"] = f, P
                    arrays[f"{tag}__dt"] = dt[ok]
    R = pd.DataFrame(rows)
    R.to_csv(OUTD / "radio_sampling.csv", index=False)
    np.savez(OUTD / "radio_sampling.npz", **arrays)
    pd.set_option("display.width", 220)
    g = R.groupby(["capture", "board", "condition", "rate"], sort=False)
    print(g.agg(ch=("channel", "size"), rate_hz=("rate_hz", "median"), short_us=("short_med_us", "median"),
                long_pct=("long_frac", lambda v: 100 * v.median()), long_us=("long_med_us", "median"),
                sd_mv=("sd_mv", "median"), tpl_mv=("template_rms_mv", "median"),
                tpl_pct=("template_var_frac", lambda v: 100 * v.median()),
                lines_pct=("lines_frac", lambda v: 100 * v.median())).round(2).to_string())


if __name__ == "__main__":
    main()
