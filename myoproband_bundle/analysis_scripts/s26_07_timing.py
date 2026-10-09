"""Sabado 26: anatomia del muestreo, latencias y presupuesto de datos.

Lo que dice el firmware (SenseAI_emg8 @ c45a7bc, main.cpp): 4 ADS1015 en dos
buses I2C, disparo simple en ronda a 3300 SPS; los crudos se convierten en
cada ronda y las envolventes cada kSLOW_DIV = 20 rondas; el limitador R1000
sostiene 1000 Hz de media. Aqui se mide lo que eso produce:

- rondas: intervalo corto/largo de E1, periodo del largo (en muestras) y si
  las muestras de envolvente caen dentro de los intervalos largos;
- desfase de cada canal (crudo y envolvente) respecto a la muestra de E1 de su
  ronda: el horario de conversion dentro de la trama;
- error de tiempo si se tratara el crudo como 1 kHz uniforme (diente de sierra)
  y lo que eso hace a un tono conocido: se genera sin(2 pi f t) en las marcas
  reales, se analiza como uniforme y se mide el mayor espurio (dBc); con
  remuestreo por marcas el mismo tono sale limpio;
- deriva del reloj del equipo frente al PC (ppm, pendiente del ajuste
  wall -> ts_us) y variacion del retardo de entrega (residuo del ajuste,
  medido contra el mas rapido);
- latencia comando -> efecto: Prest registrado en el PC -> arranque del
  barrido visto en el sEMG (incluye el retardo de entrega equipo -> PC);
- presupuesto: bytes por minuto por flujo (archivos del WAL), equivalente en
  SD (mismos registros) y tasa UDP con cabeceras (174 registros por paquete,
  vaciado cada 30 ms; net_stream.cpp).

Escribe data/s26/timing_deep.csv, timing_offsets.csv, timing_spurs.csv,
budget.csv y timing_example.npz (S01) para la figura 8.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import IMU_DTYPE, SAMPLE_DTYPE, host_to_device
from s26_common import DATA26, SUBJECT_ORDER, find_sessions, load26, prest_times, sweep_onsets

LONG_US = 1800           # un intervalo crudo por encima de esto es la ronda con envolventes
TONES = (50.0, 100.0, 150.0, 250.0)
UDP_RAW_RECS, E8_HDR, IPUDP_HDR = 174, 12, 28

def lagrange4(t, y, tg):
    """Interpolacion cubica local (Lagrange de 4 puntos) sobre marcas no
    uniformes. Sin scipy: el instalado esta roto."""
    j = np.clip(np.searchsorted(t, tg) - 2, 0, len(t) - 4)
    T = np.stack([t[j + k] for k in range(4)], 1)
    Yv = np.stack([y[j + k] for k in range(4)], 1)
    out = np.zeros(len(tg))
    for a in range(4):
        w = np.ones(len(tg))
        for b in range(4):
            if a != b:
                w *= (tg - T[:, b]) / (T[:, a] - T[:, b])
        out += w * Yv[:, a]
    return out


def welch_db(y, fs, n=8192):
    win_ = np.hanning(n)
    segs = [y[k:k + n] for k in range(0, len(y) - n + 1, n // 2)]
    P = np.mean([np.abs(np.fft.rfft((s_ - s_.mean()) * win_)) ** 2 for s_ in segs], 0)
    return np.fft.rfftfreq(n, 1 / fs), P


S = pd.read_csv(DATA26 / "sessions.csv")
full = S[S.complete].set_index("subject")
paths = {p.name: p for p in find_sessions()}
rows, offs_rows, spur_rows, bud_rows = [], [], [], []
example = {}
link = {}                 # por sesion: retardos de entrega, latencias y tasas en ventanas de 10 s

for subj in SUBJECT_ORDER:
    name = full.loc[subj, "session"]
    path = paths[name]
    x = load26(path)
    s = x.s
    pv = full.loc[subj, "preview_s"]
    t_start = s.t0_us + int((pv + 0.5) * 1e6)
    e1 = s.raw[0].ts_us.astype(np.int64)
    e1 = e1[e1 >= t_start]
    d = np.diff(e1)
    ok = d < 5000
    long_idx = np.where((d > LONG_US) & ok)[0]
    period = np.diff(long_idx)
    short = d[ok & (d <= LONG_US)]
    longv = d[long_idx]
    # envolventes dentro de un intervalo largo de E1
    env_in, env_n = 0, 0
    lo_edges, hi_edges = e1[long_idx], e1[long_idx + 1]
    for i in range(8):
        te = s.env[i].ts_us.astype(np.int64)
        te = te[(te >= e1[0]) & (te <= e1[-1])]
        j = np.searchsorted(lo_edges, te, side="right") - 1
        inside = (j >= 0) & (te < hi_edges[np.clip(j, 0, len(hi_edges) - 1)])
        env_in += int(inside.sum())
        env_n += len(te)
    # desfase de cada canal respecto a la muestra de E1 previa (crudo y envolvente)
    for kind, chans in (("raw", s.raw), ("env", s.env)):
        for i in range(8):
            tc = chans[i].ts_us.astype(np.int64)
            tc = tc[(tc >= e1[0]) & (tc <= e1[-1])]
            j = np.searchsorted(e1, tc, side="right") - 1
            off = (tc - e1[j]) / 1000.0
            # cuanto dura la ronda en la que cae (corta o larga)
            in_long = np.isin(j, long_idx)
            offs_rows.append(dict(subject=subj, kind=kind, channel=f"E{i+1}", adc=chans[i].adc, pin=chans[i].pin,
                                  off_med_ms=float(np.median(off)), off_p10_ms=float(np.percentile(off, 10)),
                                  off_p90_ms=float(np.percentile(off, 90)),
                                  off_med_short_ms=float(np.median(off[~in_long])) if (~in_long).any() else np.nan,
                                  off_med_long_ms=float(np.median(off[in_long])) if in_long.any() else np.nan,
                                  frac_in_long=float(in_long.mean())))
    # error frente a una rejilla uniforme y espurios de un tono
    seg = e1[:120000]                      # 2 minutos
    tt = (seg - seg[0]) / 1e6
    n = np.arange(len(seg))
    mean_dt = (tt[-1] - tt[0]) / (len(tt) - 1)
    err = tt - n * mean_dt
    err_ms = (err - np.median(err)) * 1e3
    N = 1 << 16
    win_ = np.hanning(N)
    for f0 in TONES:
        y = np.sin(2 * np.pi * f0 * tt[:N])
        Y = np.abs(np.fft.rfft(y * win_)) ** 2
        fr = np.fft.rfftfreq(N, mean_dt)
        car = int(np.argmin(np.abs(fr - f0)))
        pc = Y[max(0, car - 4):car + 5].sum()
        mask = np.ones_like(Y, bool)
        mask[max(0, car - 8):car + 9] = False
        mask[:3] = False
        # potencia de espurios por pico (suma de +/-4 bins alrededor del maximo)
        ksp = int(np.argmax(np.where(mask, Y, 0)))
        psp = Y[max(0, ksp - 4):ksp + 5].sum()
        tot_sp = Y[mask].sum()
        # con remuestreo por marcas: lineal y cubico (Lagrange de 4 puntos)
        tg = np.arange(N) * mean_dt
        res_db = {}
        for how, yi in (("lin", np.interp(tg, tt[:N], y)), ("cub", lagrange4(tt[:N], y, tg))):
            Yi = np.abs(np.fft.rfft(yi * win_)) ** 2
            pci = Yi[max(0, car - 4):car + 5].sum()
            ksi = int(np.argmax(np.where(mask, Yi, 0)))
            res_db[how] = (10 * np.log10(Yi[max(0, ksi - 4):ksi + 5].sum() / pci), float(fr[ksi]))
        spur_rows.append(dict(subject=subj, tone_hz=f0, spur_hz=float(fr[ksp]), spur_dbc=10 * np.log10(psp / pc),
                              all_spurs_dbc=10 * np.log10(tot_sp / pc), resampled_spur_dbc=res_db["lin"][0],
                              resampled_spur_hz=res_db["lin"][1], cubic_spur_dbc=res_db["cub"][0]))
        if subj == "S01" and f0 == 100.0:
            # rejilla dentro del tramo con marcas: nada de extrapolar
            M = len(tt)
            y2 = np.sin(2 * np.pi * f0 * tt)
            tg2 = np.arange(int(tt[-1] / mean_dt) - 2) * mean_dt
            fw, Pn = welch_db(y2, 1 / mean_dt)
            _, Pl = welch_db(np.interp(tg2, tt, y2), 1 / mean_dt)
            _, Pc = welch_db(lagrange4(tt, y2, tg2), 1 / mean_dt)
            kc = int(np.argmin(np.abs(fw - f0)))
            ref = lambda P_: P_[max(0, kc - 4):kc + 5].sum()
            example.update(tone_f=fw, tone_naive=10 * np.log10(Pn / ref(Pn) + 1e-20),
                           tone_resampled=10 * np.log10(Pl / ref(Pl) + 1e-20),
                           tone_cubic=10 * np.log10(Pc / ref(Pc) + 1e-20))
    # reloj y entrega
    f, resid_ms = host_to_device(s)
    ppm = f.slope - 1e6
    pts = [(e["wall"], e["last_ts_us"]) for e in s.events if e.get("last_ts_us") is not None]
    w = np.array([p_[0] for p_ in pts]); dv = np.array([p_[1] for p_ in pts], float)
    res = (dv - f(w)) / 1000.0
    keep = np.abs(res - np.median(res)) < 200
    res = res[keep]
    delay_var = np.max(res) - res          # 0 = la entrega mas rapida vista
    # latencia comando -> efecto
    pr = prest_times(x)
    ons = sweep_onsets(x)
    lat = []
    for p_ in pr:
        o = [o_ for o_ in ons if 0.03 <= o_ - p_ <= 0.40]
        if o:
            lat.append((o[0] - p_) * 1000)
    lat = np.array(lat)
    link[f"delay_{subj}"] = delay_var
    link[f"lat_{subj}"] = lat
    # tasas en ventanas de 10 s (crudo E1, envolvente E1, IMU), desde la tasa completa
    edges = np.arange(t_start, s.t0_us + int(s.duration_s * 1e6), 10_000_000)
    for key, ts_ in (("raw", s.raw[0].ts_us), ("env", s.env[0].ts_us), ("imu", np.sort(s.imu["ts_us"].astype(np.int64)))):
        h, _ = np.histogram(np.asarray(ts_, np.int64), bins=edges)
        link[f"rate_{key}_{subj}"] = h / 10.0
    link[f"rate_t_{subj}"] = (edges[:-1] - s.t0_us) / 60e6
    # presupuesto
    dur_min = s.duration_s / 60.0
    sizes = {k: (path / f).stat().st_size for k, f in (("raw_env", "raw.bin"), ("imu", "imu.bin"),
                                                       ("aux", "aux.jsonl"), ("events", "events.jsonl"),
                                                       ("meta", "metadata.json"))}
    n_raw = sum(len(c.ts_us) for c in s.raw.values())
    n_env = sum(len(c.ts_us) for c in s.env.values())
    bud_rows.append(dict(subject=subj, minutes=dur_min,
                         raw_mb_min=n_raw * SAMPLE_DTYPE.itemsize / 1e6 / dur_min,
                         env_mb_min=n_env * SAMPLE_DTYPE.itemsize / 1e6 / dur_min,
                         imu_mb_min=sizes["imu"] / 1e6 / dur_min,
                         aux_mb_min=sizes["aux"] / 1e6 / dur_min,
                         events_mb_min=(sizes["events"] + sizes["meta"]) / 1e6 / dur_min,
                         total_mb_min=sum(sizes.values()) / 1e6 / dur_min))
    rows.append(dict(subject=subj,
                     short_us_med=float(np.median(short)), long_us_med=float(np.median(longv)),
                     long_frac=float(len(long_idx) / ok.sum()),
                     long_period_mode=int(np.bincount(period).argmax()) if len(period) else 0,
                     long_period_frac=float(np.mean(period == np.bincount(period).argmax())) if len(period) else np.nan,
                     env_in_long_frac=env_in / max(env_n, 1),
                     err_p2p_ms=float(np.percentile(err_ms, 99.5) - np.percentile(err_ms, 0.5)),
                     clock_ppm=float(ppm), resid_ms=float(resid_ms),
                     delay_p50_ms=float(np.percentile(delay_var, 50)), delay_p95_ms=float(np.percentile(delay_var, 95)),
                     delay_max_ms=float(np.max(delay_var)), n_events=int(keep.sum()),
                     cmd_lat_med_ms=float(np.median(lat)) if len(lat) else np.nan,
                     cmd_lat_p10_ms=float(np.percentile(lat, 10)) if len(lat) else np.nan,
                     cmd_lat_p90_ms=float(np.percentile(lat, 90)) if len(lat) else np.nan, n_cmd=len(lat)))
    if subj == "S01":
        k0 = long_idx[10] - 45
        example.update(e1_ts=e1[k0:k0 + 130], err_t=tt[:4000], err_ms=err_ms[:4000],
                       delay_var=delay_var, cmd_lat=lat)
        for kind, chans in (("raw", s.raw), ("env", s.env)):
            for i in range(8):
                tc = chans[i].ts_us.astype(np.int64)
                m = (tc >= e1[k0]) & (tc <= e1[k0 + 129])
                example[f"{kind}{i}"] = tc[m]
    r = rows[-1]
    print(f"{subj}: corto {r['short_us_med']:.0f} us, largo {r['long_us_med']:.0f} us ({100*r['long_frac']:.2f} %, "
          f"cada {r['long_period_mode']} muestras en el {100*r['long_period_frac']:.1f} % de los casos); "
          f"envolventes dentro del largo {100*r['env_in_long_frac']:.1f} %; error vs rejilla p2p {r['err_p2p_ms']:.2f} ms")
    print(f"    reloj {r['clock_ppm']:+.1f} ppm; entrega p50/p95/max {r['delay_p50_ms']:.1f}/{r['delay_p95_ms']:.1f}/"
          f"{r['delay_max_ms']:.1f} ms; Prest->barrido {r['cmd_lat_med_ms']:.0f} ms (p10-p90 {r['cmd_lat_p10_ms']:.0f}-"
          f"{r['cmd_lat_p90_ms']:.0f}, n={r['n_cmd']})")

D = pd.DataFrame(rows)
D.to_csv(DATA26 / "timing_deep.csv", index=False)
O = pd.DataFrame(offs_rows)
O.to_csv(DATA26 / "timing_offsets.csv", index=False)
SP = pd.DataFrame(spur_rows)
SP.to_csv(DATA26 / "timing_spurs.csv", index=False)
BU = pd.DataFrame(bud_rows)
BU.to_csv(DATA26 / "budget.csv", index=False)
np.savez(DATA26 / "timing_example.npz", **example)
np.savez(DATA26 / "link_arrays.npz", **link)

print("\ndesfase por canal (mediana entre sesiones, ms; ronda corta / ronda larga):")
print(O.groupby(["kind", "channel"])[["off_med_short_ms", "off_med_long_ms", "frac_in_long"]].median().round(3).to_string())
print("\nespurios de un tono tratado como 1 kHz uniforme (mediana entre sesiones):")
print(SP.groupby("tone_hz")[["spur_hz", "spur_dbc", "all_spurs_dbc", "resampled_spur_dbc", "cubic_spur_dbc"]]
      .median().round(1).to_string())
b = BU.mean(numeric_only=True)
pkt_raw = 8000 / UDP_RAW_RECS
udp_bps = (8000 * 8 + pkt_raw * (E8_HDR + IPUDP_HDR)) + (400 * 8 + 33.3 * (E8_HDR + IPUDP_HDR)) + \
          (full.imu_hz.mean() * IMU_DTYPE.itemsize + 33.3 * (E8_HDR + IPUDP_HDR))
print(f"\npresupuesto: crudo {b.raw_mb_min:.2f} + env {b.env_mb_min:.3f} + IMU {b.imu_mb_min:.3f} + aux {b.aux_mb_min:.4f} + "
      f"eventos {b.events_mb_min:.4f} = {b.total_mb_min:.2f} MB/min ({b.total_mb_min*60/1000:.2f} GB/h); "
      f"UDP ~{udp_bps*8/1e6:.2f} Mbit/s con cabeceras ({pkt_raw:.0f} paquetes crudos/s)")
