"""Sabado 26: modalidades auxiliares por barrido y por ensayo.

- Impedancia: cada barrido con el Prest que lo disparo (el que llega ~5 s
  antes que el barrido). Contexto: 'post' (al soltar un agarre) o 'break'
  (el segundo barrido de cada pausa, ya en reposo). |Z| a 2, 5, 10, 20, 50 y
  100 kHz; los espectros completos van a impedance.npz.
- Presion y temperatura: la placa auxiliar solo lee mientras se agarra, cada
  2 s desde el Pgrasp (tres por ensayo; llegan al PC en lotes de 1 s). Las
  presiones salen de un filtro de mediana de 3 lecturas que pasa de un
  agarre al siguiente: la 1a (y a veces la 2a) repite el agarre anterior, asi
  que solo la 3a es del agarre en curso (no hay linea base dentro del ensayo).
  -126.79 C es la entrada del ADC a fondo de escala (1.75 V): sensor ausente
  (las tres primeras sesiones).
- IMU: la gravedad media durante el sostener da la postura del antebrazo
  (inclinacion y giro); el giroscopio, sin su sesgo, da el movimiento al
  agarrar y al soltar.

Escribe data/s26/{impedance.csv, impedance.npz, sensors_trials.csv, imu_trials.csv}.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s26_common import DATA26, GRASP_NAMES, find_sessions, imu_arrays, load26, prest_times

N_TRIALS = 42
TEMP_SENTINEL = -126.0      # por debajo: sensor ausente
FREQS = (2e3, 5e3, 10e3, 20e3, 50e3, 100e3)

rows_z, rows_p, rows_i = [], [], []
spectra = {}
for path in find_sessions():
    x = load26(path)
    if len(x.trials) != N_TRIALS:
        continue
    pr = prest_times(x)
    t_session_min = lambda t: t / 60.0
    # --- impedancia
    Zs = []
    for k, w in enumerate(x.sweeps):
        cands = [p for p in pr if 3.5 < w["t_arrival"] - p < 7.5]
        p = max(cands) if cands else np.nan
        post = any(abs(p - t.t_rest) < 0.3 for t in x.trials) if np.isfinite(p) else False
        ctx = "post" if post else ("break" if np.isfinite(p) and p > 1.0 else "start")
        prev = [t for t in x.trials if np.isfinite(p) and t.t_rest <= p + 0.3]
        g = prev[-1].grasp if prev else 0
        z = w["z"]
        Zs.append(z)
        rows_z.append(dict(session=x.name, subject=x.subject, k=k, t_prest=round(p, 3) if np.isfinite(p) else np.nan,
                           t_arrival=round(w["t_arrival"], 3), context=ctx, prev_grasp=g,
                           **{f"z{int(f/1e3)}k": float(np.interp(f, w["freq"], z)) for f in FREQS}))
    spectra[x.subject] = np.array(Zs)
    spectra["freq"] = x.sweeps[0]["freq"]
    spectra[x.subject + "_t"] = np.array([r["t_prest"] for r in rows_z if r["session"] == x.name])
    # --- presion y temperatura por ensayo
    sen = x.sensors
    for tr in x.trials:
        k = np.where((sen["t"] >= tr.t_grasp - 0.3) & (sen["t"] < tr.t_rest + 0.6))[0]
        row = dict(session=x.name, subject=x.subject, trial=tr.index, grasp=tr.grasp,
                   grasp_name=GRASP_NAMES.get(tr.grasp, str(tr.grasp)), rep=tr.rep, block=tr.index // 6,
                   t_grasp=round(tr.t_grasp, 3), n=len(k))
        # tres lecturas por sostener: el sabado eran todas (0, 2 y 4 s); con la placa auxiliar que lee
        # ~3 por segundo (sesion del 1-10) se toman la primera, la del medio y la ultima
        pick = k[np.round(np.linspace(0, len(k) - 1, 3)).astype(int)] if len(k) > 3 else k[:3]
        for j, kk in enumerate(pick):
            tc = sen["temp"][kk]
            row.update({f"dt{j}": round(sen["t"][kk] - tr.t_grasp, 3), f"p1_{j}": sen["p1"][kk],
                        f"p2_{j}": sen["p2"][kk], f"temp_{j}": tc if tc > TEMP_SENTINEL else np.nan})
        rows_p.append(row)
    # --- IMU
    t, acc, gyr = imu_arrays(x)
    bias = np.median(gyr, 0)
    g = gyr - bias
    gm = np.linalg.norm(g, axis=1)

    def seg(a, b):
        return (t >= a) & (t < b)

    for tr in x.trials:
        h = seg(tr.t_grasp + 1.0, tr.t_rest - 0.5)
        a = acc[h].mean(0) if h.any() else np.full(3, np.nan)
        n = np.linalg.norm(a)
        rows_i.append(dict(session=x.name, subject=x.subject, trial=tr.index, grasp=tr.grasp,
                           grasp_name=GRASP_NAMES.get(tr.grasp, str(tr.grasp)), rep=tr.rep, block=tr.index // 6,
                           ax=a[0], ay=a[1], az=a[2],
                           incl_deg=float(np.degrees(np.arccos(np.clip(a[2] / n, -1, 1)))),
                           roll_deg=float(np.degrees(np.arctan2(a[1], a[2]))),
                           pitch_deg=float(np.degrees(np.arctan2(-a[0], np.hypot(a[1], a[2])))),
                           gyr_onset_rms=float(np.sqrt(np.mean(gm[seg(tr.t_grasp - 0.2, tr.t_grasp + 1.2)] ** 2))),
                           gyr_hold_rms=float(np.sqrt(np.mean(gm[h] ** 2))) if h.any() else np.nan,
                           gyr_release_rms=float(np.sqrt(np.mean(gm[seg(tr.t_rest - 0.2, tr.t_rest + 1.2)] ** 2))),
                           gyr_rest_rms=float(np.sqrt(np.mean(gm[seg(tr.t_rest + 2.5, tr.t_rest + 4.5)] ** 2))),
                           acc_dyn_hold_mg=float(1e3 * np.std(np.linalg.norm(acc[h], axis=1))) if h.any() else np.nan,
                           imu_temp=float(np.median(x.s.imu["temp100"][np.searchsorted(t, tr.t_grasp):
                                                                     np.searchsorted(t, tr.t_rest)] / 100.0))))
    P = pd.DataFrame([r for r in rows_p if r["session"] == x.name])
    I = pd.DataFrame([r for r in rows_i if r["session"] == x.name])
    Z = pd.DataFrame([r for r in rows_z if r["session"] == x.name])
    print(f"{x.subject}: barridos {len(Z)} ({dict(Z.context.value_counts())})  |Z|10k {Z.z10k.iloc[1]/1e3:.1f} -> {Z.z10k.iloc[-1]/1e3:.1f} kOhm")
    print(f"    presion: muestras por ensayo {dict(P.n.value_counts())}; dt {P.dt0.median():.2f}/{P.dt1.median():.2f}/{P.dt2.median():.2f} s; "
          f"p1 {P.p1_0.median():.1f} -> {P.p1_1.median():.1f} -> {P.p1_2.median():.1f} kPa; "
          f"p2 {P.p2_0.median():.1f} -> {P.p2_1.median():.1f} -> {P.p2_2.median():.1f} kPa; "
          f"temp valida {int(P.temp_0.notna().sum())}/{len(P)}")
    print(f"    IMU: inclinacion {I.incl_deg.median():.1f} deg (rango {I.incl_deg.min():.1f}-{I.incl_deg.max():.1f}); "
          f"giro dps al agarrar {I.gyr_onset_rms.median():.2f}, sostener {I.gyr_hold_rms.median():.2f}, "
          f"soltar {I.gyr_release_rms.median():.2f}, reposo {I.gyr_rest_rms.median():.2f}; "
          f"temp IMU {I.imu_temp.iloc[0]:.1f} -> {I.imu_temp.iloc[-1]:.1f} C")

pd.DataFrame(rows_z).to_csv(DATA26 / "impedance.csv", index=False)
np.savez(DATA26 / "impedance.npz", **spectra)
pd.DataFrame(rows_p).to_csv(DATA26 / "sensors_trials.csv", index=False)
pd.DataFrame(rows_i).to_csv(DATA26 / "imu_trials.csv", index=False)
