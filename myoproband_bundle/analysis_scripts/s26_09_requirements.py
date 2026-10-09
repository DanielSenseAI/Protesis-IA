"""Tabla de requisitos de desempeno del articulo (sabado 26), con los
objetivos que fijo el usuario el 29-09 (interchannel <= 1 ms, alineacion
multimodal <= 100 ms, perdida <= 0.1 %, ...).

Cada cifra sale de las tablas de data/s26; lo que estas grabaciones no miden
(ancho de banda analogico, bateria, 120 min seguidos, medidas mecanicas) no se
rellena: queda como "not tested" o "to measure". La segunda tabla son
requisitos candidatos (objetivo propuesto, no fijado por el usuario) con lo
que ya se midio.

Escribe requirements_s26.md (para pegar en el borrador) y
data/s26/requirements.csv (el informe lee su tabla de ahi).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s26_common import DATA26, DATASET, OUT, bench_long_runs

# La tabla lleva ≥ y superindices; fuera del runner la consola es cp1252.
sys.stdout.reconfigure(encoding="utf-8")

S = pd.read_csv(DATA26 / "sessions.csv")
full = S[S.complete].set_index("subject")
N = len(full)
TD = pd.read_csv(DATA26 / "timing_deep.csv").set_index("subject")
P = pd.read_csv(DATA26 / "sensors_trials.csv")
P = P[P.session.isin(full.session)]
Z = pd.read_csv(DATA26 / "impedance.csv")
Z = Z[Z.session.isin(full.session)]
W = pd.read_csv(DATA26 / "sweeps.csv")
W = W[W.session.isin(full.session)]
OV = pd.read_csv(DATA26 / "hold_overlap.csv")
A = pd.read_csv(DATA26 / "sensors.csv")
A = A[A.session.isin(full.session)]
C = pd.read_csv(DATA26 / "channels.csv")
C = C[C.session.isin(full.session)]
SE = pd.read_csv(DATA26 / "semg_channels.csv")
LK = np.load(DATA26 / "link_arrays.npz")


def rng(v, fmt=".1f", unit=""):
    """'a–b unit' (o 'a unit' si coinciden al redondear)."""
    a, b = format(min(v), fmt), format(max(v), fmt)
    return (a if a == b else f"{a}–{b}") + unit


# ------------------------------------------------------------------ cifras
n_raw = C[C.kind == "raw"].groupby("subject").size()
rate_full = 1000.0 / pd.read_csv(DATA26 / "timing.csv").raw_dt_mean_ms        # tasa a velocidad plena
drop = full[full.raw_missing > 0]
loss_pool = float(np.average(full.raw_loss_pct, weights=full.full_rate_s))
spread_med = float(np.median(full.frame_spread_med_us)) / 1000
lab_sd = full.host_jitter_ms                                                    # DE del ajuste anfitrion->equipo
aux_q = P.dt0.quantile([0.02, 0.98])                                            # llegada de la 1a lectura tras la senal
wrap_min = 2 ** 32 / 60e6
# grabaciones largas de banco: la ultima que corrio (no cortada a mano) es la que cuenta
long_runs = bench_long_runs()
bench = dict(l.split(None, 1) for l in (OUT / "data" / "sweep_timing.txt").read_text().splitlines() if l.strip())
sweep_s = [float(v) for v in bench["duration_s"].split()]
ctx = Z.context.value_counts()
lat = np.concatenate([LK[k] for k in LK.files if k.startswith("lat_")])
dly = np.concatenate([LK[k] for k in LK.files if k.startswith("delay_")])
ov = OV[OV.overlap_s > 0]
first_only = bool((ov.trial == 0).all())
dense = [s_ for s_ in full.index if P[P.subject == s_].n.median() > 3]   # lecturas densas (S07)
short_holds = P[P.n < 3].merge(ov[["subject", "trial"]], on=["subject", "trial"])
ok_load = (A.p1_kpa > 0.05) & (A.p2_kpa > 0.05)
load_bad = A.assign(ok=ok_load).groupby("subject").ok.apply(lambda s: int((~s).sum()))
temp_ok = A.assign(t=A.temp_c > -126).groupby("subject").t.any()
long_frac = TD.long_frac.reindex(full.index)
noise_mv = float(np.median(SE.base_rms_mv))
LSB_MV = 8192 / 4096                                                            # ADS1015 a +/-4.096 V, 12 bits

# ------------------------------------------------------------------ tabla del articulo
# (requisito, objetivo, medido, estado, nota); el objetivo es el que dio el usuario
rows = [
    ("sEMG sites", "8", f"{int(n_raw.min())}", "Met",
     f"All {N} sessions recorded 8 raw and 8 envelope channels."),
    ("Raw sampling", "≥900 Hz/ch", f"{rate_full.mean():.1f} Hz/ch", "Met",
     f"Mean sample interval {1000 / rate_full.mean():.3f} ms in all {N} sessions. Effective rate including dropouts: "
     f"{full.raw_hz.min():.1f}–{full.raw_hz.max():.1f} Hz/ch."),
    ("Analog bandwidth", "10–450 Hz nominal", "not measured", "Not tested",
     "These recordings carry no known test input; needs a swept-sine bench test."),
    ("Interchannel spread", "≤1 ms", f"{spread_med:.2f} ms (median)", "Partly met",
     f"Spread across the 8 channels of one frame: median {rng(full.frame_spread_med_us / 1000, '.2f')} ms by session, "
     f"p95 {rng(full.frame_spread_p95_us / 1000, '.2f')} ms; {rng(full.frame_spread_le1ms_pct, '.0f')} % of frames "
     "within 1 ms. The channels are not converted simultaneously; every sample carries its own timestamp, so the "
     "skew is known and correctable."),
    ("Multimodal alignment", "≤100 ms", f"labels: SD {rng(lab_sd, '.0f')} ms; auxiliary: ≤{aux_q.iloc[1]:.1f} s",
     "Partly met",
     f"sEMG (raw and envelope) and IMU share one device clock with per-sample µs timestamps. Task labels "
     f"(host-logged) map onto that clock with a residual SD of {rng(lab_sd, '.1f')} ms. Contact load, temperature "
     f"and impedance carry no device timestamp: the bracelet relays them on a 1 s heartbeat and the host stamps "
     f"them on arrival, {aux_q.iloc[0]:.2f}–{aux_q.iloc[1]:.2f} s after the task cue (2nd–98th percentile). Fix: "
     "firmware-upgrades.md B2."),
    ("Acquisition loss", "≤0.1%", f"{loss_pool:.2f}% pooled", "Met (pooled)",
     f"0 % in {N - len(drop)} of {N} sessions; "
     + "; ".join(f"{s_} {r.raw_loss_pct:.2f} % ({int(r.raw_gaps)} gaps, {r.raw_missing / 8 / 1000:.1f} s)"
                 for s_, r in drop.iterrows())
     + ". Pooled = weighted by full-rate time."),
    ("Continuous recording", "≥120 min", f"{rng(full.span_s / 60, '.1f')} min per session", "Not tested",
     f"{full.full_rate_s.sum() / 60:.1f} min at full rate over {N} sessions, none longer than "
     f"{full.span_s.max() / 60:.1f} min. The 32-bit µs timestamps wrap every {wrap_min:.1f} min of device time "
     "(firmware-upgrades.md B4).") if not long_runs else
    ("Continuous recording", "≥120 min", f"{long_runs[-1]['minutes']:.1f} min (bench)",
     "Met" if long_runs[-1]["minutes"] >= 120 else "Not met",
     f"Bench board, all channels at 1000 Hz to the SD card, {long_runs[-1]['day']}: "
     + (f"the firmware stopped the recording ({long_runs[-1]['cause']}) after {long_runs[-1]['minutes']:.1f} min, "
        f"when one SD sync took {long_runs[-1]['sync_s']:.2f} s against a 1.5 s raw buffer "
        "(firmware-upgrades.md B11). " if long_runs[-1]["minutes"] < 120 else "ran to the planned stop. ")
     + f"Participant sessions: {N}, none longer than {full.span_s.max() / 60:.1f} min. The 32-bit µs "
     f"timestamps wrap every {wrap_min:.1f} min of device time (firmware-upgrades.md B4)."),
    # el reloj del equipo no siempre arranca en cero al grabar (S02 y S03 empiezan en 356 s y 62 s), asi que
    # tampoco sirve para deducir cuanto estuvo encendido
    ("Battery autonomy", "≥2 h", "not logged", "Not tested",
     "Battery state is not in the recordings."),
    ("Circumference", "18–40 cm", "18–40 cm", "To confirm",
     "Mechanical: not in the recordings. Keep only if measured on the band."),
    ("Longitudinal footprint", "≤12 cm", "to measure", "To measure",
     "Mechanical. State whether it includes the distal auxiliary armband."),
]

# ------------------------------------------------------------------ candidatos (objetivo propuesto)
cand = [
    ("Impedance sweep", "2–100 kHz, 99 points, ≤5 s, every rest", "Point count and duration per sweep",
     f"{int((W.finite == 99).sum())}/{len(W)} complete; {rng(sweep_s, '.1f')} s", "Met",
     f"{int(W.f_min.min() / 1e3)}–{int(W.f_max.max() / 1e3)} kHz. Per session: 1 at start, {int(ctx['post'] / N)} "
     f"after the holds, {int(ctx['break'] / N)} between blocks. Duration from the bench timing "
     f"(data/sweep_timing.txt). Starts {np.median(lat):.0f} ms after the rest cue (median, n = {len(lat)})."),
    ("Excitation outside holds", "0 holds with a sweep", "Sweep windows vs. hold windows",
     f"{len(ov)}/{len(OV)} holds", "Not met",
     ("All are the first hold of a session: a sweep started at the beginning of the session ("
      + ", ".join(f"{k} in {int(v)}" for k, v in ov.source.value_counts().items())
      + ") was still running" if first_only else "Holds with a sweep: see data/s26/hold_overlap.csv")
     + f"; in those {len(short_holds)} holds the auxiliary board also took 1 load reading instead of 3. "
     "Fix: firmware-upgrades.md B8."),
    # S07 lee ~3 por s y llegan en tandas por latido: su intervalo de llegada (0 s dentro de la tanda)
    # no dice nada de la cadencia, que se saca de cuantas lecturas caben en el sostener de 5 s
    ("Load/temperature cadence", "every 2 s during holds", "Reading intervals",
     f"{rng(full.drop(dense).sensor_dt_med_s, '.2f')} s"
     + "".join(f"; {s_} ~{5 / P[P.subject == s_].n.median():.2f} s" for s_ in dense), "Met",
     f"3 readings in {int((P[~P.subject.isin(dense)].n == 3).sum())}/{int((~P.subject.isin(dense)).sum())} holds"
     + "".join(f"; {s_}: ~{int(P[P.subject == s_].n.median())} readings per hold, relayed in batches of 3 on the "
               "1 s heartbeat" for s_ in dense) + "."),
    ("Auxiliary sensor availability", "valid readings in every session", "Sensor presence and range checks",
     f"load {int(ok_load.sum())}/{len(A)}; temperature {int(temp_ok.sum())}/{N} sessions", "Not met",
     f"Load: {int(ok_load.sum())} of {len(A)} readings with both sensors above 0 kPa ("
     + " and ".join(f"{s_} {v}" for s_, v in load_bad.items() if v) + " at 0 kPa). Temperature: sensor not detected in "
     + ", ".join(temp_ok.index[~temp_ok]) + " (reads −126.8 °C)."),
    ("Clock synchronisation", "residual ≤20 ms after drift correction", "Host–device clock fit per session",
     f"SD {rng(lab_sd, '.1f')} ms", "Met",
     f"Device clock +{TD.clock_ppm.min():.0f} to +{TD.clock_ppm.max():.0f} ppm vs. PC "
     f"({rng(TD.clock_ppm * 3.6, '.0f')} ms/h), removed by a linear fit per session. Uncorrected, two hours "
     f"would accumulate {rng(TD.clock_ppm * 7.2, '.0f')} ms against the 100 ms alignment target."),
    ("Stream delivery jitter", "p95 ≤50 ms", "Arrival vs. device time",
     f"p95 {np.percentile(dly, 95):.0f} ms", "Met",
     f"p50 {np.percentile(dly, 50):.0f}, p99 {np.percentile(dly, 99):.0f}, max {dly.max():.0f} ms, above the "
     "fastest delivery (variation, not absolute latency)."),
    ("Local storage", "complete SD copy of every recording", "SD files vs. host counters",
     "1 of 8 recordings on SD", "Not met",
     f"Operator's check; cause not established. The host copy was complete ({int(S.wal_consistent.sum())}/{len(S)} "
     "recordings match their counters). Fix: firmware-upgrades.md B1. Already in the draft's Table 3."),
    ("Per-sample timestamps", "every sample, 1 µs resolution", "Timestamp continuity",
     "all samples", "Met",
     f"Intervals {rng(TD.short_us_med / 1000, '.2f')} ms ({100 * (1 - long_frac.mean()):.0f} %) and "
     f"{rng(TD.long_us_med / 1000, '.1f')} ms ({100 * long_frac.mean():.0f} %, every "
     f"{int(TD.long_period_mode.mode().iloc[0])}th round): resample on "
     "the timestamps (firmware-upgrades.md B6)."),
    ("Resting noise floor", "≤ X µV RMS referred to input", "Rest segments, 20–450 Hz",
     f"{noise_mv:.1f} mV at ADC input", "Pending",
     f"= {noise_mv / LSB_MV:.1f} LSB. Needs the total analog gain to be referred to the electrodes."),
]

# ------------------------------------------------------------------ salida
# marcas de nota en superindice Unicode: se pegan igual en Word, Docs o LaTeX
def sup(i):
    return "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(d)] for d in str(i + 1))


def supl(i):
    return "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏ"[i]


R = pd.DataFrame(rows, columns=["requirement", "target", "measured", "status", "note"])
R.to_csv(DATA26 / "requirements.csv", index=False)
K = pd.DataFrame(cand, columns=["requirement", "target", "verification", "measured", "status", "note"])
K.to_csv(DATA26 / "requirements_candidates.csv", index=False)

md = [f"# Performance requirements — {'Saturday 26 sessions' if DATASET == 's26' else DATASET}", "",
      ("Six complete sessions (S01–S06)" if DATASET == "s26" else
       f"{N} complete sessions ({', '.join(sorted(full.index))})")
      + f". Generated by `scripts/s26_09_requirements.py`; every value comes from "
      f"`data/{DATASET}/`. Targets as set on 29 Sep 2026.", "",
      "| Requirement | Target | Measured result | Status |", "|---|---:|---:|---|"]
md += [f"| {r.requirement} | {r.target} | {r.measured}{sup(i)} | {r.status} |" for i, r in R.iterrows()]
md += [""] + [f"{sup(i)} {r.note}  " for i, r in R.iterrows()]
md += ["", "## Candidate requirements (targets are proposals)", "",
       "| Requirement | Target | Verification | Measured result | Status |", "|---|---:|---|---:|---|"]
md += [f"| {r.requirement} | {r.target} | {r.verification} | {r.measured}{supl(i)} | {r.status} |"
       for i, r in K.iterrows()]
md += [""] + [f"{supl(i)} {r.note}  " for i, r in K.iterrows()]
md += ["", "Not measurable from these recordings, but usually expected for a wearable: mass of both armbands and "
       "donning time. If the IMU stays in the paper: "
       f"{rng(full.imu_hz, '.0f')} Hz by session against 200 Hz nominal (firmware-upgrades.md B7).", ""]
(OUT / f"requirements_{DATASET}.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
