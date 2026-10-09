"""Informe del conjunto elegido (report_<conjunto>.html): cifras, figuras y texto de borrador.

    EMG8_DATASET=s26-01 python s26_06_report.py [--out <ruta.html>]

Todo sale de data/<conjunto>/*.csv|npz para que el texto no se desvie de las
figuras. En ingles, como el articulo. Solo codigos de sujeto. Con s26 da el
informe del sabado; con s26-01 suma la sesion del 1-10 (S07) y su seccion.
"""
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT
from s26_common import DATA26, DATASET, FIG26, FIGP, GRASP_SHORT, SUBJECT_ORDER, bench_long_runs, find_sessions

try:
    from PIL import Image
except ImportError:          # sin PIL se incrustan tal cual
    Image = None

S = pd.read_csv(DATA26 / "sessions.csv")
full = S[S.complete].set_index("subject").reindex(SUBJECT_ORDER)
OUT_HTML = (Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv
            else OUT / f"report_{DATASET}.html")

# Cuantos son, en palabras y en canales: el texto ya no da por hecho "seis" ni "48"
WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
N = len(full)
NW, NCH = WORDS[N], 8 * N
DAY_TXT = {"20260926": "26 September 2026", "20261001": "1 October 2026"}
DAY_SHORT = {"20260926": "26 Sep", "20261001": "1 Oct"}
day_of = lambda sess: sess.split("_")[2]
S["day"] = S.session.map(day_of)
full["day"] = full.session.map(day_of)
days = sorted(S.day.unique())
EXTRA = full[full.day != days[0]]          # sesiones de otros dias que el sabado (en s26-01, S07)


def span(subs):
    """'S01–S06' si los codigos son consecutivos, si no 'S01, S03'."""
    k = sorted(subs)
    n = [int(s_[1:]) for s_ in k]
    if len(k) > 2 and n == list(range(n[0], n[0] + len(n))):
        return f"{k[0]}–{k[-1]}"
    return ", ".join(k)


# Protocolo por sesion (metadata.json): semilla del orden, entrada previa y colocacion
meta = {}
for p_ in find_sessions():
    m_ = json.loads((p_ / "metadata.json").read_text(encoding="utf-8"))
    meta[p_.name] = dict(seed=m_.get("seed"), lead_in=(m_.get("timing") or {}).get("lead_in_s"),
                         per_subject=bool(m_.get("seed_per_subject")), timing=m_.get("timing") or {})
full["seed"] = full.session.map(lambda x: meta.get(x, {}).get("seed"))
full["lead_in"] = full.session.map(lambda x: meta.get(x, {}).get("lead_in"))
seed_groups = full.groupby("seed").apply(lambda g: list(g.index)).sort_values(key=lambda c: -c.map(len))
same_order = len(seed_groups) == 1
main_seed, main_subs = seed_groups.index[0], seed_groups.iloc[0]
other_subs = [s_ for s_ in SUBJECT_ORDER if s_ not in main_subs]
TR0 = pd.read_csv(DATA26 / "trials.csv")
blocks_of = lambda s_: [GRASP_SHORT.get(g, str(g)) for g in
                        TR0[TR0.session == full.loc[s_, "session"]].sort_values("trial").grasp.iloc[::6]]
first_last = lambda s_: (blocks_of(s_)[0], blocks_of(s_)[-1])
# frases sobre el orden de los agarres (el sabado era uno solo para todos)
if same_order:
    order_txt = f"All participants used the same grasp order (seed {int(main_seed)})"
    order_short = "every participant had the same grasp order"
else:
    order_txt = (f"{span(main_subs)} used the same grasp order (seed {int(main_seed)}); "
                 + "; ".join(f"{s_} had its own (per-participant seed)" for s_ in other_subs))
    order_short = f"{span(main_subs)} had the same grasp order"
f0, l0 = first_last(main_subs[0])
if same_order:
    order_review = (f"every participant received the same grasp order (seed {int(main_seed)}; {f0} always first, "
                    f"{l0} always last), so grasp-related differences in contact load, sEMG or separability cannot be "
                    "separated from time in session (Figure 11b). Vary the seed per participant in the next sessions; "
                    "the monitor already stores it.")
else:
    order_review = (f"{span(main_subs)} received the same grasp order (seed {int(main_seed)}; {f0} first, {l0} last), "
                    "so for them grasp-related differences in contact load, sEMG or separability cannot be separated "
                    "from time in session (Figure 11b). "
                    + " ".join(f"{s_} already had its own order (seed {int(full.loc[s_, 'seed'])}, per participant; "
                               f"{first_last(s_)[0]} first, {first_last(s_)[1]} last)." for s_ in other_subs)
                    + " Keep a per-participant seed in the next sessions.")
C = pd.read_csv(DATA26 / "semg_channels.csv")
T = pd.read_csv(DATA26 / "semg_trials.csv")
Bk = pd.read_csv(DATA26 / "semg_breaks.csv")
TM = pd.read_csv(DATA26 / "timing.csv").set_index("subject")
BD = pd.read_csv(DATA26 / "sweep_bands.csv")
EV = pd.read_csv(DATA26 / "sweep_env.csv")
W = pd.read_csv(DATA26 / "wifi_lines.csv")
Z = pd.read_csv(DATA26 / "impedance.csv")
P = pd.read_csv(DATA26 / "sensors_trials.csv")
I = pd.read_csv(DATA26 / "imu_trials.csv")
CL = pd.read_csv(DATA26 / "classify.csv")
TR = pd.read_csv(DATA26 / "trials.csv")
TD = pd.read_csv(DATA26 / "timing_deep.csv").set_index("subject").reindex(SUBJECT_ORDER)
TO = pd.read_csv(DATA26 / "timing_offsets.csv")
SPU = pd.read_csv(DATA26 / "timing_spurs.csv")
BU = pd.read_csv(DATA26 / "budget.csv").set_index("subject")
LK = np.load(DATA26 / "link_arrays.npz")
LG = pd.read_csv(DATA26 / "envelope_lag.csv")
IFc = pd.read_csv(DATA26 / "interface.csv")
CZ = np.load(DATA26 / "corr.npz")

# anatomia del muestreo, enlace, presupuesto (s26_07) e instrumento (s26_08)
imu_win = np.concatenate([LK[f"rate_imu_{s_}"] for s_ in SUBJECT_ORDER])
imu_med = {s_: float(np.median(LK[f"rate_imu_{s_}"])) for s_ in SUBJECT_ORDER}
lat_all = np.concatenate([LK[f"lat_{s_}"] for s_ in SUBJECT_ORDER])
spur = SPU.groupby("tone_hz")[["spur_dbc", "resampled_spur_dbc", "cubic_spur_dbc"]].median()
bud = BU.mean(numeric_only=True)
own_adc = TO[(TO.kind == "raw") & (TO.channel == "E2")].off_med_ms.median()


def neigh(key, s_):
    Cm = CZ[f"{key}_{s_}"]
    return float(np.mean([Cm[i, (i + 1) % 8] for i in range(8)]))


def far(key, s_):
    Cm = CZ[f"{key}_{s_}"]
    return float(np.mean([Cm[i, (i + 4) % 8] for i in range(8)]))


r_hold = [neigh("hold", s_) for s_ in SUBJECT_ORDER]
r_rest = [neigh("rest", s_) for s_ in SUBJECT_ORDER]
r_restnb = [neigh("restnb", s_) for s_ in SUBJECT_ORDER]
r_far_rest = [far("rest", s_) for s_ in SUBJECT_ORDER]
# todos los sostenes que califican, como en las figuras 10 y 13 (el cruce sostenido ya no deja
# los valores sueltos que antes obligaban a recortar a -100..500 ms)
lag_on = LG.lag_on_ms.dropna()
# retardo por correlacion cruzada de todo el sostener (envlag.py): el estimado robusto
lag_x = LG.lag_xcorr_ms.dropna()
lag_off = LG.lag_off_ms.dropna()
WCL = pd.read_csv(DATA26 / "wifi_clean.csv")
GAIN = pd.read_csv(DATA26 / "channel_gain.csv").set_index("channel")
_o = WCL[WCL.method == "original"].set_index(["subject", "channel"])
wifi_stats = {m_: dict(lines=100 * WCL[WCL.method == m_].lines_share_rest.median(),
                       hold_db=float(np.median(20 * np.log10(WCL[WCL.method == m_].set_index(["subject", "channel"])
                                                              .hold_rms_mv / _o.hold_rms_mv))))
              for m_ in ("original", "notch", "template", "repair")}
rep_frac = 100 * WCL[WCL.method == "repair"].frac_repaired.median()


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


rho_if = {s_: spearman(IFc[IFc.subject == s_].z5k, IFc[IFc.subject == s_].rest_db) for s_ in SUBJECT_ORDER}
rho_zt = {s_: spearman(IFc[IFc.subject == s_].z5k, IFc[IFc.subject == s_].t_min) for s_ in SUBJECT_ORDER}

# Cifras del texto que salen de los datos (antes estaban escritas a mano)
OV = pd.read_csv(DATA26 / "hold_overlap.csv")
ov = OV[OV.overlap_s > 0]
ov_n, ov_total = len(ov), len(OV)
ov_rng = f"{ov.overlap_s.min():.1f}–{ov.overlap_s.max():.1f}" if len(ov) else "0"
ov_subjects = sorted(ov.subject.unique(), key=SUBJECT_ORDER.index)
ov_clear = [s_ for s_ in SUBJECT_ORDER if s_ not in ov_subjects]
ov_by_rest = sorted(ov[ov.source == "rest command"].subject.unique(), key=SUBJECT_ORDER.index)
aux_first = (P.dt0.quantile(0.02), P.dt0.quantile(0.98))
span_rng = f"{full.span_s.min() / 60:.1f}–{full.span_s.max() / 60:.1f}"
# cortes de verdad (huecos); unas pocas muestras de menos sin hueco (S07: 5) van aparte
drop = full[full.raw_gaps > 0]
tiny = full[(full.raw_missing > 0) & (full.raw_gaps == 0)]
tiny_txt = "".join(f" ({s_} lacked {int(r.raw_missing)} sample{'s' if r.raw_missing != 1 else ''} across its eight "
                   "channels, with no gap)" for s_, r in tiny.iterrows())
drop_s = {s_: float(r.raw_missing / 8 / r.raw_hz) for s_, r in drop.iterrows()}
drop_txt = "; ".join(f"{s_} {v:.1f} s ({full.loc[s_, 'raw_loss_pct']:.2f} %)" for s_, v in drop_s.items()) or "none"
env_max_v = EV.sweep_p95.max() / 1000
prev_rng = f"{full.preview_s.min():.1f}–{full.preview_s.max():.1f}"
noisy = C[C.base_rms_mv > 24]
rest_dev = C.rest_db_med.abs().max()
# Arranques abortados: minutos hasta la sesion completa del mismo sujeto
aborted = S[~S.complete]
hhmm = lambda n: int(n[-6:-4]) * 60 + int(n[-4:-2]) + int(n[-2:]) / 60
restart_min = [hhmm(full.loc[r.subject, "session"]) - hhmm(r.session) for _, r in aborted.iterrows()
               if r.subject in full.index]
# lecturas de carga/temperatura del primer sostener de las sesiones con barrido encima
_p0 = P[P.session.isin(S[S.complete].session) & (P.trial == 0) & P.subject.isin(ov_subjects)]
_n0 = sorted(set(_p0.n.astype(int)))
first_hold_txt = ((f"a single reading ({_p0.dt0.min():.1f}–{_p0.dt0.max():.1f} s after the cue)" if _n0[0] == 1
                   else f"{WORDS[_n0[0]]} readings") if len(_n0) == 1 else f"{'/'.join(str(v) for v in _n0)} readings")
# muestreo: la ronda larga (envolventes) y la tasa en ventanas de 10 s
long_ms = TD.long_us_med.median() / 1000
long_pct = 100 * TD.long_frac.median()
off_win = {s_: int((np.abs(LK[f"rate_raw_{s_}"] - 1000) > 0.5).sum()) for s_ in SUBJECT_ORDER}
n_off_win = sum(off_win.values())
rate_ok = float(np.median(np.concatenate([LK[f"rate_raw_{s_}"] for s_ in SUBJECT_ORDER])))
wrap_min = 2 ** 32 / 60e6
# ensayos sin tasa completa en las sesiones completas, con su causa
low = TR[TR.session.isin(full.session) & (TR.coverage < 0.95)].copy()
low["cause"] = ["start-up preview" if r.t_grasp < full.loc[r.subject, "preview_s"] else "dropout"
                for _, r in low.iterrows()]
low = low.sort_values("subject", key=lambda c: c.map(SUBJECT_ORDER.index))
low_txt = ", and ".join(f"trial {r.trial + 1} of {r.subject}, " + ("still in the start-up preview"
                         if r.cause == "start-up preview" else f"during the {drop_s.get(r.subject, 0):.1f} s dropout")
                         for _, r in low.iterrows())
prev_hit = low[low.cause == "start-up preview"]
prev_hit_txt = "; ".join(f"in {r.subject} this overlapped trial {r.trial + 1}" for _, r in prev_hit.iterrows())
# barridos: completos, rejilla de frecuencias y duracion (4.2 s medidos en banco, data/sweep_timing.txt)
n_sweeps, n_sweeps_ok = int(full.sweeps.sum()), int(full.sweeps_99pts.sum())
wal_ok = int(S.wal_consistent.sum())
ZN = np.load(DATA26 / "impedance.npz")
freq = ZN["freq"]
SWEEP_S = 4.2
short_band = (freq >= 2e3) & (freq <= 30e3)
short_s = short_band.sum() / len(freq) * SWEEP_S
# rodilla de |Z|: primera frecuencia a menos de 10 % del piso (|Z| a 100 kHz), barrido por barrido
knees = {}
for s_ in SUBJECT_ORDER:
    M = ZN[s_][np.isfinite(ZN[s_]).all(axis=1)]
    knees[s_] = np.array([freq[np.argmax(row <= 1.1 * row[-1])] / 1e3 for row in M])
knee_med = pd.Series({s_: np.median(k) for s_, k in knees.items()})
knee_max_s = max(knees, key=lambda s_: knees[s_].max())
knee_max = knees[knee_max_s].max()
# sensores: cargas que caen a ~0 kPa para el resto de la sesion (la temperatura, mas abajo con TT)


def load_lost(s_, col, floor=1.0):
    # primer ensayo desde cuya lectura todas las siguientes quedan bajo 1 kPa
    p = P[P.subject == s_].sort_values("trial")
    v = p[[f"{col}_0", f"{col}_1", f"{col}_2"]].to_numpy().ravel()
    tr = np.repeat(p.trial.to_numpy(), 3)
    ok = ~np.isnan(v)
    v, tr = v[ok], tr[ok]
    if not len(v) or v[-1] >= floor:
        return None
    good = np.where(v >= floor)[0]
    i = good[-1] + 1 if len(good) else 0
    return int(tr[i]) if len(v) - i >= 3 else None


lost = [(s_, col, t) for s_ in SUBJECT_ORDER for col in ("p1", "p2") if (t := load_lost(s_, col)) is not None]
_lost = [f"from trial {t + 1} of 42 in {s_} ({col})" for s_, col, t in lost]
lost_txt = ", ".join(_lost[:-1]) + (" and " if len(_lost) > 1 else "") + (_lost[-1] if _lost else "")
trials_all = (f"all {NW} sessions" if (full.trials == 42).all()
              else f"{int((full.trials == 42).sum())} of {len(full)} sessions")
sweeps_txt = (f"All {n_sweeps} impedance sweeps arrived complete ({len(freq)} points)." if n_sweeps_ok == n_sweeps
              else f"{n_sweeps_ok} of {n_sweeps} impedance sweeps arrived complete ({len(freq)} points).")
wal_txt = f"all {WORDS[len(S)]} folders" if wal_ok == len(S) else f"{wal_ok} of {len(S)} folders"
drop_who_txt = (f"the single dropout in {drop.index[0]} accounts" if len(drop) == 1
                else f"the dropouts in {', '.join(drop.index)} account")

# ------------------------------------------------------------------ cifras
n_sess = len(S)
fr_min = full.full_rate_s.sum() / 60
loss_all = np.average(full.raw_loss_pct, weights=full.full_rate_s)
trials_fr = int(full.trials_full_rate.sum())
spread_med = float(np.median(TM.spread_med_ms))
spread_p95 = float(np.median(TM.spread_p95_ms))
jit = (full.host_jitter_ms.min(), full.host_jitter_ms.max())
imu_hz = (full.imu_hz.min(), full.imu_hz.max())

# envolvente: umbral de respuesta
T2 = T.dropna(subset=["act_db", "env_delta"])
bins = np.arange(-3, 25, 1.0)
med = [T2.env_delta[(T2.act_db >= a) & (T2.act_db < b)].median() if ((T2.act_db >= a) & (T2.act_db < b)).sum() > 15
       else np.nan for a, b in zip(bins[:-1], bins[1:])]
cen = (bins[:-1] + bins[1:]) / 2
env_thr = float(cen[np.nanargmax(np.array(med) > 20)])

# barridos: celdas afectadas (dB de todo el barrido en pausas)
K = pd.read_csv(DATA26 / "sweep_steps.csv")
whole = (K.groupby(["subject", "channel", "block"]).db.apply(lambda d: 10 * np.log10(np.mean(10 ** (d / 10))))
         .groupby(["subject", "channel"]).median())
hot = whole[whole > 8]
flag_env = EV[(EV.sweep_p95 > EV.hold_p95) & (EV.sweep_p95 > 100)]
band_hot = BD.set_index(["subject", "channel"]).loc[hot.index].reset_index()
band_cold = BD.set_index(["subject", "channel"]).drop(hot.index).reset_index()
# canales calientes que siguen recogiendo por encima de 30 kHz (recogida de banda ancha)
broad = band_hot[(band_hot.band == "30-100") & (band_hot.db > 8)]
broad_txt = ", ".join(f"{r.subject} {r.channel}" for _, r in broad.iterrows())

# sEMG por sujeto
qual = []
for s_ in SUBJECT_ORDER:
    c = C[C.subject == s_]
    t = T[T.subject == s_]
    cv = (t.groupby(["grasp", "channel"]).rms_hold.agg(lambda v: np.nanstd(v) / np.nanmean(v))).median()
    rest_first = t[t.block == 0].groupby("channel").rms_rest.median()
    rest_last = t[t.block == 6].groupby("channel").rms_rest.median()
    drift = float(np.median(20 * np.log10(rest_last / rest_first)))
    qual.append(dict(subject=s_, responsive=int((c.act_db_med >= 6).sum()),
                     act_top3=float(np.sort(c.act_db_med.values)[-3:].mean()),
                     noise_mv=float(c.base_rms_mv.median()), mdf=float(c.mdf_med.median()),
                     rep_cv=float(cv) * 100, rest_drift_db=drift,
                     sweep_cells=int((whole.loc[s_] > 8).sum())))
Q = pd.DataFrame(qual).set_index("subject")

# carga de contacto. La placa auxiliar filtra cada FSR con una mediana de las
# 3 ultimas lecturas y ese bufer sigue de un sostener al siguiente: la primera
# lectura de un sostener suele repetir la ultima del anterior (se mide abajo).
# Solo la tercera lectura es del sostener en curso; la variacion dentro del
# sostener no se puede medir con estos datos.
pp = []
for s_ in SUBJECT_ORDER:
    p = P[P.subject == s_]
    for col in ("p1", "p2"):
        v = p[[f"{col}_0", f"{col}_1", f"{col}_2"]]
        last = p[f"{col}_2"]
        ok = last > 0.05
        blocks = p[ok].groupby("block")[f"{col}_2"].median()
        pp.append(dict(subject=s_, sensor=col, between_kpa=float(blocks.max() - blocks.min()),
                       zero=int(((v <= 0.05) & v.notna()).sum().sum())))
PP = pd.DataFrame(pp)
carry = {}
for col in ("p1", "p2", "temp"):
    same = n = 0
    for s_ in SUBJECT_ORDER:
        g = P[P.subject == s_].sort_values("trial")
        a, b = g[f"{col}_0"].to_numpy()[1:], g[f"{col}_2"].to_numpy()[:-1]
        k = np.isfinite(a) & np.isfinite(b)
        same += int(np.sum(np.isclose(a[k], b[k], atol=0.005)))
        n += int(k.sum())
    carry[col] = 100 * same / max(n, 1)

# Lecturas densas (el 1-10 la placa auxiliar leyo ~3 por s, ~15 por sostener): con ellas si se
# puede ver la variacion dentro de un sostener. Las 2 primeras de cada uno son arrastre del filtro.
A = pd.read_csv(DATA26 / "sensors.csv")
dense = [s_ for s_ in SUBJECT_ORDER if P[P.subject == s_].n.median() > 3]
DN = {}
for s_ in dense:
    a = A[A.subject == s_]
    rows_ = []
    for _, r in TR0[TR0.session == full.loc[s_, "session"]].iterrows():
        # llegan en tandas por latido de 1 s: la ultima tanda del sostener cae hasta 1 s despues
        x = a[(a.t >= r.t_grasp) & (a.t < r.t_rest + 1.0)]
        if len(x) >= 5:
            y = x.iloc[2:]
            rows_.append(dict(block=int(r.trial) // 6, n=len(x), T_sd=y.temp_c.std(), p2_sd=y.p2_kpa.std(),
                              T=y.temp_c.median(), p2=y.p2_kpa.median(), p1=y.p1_kpa.median()))
    d_ = pd.DataFrame(rows_)
    bl = d_.groupby("block")[["T", "p2"]].median()
    DN[s_] = dict(n=int(d_.n.median()), rate=float(d_.n.median() / 5.0), T_sd=float(d_.T_sd.median()),
                  T_rng=float(np.ptp(A[A.subject == s_].temp_c)), p2_sd=float(d_.p2_sd.median()),
                  p1=float(d_.p1.median()), p2_lo=float(d_.p2.min()), p2_hi=float(d_.p2.max()),
                  T_prev=float(bl["T"].iloc[-2]), T_last=float(bl["T"].iloc[-1]),
                  p2_prev=float(bl.p2.iloc[-2]), p2_last=float(bl.p2.iloc[-1]),
                  last_grasp=blocks_of(s_)[-1])
T_STEP = float(np.min(np.diff(np.unique(np.round(A.temp_c[A.temp_c > -100].dropna(), 3)))))

# temperatura
temp = []
for s_ in SUBJECT_ORDER:
    p = P[P.subject == s_]
    ok_t = bool(p[["temp_0", "temp_1", "temp_2"]].notna().any().any())
    i = I[I.subject == s_]
    it = i.imu_temp[i.imu_temp > 5]
    temp.append(dict(subject=s_, valid=ok_t, t0=float(p.temp_0.dropna().iloc[:3].median()) if ok_t else np.nan,
                     t1=float(p.temp_2.dropna().iloc[-3:].median()) if ok_t else np.nan,
                     imu0=float(it.iloc[:3].median()), imu1=float(it.iloc[-3:].median())))
TT = pd.DataFrame(temp).set_index("subject")
temp_missing = [s_ for s_ in SUBJECT_ORDER if not TT.loc[s_, "valid"]]
# SUBJECT_ORDER es el orden de grabacion: se dice "las primeras N" solo si lo son
temp_when = (f"the first {WORDS[len(temp_missing)]} sessions" if temp_missing == SUBJECT_ORDER[:len(temp_missing)]
             else f"{WORDS[len(temp_missing)]} sessions")

# impedancia
zz = []
for s_ in SUBJECT_ORDER:
    z = Z[(Z.subject == s_) & (Z.t_prest > 1.0)].sort_values("t_prest")
    zz.append(dict(subject=s_, z2_0=z.z2k.iloc[0] / 1e3, z2_1=z.z2k.iloc[-1] / 1e3,
                   z5_change=100 * (z.z5k.iloc[-1] / z.z5k.iloc[0] - 1), z100=z.z100k.median() / 1e3))
ZZ = pd.DataFrame(zz).set_index("subject")
z100_all = Z.z100k / 1e3

# IMU
imu = I.groupby("subject")[["gyr_onset_rms", "gyr_hold_rms", "gyr_release_rms", "gyr_rest_rms"]].median()
posture_max = I.groupby("subject").incl_deg.max() - I.groupby("subject").incl_deg.min()   # cambio, no valor absoluto
incl_txt = ", ".join(f"{s_} {posture_max[s_]:.0f}°" for s_ in SUBJECT_ORDER if posture_max[s_] > 5)
short_starts_txt = " and ".join(f"{v:.0f} s" for v in aborted.span_s)


def img(stem, width=1600):
    stem = stem.replace("fig_s26_", FIGP + "_", 1)    # claves del sabado, archivo del conjunto
    pth = FIG26 / f"{stem}.png"
    if Image is not None:
        im = Image.open(pth)
        if im.width > width:
            im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
    else:
        data = pth.read_bytes()
    return f'<img src="data:image/png;base64,{base64.b64encode(data).decode()}" alt="{stem}">'


def f1(v):
    return f"{v:.1f}"


# ------------------------------------------------------------------ tablas
# requisitos: la misma tabla que va al articulo (s26_09_requirements.py), con sus notas
RQ = pd.read_csv(DATA26 / "requirements.csv")
RK = pd.read_csv(DATA26 / "requirements_candidates.csv")


def req_table(D):
    return "".join(f"<tr><td>{html.escape(r.requirement)}</td><td>{html.escape(r.target)}</td>"
                   f"<td>{html.escape(r.measured)}</td><td class='st'>{html.escape(r.status)}</td>"
                   f"<td class='nt'>{html.escape(r.note)}</td></tr>" for r in D.itertuples())


req_html, cand_html = req_table(RQ), req_table(RK)

ses_html = "".join(
    f"<tr><td>{s_}</td><td>{full.loc[s_, 'session'][-6:-4]}:{full.loc[s_, 'session'][-4:-2]}</td>"
    f"<td>{full.loc[s_, 'span_s']/60:.1f}</td><td>{full.loc[s_, 'preview_s']:.1f}</td>"
    f"<td>{full.loc[s_, 'raw_hz']:.1f}</td><td>{full.loc[s_, 'raw_loss_pct']:.2f}</td>"
    f"<td>{int(full.loc[s_, 'trials'])}/42 ({int(full.loc[s_, 'trials_full_rate'])})</td>"
    f"<td>{int(full.loc[s_, 'sweeps_99pts'])}/{int(full.loc[s_, 'sweeps'])}</td>"
    f"<td>{int(full.loc[s_, 'sensors'])}</td><td>{'yes' if TT.loc[s_, 'valid'] else 'not detected'}</td>"
    f"<td>{full.loc[s_, 'imu_hz']:.0f}</td><td>{full.loc[s_, 'host_jitter_ms']:.1f}</td></tr>"
    for s_ in SUBJECT_ORDER)

q_html = "".join(
    f"<tr><td>{s_}</td><td>{Q.loc[s_, 'responsive']}/8</td><td>{Q.loc[s_, 'act_top3']:.1f}</td>"
    f"<td>{Q.loc[s_, 'noise_mv']:.1f}</td><td>{Q.loc[s_, 'mdf']:.0f}</td><td>{Q.loc[s_, 'rep_cv']:.0f}</td>"
    f"<td>{Q.loc[s_, 'rest_drift_db']:+.1f}</td><td>{Q.loc[s_, 'sweep_cells']}</td></tr>" for s_ in SUBJECT_ORDER)

aux_html = "".join(
    f"<tr><td>{s_}</td><td>{ZZ.loc[s_, 'z2_0']:.0f} → {ZZ.loc[s_, 'z2_1']:.0f}</td>"
    f"<td>{ZZ.loc[s_, 'z5_change']:+.0f}</td><td>{ZZ.loc[s_, 'z100']:.2f}</td>"
    f"<td>{PP[(PP.subject == s_) & (PP.sensor == 'p1')].between_kpa.iloc[0]:.1f}</td>"
    f"<td>{PP[(PP.subject == s_) & (PP.sensor == 'p2')].between_kpa.iloc[0]:.1f}</td>"
    f"<td>{int(PP[PP.subject == s_].zero.sum())}</td>"
    f"<td>{(f'{TT.loc[s_, chr(116)+chr(48)]:.1f} → {TT.loc[s_, chr(116)+chr(49)]:.1f}') if TT.loc[s_, 'valid'] else 'n/a'}</td>"
    f"<td>{TT.loc[s_, 'imu0']:.1f} → {TT.loc[s_, 'imu1']:.1f}</td>"
    f"<td>{imu.loc[s_, 'gyr_onset_rms']:.2f} / {imu.loc[s_, 'gyr_rest_rms']:.2f}</td>"
    f"<td>{posture_max.loc[s_]:.0f}</td></tr>" for s_ in SUBJECT_ORDER)

hot_list = ", ".join(f"{s_} {c_} ({v:.0f} dB)" for (s_, c_), v in hot.sort_values(ascending=False).items())
envf_list = ", ".join(f"{r.subject} {r.channel}" for _, r in flag_env.sort_values("sweep_p95", ascending=False).iterrows())
bands_hot = BD.set_index(["subject", "channel"]).loc[hot.index].groupby("band").db.median()
bands_cold = band_cold.groupby("band").db.median()

caveats = [
    ("Bioimpedance excitation reaches some sEMG channels.",
     f"In relaxed breaks, {len(hot)} of {NCH} participant-channels rose > 8 dB during a sweep ({hot_list}); the "
     f"other {NCH-len(hot)} rose by a median {band_cold.db.median():.1f} dB. The pickup is frequency-selective: on the "
     f"affected channels it peaks at 10–30 kHz excitation (median {bands_hot.get('10-20', np.nan):.0f} dB at "
     f"10–20 kHz, {bands_hot.get('20-30', np.nan):.0f} dB at 20–30 kHz) and vanishes above 30 kHz"
     + (f" except on {broad_txt}, where it is broadband. " if len(broad) else ". ") +
     f"On {len(flag_env)} channels ({envf_list}) the hardware envelope rose more during a sweep than during any "
     f"grasp (up to {env_max_v:.1f} V): a false activation for envelope-based use.",
     "Keep sweeps in rest (as scheduled here), mark or blank the sweep window in downstream processing, and "
     "investigate the 10–30 kHz coupling path (electrode proximity/shielding) per channel."),
    ("The Wi-Fi radio leaves spectral lines in the sEMG.",
     f"Lines at n × 9.766 Hz, i.e. every 102.4 ms, the default beacon interval of the bracelet's own Wi-Fi access "
     f"point (the most likely source; not yet confirmed by changing the interval), stand up to {W.peak_line_db.median():.0f} dB "
     f"above the resting floor (median over channels; highest {W.peak_line_db.max():.0f} dB) and carry {W.comb_power_pct.median():.0f} % of resting power (p10–p90 "
     f"{W.comb_power_pct.quantile(0.1):.0f}–{W.comb_power_pct.quantile(0.9):.0f} %), i.e. "
     f"+{W.comb_rms_db.median():.1f} dB on the resting RMS. The radio has to be on because full-rate sEMG is only "
     "delivered over UDP.",
     "Longer beacon interval (which would also confirm the source) or USB/UART full-rate transport; supply "
     "decoupling of the radio; optional comb notch."),
    ("Auxiliary data are timestamped by the host, not by the device.",
     "The auxiliary board reads contact load and temperature at the grasp command and every 2 s until the rest "
     "command, and keeps synchronised timestamps on its own SD, but the lines it relays carry none; the bracelet "
     f"forwards them on a 1 s heartbeat, so the first sample reaches the host {aux_first[0]:.1f}–{aux_first[1]:.1f} s "
     f"after the cue and a sweep {np.median(TM.sweep_arrival_med_s):.1f} s after the rest cue. Alignment with sEMG is "
     "therefore ~1 s for these modalities.",
     "Stamp every auxiliary line on reception with the bracelet clock and relay the auxiliary board's own time "
     "(firmware-upgrades.md B2, A1)."),
    ("Raw sEMG is not uniformly sampled.",
     f"Each ADC converts its two raw channels every {TD.short_us_med.median()/1000:.2f} ms and, every 20th round, "
     f"also its two envelope channels, which stretches that round to {long_ms:.2f} ms ({long_pct:.2f} % "
     f"of intervals, every {int(TD.long_period_mode.mode()[0])}th interval in at least "
     f"{100 * TD.long_period_frac.min():.2f} % of cases; the average is {rate_ok:.1f} Hz). Treated as uniform 1 kHz, the stream "
     f"carries a 50 Hz sawtooth timing error of {TD.err_p2p_ms.min():.1f}–{TD.err_p2p_ms.max():.1f} ms p–p: a "
     f"100 Hz tone shows a spur at {spur.loc[100, 'spur_dbc']:.0f} dB relative to the tone "
     f"({spur.loc[250, 'spur_dbc']:.0f} dB at 250 Hz). Resampling on the timestamps reduces it to "
     f"{spur.loc[100, 'resampled_spur_dbc']:.0f} dB (linear) or {spur.loc[100, 'cubic_spur_dbc']:.0f} dB (cubic), "
     f"but not above ~200 Hz ({spur.loc[250, 'cubic_spur_dbc']:.0f} dB at 250 Hz): the 2.5 ms hole limits "
     "reconstruction.",
     "Users: always resample on the timestamps. Firmware: spread the envelope conversions so no raw interval "
     "exceeds ~1.1 ms (firmware-upgrades.md B6)."),
    ("The IMU runs below its nominal rate.",
     f"Polled at a nominal 200 Hz, it delivered {min(imu_med.values()):.0f}–{max(imu_med.values()):.0f} Hz "
     f"(session medians; 10 s windows {imu_win.min():.0f}–{imu_win.max():.0f} Hz), stamped at read time.",
     "Data-ready or FIFO-timestamped acquisition at a fixed ODR (firmware-upgrades.md B7)."),
    ("Part of the resting sEMG is common to all channels.",
     f"At relaxed rest, neighbouring channels correlate at r = {np.mean(r_rest):.2f} and opposite ones at "
     f"{np.mean(r_far_rest):.2f}: no decay with distance, which points to interference (reference, supply or "
     f"radio) rather than muscle crosstalk. Removing the Wi-Fi beacon lines lowers it to {np.mean(r_restnb):.2f}. During "
     f"holds the channels are nearly independent (r = {np.mean(r_hold):.2f}).",
     "Review reference/ground routing and radio supply decoupling; common-average referencing in processing."),
    ("Auxiliary sensor problems were not flagged during acquisition.",
     f"The skin-temperature sensor read -126.8 °C (not detected) in {temp_when} ({', '.join(temp_missing)}); "
     + (f"one contact-load sensor fell to ~0 kPa (below 1 kPa for the rest of the session) {lost_txt}. "
        if len(lost) == 1 else
        f"{WORDS[len(lost)]} contact-load sensors read ~0 kPa (below 1 kPa for the rest of the session), {lost_txt}. ")
     + "A reading "
     "at 0 kPa is below the sensor's boot-time tare: the data cannot tell a lifted or displaced sensor from a "
     "failed one.",
     "Pre-session and live checks for sentinel values and flat/zero auxiliary channels in the monitor."),
    ("Contact-load readings are filtered across holds, zeroed at boot, and p2 is provisionally calibrated.",
     "The auxiliary firmware takes the median of each force sensor's last three readings, and that buffer carries "
     f"over from one hold to the next: the first reading of a hold equalled the last reading of the previous hold "
     f"in {carry['p1']:.0f} % of trials (p1) and {carry['p2']:.0f} % (p2), against {carry['temp']:.0f} % for the "
     "unfiltered temperature. "
     + ("Only the third reading of each hold is used here, so stability within a hold cannot be assessed. "
        if not dense else
        f"Only the last reading of each hold is used here (the third on {DAY_SHORT[days[0]]}, when three readings "
        "per hold could not show stability within a hold); "
        + "; ".join(f"in {s_}, with ~{DN[s_]['n']} readings per hold, p2 varied by {DN[s_]['p2_sd']:.1f} kPa (SD) "
                    "within a hold" for s_ in dense) + ". ")
     + "Both sensors are zeroed (tared) when the auxiliary board boots, so the values are relative to "
     "the load at that moment. The p2 conversion to kPa is a provisional two-weight fit that the firmware itself "
     "marks as a reference, not a reliable measurement; p2 values are shown but not interpreted in kPa.",
     "Reset the median filter at each grasp command; tare explicitly with the band off; recalibrate p2 "
     "(firmware-upgrades.md)."),
    ("The start of a session costs the first trial.",
     f"The full-rate stream began {prev_rng} s after recording started (until then the host logged only the 1 Hz "
     "UART preview; the cause of the delay was not established); "
     f"{prev_hit_txt or 'no trial fell in it'}. The auxiliary board also runs a full sweep when a test starts, and the host began "
     f"the first hold right at the start of recording, so in {len(ov_subjects)} of {NW} sessions (all but "
     f"{', '.join(ov_clear)}) the first hold ran on top of a sweep for {ov_rng} s ({ov_n}/{ov_total} holds); in "
     f"{WORDS[len(ov_by_rest)]} of them ({', '.join(ov_by_rest)}), opened while the bracelet was already recording, "
     "the sweep was the one requested by the session-start rest command. While it swept, the auxiliary board did not "
     f"read its sensors: the first hold of those sessions got {first_hold_txt} instead of three."
     + "".join(f" {s_} ({DAY_SHORT[full.loc[s_, 'day']]}), run with a {full.loc[s_, 'lead_in']:.0f} s lead-in, had no "
               "overlap." for s_ in SUBJECT_ORDER
               if pd.notna(full.loc[s_, "lead_in"]) and full.loc[s_, "lead_in"] > 0 and s_ not in ov_subjects),
     "Hold a ≥ 5 s lead-in after the recording starts and start the protocol only once full-rate data are flowing "
     "(monitor); let a grasp command abort a running sweep (firmware-upgrades.md B8)."),
    ("The impedance magnitude has a fixed floor.",
     f"Above the knee (where |Z| comes within 10 % of its 100 kHz value: {knee_med.min():.0f}–{knee_med.max():.0f} kHz, "
     f"session medians; up to {knee_max:.0f} kHz in the first sweeps of {knee_max_s}) every sweep converged on "
     f"{z100_all.min():.2f}–{z100_all.max():.2f} kΩ in all participants, which points to the measurement chain "
     "rather than the skin.",
     "Report |Z| below the knee only; review the AD5933 range/calibration and series resistance."),
    ("The hardware envelope has a dead zone and a lag.",
     f"The analog envelope stayed at its floor until the raw activation exceeded ~{env_thr:.1f} dB above rest, "
     f"then rose monotonically. Over the whole hold it trailed the raw RMS by {lag_x.median():.0f} ms "
     f"(cross-correlation, IQR {lag_x.quantile(0.25):.0f}–{lag_x.quantile(0.75):.0f} ms, n = {len(lag_x)}); its 50 % "
     f"crossing came {lag_on.median():.0f} ms after the raw one at the grasp (the dead zone stretches the rise) and "
     f"{lag_off.median():.0f} ms at the release (Figure 13). Weak activations are invisible in the envelope.",
     "Offset/gain review of the envelope stage, or rely on the raw channel for low-effort tasks and fast control."),
    ("The impedance electrodes do not report the sEMG interface.",
     "Within sessions, |Z| at 5 kHz changed strongly (Spearman ρ with time " + ", ".join(
         f"{rho_zt[s_]:+.2f}" for s_ in SUBJECT_ORDER) + ") but did not track the resting sEMG noise (ρ " +
     ", ".join(f"{rho_if[s_]:+.2f}" for s_ in SUBJECT_ORDER) + "): the impedance pair sits on the distal auxiliary armband, "
     "over the flexor region, and measures its own interface.",
     "To monitor the sEMG electrodes themselves, route the AD5933 to sEMG electrode pairs through an analog "
     "switch (the auxiliary board already uses an ADG849 to select the AD5933 feedback resistor)."),
    ("Local storage did not retain the sessions.",
     f"After {'the day' if not len(EXTRA) else DAY_SHORT[days[0]]}, the SD card held one set of files (one "
     "recording) instead of eight (operator's check); the "
     "host copy was complete. The cause was not established. In this firmware any SD error disables the card until "
     "the next boot while acquisition continues over the link, and during the day the status LED never showed an "
     "SD fault (operator's observation).",
     "Firmware: retry the card at each recording start and show SD faults unmistakably on the LED; monitor: "
     "end-of-session 'saved to SD / PC' confirmation (FIRMWARE-CONTRACT §2.5)."),
]
# prueba larga de banco (vale para el equipo, no para un conjunto): si no llego a 120 min, va como salvedad
LR = bench_long_runs()
if LR and LR[-1]["minutes"] < 120:
    lr = LR[-1]
    caveats.append((
        "A slow SD write ends the recording.",
        f"In a bench recording of all eight channels at 1000 Hz to the SD card ({lr['day']}), the firmware stopped "
        f"after {lr['minutes']:.1f} min, when one sync of the four files took {lr['sync_s']:.2f} s, longer than the "
        f"1.5 s raw buffer: {lr['raw_drops']} raw samples were lost, the files closed cleanly ({lr['new_mb']:.0f} MB) "
        "and the earlier files on the card were unchanged. In this firmware any lost sample ends the recording, so "
        "one stall decides how long a session can be; continuous recording ≥ 120 min was not reached.",
        "Keep recording through a buffer overflow and log the gap; a buffer of ≥ 3 s; fewer syncs or pre-allocated "
        "files (firmware-upgrades.md B11)."))
cav_html = "".join(f"<li><b>{a}</b> {b}<br><span class='fix'>Fix:</span> {c}</li>" for a, b, c in caveats)

# frases que dependen de la prueba larga y de cuantos dias hay en el conjunto
if not LR:
    claim_cont = "Continuous recording ≥ 120 min and battery ≥ 2 h: not tested here."
    bench_res = ""
else:
    lr = LR[-1]
    ok_ = lr["minutes"] >= 120
    claim_cont = (f"Continuous recording ≥ 120 min: {'met' if ok_ else 'not met'} in the bench test "
                  f"({lr['minutes']:.1f} min" + ("" if ok_ else ", stopped by an SD write stall; firmware-upgrades.md "
                                                  "B11") + "). Battery ≥ 2 h: not tested.")
    bench_res = ("" if ok_ else
                 f" In a separate bench recording to the SD card ({lr['day']}), the firmware stopped after "
                 f"{lr['minutes']:.1f} min, when one SD sync took {lr['sync_s']:.2f} s and the 1.5 s raw buffer "
                 "overflowed, so continuous recording ≥ 120 min was not reached (firmware-upgrades.md B11).")
sat = S[S.day == days[0]]
rec_txt = (f"{WORDS[len(sat)].capitalize()} recordings were made on {DAY_TXT[days[0]]}: "
           f"{WORDS[int(sat.complete.sum())]} complete sessions (one per participant) and "
           f"{WORDS[int((~sat.complete).sum())]} short starts ({short_starts_txt}) that were repeated within "
           f"{max(restart_min):.0f} min")
for d_ in days[1:]:
    g_ = S[(S.day == d_) & S.complete]
    rec_txt += (f", and {WORDS[len(g_)]} complete session{'s' if len(g_) != 1 else ''} on {DAY_TXT[d_]} "
                f"({', '.join(g_.subject)})")
rec_txt += f". The {NW} complete sessions"
sd_day = "the day" if not len(EXTRA) else DAY_SHORT[days[0]]
TITLE = ("MyoProBand Saturday results" if not len(EXTRA) else
         "MyoProBand results, " + " and ".join(DAY_SHORT[d_] for d_ in days))
H1 = ("MyoProBand · participant session, 26 Sep 2026" if not len(EXTRA) else
      "MyoProBand · participant sessions, " + " and ".join(DAY_SHORT[d_] for d_ in days) + " 2026")
# colocacion: no esta en los datos (el metadata trae el texto por omision del monitor); lo dijo el operador
PLACEMENT_NOTE = {"S07": "The band was placed much more distally than on 26 Sep (operator's note; the session "
                         "metadata keeps the monitor's default \"proximal third\" text)."}


def extra_protocol(s_):
    tm = meta.get(full.loc[s_, "session"], {}).get("timing", {})
    dn = DN.get(s_)
    return (f"same timing ({tm.get('hold_s', 5)} s hold, {tm.get('rest_s', 7)} s rest, {tm.get('break_s', 30)} s "
            f"breaks) with a {full.loc[s_, 'lead_in']:.0f} s lead-in before the first cue, its own grasp order (seed "
            f"{int(full.loc[s_, 'seed'])}, per participant)"
            + (f", and contact load and temperature read about {dn['rate']:.0f} times per second "
               f"(~{dn['n']} readings per hold)" if dn else "") + ". " + PLACEMENT_NOTE.get(s_, ""))


note_extra = "".join(f"<br><b>{DAY_SHORT[full.loc[s_, 'day']]} ({s_})</b>: {extra_protocol(s_)}" for s_ in EXTRA.index)


def extra_section():
    """La sesion de otro dia (S07): que cambio, como salio y sus dos figuras."""
    if not len(EXTRA):
        return ""
    out = []
    for s_ in EXTRA.index:
        r = full.loc[s_]
        c = C[C.subject == s_].set_index("channel")
        act = c.act_db_med.sort_values(ascending=False)
        resp = [f"{ch} {v:.1f} dB" for ch, v in act.items() if v >= 6]
        flat = [ch for ch in sorted(c.index) if c.loc[ch, "act_db_med"] < 3]
        others_mv = float(np.median(C[C.subject != s_].base_rms_mv))
        # en reposo: muy por encima del resto (> 4 veces) suena a mal contacto; el resto solo es mas alto
        hi = c[c.base_rms_mv > 24].base_rms_mv
        bad, mid = hi[hi > 4 * others_mv], hi[hi <= 4 * others_mv]
        parts = [f"{ch} read {v:.0f} mV ({v / others_mv:.0f}× the median of the other sessions, "
                 f"{others_mv:.0f} mV), probably poor contact" for ch, v in bad.items()]
        if len(mid):
            parts.append(f"{', '.join(sorted(mid.index))} read {mid.min():.0f}–{mid.max():.0f} mV")
        noisy_txt = ("; at rest " + "; ".join(parts)) if parts else ""
        pick = [ch for ch in sorted(whole.loc[s_].index) if whole.loc[(s_, ch)] > 8]
        dn = DN.get(s_)
        f_, l_ = first_last(s_)
        g26 = GRASP_SHORT.get(int(TR0[(TR0.session == r.session) & (TR0.trial == 26)].grasp.iloc[0]), "")
        g26 = g26.lower().replace(" ", "-")                  # "power-sphere holds"
        g4 = blocks_of(s_)[4].lower().replace(" ", "-")
        out.append(
            f"<h2>The {DAY_TXT[r.day]} session ({s_})</h2>"
            f"<p><b>Protocol.</b> 7 grasps × 6 repetitions as on {DAY_SHORT[days[0]]}, {extra_protocol(s_)} "
            f"Order: {f_} first, {l_} last.</p>"
            f"<p><b>Integrity</b> was complete: {r.span_s / 60:.1f} min, full-rate data from "
            + (f"{r.preview_s:.1f} s" if r.preview_s > 0 else "the start (no 1 Hz preview)")
            + f", {int(r.raw_missing)} samples short over the eight channels with no gap, {int(r.trials)}/42 trials "
            f"({int(r.trials_full_rate)} at full rate), {int(r.sweeps_99pts)}/{int(r.sweeps)} complete sweeps"
            + (", valid skin temperature" if TT.loc[s_, "valid"] else ", no skin temperature") + ".</p>"
            f"<p><b>sEMG</b> was weaker than on {DAY_SHORT[days[0]]}: {WORDS[len(resp)]} of eight channels exceeded "
            f"6 dB ({', '.join(resp)}); {', '.join(flat)} stayed below 3 dB"
            + noisy_txt
            + (f"; the sweep reached {', '.join(pick)} in the breaks" if pick else "")
            + ". The distal placement is the likely reason for the flat channels.</p>"
            + (f"<p><b>Auxiliary readings.</b> p1 (flexor side) read {dn['p1']:.2f} kPa throughout, i.e. it carried no "
               f"load, and p2 {dn['p2_lo']:.0f}–{dn['p2_hi']:.0f} kPa (provisional calibration). Within a hold, "
               f"temperature varied by {dn['T_sd']:.2f} °C (SD; resolution {T_STEP:.2f} °C) and p2 by "
               f"{dn['p2_sd']:.1f} kPa, so the denser readings mainly show the sensors' scatter, which three readings "
               f"per hold could not. In the last block ({dn['last_grasp']}) temperature went from {dn['T_prev']:.1f} "
               f"to {dn['T_last']:.1f} °C and p2 from {dn['p2_prev']:.0f} to {dn['p2_last']:.0f} kPa together, as if "
               "the band had seated more firmly.</p>" if dn else "")
            + f"<figure>{img('fig_s26_2_multimodal_' + s_ + '_26')}<figcaption><b>{s_} as in Figure 2.</b> (a) The "
            f"whole session. (b) 23 s around two {g26} holds (the box marked detail in a). E8, E5 and E6 carry "
            "the activity; E1 and E2 burst only during the sweep after each release; the contact load and temperature "
            "rows are dense enough to read as lines.</figcaption></figure>"
            + f"<figure>{img('fig_s26_7_showcase_' + s_ + '_4_3')}<figcaption><b>{s_} as in Figure 7.</b> (a) The "
            f"{g4} block on one time axis. (b) The third repetition of each grasp with its activation ring. "
            "The flexor-side load (p1) stays near zero throughout.</figcaption></figure>")
    return "".join(out)

captions = {
    "fig_s26_7_showcase": "<b>MyoProBand acquiring each variable when the protocol asks for it (participant S01).</b> "
        "(a) One complete block (six repetitions of a power-sphere grasp) on a single time axis: continuous 8-channel "
        "sEMG (20–450 Hz, ADC input) and gyroscope; contact load and skin temperature read every 2 s during holds only "
        "(temperature: all readings, joined within each hold, never across rests; contact load: the last reading of "
        "each hold, the only one free of the auxiliary firmware's median filter; p2 provisionally calibrated); and one "
        f"bioimpedance sweep ({freq[0]/1e3:.0f}→{freq[-1]/1e3:.0f} kHz, {len(freq)} points, {SWEEP_S:.1f} s) after each "
        "release, each point at its estimated excitation time (sweep onset plus the measured step duration). Tints: "
        "hold (grasp colour), sweep (orange). (b) The third repetition of each of the seven grasps on shared scales; "
        "rings show the median activation of the eight electrodes around the forearm over the six repetitions (E1, over "
        "the flexors, at top; E2–E8 follow in numbering order).",
    "fig_s26_1_integrity": f"<b>Data availability by modality for the {NW} participants, in recording order.</b> Holds "
        "coloured by grasp; sEMG (8 raw + 8 envelope channels) with the 1 Hz start-up preview in grey and dropouts in red; "
        "IMU with gaps > 0.25 s as breaks; impedance sweeps hatched; contact-load and temperature samples as ticks "
        "(red: a sensor at 0 kPa). Right: per-session integrity.",
    "fig_s26_2_multimodal": "<b>A complete session (S02).</b> (a) Task, sEMG activation per channel (RMS in 250 ms, dB relative "
        f"to rest), impedance magnitude per sweep, contact load, skin temperature and gyroscope over "
        f"{full.loc['S02', 'span_s'] / 60:.1f} min. (b) 23 s around "
        "two holds (the box marked detail in a): raw sEMG (offset removed), hardware envelope, contact load (last "
        "reading of each hold; p2 provisionally calibrated), gyroscope and the sweep points placed at their "
        "estimated excitation time. E2 (orange) picks up the sweep at 10–30 kHz and its envelope saturates.",
    "fig_s26_3_timing": f"<b>Sampling and alignment.</b> (a) sEMG inter-sample intervals, all eight channels "
        f"({100 - long_pct:.0f} % near {TD.short_us_med.median() / 1000:.2f} ms; {long_pct:.0f} % in a second mode, the "
        f"rounds that also convert the envelopes; mean {TM.raw_dt_mean_ms.mean():.3f} ms). (b) Spread of one 8-channel frame against the ≤ 1 ms target. (c) IMU intervals (polled, "
        "stamped at read time, catch-up bursts). (d) Arrival of auxiliary data at the host relative to the task event "
        "(the bracelet relays auxiliary lines on a 1 s heartbeat).",
    "fig_s26_8_timing": "<b>How the bracelet samples.</b> (a) Every conversion of the 16 channels over 26 ms (S01), "
        f"grouped by ADC. Within an ADC the order is fixed (second raw channel +{own_adc:.2f} ms; every 20th round the two "
        f"envelope channels, which stretches that round to {long_ms:.2f} ms); the four ADCs run at independent phases. (b) E1 intervals. "
        "(c) Error of each timestamp against a uniform 1 kHz grid. (d) A 100 Hz tone sampled at the real timestamps, "
        "analysed as if uniform, and after linear and cubic resampling on the timestamps. (e) Largest spur against tone "
        f"frequency, relative to the tone (median and range over {NW} sessions).",
    "fig_s26_9_link": "<b>Link, latency and data budget.</b> (a) Effective rate in 10 s windows over each session: raw "
        f"sEMG ({WORDS[n_off_win]} window{'s' if n_off_win != 1 else ''} off-scale at the S02 dropout) and IMU (nominal 200 Hz polling). (b) Device clock against the PC "
        "clock. (c) Delivery delay to the PC above the fastest delivery of each session (not absolute latency). (d) From "
        "the rest command logged on the PC to the sweep onset seen in the sEMG (includes the delivery delay); none in S02, "
        "where the switch-on transient reached a single channel and the onset detector requires two. (e) Storage "
        "per minute of recording by stream; the SD card stores the same 8-byte records.",
    "fig_s26_10_instrument": "<b>Instrument characteristics.</b> (a, b) Zero-lag correlation between the eight raw "
        "channels (20–450 Hz) during holds and in relaxed rest, averaged over participants. (c) Mean correlation by "
        "distance around the forearm (range over participants): flat, which points to common-mode interference rather "
        "than crosstalk; part of it is the 9.766 Hz line family. (d) Lag of the hardware envelope behind the raw RMS at grasp onset (channels with ≥ 8 dB "
        "activation and no sweep pickup). (e) Impedance at 5 kHz against resting sEMG noise, one point per trial.",
    "fig_s26_4_sweeps": "<b>What an impedance sweep leaves in the sEMG, measured in relaxed breaks.</b> (a) RMS change over "
        "the whole sweep per participant and channel. (b) By excitation frequency. (c) Hardware envelope during sweeps vs "
        "during holds (p95 above rest). (d) By band, as in the bench analysis.",
    "fig_s26_5_semg": "<b>sEMG quality in participants.</b> (a) Hold activation (median over 42 trials, dB relative to the "
        "resting baseline). (b) Resting noise floor at the ADC input. (c) Power spectral density during holds and "
        "at rest, relative to the resting level at 300–400 Hz (analog 60 Hz notch visible). (d) Resting power spectral "
        "density at 0.24 Hz resolution, relative to its median: lines at multiples of 9.766 Hz (1/102.4 ms, most likely "
        "the Wi-Fi beacon). "
        "(e) Hardware envelope rise vs raw activation, trial by trial.",
    "fig_s26_6_impedance": f"<b>Skin–electrode impedance.</b> (a) The {'/'.join(str(v) for v in sorted(set(full.sweeps.astype(int))))} "
        "sweeps of each session coloured by time. "
        "(b) Change of |Z| at 5 kHz from the first sweep. (c) |Z| at 100 kHz: the same floor in every participant.",
}
captions["fig_s26_11_context"] = (
    "<b>Context variables over the session.</b> (a) Contact load per trial (the last reading of each hold, the only one "
    "free of the auxiliary firmware's median filter), flexor side (p1, filled) and extensor side (p2, open; provisional "
    "calibration), coloured by the grasp of the block; × readings at 0 kPa. (b) Flexor-side load (p1) by grasp relative "
    f"to each participant's median. {order_txt}, "
    f"so {'' if same_order else 'within ' + span(main_subs) + ' '}grasp and time in session are confounded. "
    f"(c) Skin temperature ({WORDS[int(TT.valid.sum())]} sessions with a sensor; readings joined "
    "only within a hold) and IMU die temperature. (d) Gyroscope RMS around grasp onset, during the hold, around release "
    "and at rest (median and IQR over 42 trials), and how much the band's inclination changed between holds.")
captions["fig_s26_13_envlag"] = (
    "<b>How the hardware envelope lag is measured.</b> One hold with lags at the median. (a) Raw sEMG band-passed "
    "20–450 Hz and its centred 50 ms moving RMS; orange: median level in each shaded window (rest, plateau, rest after "
    "release). (b) Hardware envelope (50 Hz) with the same windows. (c, d) Both normalised to their own rest (0) and "
    "plateau (1): the lag is the distance between the 50 % crossings, each required to hold for 250 ms (bars). (e) "
    "Cross-correlation over the whole hold: the shift that best aligns the envelope with the raw RMS. (f) All "
    f"qualifying holds: {lag_on.median():.0f} ms at the grasp, {lag_off.median():.0f} ms at the release and "
    f"{lag_x.median():.0f} ms by cross-correlation (IQR {lag_x.quantile(0.25):.0f}–{lag_x.quantile(0.75):.0f} ms). The "
    "longer lag at the grasp comes from the envelope's dead zone (Figure 5e). A first version took the first 50 % "
    "crossing without requiring it to hold; plateau dips then triggered most release crossings before the cue.")
captions["fig_s26_14_wifi"] = (
    "<b>The Wi-Fi beacon interference and how to remove it.</b> (a) Rest samples folded on the 102.4 ms beacon period "
    "of the device clock: one dip of ~3 ms on all eight channels at once, locked to the device clock. (b) A rest "
    f"segment, original and with the samples inside the dip window repaired by interpolation ({rep_frac:.1f} % of "
    f"samples). (c) Resting power spectral density of {NCH} channels: original, comb notch and repaired; the small line "
    "left at 46 Hz is the UDP data-packet rate (8000 records/s ÷ 174 per packet). (d) Per method: resting power left "
    f"in the lines (original {wifi_stats['original']['lines']:.1f} %, notch {wifi_stats['notch']['lines']:.1f} %, "
    f"template {wifi_stats['template']['lines']:.1f} %, repaired {wifi_stats['repair']['lines']:.1f} %) and change "
    f"of the hold RMS ({wifi_stats['notch']['hold_db']:+.2f}, {wifi_stats['template']['hold_db']:+.2f} and "
    f"{wifi_stats['repair']['hold_db']:+.2f} dB). On a bench board without the sEMG modules (SD captures of 28 Sep, "
    "~9 mV input noise) the dip does not appear with the radio on or off.")
captions["fig_s26_15_channels"] = (
    "<b>Channel gain and how to display the eight channels.</b> (a) Resting floor without the Wi-Fi dips, relative to "
    f"each session's median channel: E3 {GAIN.loc['E3', 'rest_floor_rel_median']:.2f}×, E4 "
    f"{GAIN.loc['E4', 'rest_floor_rel_median']:.2f}× (E3/E4 = "
    f"{GAIN.loc['E3', 'rest_floor_rel_median'] / GAIN.loc['E4', 'rest_floor_rel_median']:.2f}×). (b) Depth of the beacon "
    "dip per channel: nearly equal, so it enters after the module gains; the estimate in (a) is therefore a lower "
    "bound on the gain ratio. (c–e) One hold in mV, in multiples of each channel's resting RMS, and with each channel "
    "on its own scale.")
order = ["fig_s26_7_showcase", "fig_s26_1_integrity", "fig_s26_2_multimodal", "fig_s26_3_timing",
         "fig_s26_8_timing", "fig_s26_9_link", "fig_s26_4_sweeps", "fig_s26_5_semg", "fig_s26_14_wifi",
         "fig_s26_15_channels", "fig_s26_10_instrument", "fig_s26_13_envlag", "fig_s26_6_impedance",
         "fig_s26_11_context"]

tim_html = "".join(
    f"<tr><td>{s_}</td><td>{TD.loc[s_, 'short_us_med']:.0f} / {TD.loc[s_, 'long_us_med']:.0f}</td>"
    f"<td>{100*TD.loc[s_, 'long_frac']:.2f} (every {TD.loc[s_, 'long_period_mode']})</td>"
    f"<td>{TD.loc[s_, 'err_p2p_ms']:.2f}</td><td>{imu_med[s_]:.0f}</td><td>{TD.loc[s_, 'clock_ppm']:+.1f}</td>"
    f"<td>{TD.loc[s_, 'delay_p50_ms']:.0f} / {TD.loc[s_, 'delay_p95_ms']:.0f}</td>"
    f"<td>{('%.0f' % TD.loc[s_, 'cmd_lat_med_ms']) if np.isfinite(TD.loc[s_, 'cmd_lat_med_ms']) else 'n/a'}</td>"
    f"<td>{BU.loc[s_, 'total_mb_min']:.2f}</td></tr>" for s_ in SUBJECT_ORDER)

ins_html = "".join(
    f"<tr><td>{s_}</td><td>{neigh('hold', s_):.2f}</td><td>{neigh('rest', s_):.2f}</td>"
    f"<td>{neigh('restnb', s_):.2f}</td><td>{far('rest', s_):.2f}</td>"
    # S07 no tiene sostenes que califiquen (canales activos con captacion del barrido)
    f"<td>{('%.0f' % LG[LG.subject == s_].lag_on_ms.median()) if (LG.subject == s_).any() else 'n/a'} "
    f"(n={len(LG[LG.subject == s_])})</td>"
    f"<td>{rho_zt[s_]:+.2f}</td><td>{rho_if[s_]:+.2f}</td></tr>" for s_ in SUBJECT_ORDER)
# separabilidad: solo en la seccion final (secundaria), con la figura 12
CLs = CL.set_index("subject")
cls_html = "".join(
    f"<tr><td>{s_}</td><td>{100*CLs.loc[s_, 'acc_window']:.1f}</td><td>{100*CLs.loc[s_, 'acc_trial']:.1f}</td>"
    f"<td>{100*CLs.loc[s_, 'acc_norm_window']:.1f}</td><td>{100*CLs.loc[s_, 'acc_rest_window']:.1f}</td>"
    f"<td>{100*CLs.loc[s_, 'acc_blockorder_window']:.1f}</td><td>{int(CLs.loc[s_, 'n_windows'])}</td></tr>"
    for s_ in SUBJECT_ORDER)
fig_html = "".join(f"<figure>{img(k)}<figcaption>{captions[k]}</figcaption></figure>" for k in order)

# ------------------------------------------------------------------ texto de resultados (borrador)
res = f"""
<h3>3.4.4 Data integrity and local storage</h3>
<p>{rec_txt} lasted {span_rng} min
({fr_min:.1f} min of full-rate data). Every one of the 42 programmed trials was performed in {trials_all}; {trials_fr}
of {int(full.trials.sum())} holds were covered at full rate (the exceptions: {low_txt}). sEMG continuity was complete in
{len(full) - len(drop)} sessions{tiny_txt}; {drop_who_txt} for {'all' if not len(tiny) else f'{int(drop.raw_missing.sum()):,} of the'}
{int(full.raw_missing.sum()):,} missing samples ({loss_all:.3f} % pooled). {sweeps_txt} The host copy matched its own
sample, IMU-sample and event counters in {wal_txt}. The SD card, however, held one set of files (one recording)
instead of eight at the end of {sd_day}. The cause was not established: in this firmware an SD error disables the card
until the next boot while acquisition continues over the link, and the status LED did not show an SD fault during the
day. Local storage therefore did not meet its requirement in this study.{bench_res}</p>
<h3>3.4.6 Power and thermal behaviour</h3>
<p>Battery state was not logged. The IMU die temperature, a proxy for the temperature inside the device, rose by
{np.median(TT.imu1 - TT.imu0):.1f} °C (median; range {(TT.imu1 - TT.imu0).min():+.1f} to
{(TT.imu1 - TT.imu0).max():+.1f} °C) over a session.</p>
<h3>3.5.1 Session completion and acquisition reliability</h3>
<p>See Figure 1 and the per-session table. Raw sEMG was delivered at {full.raw_hz.min():.1f}–{full.raw_hz.max():.1f}
Hz per channel, the envelope at 50 Hz and the IMU at {imu_hz[0]:.0f}–{imu_hz[1]:.0f} Hz. The eight channels of one
frame were sampled within {spread_med:.2f} ms (median; p95 {TM.spread_p95_ms.min():.2f}–{TM.spread_p95_ms.max():.2f}
ms by session). Host delivery jitter was
{jit[0]:.1f}–{jit[1]:.1f} ms.</p>
<h3>3.5.2 sEMG in participants</h3>
<p>Referred to a resting baseline taken in the 30 s breaks, the steady part of each hold raised the sEMG RMS by a median
of {max(0.0, C.act_db_med.min()):.0f}–{C.act_db_med.max():.0f} dB per channel (90th percentile up to
{C.act_db_p90.max():.0f} dB); {int((C.act_db_med >= 6).sum())} of {NCH} participant-channels
exceeded 6 dB. The resting noise floor was {np.median(C.base_rms_mv):.1f} mV at the ADC input
({np.median(C.base_rms_mv)/2:.1f} LSB); {len(noisy)} channels were noisier ({noisy.base_rms_mv.min():.0f}–{noisy.base_rms_mv.max():.0f}
mV). Median frequency during holds was
{C.mdf_med.min():.0f}–{C.mdf_med.max():.0f} Hz. Resting windows within trials sat within ±{rest_dev:.1f} dB of the break
baseline. Two integration effects are visible at rest: pickup of the impedance excitation on specific channels
(Figure 4) and lines at multiples of 9.766 Hz, most likely from the Wi-Fi beacon ({W.comb_power_pct.median():.0f} % of resting
power, Figure 5d). The analog envelope responded only above ~{env_thr:.1f} dB of raw activation (Figure 5e).</p>
<h3>3.5.3 Auxiliary modalities</h3>
<p>Impedance magnitude at 2 kHz ranged {ZZ.z2_0.min():.0f}–{ZZ.z2_0.max():.0f} kΩ at the start of the sessions and
changed over the session by {ZZ.z5_change.min():+.0f} to {ZZ.z5_change.max():+.0f} % at 5 kHz; above the knee all
participants converged on the same {z100_all.min():.2f}–{z100_all.max():.2f} kΩ floor. Contact load, taken from the
last reading of each hold (the auxiliary firmware median-filters readings across holds), differed between grasp blocks
by a median range of {PP.between_kpa.median():.0f} kPa; the values are relative to the tare taken when the auxiliary
board boots, and the p2 calibration is provisional. Skin temperature was available in {WORDS[int(TT.valid.sum())]}
sessions and rose {np.nanmedian(TT.t1 - TT.t0):.1f} °C (median) over a session; the sensor was not detected in the
other {WORDS[int((~TT.valid).sum())]}. The IMU showed a still forearm during the trials (gyroscope RMS
{imu.gyr_rest_rms.median():.2f} °/s at rest, {imu.gyr_onset_rms.median():.2f} °/s around grasp onset), although the
band's inclination changed between holds by up to {posture_max.max():.0f}° ({incl_txt}), i.e. the forearm or the band
was repositioned at some point.</p>
<h3>3.4.1–3.4.3 (participant data) How the bracelet samples</h3>
<p>Each ADS1015 converts its two raw channels in single-shot mode every {TD.short_us_med.median()/1000:.2f} ms, the
second {own_adc:.2f} ms after the first; every 20th round it also converts its two envelope channels, and that round
takes {TD.long_us_med.median()/1000:.2f} ms. In all {NW} sessions this pattern repeated exactly every 20 samples
({long_pct:.2f} % of intervals) and the effective rate was {rate_ok:.1f} Hz per channel in every 10 s window except the
{WORDS[n_off_win]} containing the S02 dropout. The four ADCs (two per I2C bus) run at independent phases, which gives the 8-channel frame spread of
{spread_med:.2f} ms (median). Because every sample carries its own device timestamp, this skew and the periodic
2.5 ms gap are known exactly. Analysed as if uniformly sampled, the stream carries a 50 Hz sawtooth timing error of
{TD.err_p2p_ms.min():.1f}–{TD.err_p2p_ms.max():.1f} ms p–p, which turns a 100 Hz tone into a spectrum with a spur
at {spur.loc[100, 'spur_dbc']:.0f} dB relative to the tone; resampling on the timestamps lowers it to
{spur.loc[100, 'resampled_spur_dbc']:.0f} dB (linear) and {spur.loc[100, 'cubic_spur_dbc']:.0f} dB (cubic), but
above ~200 Hz the 2.5 ms gap limits any reconstruction ({spur.loc[250, 'cubic_spur_dbc']:.0f} dB at 250 Hz;
Figure 8). The IMU, polled at a nominal 200 Hz, delivered {min(imu_med.values()):.0f}–{max(imu_med.values()):.0f} Hz.</p>
<h3>3.4.5 Link, latency and data volume</h3>
<p>The device clock ran +{TD.clock_ppm.min():.0f} to +{TD.clock_ppm.max():.0f} ppm fast relative to the PC
({TD.clock_ppm.min()*3.6:.0f}–{TD.clock_ppm.max()*3.6:.0f} ms per hour), absorbed by a per-session linear mapping
(residual {jit[0]:.1f}–{jit[1]:.1f} ms). Data reached the PC with a delivery delay that varied by a median of
{TD.delay_p50_ms.min():.0f}–{TD.delay_p50_ms.max():.0f} ms (p95 {TD.delay_p95_ms.min():.0f}–{TD.delay_p95_ms.max():.0f}
ms) above the fastest delivery; the absolute latency could not be measured without a common clock, but the firmware
fills a UDP packet with 174 raw records (21.8 ms of data) or flushes it after 30 ms. From a rest command logged on the
PC to the start of the impedance excitation visible in the sEMG took a median of {np.median(lat_all):.0f} ms
(n = {len(lat_all)}). One minute of recording occupied {bud.total_mb_min:.2f} MB ({bud.total_mb_min*60/1000:.2f} GB/h;
raw sEMG {100*bud.raw_mb_min/bud.total_mb_min:.0f} %), on the PC and on the SD card alike (same 8-byte records), and
{bud.total_mb_min * 8 / 60:.2f} Mbit/s of payload over UDP. The 32-bit microsecond timestamps wrap after
{wrap_min:.1f} min.</p>
<h3>Instrument characteristics</h3>
<p>During holds the eight raw channels were nearly independent (neighbouring channels r = {np.mean(r_hold):.2f}). At
relaxed rest they shared a common component (r = {np.mean(r_rest):.2f} for neighbours, {np.mean(r_far_rest):.2f} for
opposite electrodes; {np.mean(r_restnb):.2f} after removing the 9.766 Hz lines), which points to interference rather
than crosstalk. The hardware envelope trailed the raw RMS by {lag_x.median():.0f} ms (cross-correlation over each hold,
IQR {lag_x.quantile(0.25):.0f}–{lag_x.quantile(0.75):.0f} ms); its 50 % crossing lagged by {lag_on.median():.0f} ms at the grasp and
{lag_off.median():.0f} ms at the release (Figure 13). Impedance measured at the separate impedance electrodes did
not track the resting sEMG noise within sessions (Spearman ρ between {min(rho_if.values()):+.2f} and
{max(rho_if.values()):+.2f}), although it changed markedly over time.</p>
<h3>3.6.1–3.6.3 Multimodal behaviour</h3>
<p>Figure 7 (S01, the most complete session) shows the acquisition schedule used here: sEMG and IMU continuous, contact
load and temperature during holds, one impedance sweep after every release, so that no hold was exposed to excitation
except the first hold of {ov_n} sessions, which overlapped the auxiliary board's start-of-test sweep. The schedule is
driven by the host's grasp and rest commands, so it follows the protocol the host runs; in this firmware each rest
command always triggers a sweep and each grasp command the 2 s sensor readings, and making that mapping
configurable (sweep every N rests, on grasp change or on demand) is requested in firmware-upgrades.md A5. Within a session, impedance settled (Figure 6b), skin temperature
rose under the band, and contact load changed between grasp blocks (confounded with time in session:
{order_short}).</p>
"""

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>
:root {{ --ink:#1f2328; --ink2:#57606a; --grid:#d8dee4; --bg:#ffffff; --accent:#eb6834; }}
body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; color:var(--ink); background:var(--bg);
       max-width: 980px; margin: 0 auto; padding: 16px; line-height: 1.5; }}
h1 {{ font-size: 1.5rem; margin-bottom: 0.2rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid var(--grid); }}
h3 {{ margin-bottom: 0.2rem; }}
.sub {{ color: var(--ink2); }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; display: block; overflow-x: auto; }}
th, td {{ border-bottom: 1px solid var(--grid); padding: 4px 6px; text-align: left; vertical-align: top; }}
th {{ color: var(--ink2); font-weight: 600; }}
td.st {{ white-space: nowrap; font-weight: 600; }}
td.nt {{ font-size: 0.85em; color: var(--ink2); }}
figure {{ margin: 1.5rem 0; }} figure img {{ width: 100%; height: auto; border: 1px solid var(--grid); }}
figcaption {{ font-size: 0.85rem; color: var(--ink2); margin-top: 0.4rem; }}
li {{ margin-bottom: 0.6rem; }} .fix {{ color: var(--accent); font-weight: 600; }}
.note {{ background: #f6f8fa; border-left: 3px solid var(--accent); padding: 8px 12px; }}
</style></head><body>
<h1>{H1}</h1>
<p class="sub">Technical evaluation of the multimodal armband under development. {NW.capitalize()} participants (codes only),
7 grasps × 6 repetitions each. Generated from <code>EMG8_article_analysis/data/{DATASET}</code>.</p>
<div class="note"><b>Protocol as run</b> (the draft's §2.8.3 says otherwise): 7 grasps × <b>6</b> repetitions,
<b>5 s</b> hold, 12 s per trial (5 s hold + 7 s rest; no separate preparation phase was logged), grouped by grasp in a seeded random order, <b>30 s</b>
break after each grasp; one impedance sweep at every rest command; contact load and temperature every 2 s during holds only.{note_extra}</div>
<h2>Performance requirements (article table, targets of 29 Sep) vs. measured</h2>
<table><tr><th>Requirement</th><th>Target</th><th>Measured</th><th>Status</th><th>Note</th></tr>{req_html}</table>
<p>Candidate requirements (targets proposed here, not adopted yet):</p>
<table><tr><th>Requirement</th><th>Target</th><th>Measured</th><th>Status</th><th>Note</th></tr>{cand_html}</table>
<h2>Sessions</h2>
<table><tr><th>Subj.</th><th>Start</th><th>min</th><th>preview s</th><th>sEMG Hz/ch</th><th>loss %</th>
<th>trials (full rate)</th><th>sweeps</th><th>load/temp samples</th><th>skin temp.</th><th>IMU Hz</th>
<th>host jitter ms</th></tr>{ses_html}</table>
{extra_section()}
<h2>sEMG quality per participant</h2>
<table><tr><th>Subj.</th><th>channels ≥ 6 dB</th><th>top-3 activation dB</th><th>resting RMS mV</th>
<th>median freq. Hz</th><th>repetition CV %</th><th>rest drift dB (last vs first block)</th>
<th>channels with sweep pickup</th></tr>{q_html}</table>
<h2>Auxiliary modalities per participant</h2>
<table><tr><th>Subj.</th><th>|Z| 2 kHz first→last (kΩ)</th><th>|Z| 5 kHz change %</th><th>|Z| 100 kHz</th>
<th>p1 range between blocks (kPa)</th><th>p2 range between blocks (kPa, provisional)</th><th>readings at 0 kPa</th>
<th>skin temp. °C</th><th>device (IMU) °C</th><th>gyro onset/rest °/s</th><th>inclination change between holds °</th></tr>{aux_html}</table>
<h2>Acquisition timing, link and storage per session</h2>
<table><tr><th>Subj.</th><th>raw interval short / long (µs)</th><th>long intervals %</th>
<th>error vs uniform grid, p–p (ms)</th><th>IMU Hz (median of 10 s)</th><th>device clock vs PC (ppm)</th>
<th>delivery delay p50 / p95 (ms)</th><th>rest cue → sweep onset (ms)</th><th>MB/min</th></tr>{tim_html}</table>
<h2>Instrument characteristics per participant</h2>
<table><tr><th>Subj.</th><th>r neighbours, hold</th><th>r neighbours, rest</th><th>rest, beacon removed</th>
<th>r opposite, rest</th><th>envelope lag at onset (ms)</th><th>ρ(|Z|, time)</th><th>ρ(|Z|, resting sEMG)</th>
</tr>{ins_html}</table>
<h2>Caveats of integrating the modalities, and fixes</h2>
<p>Firmware changes are itemised with evidence and acceptance tests in <code>firmware-upgrades.md</code>.</p>
<ol>{cav_html}</ol>
<h2>Reading this as a reviewer and as the special-issue editor</h2>
<h3>What the data support</h3>
<ul>
<li>Complete multimodal sessions in {NW} participants: 8 raw + 8 envelope sEMG channels at {rate_ok:.1f} Hz with
per-sample device timestamps, IMU, {n_sweeps} impedance sweeps, contact load and, in {WORDS[int(TT.valid.sum())]}
sessions, skin temperature, with {'0 % sEMG loss' if not len(tiny) else 'no sEMG gap'} in
{WORDS[len(full) - len(drop)]} of {WORDS[len(full)]} sessions.</li>
<li>Scheduling as a design feature: excitation confined to rests ({ov_total - ov_n} of {ov_total} holds free of
excitation; the other {ov_n} are the first hold of {len(ov_subjects)} sessions, fixable with a lead-in), contact load
and temperature during holds, all driven by the host's commands, which is the article's central argument (Figure 7);
the mapping from command to measurement is still fixed in the auxiliary firmware (firmware-upgrades.md A5).</li>
<li>Quantified integration effects, each with a likely cause (impedance pickup at 10–30 kHz, the 9.766 Hz lines most
likely from the Wi-Fi beacon, envelope dead zone, host-level auxiliary alignment, periodic sampling gap). The sampling
gap and the auxiliary alignment are confirmed in the firmware source; the other causes are inferred from the
recordings.</li>
</ul>
<h3>Claims to state carefully</h3>
<ul>
<li>"sEMG unaffected during integrated operation": the impedance excitation stayed out of the holds (except the first
hold of {ov_n} sessions) but reached the rests on {len(hot)} of {NCH} channels, and the 9.766 Hz lines are present
throughout; say so.</li>
<li>Interchannel spread ≤ 1 ms: met at the median ({np.median(full.frame_spread_med_us) / 1000:.2f} ms), not at
p95 ({full.frame_spread_p95_us.min() / 1000:.2f}–{full.frame_spread_p95_us.max() / 1000:.2f} ms;
{full.frame_spread_le1ms_pct.min():.0f}–{full.frame_spread_le1ms_pct.max():.0f} % of frames within 1 ms). Multimodal
alignment ≤ 100 ms: met for sEMG and task labels, not for the auxiliary modalities (stamped on arrival, up to
~{aux_first[1]:.1f} s) in this firmware.</li>
<li>Complete local storage: not met {'in this study' if not len(EXTRA) else 'on ' + DAY_SHORT[days[0]]} (the host copy was complete).</li>
<li>{claim_cont}</li>
</ul>
<h3>What a reviewer will ask for</h3>
<ul>
<li>Total analog gain, to express the resting noise ({np.median(C.base_rms_mv):.1f} mV at the ADC input) in µV referred
to input.</li>
<li>Participant summary (age, sex, forearm circumference, tested arm) as aggregate statistics, from the subject
sheet, without names.</li>
<li>The protocol as run (6 repetitions, 5 s hold, 30 s breaks) and the two restarted sessions.</li>
<li>Test–retest (one session per participant here) and movement conditions (the IMU shows a still forearm; state
from the protocol whether it was supported).</li>
<li>Order effects: {order_review}</li>
</ul>
<h3>Worth adding for readers who would build or use such a device</h3>
<ul>
<li>A short design-guidelines table: schedule active excitation outside analysis windows; characterise radio lines;
timestamp every sample and every auxiliary record on one clock; check sensor presence before and during sessions;
never rely on a single storage path; sweep only the informative band (e.g. 2–30 kHz: {int(short_band.sum())} of
{len(freq)} points, {short_s:.1f} s instead of {SWEEP_S:.1f} s).</li>
<li>The sampling anatomy (Figure 8) and data budget (Figure 9e) as a data-description section, with the
recommendation to resample on the timestamps.</li>
<li>For the special issue: Figure 7 works as the graphical abstract; a data and code availability statement
(anonymised recordings in the WAL format, analysis scripts).</li>
</ul>
<h2>Figures</h2>
<p class="sub">The figures carry no panel letters; (a), (b)… in the captions refer to the panel files <code>figures/{DATASET}/panels/&lt;figure&gt;_a</code>, <code>_b</code>… (PNG, SVG, PDF), each exported on its own at the same scale. Versions without the IMU: <code>python scripts/s26_run_all.py{'' if DATASET == 's26' else ' --dataset ' + DATASET} --no-imu</code> → <code>figures/{DATASET}_no_imu/</code>.</p>{fig_html}
<h2>Draft results text</h2>{res}
<h2>Secondary: grasp separability (models)</h2>
<p>A shallow check of the information the eight channels carry, not part of the technical evaluation. A grasp
classifier (LDA on Hudgins features, leave-one-repetition-out) reached {100*CL.acc_window.mean():.1f} % per window and
{100*CL.acc_trial.mean():.1f} % per trial (chance 14.3 %). Normalised to each trial's own pre-cue rest it reached
{100*CL.acc_norm_window.mean():.1f} %, while rest windows were classified by block at {100*CL.acc_rest_window.mean():.1f} %.
With labels from the wrong ordering hypothesis it fell to chance ({100*CL.acc_blockorder_window.mean():.1f} %),
which confirms the reconstructed grasp order. Results are kept in <code>data/{DATASET}/classify.csv</code> and
<code>confusion.npz</code>. The paragraph can serve as a short secondary subsection at the end of the results.</p>
<table><tr><th>Subj.</th><th>window %</th><th>trial %</th><th>normalised to own rest, window %</th>
<th>control: rest windows by block %</th><th>control: wrong-order labels %</th><th>windows</th></tr>{cls_html}</table>
<figure>{img("fig_s26_12_models")}<figcaption><b>Grasp separability, a secondary metric.</b> (a) Accuracy per
participant: per 250 ms window and per trial (majority vote), raw features and features normalised to each trial's own
pre-cue rest. Two controls: rest windows labelled by their block, and labels taken from the wrong ordering hypothesis.
Linear discriminant analysis with shrinkage on the Hudgins time-domain set per channel (log mean absolute value, log
waveform length, zero crossings, slope-sign changes; 32 features), leave-one-repetition-out (six folds) within each
participant. (b) Per-trial confusion pooled over the {NW} participants, in presentation order. Repetitions of a grasp were
consecutive and {order_short}, so these figures are an upper bound on within-session
separability, not a generalisation estimate.</figcaption></figure>
</body></html>"""
out = OUT_HTML
out.write_text(html, encoding="utf-8")
print("ok", out, f"{out.stat().st_size/1e6:.1f} MB")
print(Q.round(1).to_string())
print(PP.round(1).to_string())
print(TT.round(1).to_string())
