"""Synthetic session folders with the same on-disk structure as the real ones.

Used only by selftest.py, so the pipeline can be exercised without the
participant data. Nothing here is a model of physiology: the signals are noise
plus gated bursts and sweep transients, and exist to give the loaders, the
segmentation and the sweep-onset detector something with the real structure:

  * raw.bin    8-byte Sample records, four ADCs at independent phases, raw pins at 0.913 ms
               rounds (second pin +0.46 ms), every 20th round also the two envelope pins and
               a 2.48 ms round; a 1 Hz start-up preview; one dropout. Channel pins follow
               records.RAW_CH_BY_ADC / ENV_CH_BY_ADC.
  * imu.bin    20-byte records at ~170 Hz with jitter, mg and 0.1 deg/s.
  * events.jsonl  phase commands and acknowledgements, a 10 s polling echo of the current
               phase, labels; each event carries wall time and the last device ts_us seen.
  * aux.jsonl  contact load / temperature readings and 99-point impedance sweeps, stamped with
               host time only (they arrive 0.3-1.1 s after they happen).
  * metadata.json  as written by the monitor (seed, order, grasps, reps, timing, sequence).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SAMPLE_DTYPE = np.dtype([("ts_us", "<u4"), ("adc", "u1"), ("ch", "u1"), ("value", "<i2")])
IMU_DTYPE = np.dtype([("ts_us", "<u4"), ("ax", "<i2"), ("ay", "<i2"), ("az", "<i2"),
                      ("gx", "<i2"), ("gy", "<i2"), ("gz", "<i2"), ("temp100", "<i2"), ("_pad", "<u2")])
RAW_CH_BY_ADC = ((0, 3), (1, 3), (0, 3), (1, 3))
ENV_CH_BY_ADC = ((1, 2), (0, 2), (1, 2), (0, 2))
GRASPS = [25, 37, 22, 31, 27, 30, 40]
SPIKE_CHANNELS = (0, 1, 6, 7)          # E1, E2, E7, E8


def _timeline(n_trials: int, reps: int, lead_in: float, hold: float, rest: float, break_rest: float,
              break_len: float, rng):
    """Returns trials [(t_grasp, t_rest, t_next)], prest times, breaks."""
    t = lead_in
    trials, prest = [], []
    for i in range(n_trials):
        tg, tr = t, t + hold
        prest.append(tr)
        if (i + 1) % reps == 0 and i + 1 < n_trials:
            prest.append(tr + break_rest)             # second Prest: the break
            tn = tr + break_rest + break_len
        else:
            tn = tr + rest
        trials.append((tg, tr, tn))
        t = tn
    return trials, prest, t + rest


def make_session(root: Path, name: str, subject: str, *, n_trials=42, all_labels=True, seed=712,
                 hold=5.0, rest=7.0, dropout=None, wall0=1.7907e9, rng_seed=0) -> dict:
    rng = np.random.default_rng(rng_seed)
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    reps = 6
    grasps = GRASPS
    # the same shuffle the monitor uses; imported lazily so this file stays standalone-readable
    import s26_common as S
    seq = S.grasp_sequence(grasps, reps, seed, "grouped")[:n_trials]
    trials, prest, end = _timeline(n_trials, reps, 6.0, hold, rest, 7.0, 23.0, rng)
    sweep_on = [0.8] + [p + 0.14 for p in prest]       # start-of-test sweep + one per Prest (0.11 s after the command event)
    T = end + 1.0

    # ------------------------------------------------------------ raw.bin
    gate = np.zeros(int(T * 1000) + 10)
    for (tg, tr, _), g in zip(trials, seq):
        gate[int(tg * 1000):int(tr * 1000)] = 1.0
    gate = np.convolve(gate, np.ones(30) / 30, "same")
    gains = rng.uniform(0.4, 1.4, 8)
    parts_ts, parts_adc, parts_pin, parts_chan, parts_raw = [], [], [], [], []
    for adc in range(4):
        ph = rng.uniform(0, 900)
        n_r = int(T * 1e6 / 913.0) + 50
        r = np.arange(n_r)
        period = np.where(r % 20 == 19, 2480.0, 913.0)
        starts = ph + np.cumsum(period) - period
        starts = starts[starts < T * 1e6]
        r = r[:len(starts)]
        for slot in (0, 1):
            parts_ts.append(starts + 460.0 * slot)
            parts_pin.append(np.full(len(starts), RAW_CH_BY_ADC[adc][slot]))
            parts_chan.append(np.full(len(starts), adc * 2 + slot))
            parts_raw.append(np.ones(len(starts), bool))
            parts_adc.append(np.full(len(starts), adc))
            e = r % 20 == 19
            parts_ts.append(starts[e] + 920.0 + 460.0 * slot)
            parts_pin.append(np.full(int(e.sum()), ENV_CH_BY_ADC[adc][slot]))
            parts_chan.append(np.full(int(e.sum()), adc * 2 + slot))
            parts_raw.append(np.zeros(int(e.sum()), bool))
            parts_adc.append(np.full(int(e.sum()), adc))
    ts = np.concatenate(parts_ts)
    adcs = np.concatenate(parts_adc).astype(np.uint8)
    pins = np.concatenate(parts_pin).astype(np.uint8)
    chan = np.concatenate(parts_chan)
    is_raw = np.concatenate(parts_raw)
    order = np.argsort(ts, kind="stable")
    ts, adcs, pins, chan, is_raw = ts[order], adcs[order], pins[order], chan[order], is_raw[order]
    tsec = ts / 1e6
    g = gate[np.clip((tsec * 1000).astype(int), 0, len(gate) - 1)]
    val = np.where(is_raw, 300 + rng.normal(0, 5, len(ts)) + g * 60 * gains[chan] * rng.normal(0, 1, len(ts)),
                   20 + g * 150 * gains[chan] + rng.normal(0, 1, len(ts)))
    for on in sweep_on:                               # sweep onset transient on E1, E2, E7, E8, raw only
        m = is_raw & np.isin(chan, SPIKE_CHANNELS) & (tsec >= on) & (tsec < on + 0.008)
        val[m] += 1500 * np.exp(-(tsec[m] - on) / 0.004)
    keep = np.ones(len(ts), bool)
    pre = tsec < 3.0                                   # 1 Hz start-up preview: first sample of each second
    sec = tsec.astype(int)
    key = (chan * 2 + is_raw.astype(int)) * 10 + sec
    _, first_idx = np.unique(key, return_index=True)   # ts is sorted, so first index = first sample
    first = np.zeros(len(ts), bool)
    first[first_idx] = True
    keep &= ~pre | first
    if dropout:
        keep &= ~((tsec >= dropout[0]) & (tsec < dropout[1]))
    arr = np.zeros(int(keep.sum()), dtype=SAMPLE_DTYPE)
    arr["ts_us"] = np.round(ts[keep]).astype(np.uint32)
    arr["adc"], arr["ch"] = adcs[keep], pins[keep]
    arr["value"] = np.round(val[keep]).astype(np.int16)
    arr.tofile(d / "raw.bin")

    # ------------------------------------------------------------ imu.bin
    n_imu = int(T * 170)
    t_imu = np.cumsum(rng.normal(5900.0, 1500.0, n_imu).clip(1500, 14000)) + 3000.0
    t_imu = t_imu[t_imu < T * 1e6]
    imu = np.zeros(len(t_imu), dtype=IMU_DTYPE)
    imu["ts_us"] = np.round(t_imu).astype(np.uint32)
    imu["ax"] = (30 * np.sin(t_imu / 3e6)).astype(np.int16)
    imu["ay"] = (20 * np.cos(t_imu / 4e6)).astype(np.int16)
    imu["az"] = (1000 + rng.normal(0, 4, len(imu))).astype(np.int16)
    imu["gx"], imu["gy"], imu["gz"] = (rng.normal(0, 1.5, (3, len(imu)))).astype(np.int16)
    imu["temp100"] = (2800 + (t_imu / 1e6) * 0.4).astype(np.int16)
    imu.tofile(d / "imu.bin")

    # ------------------------------------------- events.jsonl and aux.jsonl
    def wall(t):                                       # host clock vs device clock: 15 ppm, 4 ms jitter
        return wall0 + t * (1 - 15e-6) + rng.normal(0, 0.004)

    events = []

    def ev(t, payload):
        lag = rng.uniform(0.0, 0.02)
        events.append({"wall": wall(t), "mono": t + 100.0, "last_ts_us": int(1e6 * max(t - lag, 0.0)),
                       "event": payload})
    ev(0.0, ["recording", True])
    cmds = []                                         # (t, phase) real commands
    for i, ((tg, tr, tn), gr) in enumerate(zip(trials, seq)):
        if all_labels or i == 0:
            ev(tg - 2.0, ["label", gr, sum(1 for q in seq[:i + 1] if q == gr)])
        cmds.append((tg + 0.03, "grasp"))
        cmds.append((tr + 0.03, "rest"))
    for p in prest:                                   # break Prest (second of a pair)
        if not any(abs(c[0] - 0.03 - p) < 1e-6 for c in cmds):
            cmds.append((p + 0.03, "rest"))
    cmds.sort()
    for t, ph in cmds:
        ev(t, ["phase", ph])
    # polling echo every 10 s repeating the current phase (not a command)
    t_echo = 7.0                                      # after the first command, as in the real logs
    while t_echo < T:
        cur = [ph for t, ph in cmds if t <= t_echo]
        ev(t_echo, ["phase", cur[-1] if cur else "rest"])
        t_echo += 10.0
    ev(T - 0.5, ["recording", False])
    events.sort(key=lambda e: e["wall"])
    with open(d / "events.jsonl", "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")

    aux = []
    freqs = (np.arange(2, 101) * 1000.0)
    for n, on in enumerate(sweep_on, start=1):
        z = 7200.0 + 40000.0 / (1 + freqs / 15000.0) + rng.normal(0, 30, len(freqs))
        aux.append({"kind": "impedance_sweep", "wall": wall(on + 4.2 + rng.uniform(0.3, 1.0)),
                    "mono": on + 4.5 + 100.0, "mac": "AA:BB", "session": name, "id": n,
                    "freq_hz": freqs.tolist(), "z_ohm": np.round(z, 2).tolist(), "points": 99})
    aux.append({"kind": "aux_sensors", "wall": wall(0.9), "mono": 100.9, "mac": "AA:BB", "session": name,
                "p1_kpa": 0.0, "p2_kpa": 0.0, "temp_c": 34.8})
    for (tg, tr, _) in trials:
        k = 0
        while tg + 0.3 + k * 2.0 < tr:
            t = tg + 0.3 + k * 2.0
            aux.append({"kind": "aux_sensors", "wall": wall(t + rng.uniform(0.1, 1.1)), "mono": t + 100.5,
                        "mac": "AA:BB", "session": name, "p1_kpa": round(float(rng.uniform(5, 20)), 3),
                        "p2_kpa": round(float(rng.uniform(5, 20)), 3), "temp_c": 34.8 + 0.001 * t})
            k += 1
    aux.sort(key=lambda a: a["wall"])
    with open(d / "aux.jsonl", "w", encoding="utf-8") as fh:
        for a in aux:
            fh.write(json.dumps(a) + "\n")

    meta = {"protocol": "UdeA 8 - exercise C", "subject": subject, "session": 1, "arm": "right", "mode": 1,
            "seed": seed, "seed_base": seed, "seed_per_subject": False, "order": "grouped",
            "grasps": grasps, "reps": reps, "trials": 42,
            "timing": {"prepare_s": 0, "hold_s": hold, "rest_s": rest, "break_every": 6, "break_s": 30,
                       "lead_in_s": 6},
            "sequence": [[gr, 0] for gr in seq]}
    (d / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"dir": d, "n_trials": n_trials, "n_sweeps": len(sweep_on), "seq": seq, "sweep_on": sweep_on,
            "trials": trials}
