# MyoProBand: raw recordings and synchronized databases (26 Sep and 1 Oct 2026)

Raw session recordings from the MyoProBand armband (8 raw sEMG channels, 8 hardware-envelope
channels, IMU, contact load, skin temperature, bioimpedance sweeps), the databases derived from
them, and everything needed to rebuild the databases and check them against the raw files. You do
not need the recording software or its data store.

## What is in this bundle

```
raw/sessions/<folder>/         the recordings exactly as the monitor wrote them
                               raw.bin  imu.bin  events.jsonl  aux.jsonl  metadata.json  [qc.json]
databases/
  raw_1khz/<tag>_raw_1khz.csv.gz        raw sEMG on a 1 kHz grid + all auxiliary variables
  envelope_50hz/<tag>_env_50hz.csv.gz   hardware envelope on a 50 Hz grid + the same auxiliary variables
  sweeps/<tag>/<tag>_sweep_NN.csv       one file per impedance sweep (99 points each)
  sessions_index.csv                    every session folder: complete or not, rows, checks, SHA-256
  participants.csv                      folder code <-> participant label
analysis_scripts/              the article's analysis code (one small path change in common.py, see REPRODUCE_REPORT.md)
tools/                         build_databases.py, verify_databases.py, selftest.py, make_bundle.py
reference/                     the report and figures the analysis produced, for comparison
SHA256SUMS.txt                 checksum of every file in the bundle
```

`<tag>` is the participant label plus the folder's start date and time, for example
`S01_20260926_113055`, so two sessions of one participant never share a file name.

## Quick start

```sh
pip install -r requirements.txt
python tools/verify_databases.py --sessions raw/sessions --databases databases   # checks the databases against raw.bin
python tools/build_databases.py  --sessions raw/sessions --out my_databases       # rebuild; output is byte-identical
python tools/check_checksums.py                                                   # integrity of the bundle (or: sha256sum -c SHA256SUMS.txt)
python tools/selftest.py                                                          # code check on synthetic sessions
```

```python
import pandas as pd
d = pd.read_csv("databases/raw_1khz/S01_20260926_113055_raw_1khz.csv.gz")
d[d.phase == "grasp"].groupby(["trial", "grasp_name"]).raw_E1.std()
```

## Participants

| Label | Folder | Notes |
|---|---|---|
| S01 to S06 | `sS01 ... sS06_n1_20260926_*` | 26 Sep 2026. S01 was recorded under code S00 and the folder was renamed to S01 afterwards. |
| S07 | `sS00_n1_20261001_173409` and `sS00_n1_20261001_174831` | 1 Oct 2026, two sessions of the same participant (17:34 and 17:48), both recorded under the monitor's default code S00. The analysis code labels the first S07 (`s26_common._DATASETS["s26-01"]`); the second is assigned S07 by the operator in `tools/participant_labels.csv`. |

Two recordings on 26 Sep (S05, S06) were aborted starts that the operator repeated within 3 minutes.
They are kept in `raw/` but are not in the databases: a session is included only if it has all 42
programmed trials. `sessions_index.csv` states for every folder whether it is included and why.

## Time base and synchronization

All streams are placed on the device clock, in seconds since the first raw sample of the
recording (`t_s`). `device_ts_us` gives the same instant in the device's own microsecond counter,
so any row can be matched to the records in `raw.bin`.

| Variable | Clock it was recorded on | How it reaches `t_s` | Accuracy |
|---|---|---|---|
| raw sEMG, envelope, IMU | device (every record has `ts_us`) | direct | exact to the record; the IMU is stamped when read |
| task labels, phases (grasp, rest, break) | PC (events logged with PC time and the last device `ts_us`) | linear fit PC to device per session (`clock_fit_resid_ms` in the index) | about 8 to 10 ms |
| contact load, skin temperature | PC arrival time only; the bracelet relays them on a 1 s heartbeat | estimated reading time (`t_est`, batch spread over the previous second) | about 1 s |
| impedance sweeps | PC arrival time only | sweep onset detected in the sEMG (the excitation leaves a transient on some channels), 55 ms to the first point, then 41.9 ms per point | tens of ms when detected in the sEMG; see `onset_source` |

The synchronization code is the article's own (`analysis_scripts/s26_common.py`, `sweeps.py`);
`build_databases.py` only resamples what it produces.

## Resampling: hold, not interpolation

At each grid time the database holds the **newest sample at or before that time**. Nothing is
interpolated. The age of the held sample is stored (`emg_age_us` or `env_age_us`, `imu_age_ms`,
`aux_age_s`) so stale values can be masked.

* raw sEMG is blank where the newest sample is older than 5 ms (the article's threshold): this covers
  the 1 Hz start-up preview (the first 0 to 16.8 s of some sessions) and dropouts (S02 has one of 3.2 s).
* envelope is blank where older than 60 ms.
* The raw stream is not uniform: each ADC converts every 0.91 ms, with a 2.48 ms round every 20th
  sample, and averages 1000 Hz. Holding it on a 1 kHz grid gives a timing quantization of up to
  1 ms. For spectral work use the exact timestamps in `raw.bin` (see the report, figure 8).
* `python tools/build_databases.py --emg-resample linear` reproduces the interpolation the article's
  figures use instead.

## Columns (both grid databases)

| Column | Meaning |
|---|---|
| `participant`, `session_dir` | label and source folder |
| `t_s`, `device_ts_us` | grid time, and the same instant on the device counter |
| `trial` (1 to 42), `grasp_id`, `grasp_name`, `rep` (1 to 6) | blank before the first grasp command |
| `phase` | `pre`, `grasp` (command to release), `rest`, `break` (second rest command to next grasp), `post` |
| `label_source` | how the grasp identity was obtained: `labels` (logged), `seed` (rebuilt from the protocol seed), or a mix, as in the article's code |
| `raw_E1` ... `raw_E8` (1 kHz file) | raw sEMG, signed 12-bit ADC counts |
| `env_E1` ... `env_E8` (50 Hz file) | hardware envelope, ADC counts |
| `emg_age_us` / `env_age_us` | age of the oldest newest-sample across the eight channels |
| `ax_g` `ay_g` `az_g`, `gx_dps` `gy_dps` `gz_dps`, `imu_temp_c`, `imu_age_ms` | IMU, held |
| `p1_kpa`, `p2_kpa` | contact load, flexor and extensor side |
| `load_ok` | 1 when the reading is the third or later of the hold (the auxiliary board's median filter carries values across holds, so earlier readings can repeat the previous hold) |
| `skin_temp_c`, `skin_temp_valid` | skin temperature, 0 when the sensor reads about -126.8 (not detected) |
| `aux_age_s` | seconds since the held load/temperature reading's estimated time |
| `sweep_n` | sweep counter within the session (1 to 49), set during the whole sweep including the first 55 ms |
| `sweep_id` | the id the monitor gave the sweep (matches `sweep_id` in the sweep files) |
| `sweep_point` (0 to 98), `impedance_freq_khz` (2 to 100) | which point is being excited at this instant |
| `impedance` | impedance magnitude in ohm of that point, for the 41.9 ms it is excited; blank outside sweeps |

Units: sEMG and envelope are raw ADC counts. The article converts at **2.0 mV per count** (ADS1015
with gain "One", +-4.096 V; `common.LSB_MV`); the monitor's own default is 1.0, so check the front-end
gain before quoting millivolts. IMU is in g and degrees per second, converted from the firmware's mg
and 0.1 deg/s. Temperatures in degrees C, load in kPa, impedance in ohm.

## Sweep files

One CSV per sweep with its 99 points: `point`, `freq_khz`, `impedance_ohm`, `t_start_s`, `t_center_s`,
`t_end_s`, `device_ts_us_center`, plus `sweep_onset_s`, `onset_source`, `prest_s` (the rest command
that triggered it), `arrival_s` (when the PC received it). `onset_source` is one of:

* `semg`: onset detected in the sEMG after a rest command (the normal case, 0.08 to 0.30 s after it);
* `nominal_prest+0.15s`: no onset found; the rest command plus the article's 0.15 s;
* `start_of_test_arrival-4.9s`: the sweep that starts with the recording has no rest command; its onset
  is estimated from arrival time and is only good to about 1 s.

The sweep geometry (55 ms, 41.9 ms per point, 99 points from 2 to 100 kHz, 4.2 s) is the article's own,
measured on a bench recording (`02_sweep_timing.py`).

## Things to know before analysing

Taken from the report in `reference/`; check them against the data:

* Contact load, temperature and sweeps are aligned to the sEMG only to about 1 s (sweeps better when
  the onset comes from the sEMG).
* Bioimpedance excitation reaches some sEMG channels at 10 to 30 kHz: mask rows where `sweep_n` is set
  if you analyse sEMG at rest.
* Skin temperature was not detected in three sessions; one load sensor stayed at 0 kPa in two sessions.
* All 26 Sep participants received the same grasp order (seed 712), so grasp and time in session are confounded.
* `rep` in these databases is the occurrence order of the grasp among the session's trials. The
  article's `Trial.rep` is the same unless only some trials had a logged label; then it can repeat a
  number (1, 1, 2, 3, 4, 5). `sessions_index.csv` column `rep_mismatches_vs_analysis` counts the trials
  where the two differ.

## Checks that were run

`tools/verify_databases.py` re-reads `raw.bin`, `imu.bin` and `aux.jsonl` with its own code (it does not
import the analysis or the builder) and checks row counts, the time grid, 3000 random grid times of every
raw and envelope channel against the newest raw record, the IMU, every sweep file against `aux.jsonl`,
the 1 kHz database during each sweep, the trial/repetition structure, and that the 1 kHz and 50 Hz
databases agree on every shared column. `tools/selftest.py` runs the whole pipeline on synthetic sessions
and checks that two builds are byte-identical.

## Citation and licence

To be completed by the authors.
