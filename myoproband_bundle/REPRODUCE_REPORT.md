# Rebuilding the report from this bundle

The report in `reference/` (`report_s26_V2.html`) and its figures were produced by the scripts in
`analysis_scripts/` from the session folders in `raw/sessions/`. You do not need the monitor or its
data store.

## Setup

```sh
pip install -r requirements.txt            # numpy, pandas, matplotlib
export EMG8_OUT=$PWD/work                  # Windows: set EMG8_OUT=%CD%\work
export EMG8_SESSIONS=$PWD/raw/sessions     # Windows: set EMG8_SESSIONS=%CD%\raw\sessions
mkdir -p $EMG8_OUT/data
```

The only change to the analysis code is in `analysis_scripts/common.py`: the paths that were fixed to
`D:/PhD/...` now come from `EMG8_OUT` and `EMG8_SESSIONS`, and fall back to the old values if unset.
`diff` it against your original to confirm; nothing else was edited.

## Run

```sh
cd analysis_scripts
python s26_run_all.py --dataset s26 --from s26_01     # the six 26 Sep participants (the report)
python s26_run_all.py --dataset s26-01 --from s26_01  # the same plus the 1 Oct participant (S07)
python s26_run_all.py --list                           # the steps
```

For `--dataset s26-01`, the second 1 Oct session (`sS00_n1_20261001_174831`, label S00) is not known to
`s26_common._DATASETS`: add `"S00"` to the `order` list of `"s26-01"` there before running the figure scripts, or
they cannot place that participant. The published report uses the `s26` dataset and does not need this.

Tables go to `$EMG8_OUT/data/<dataset>/`, figures to `$EMG8_OUT/figures/<dataset>/`, and the report to
`$EMG8_OUT/report_<dataset>.html`. Compare against `reference/`.

## Inputs the report needs that are not session folders

| Step | Needs | Bundled | If missing |
|---|---|---|---|
| `s26_00_extract.py` | the Drive zip of the 26 Sep sessions | no, not needed | skip it with `--from s26_01` |
| `s26_09_requirements.py` | `data/sweep_timing.txt` (output of `02_sweep_timing.py` on the bench session `sS04_n1_20260921_231342`) | only if you pass it with `make_bundle.py --extras` | the step fails; copy the file to `$EMG8_OUT/data/` |
| `bench_radio_sampling.py` | the bench SD captures of the firmware repo | no | it prints "sin capturas de banco" and does nothing |
| `s26_06_report.py`, `s26_09_requirements.py` | long bench runs `data_raw/bench_*/long-*/summary.json` | no | the continuous-recording claim is reported as "not tested" |

Report paragraphs that cite bench captures (the 28 Sep Wi-Fi bench board, the long bench run) will
therefore differ from `reference/` unless those inputs are added. Figures and tables computed from the
session folders do not depend on them.

## What has and has not been checked

The databases are checked by `tools/verify_databases.py` against the raw files. The analysis scripts
were not re-run on the participant data by the person who assembled this bundle, so agreement between a
rebuilt report and `reference/` is for you to confirm. Differences you should expect even when
everything is right: figure rendering (matplotlib version) and anything that depends on the bench
inputs above. The databases use hold resampling; the report's figures use linear interpolation on the
raw timestamps, so numbers computed from the databases will not match the report to the last digit.
