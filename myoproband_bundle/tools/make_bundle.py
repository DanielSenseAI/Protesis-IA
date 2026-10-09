#!/usr/bin/env python3
"""Assemble the shareable bundle from the real session folders. Run this on the PC that has them.

    python tools/make_bundle.py --sessions "C:/Users/<you>/Desktop/IA-Arm_Monitor/sessions" ^
        --report report_s26_V2.html --figures s26-noLetters.zip --out dist

What it does, in order (it stops at the first problem):
  1. finds the session folders of the study days (26 Sep and 1 Oct 2026), by folder name;
  2. copies ONLY the recording files of each (raw.bin, imu.bin, events.jsonl, aux.jsonl,
     metadata.json, qc.json). Subject sheets, the shared sessions/_aux store and anything else
     are never copied;
  3. checks that every metadata.json subject is a code (S01...), not a name;
  4. builds the databases from the COPIES and verifies them against the copies;
  5. adds the analysis scripts, tools, README, the report and figures you pass in;
  6. writes SHA256SUMS.txt and a zip in which files have fixed timestamps and order.
Nothing is read from or written to --sessions except the copies it makes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DAYS = ("20260926", "20261001")
KEEP = ("raw.bin", "imu.bin", "events.jsonl", "aux.jsonl", "metadata.json", "qc.json")
FOLDER_RE = re.compile(r"^sS\d{2}_n\d+_(\d{8})_(\d{6})$")
SUBJECT_RE = re.compile(r"^[A-Z]\d{2,3}$")
KNOWN_META = {"protocol", "subject", "session", "arm", "mode", "seed", "seed_base", "seed_per_subject",
              "order", "grasps", "reps", "trials", "timing", "sequence", "placement", "preflight", "wal"}
BUNDLE = "myoproband_s26_dataset"
ZIP_DATE = (2026, 10, 9, 0, 0, 0)          # fixed so the zip does not change between runs


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_folders(src: Path) -> list[Path]:
    out = []
    for p in sorted(src.iterdir()):
        m = FOLDER_RE.match(p.name)
        if p.is_dir() and m and m.group(1) in DAYS and (p / "raw.bin").exists():
            out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", type=Path, required=True, help="the monitor's sessions folder")
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--report", type=Path, default=None, help="report_s26_V2.html (reference output)")
    ap.add_argument("--figures", type=Path, default=None, help="s26-noLetters.zip (reference figures)")
    ap.add_argument("--extras", type=Path, nargs="*", default=[],
                    help="extra files for the report that are not session folders (see REPRODUCE_REPORT.md)")
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args(argv)

    stage = a.out / BUNDLE
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "raw" / "sessions").mkdir(parents=True)

    folders = find_folders(a.sessions)
    if not folders:
        sys.exit(f"no session folders for {DAYS} with a raw.bin in {a.sessions}")
    print(f"{len(folders)} session folders found:")
    for f in folders:
        meta = json.loads((f / "metadata.json").read_text(encoding="utf-8")) if (f / "metadata.json").exists() else {}
        subj = str(meta.get("subject", "?"))
        unknown = sorted(set(meta) - KNOWN_META)
        print(f"  {f.name}  subject={subj!r}  raw.bin={(f / 'raw.bin').stat().st_size / 1e6:.1f} MB"
              + (f"  UNKNOWN metadata keys: {unknown}" if unknown else ""))
        if not SUBJECT_RE.match(subj):
            sys.exit(f"{f.name}: subject {subj!r} is not a code like S01. Fix it before sharing; nothing was written.")
        if unknown:
            sys.exit(f"{f.name}: metadata.json has keys this script does not know {unknown}. "
                     f"Read them for personal data, then add them to KNOWN_META.")
        dst = stage / "raw" / "sessions" / f.name
        dst.mkdir()
        for name in KEEP:
            if (f / name).exists():
                shutil.copy2(f / name, dst / name)

    for sub in ("analysis_scripts", "tools"):
        shutil.copytree(ROOT / sub, stage / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("README.md", "REPRODUCE_REPORT.md", "requirements.txt"):
        shutil.copy2(ROOT / name, stage / name)
    ref = stage / "reference"
    ref.mkdir()
    for f in (a.report, a.figures):
        if f:
            shutil.copy2(f, ref / f.name)
    if a.extras:
        (stage / "extras").mkdir()
        for f in a.extras:
            shutil.copy2(f, stage / "extras" / f.name)

    print("\nbuilding databases from the copies ...")
    py = sys.executable
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}          # keep the staging folder free of .pyc
    run = lambda *c: subprocess.run([py, "-I", *map(str, c)], check=True, env=env)
    run(stage / "tools" / "build_databases.py", "--sessions", stage / "raw" / "sessions",
        "--out", stage / "databases")
    if not a.skip_verify:
        print("\nverifying databases against the copies ...")
        run(stage / "tools" / "verify_databases.py", "--sessions", stage / "raw" / "sessions",
            "--databases", stage / "databases")
    for junk in list(stage.rglob("__pycache__")):
        shutil.rmtree(junk, ignore_errors=True)
    shutil.rmtree(stage / "databases" / "_analysis_work", ignore_errors=True)
    shutil.rmtree(stage / "_analysis_work", ignore_errors=True)

    files = sorted(p for p in stage.rglob("*") if p.is_file())
    (stage / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(stage).as_posix()}\n" for p in files), encoding="utf-8")
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    zpath = a.out / f"{BUNDLE}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            zi = zipfile.ZipInfo(f"{BUNDLE}/{p.relative_to(stage).as_posix()}", ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, p.read_bytes())
    print(f"\n{len(files)} files, {zpath.stat().st_size / 1e6:.0f} MB -> {zpath}")
    print(f"zip sha256: {sha256(zpath)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
