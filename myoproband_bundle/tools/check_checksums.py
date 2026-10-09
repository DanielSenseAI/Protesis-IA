#!/usr/bin/env python3
"""Check every file in the bundle against SHA256SUMS.txt (works on Windows, where sha256sum is not available).

    python tools/check_checksums.py [bundle_folder]
"""
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
sums = root / "SHA256SUMS.txt"
bad = missing = n = 0
for line in sums.read_text(encoding="utf-8").splitlines():
    digest, _, name = line.partition("  ")
    p = root / name
    n += 1
    if not p.exists():
        print("MISSING", name)
        missing += 1
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != digest:
        print("DIFFERS", name)
        bad += 1
extra = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {
    l.partition("  ")[2] for l in sums.read_text(encoding="utf-8").splitlines()} - {"SHA256SUMS.txt"}
for name in sorted(extra):
    print("NOT IN LIST (ignored if it is a __pycache__ file)", name)
print(f"{n} files listed, {bad} differ, {missing} missing")
raise SystemExit(1 if bad or missing else 0)
