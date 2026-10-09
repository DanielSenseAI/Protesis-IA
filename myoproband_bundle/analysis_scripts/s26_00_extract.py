"""Sabado 26: del zip de Drive a data_raw, sin la hoja de sujetos.

El zip (26-09-2026-full.zip, bajado a mano de la carpeta MyoProband de Drive)
se deja en data_raw/2026-09-26/. Se extraen solo las carpetas de sesion; nada
con extension de hoja de calculo (.xlsx, .xls, .csv) sale del zip, porque la
hoja de sujetos lleva nombres y los nombres no entran en este analisis.

Despues se listan las claves de cada metadata.json y se comprueba que el
sujeto es un codigo (S01, S02...), no un nombre. Solo lectura del zip.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import RAW26

SHEETS = (".xlsx", ".xls", ".csv", ".ods")

zips = sorted(RAW26.glob("*.zip"))
if not zips:
    sys.exit(f"no hay zip en {RAW26}")
z = zipfile.ZipFile(zips[0])
names = z.namelist()
skipped = [n for n in names if n.lower().endswith(SHEETS)]
keep = [n for n in names if not n.lower().endswith(SHEETS)]
z.extractall(RAW26, members=keep)
print(f"{zips[0].name}: {len(keep)} archivos extraidos, {len(skipped)} hojas omitidas")

bad = []
for p in sorted(RAW26.rglob("metadata.json")):
    m = json.loads(p.read_text(encoding="utf-8"))
    subj = str(m.get("subject", ""))
    ok = bool(re.fullmatch(r"[A-Z]\d{2,3}", subj))
    if not ok:
        bad.append(p.parent.name)
    print(f"  {p.parent.name}: sujeto {subj!r} ({'codigo' if ok else 'REVISAR'}), claves {sorted(m)}")
if bad:
    sys.exit(f"sujetos que no son codigos: {bad}")
