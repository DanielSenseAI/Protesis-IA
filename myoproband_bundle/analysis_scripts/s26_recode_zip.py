"""Sabado 26: el sujeto S00 pasa a ser S01 (2026-09-28, pedido del usuario).

Los codigos del articulo van seguidos, S01-S06. Se cambia en la fuente, el zip
de data_raw/2026-09-26, para que todo lo demas salga igual al rehacerlo:

- la carpeta de la sesion sS00_n1_20260926_113055 pasa a sS01_n1_20260926_113055;
- metadata.json: "subject" y la ruta del WAL en el portatil 2;
- aux.jsonl: el campo "session" de cada linea.

raw.bin, imu.bin y events.jsonl no llevan el codigo y se copian byte a byte;
las demas sesiones, tal cual. El zip original queda en
data_raw/_archivo/26-09-2026-full_original-S00_2026-09-28.zip. En el portatil 2,
el Drive y la SD la sesion sigue llamandose sS00.

Idempotente: si el zip ya no tiene S00, no hace nada.
"""
import hashlib
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, RAW26

OLD_ID, NEW_ID = "sS00_n1_20260926_113055", "sS01_n1_20260926_113055"
SRC = RAW26 / "26-09-2026-full.zip"
BACKUP = OUT / "data_raw" / "_archivo" / "26-09-2026-full_original-S00_2026-09-28.zip"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


with zipfile.ZipFile(SRC) as z:
    if not any(OLD_ID in n for n in z.namelist()):
        sys.exit(f"{SRC.name} ya no tiene {OLD_ID}: nada que hacer")
if not BACKUP.exists() or sha(BACKUP) != sha(SRC):
    sys.exit(f"falta la copia del original en {BACKUP} (o no coincide): no se toca nada")

tmp = SRC.with_suffix(".recode.zip")
cambios = {}
with zipfile.ZipFile(SRC) as zi, zipfile.ZipFile(tmp, "w") as zo:
    for info in zi.infolist():
        data = zi.read(info)
        name = info.filename
        if OLD_ID in name:
            name = name.replace(OLD_ID, NEW_ID)
            if name.endswith((".json", ".jsonl")):
                text = data.decode("utf-8")
                n_id = text.count(OLD_ID)
                text = text.replace(OLD_ID, NEW_ID)
                n_subj = 0
                if name.endswith("metadata.json"):
                    n_subj = text.count('"subject": "S00"')
                    assert n_subj == 1, f"metadata.json: {n_subj} campos subject S00"
                    text = text.replace('"subject": "S00"', '"subject": "S01"')
                assert "S00" not in text, f"{name}: queda S00"
                data = text.encode("utf-8")
                cambios[name] = (n_id, n_subj)
            else:
                assert OLD_ID.encode() not in data
                cambios[name] = "igual (binario)"
        out = zipfile.ZipInfo(name, date_time=info.date_time)
        out.compress_type = info.compress_type
        out.external_attr = info.external_attr
        zo.writestr(out, data)

# Verificacion antes de reemplazar: mismas entradas, binarios identicos, sin S00
with zipfile.ZipFile(SRC) as zi, zipfile.ZipFile(tmp) as zn:
    a, b = zi.infolist(), zn.infolist()
    assert len(a) == len(b) == 40, (len(a), len(b))
    for x, y in zip(a, b):
        assert y.filename == x.filename.replace(OLD_ID, NEW_ID)
        if not y.filename.endswith((".json", ".jsonl")) or OLD_ID not in x.filename:
            assert x.CRC == y.CRC and x.file_size == y.file_size, y.filename
    assert not any("S00" in n for n in zn.namelist())
    assert zn.testzip() is None
os.replace(tmp, SRC)
for k, v in cambios.items():
    print(f"  {k}: {v}")
print(f"listo: {SRC.name} recodificado; original en {BACKUP}")
