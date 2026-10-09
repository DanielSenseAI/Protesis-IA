"""Sabado 26: rehace todo, de los archivos crudos al informe.

    python s26_run_all.py                 # todo, en orden
    python s26_run_all.py --from s26_07   # desde un paso (los anteriores ya estan)
    python s26_run_all.py --list          # solo la lista
    python s26_run_all.py --no-imu        # solo las figuras, sin IMU, en figures/s26_no_imu
    python s26_run_all.py --review        # solo las figuras, variantes de revision, en figures/s26_review
    python s26_run_all.py --review --no-imu   # las dos cosas, en figures/s26_review_no_imu
    python s26_run_all.py --dataset s26-01    # otro conjunto (s26 + 1-10): data/s26-01, figures/s26-01...

Cada paso lee lo que dejaron los anteriores en data/s26/ y escribe sus tablas
y figuras; si uno falla se para ahi. Los espectros de la figura 5 se
recalculan (--recompute) para que nada salga de una cache vieja.

Con --no-imu corren solo los pasos que dibujan figuras (las tablas ya deben
estar), con EMG8_NO_IMU=1: sin giroscopio, tasas ni temperatura del IMU; el
informe no se rehace. --review es igual pero con EMG8_REVIEW=1 (ver figstyle.REVIEW).
Cada figura deja ademas sus paneles sueltos en <carpeta de figuras>/panels/.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
STEPS = [
    ("s26_00_extract.py", [], "zip de Drive -> data_raw, sin la hoja de sujetos"),
    ("s26_01_integrity.py", [], "integridad y desempeno por sesion"),
    ("s26_02_semg.py", [], "sEMG por ensayo y canal"),
    ("s26_03_classify.py", [], "separabilidad de agarres (secundario)"),
    ("s26_04_aux.py", [], "impedancia, carga, temperatura, IMU por ensayo"),
    ("s26_05_sweeps.py", [], "barrido -> sEMG, sostenes pisados; figura 4"),
    ("s26_07_timing.py", [], "anatomia del muestreo, enlace, presupuesto"),
    ("s26_08_instrument.py", [], "correlacion entre canales, retardo de envolvente, interfaz"),
    ("s26_10_wifi.py", [], "interferencia de la baliza Wi-Fi: notch, plantilla, reparar"),
    ("bench_radio_sampling.py", [], "capturas de banco: radio apagada/encendida, tope 1000/max"),
    ("s26_fig1_integrity.py", [], "figura 1"),
    ("s26_fig2_multimodal.py", ["S02", "26"], "figura 2"),
    ("s26_fig3_timing.py", [], "figura 3"),
    ("s26_fig5_semg.py", ["--recompute"], "figura 5"),
    ("s26_fig6_impedance.py", [], "figura 6"),
    ("s26_fig7_showcase.py", ["S01", "4", "3"], "figura 7 (la principal)"),
    ("s26_fig8_timing.py", [], "figura 8"),
    ("s26_fig9_link.py", [], "figura 9"),
    ("s26_fig10_instrument.py", [], "figura 10"),
    ("s26_fig11_context.py", [], "figura 11"),
    ("s26_fig12_models.py", [], "figura 12 (modelos, secundario)"),
    ("s26_fig13_envlag.py", [], "figura 13 (como se mide el retardo de la envolvente)"),
    ("s26_fig14_wifi.py", [], "figura 14 (baliza Wi-Fi y como quitarla)"),
    ("s26_fig15_channels.py", [], "figura 15 (ganancia por canal y escalas de los crudos)"),
    ("s26_09_requirements.py", [], "tabla de requisitos del articulo (requirements_s26.md)"),
    ("s26_06_report.py", [], "informe report_<conjunto>.html"),
]


def main() -> int:
    args = sys.argv[1:]
    if "--list" in args:
        for name, extra, what in STEPS:
            print(f"  {name:26s} {' '.join(extra):12s} {what}")
        return 0
    start = 0
    no_imu, review = "--no-imu" in args, "--review" in args
    env = dict(os.environ)
    dataset = args[args.index("--dataset") + 1] if "--dataset" in args else env.get("EMG8_DATASET", "s26")
    env["EMG8_DATASET"] = dataset
    if no_imu:
        env["EMG8_NO_IMU"] = "1"
    if review:
        env["EMG8_REVIEW"] = "1"
    if "--from" in args:
        key = args[args.index("--from") + 1]
        start = next(i for i, (n, _, _) in enumerate(STEPS) if n.startswith(key))
    t_all = time.time()
    steps = STEPS[start:]
    if no_imu or review:
        steps = [st for st in steps if st[0].startswith("s26_fig") or st[0] == "s26_05_sweeps.py"]
    if dataset != "s26":
        # el zip del sabado y las capturas de banco del repo del firmware (iguales para todo conjunto)
        # solo corren con s26; el informe ya sigue al conjunto (report_<conjunto>.html)
        steps = [st for st in steps if st[0] not in ("s26_00_extract.py", "bench_radio_sampling.py")]
    print(f"conjunto de datos: {dataset}", flush=True)
    for name, extra, what in steps:
        t0 = time.time()
        print(f"== {name} {' '.join(extra)}  ({what})", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", str(HERE / name), *extra], cwd=HERE, env=env)
        if r.returncode != 0:
            print(f"!! {name} fallo (codigo {r.returncode}); se para aqui.")
            return r.returncode
        print(f"   {time.time() - t0:.0f} s", flush=True)
    print(f"\nlisto en {(time.time() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
