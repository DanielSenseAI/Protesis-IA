"""Estilo comun de las figuras del articulo: fondo blanco, tinta gris
oscura, marcas finas, rejilla recesiva, sin ejes dobles. La paleta
categorica es la validada del skill de dataviz y se asigna por significado,
siempre igual en todas las figuras.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

INK = "#1f2328"
INK2 = "#57606a"
GRID = "#d8dee4"

# Significado fijo (no por orden de aparicion)
C_SWEEP = "#eb6834"      # barrido / inyeccion
C_CTRL = "#2a78d6"       # control sin barrido (Pgrasp, eco)
C_MON_2313 = "#2a78d6"   # lunes 23:13, con tierra
C_MON_1813 = "#1baf7a"   # lunes 18:13, con tierra
C_MON_1607 = "#4a3aa7"   # lunes 16:07, con tierra
C_MON_1606 = "#e87ba4"   # lunes 16:06, con tierra
C_SAT = "#eb6834"        # sabado, sin tierra
C_ENV = "#2a78d6"        # canal de envolvente
C_RAW = INK              # crudo

SESSION_COLOR = {
    "sS04_n1_20260921_231342": C_MON_2313,
    "sS04_n1_20260921_181328": C_MON_1813,
    "sS04_n1_20260921_160737": C_MON_1607,
    "sS04_n1_20260921_160653": C_MON_1606,
    "sat": C_SAT,
}
SESSION_LABEL = {
    "sS04_n1_20260921_231342": "Mon 21 · 23:13 · earth",
    "sS04_n1_20260921_181328": "Mon 21 · 18:13 · earth",
    "sS04_n1_20260921_160737": "Mon 21 · 16:07 · earth",
    "sS04_n1_20260921_160653": "Mon 21 · 16:06 · earth",
    "sat": "Sat 19 · no earth",
}


def apply():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "lines.linewidth": 1.2, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


# Sin IMU: el corredor (s26_run_all.py --no-imu) pone EMG8_NO_IMU=1 y cada
# figura deja fuera giroscopio, temperatura del IMU y tasas del IMU.
NO_IMU = os.environ.get("EMG8_NO_IMU") == "1"
# Variantes de revision (s26_run_all.py --review, EMG8_REVIEW=1), a figures/s26_review: escalas y
# comunes entre paneles de una figura, histograma de intervalos en escala lineal, las 3 lecturas de
# carga y temperatura por sostener, envolvente sobre cada crudo y canales en veces su RMS en reposo.
REVIEW = os.environ.get("EMG8_REVIEW") == "1"


def tones(n):
    """Un gris por sujeto, de oscuro a claro. Con seis son exactamente los de
    siempre; con otro numero se interpolan entre el primero y el ultimo."""
    base = ["#1f2328", "#3b4046", "#57606a", "#7d858e", "#a3aab2", "#c5cbd1"]
    if n == len(base):
        return base
    a = [int(base[0][i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(base[-1][i:i + 2], 16) for i in (1, 3, 5)]
    out = []
    for k in range(n):
        f = k / max(n - 1, 1)
        out.append("#" + "".join(f"{round(a[i] + f * (b[i] - a[i])):02x}" for i in range(3)))
    return out


def num(v, fmt=".0f"):
    """Numero para texto dentro de una figura: signo menos tipografico, como el
    de las marcas de los ejes, y sin "-0" cuando el valor redondea a cero."""
    s = format(v, fmt)
    if float(s.replace("%", "")) == 0:
        s = s.replace("-", "").replace("+", "")
    return s.replace("-", "−")


def save(fig, path_stem, dpi=300, panels=None, extras=None):
    """PNG a 300 dpi para el borrador y SVG/PDF vectorial para la revista, sin
    letras de panel (el articulo las pone al armar la figura).

    panels: clave -> eje o lista de ejes; cada panel sale ademas solo, en
    <carpeta>/panels/<figura>_<clave>.{png,svg,pdf}, con el mismo tamano y
    estilo, para elegir paneles sueltos. Las barras de color hechas con
    fig.colorbar(..., ax=...) siguen a su eje solas; extras: clave -> artistas
    de la figura (leyendas o textos de figura, barras de color sueltas) que
    van con ese panel. Antes de guardar se buscan textos encimados (figcheck)."""
    try:
        import figcheck
        figcheck.check(fig, str(path_stem).replace("\\", "/").split("/")[-1])
    except Exception as e:          # la comprobacion nunca impide guardar
        print(f"  [figcheck] no se pudo comprobar: {e!r}")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=dpi if ext == "png" else None)
    if panels:
        _save_panels(fig, Path(path_stem), panels, extras or {}, dpi)
    plt.close(fig)


def _save_panels(fig, stem, panels, extras, dpi):
    """Cada panel en su archivo: se ocultan los demas ejes, textos y leyendas
    de la figura y se recorta al contorno del panel (con sus etiquetas)."""
    out = stem.parent / "panels"
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    loose = list(fig.texts) + list(fig.legends) + list(fig.artists)
    for key, axs in panels.items():
        mine = set(axs if isinstance(axs, (list, tuple)) else [axs])
        # una barra de color hecha desde un ScalarMappable suelto no tiene eje
        # propio: esa no se adivina, va en extras
        mine |= {a for a in fig.axes
                 if getattr(getattr(getattr(a, "_colorbar", None), "mappable", None), "axes", None) in mine}
        ext = list(extras.get(key, []))
        mine |= {e for e in ext if e in fig.axes}
        hidden = [a for a in fig.axes if a not in mine and a.get_visible()]
        hidden += [t for t in loose if t not in ext and t.get_visible()]
        for h in hidden:
            h.set_visible(False)
        fig.canvas.draw()
        bbs = [a.get_tightbbox(r) for a in mine if a.get_visible()]
        bbs += [e.get_window_extent(r) for e in ext if e not in fig.axes]
        bb = Bbox.union(bbs).transformed(fig.dpi_scale_trans.inverted()).padded(0.05)
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(out / f"{stem.name}_{key}.{suffix}", dpi=dpi if suffix == "png" else None, bbox_inches=bb)
        for h in hidden:
            h.set_visible(True)
