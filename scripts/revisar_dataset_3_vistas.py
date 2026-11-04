#!/usr/bin/env python3
"""
revisar_dataset_3_vistas.py
---------------------------
Visor de SOLO LECTURA para revisar, imagen por imagen, que el matching y las
anotaciones cuadren en las 3 versiones del mismo frame:

    1. ORIGEN       la imagen tal como venia en el dataset del que salio
                    (dataset_original/_referencia_origen/images/<nombre>.jpg)
    2. ORIGINAL     el frame exacto del video que quedo en dataset_original
    3. SIN HUD      el mismo frame en dataset_sin_hud

Las 3 se muestran lado a lado con sus anotaciones YOLO dibujadas encima
(mismos colores que annotate_images.py). Nunca escribe etiquetas.

Modo ventana (en tu PC, o en Hypatia con ``ssh -X``):
    python scripts/revisar_dataset_3_vistas.py \\
        --original datasets/dataset_original --sin-hud datasets/dataset_sin_hud

    Teclas:  ->  / espacio  siguiente      <-  anterior
             Inicio / Fin   primera / ultima
             b              mostrar / ocultar cajas
             g              ir a un numero de imagen
             Esc / q        salir

Modo exportar (sin pantalla, por ejemplo en un job de Slurm): guarda un JPG por
imagen con las 3 vistas lado a lado:
    python scripts/revisar_dataset_3_vistas.py --original ... --sin-hud ... \\
        --exportar datasets/revision_3_vistas

Si todavia no existe dataset_sin_hud, se puede omitir ``--sin-hud``: la tercera
vista aparece vacia.

Dependencias: Pillow (y tkinter solo para el modo ventana).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLASES_DEFECTO = ["vehicle", "building", "road", "river", "SDZI"]
CLASS_COLORS = ["#FF4C4C", "#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
TITULOS = ["1. ORIGEN (dataset de procedencia)", "2. dataset_original (frame del video)", "3. dataset_sin_hud"]
FONDO = "#1a1a2e"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def leer_clases(carpeta: Path) -> list[str]:
    for nombre in ("clases.yaml", "dataset.yaml", "data.yaml"):
        p = carpeta / nombre
        if not p.exists():
            continue
        try:
            import yaml
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            names = data.get("names", {})
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names)]
            if isinstance(names, list):
                return [str(n) for n in names]
        except Exception:
            pass
    return list(CLASES_DEFECTO)


def leer_cajas(ruta: Path) -> list[tuple[int, float, float, float, float]]:
    if not ruta.exists():
        return []
    cajas = []
    for ln in ruta.read_text().splitlines():
        p = ln.split()
        if len(p) == 5:
            try:
                cajas.append((int(p[0]), *map(float, p[1:])))
            except ValueError:
                pass
    return cajas


def fuente(tam: int):
    for f in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, tam)
        except Exception:
            continue
    return ImageFont.load_default()


def panel(ruta: Path | None, cajas, clases, ancho: int, titulo: str, subtitulo: str,
          con_cajas: bool = True) -> Image.Image:
    """Imagen escalada a `ancho` con cajas, titulo arriba. Si no existe, panel gris."""
    alto_titulo = 44
    f_t, f_s, f_c = fuente(15), fuente(12), fuente(12)
    if ruta is not None and ruta.exists():
        im = Image.open(ruta).convert("RGB")
        w, h = im.size
        esc = ancho / w
        im = im.resize((ancho, max(1, int(h * esc))), Image.LANCZOS)
        if con_cajas:
            d = ImageDraw.Draw(im)
            W, H = im.size
            for c, x, y, bw, bh in cajas:
                col = CLASS_COLORS[c % len(CLASS_COLORS)]
                x1, y1, x2, y2 = (x - bw / 2) * W, (y - bh / 2) * H, (x + bw / 2) * W, (y + bh / 2) * H
                d.rectangle([x1, y1, x2, y2], outline=col, width=2)
                nombre = clases[c] if c < len(clases) else str(c)
                tx, ty = max(x1, 0), max(y1, 0)
                tw = d.textlength(nombre, font=f_c) + 6
                d.rectangle([tx, ty, tx + tw, ty + 16], fill=col)
                d.text((tx + 3, ty + 1), nombre, fill="white", font=f_c)
        res = f"{w}x{h}"
    else:
        im = Image.new("RGB", (ancho, int(ancho * 9 / 16)), "#333344")
        ImageDraw.Draw(im).text((20, 20), "no disponible", fill="#bbbbbb", font=f_t)
        res = "-"
    lienzo = Image.new("RGB", (ancho, im.size[1] + alto_titulo), FONDO)
    d = ImageDraw.Draw(lienzo)
    d.text((6, 4), titulo, fill="#e94560", font=f_t)
    d.text((6, 24), f"{subtitulo}  [{res}, {len(cajas)} cajas]", fill="#a8dadc", font=f_s)
    lienzo.paste(im, (0, alto_titulo))
    return lienzo


class Datos:
    def __init__(self, original: Path, sin_hud: Path | None):
        self.original, self.sin_hud = original, sin_hud
        self.clases = leer_clases(original)
        with open(original / "correspondencia.csv", encoding="utf-8-sig", newline="") as f:
            self.filas = list(csv.DictReader(f))
        self.estado_sin_hud = {}
        if sin_hud and (sin_hud / "correspondencia.csv").exists():
            with open(sin_hud / "correspondencia.csv", encoding="utf-8-sig", newline="") as f:
                self.estado_sin_hud = {r["imagen"]: r for r in csv.DictReader(f)}

    def __len__(self):
        return len(self.filas)

    def mosaico(self, i: int, ancho_panel: int, con_cajas=True) -> Image.Image:
        r = self.filas[i]
        stem = Path(r["imagen"]).stem
        lab_o = leer_cajas(self.original / "labels" / f"{stem}.txt")
        ref = self.original / "_referencia_origen" / "images" / f"{stem}.jpg"
        p1 = panel(ref, lab_o, self.clases, ancho_panel, TITULOS[0],
                   f"{r.get('dataset_origen', '')}: {r.get('nombre_original', '')}", con_cajas)
        p2 = panel(self.original / "images" / r["imagen"], lab_o, self.clases, ancho_panel, TITULOS[1],
                   f"{r['imagen']}", con_cajas)
        if self.sin_hud:
            lab_s = leer_cajas(self.sin_hud / "labels" / f"{stem}.txt")
            est = self.estado_sin_hud.get(r["imagen"], {}).get("estado", "")
            p3 = panel(self.sin_hud / "images" / f"{stem}.png", lab_s, self.clases, ancho_panel, TITULOS[2],
                       f"{stem}.png  {est}", con_cajas)
        else:
            p3 = panel(None, [], self.clases, ancho_panel, TITULOS[2], "(sin --sin-hud)", con_cajas)
        alto = max(p.size[1] for p in (p1, p2, p3))
        pie = 26
        m = Image.new("RGB", (ancho_panel * 3 + 8, alto + pie), FONDO)
        for k, p in enumerate((p1, p2, p3)):
            m.paste(p, (k * (ancho_panel + 4), 0))
        info = (f"[{i + 1}/{len(self)}]  {r['video']}  frame {r['frame']}  ({r.get('segundo', '')} s)   "
                f"verificado: {r.get('verificado_por', '')}   imagen origen: {r.get('imagen_original_vs_frame', '')}   "
                f"recortada: {r.get('caja_recortada_al_borde', '')}")
        ImageDraw.Draw(m).text((8, alto + 5), info, fill="white", font=fuente(13))
        return m


# ---------------------------------------------------------------------------
# Modo exportar
# ---------------------------------------------------------------------------

_DATOS: Datos | None = None


def _init(original, sin_hud):
    global _DATOS
    _DATOS = Datos(Path(original), Path(sin_hud) if sin_hud else None)


def _exportar_uno(args):
    i, carpeta, ancho = args
    r = _DATOS.filas[i]
    destino = Path(carpeta) / f"{i + 1:04d}_{Path(r['imagen']).stem}.jpg"
    _DATOS.mosaico(i, ancho).save(destino, quality=85)
    return str(destino)


def exportar(original: Path, sin_hud: Path | None, carpeta: Path, ancho: int, workers: int):
    carpeta.mkdir(parents=True, exist_ok=True)
    n = len(Datos(original, sin_hud))
    print(f"Exportando {n} mosaicos a {carpeta} con {workers} procesos...")
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(original), str(sin_hud) if sin_hud else None)) as ex:
        for k, _ in enumerate(ex.map(_exportar_uno, [(i, str(carpeta), ancho) for i in range(n)], chunksize=8), 1):
            if k % 100 == 0 or k == n:
                print(f"  {k}/{n}", flush=True)


# ---------------------------------------------------------------------------
# Modo ventana
# ---------------------------------------------------------------------------

def ventana(datos: Datos, inicio: int):
    import tkinter as tk
    from tkinter import simpledialog
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Revision 3 vistas: origen | dataset_original | dataset_sin_hud")
    root.configure(bg=FONDO)
    root.geometry("1800x700")
    estado = {"i": max(0, min(inicio, len(datos) - 1)), "cajas": True, "tk": None}
    lbl = tk.Label(root, bg=FONDO)
    lbl.pack(fill="both", expand=True)

    def mostrar(*_):
        ancho = max(300, (root.winfo_width() - 16) // 3)
        img = datos.mosaico(estado["i"], ancho, estado["cajas"])
        alto_max = root.winfo_height() - 10
        if img.size[1] > alto_max > 100:
            img = img.resize((int(img.size[0] * alto_max / img.size[1]), alto_max), Image.LANCZOS)
        estado["tk"] = ImageTk.PhotoImage(img)
        lbl.configure(image=estado["tk"])

    def mover(d):
        estado["i"] = max(0, min(len(datos) - 1, estado["i"] + d))
        mostrar()

    def ir():
        n = simpledialog.askinteger("Ir a", f"Numero de imagen (1-{len(datos)}):", parent=root)
        if n:
            estado["i"] = max(0, min(len(datos) - 1, n - 1))
            mostrar()

    def cajas():
        estado["cajas"] = not estado["cajas"]
        mostrar()

    root.bind("<Right>", lambda e: mover(1))
    root.bind("<space>", lambda e: mover(1))
    root.bind("<Left>", lambda e: mover(-1))
    root.bind("<Home>", lambda e: mover(-len(datos)))
    root.bind("<End>", lambda e: mover(len(datos)))
    root.bind("b", lambda e: cajas())
    root.bind("g", lambda e: ir())
    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("q", lambda e: root.destroy())
    root.bind("<Configure>", lambda e: root.after_idle(mostrar) if e.widget is root else None)
    root.after(100, mostrar)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default="datasets/dataset_original")
    ap.add_argument("--sin-hud", default="datasets/dataset_sin_hud")
    ap.add_argument("--exportar", metavar="CARPETA", help="Guardar mosaicos JPG en vez de abrir la ventana")
    ap.add_argument("--ancho-panel", type=int, default=900, help="Ancho de cada vista al exportar (px)")
    ap.add_argument("--inicio", type=int, default=1, help="Numero de imagen con el que abre la ventana")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()

    original = Path(a.original)
    sin_hud = Path(a.sin_hud) if a.sin_hud and Path(a.sin_hud).exists() else None
    if not (original / "correspondencia.csv").exists():
        sys.exit(f"No encuentro {original / 'correspondencia.csv'}")
    if sin_hud is None:
        print(f"Aviso: no existe {a.sin_hud}; la vista 3 quedara vacia.")
    if a.exportar:
        exportar(original, sin_hud, Path(a.exportar), a.ancho_panel, a.workers)
    else:
        ventana(Datos(original, sin_hud), a.inicio - 1)


if __name__ == "__main__":
    main()
