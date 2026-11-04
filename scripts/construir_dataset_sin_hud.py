#!/usr/bin/env python3
"""
Construye ``dataset_sin_hud`` a partir de ``dataset_original``.

``dataset_original`` es el dataset cuyas imagenes son los frames exactos de los
3 videos (resultado del matching contra los frames originales). Este script toma,
para cada imagen, el MISMO frame pero de la carpeta ``frames_sin_hud`` de su
video y copia las mismas anotaciones. El resultado es el mismo dataset (mismos
nombres, mismas etiquetas, mismo orden) pero sin HUD.

Nombres de las imagenes:  ``<video>_<frame>_<segundo>.png``
    - frame: numero de frame a los fps originales del video, empezando en 1
      (igual que ``videos/<video>/frames_originales/frame_N.png``)
    - segundo: (frame - 1) / fps, con 3 decimales

Estructura esperada (relativa a la raiz del repo):

    datasets/dataset_original/
        images/*.png
        labels/*.txt
        correspondencia.csv      <- columnas: imagen, video, frame, ...
        clases.yaml
    videos/<video>/frames_originales/frame_N.png
    videos/<video>/frames_sin_hud/frame_N.png   (o la carpeta que diga config/videos.json)

Salida:

    datasets/dataset_sin_hud/
        images/*.png             <- mismo nombre que en dataset_original
        labels/*.txt             <- copia exacta de dataset_original/labels
        clases.yaml
        correspondencia.csv      <- la de dataset_original + columnas de trazabilidad sin HUD
        faltantes.csv            <- imagenes que no se pudieron generar (si hay)
        resumen.json

Verificacion (activada por defecto): antes de copiar el frame sin HUD se compara
la imagen de dataset_original contra ``frames_originales/frame_N.png``. Si no
coincide, se buscan los vecinos N +- ``--vecinos`` y se usa el que coincida, para
que el frame sin HUD salga del mismo instante exacto. Todo queda registrado.

Uso:
    python scripts/construir_dataset_sin_hud.py \\
        --dataset-original datasets/dataset_original \\
        --salida datasets/dataset_sin_hud --workers 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("Se necesita opencv-python (pip install opencv-python-headless)")

RAIZ = Path(__file__).resolve().parent.parent
PATRON_NUMERO = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")
EXTENSIONES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Localizacion de carpetas
# ---------------------------------------------------------------------------

def _buscar_carpetas_en_json(nodo, clave_contiene: str, encontrados: list[str]):
    """Recorre el json y junta cualquier valor 'carpeta_salida' que contenga el texto."""
    if isinstance(nodo, dict):
        for k, v in nodo.items():
            if k == "carpeta_salida" and isinstance(v, str) and clave_contiene in v:
                encontrados.append(v)
            else:
                _buscar_carpetas_en_json(v, clave_contiene, encontrados)
    elif isinstance(nodo, list):
        for v in nodo:
            _buscar_carpetas_en_json(v, clave_contiene, encontrados)


def candidatos_carpeta(raiz: Path, config: dict, video: str, tipo: str) -> list[Path]:
    """tipo: 'frames_originales' o 'frames_sin_hud'. Orden: json -> rutas estandar."""
    desde_json: list[str] = []
    _buscar_carpetas_en_json(config.get(video, {}), tipo, desde_json)
    rutas = [raiz / p for p in desde_json]
    rutas += [raiz / "videos" / video / tipo]
    if tipo == "frames_sin_hud":
        rutas += [raiz / "videos" / video / "frames_sin_hud.tmp",
                  raiz / "videos" / video / "sin_hud"]
    vistos, salida = set(), []
    for r in rutas:
        if r not in vistos:
            vistos.add(r)
            salida.append(r)
    return salida


def indexar_carpeta(carpeta: Path) -> dict[int, str]:
    """{numero_de_frame: ruta} usando el numero al final del nombre (frame_00012.png -> 12)."""
    indice: dict[int, str] = {}
    with os.scandir(carpeta) as it:
        for e in it:
            if not e.is_file():
                continue
            if os.path.splitext(e.name)[1].lower() not in EXTENSIONES:
                continue
            m = PATRON_NUMERO.search(e.name)
            if m:
                indice[int(m.group(1))] = e.path
    return indice


def elegir_carpeta(cands: list[Path], frames_necesarios: set[int]) -> tuple[Path | None, dict[int, str]]:
    """Devuelve la primera carpeta existente que tenga mas frames de los necesarios."""
    mejor = (None, {}, -1)
    for c in cands:
        if not c.is_dir():
            continue
        idx = indexar_carpeta(c)
        cubre = len(frames_necesarios & idx.keys())
        print(f"    {c}  ->  {len(idx)} archivos, cubre {cubre}/{len(frames_necesarios)} frames del dataset")
        if cubre > mejor[2]:
            mejor = (c, idx, cubre)
        if cubre == len(frames_necesarios):
            break
    return mejor[0], mejor[1]


# ---------------------------------------------------------------------------
# Trabajo por imagen
# ---------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def procesar(tarea: dict) -> dict:
    r = dict(tarea["fila"])
    nombre = r["imagen"]
    stem = Path(nombre).stem
    frame = int(r["frame"])
    idx_orig: dict[int, str] = tarea["idx_orig"]
    idx_sin: dict[int, str] = tarea["idx_sin"]
    salida = Path(tarea["salida"])
    r.update(frame_usado="", psnr_vs_frames_originales="", archivo_frames_originales_usado="",
             archivo_sin_hud_usado="", resolucion_sin_hud="", estado="")

    frame_usado = frame
    if tarea["verificar"]:
        img_ds = cv2.imread(tarea["img_original"], cv2.IMREAD_COLOR)
        if img_ds is None:
            r["estado"] = "ERROR_LEYENDO_DATASET_ORIGINAL"
            return r
        if not idx_orig:
            r["estado"] = "FALTA_EN_FRAMES_ORIGINALES"
            return r
        p0 = psnr(img_ds, cv2.imread(idx_orig[frame], cv2.IMREAD_COLOR)) if frame in idx_orig else -1.0
        mejor = (p0, frame)
        if p0 < tarea["umbral_psnr"]:
            for d in range(1, tarea["vecinos"] + 1):
                for k in (frame - d, frame + d):
                    if k in idx_orig:
                        p = psnr(img_ds, cv2.imread(idx_orig[k], cv2.IMREAD_COLOR))
                        if p > mejor[0]:
                            mejor = (p, k)
        frame_usado = mejor[1]
        if frame_usado not in idx_orig:
            r["estado"] = "FALTA_EN_FRAMES_ORIGINALES"
            return r
        r["psnr_vs_frames_originales"] = f"{mejor[0]:.2f}"
        r["archivo_frames_originales_usado"] = os.path.basename(idx_orig[frame_usado])
        if mejor[0] < tarea["umbral_psnr"]:
            r["estado"] = "NO_VERIFICADO_PSNR_BAJO"
            if not tarea["copiar_no_verificadas"]:
                return r

    r["frame_usado"] = frame_usado
    if frame_usado not in idx_sin:
        r["estado"] = "FALTA_EN_FRAMES_SIN_HUD"
        return r
    src = idx_sin[frame_usado]
    r["archivo_sin_hud_usado"] = os.path.basename(src)
    dst = salida / "images" / f"{stem}.png"

    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        r["estado"] = "ERROR_LEYENDO_SIN_HUD"
        return r
    r["resolucion_sin_hud"] = f"{img.shape[1]}x{img.shape[0]}"
    if tarea["igualar_resolucion"]:
        h, w = tarea["res_original"]
        if (img.shape[0], img.shape[1]) != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif src.lower().endswith(".png"):
        shutil.copyfile(src, dst)
    else:
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    shutil.copyfile(tarea["label"], salida / "labels" / f"{stem}.txt")
    if not r["estado"]:
        r["estado"] = "OK" if frame_usado == frame else f"OK_FRAME_CORREGIDO_{frame_usado - frame:+d}"
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-original", default="datasets/dataset_original")
    ap.add_argument("--salida", default="datasets/dataset_sin_hud")
    ap.add_argument("--config", default="config/videos.json")
    ap.add_argument("--raiz", default=str(RAIZ), help="Raiz del repo (donde esta videos/)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--sin-verificar", action="store_true",
                    help="No comparar contra frames_originales (no recomendado)")
    ap.add_argument("--umbral-psnr", type=float, default=40.0,
                    help="PSNR minimo (dB) entre dataset_original y frames_originales para darlo por verificado")
    ap.add_argument("--vecinos", type=int, default=3, help="Frames vecinos a revisar si no coincide")
    ap.add_argument("--copiar-no-verificadas", action="store_true",
                    help="Copiar igual las imagenes cuyo PSNR queda bajo el umbral")
    ap.add_argument("--igualar-resolucion", action="store_true",
                    help="Reescalar el frame sin HUD a la resolucion del frame original "
                         "(por defecto se deja en la resolucion nativa de frames_sin_hud)")
    ap.add_argument("--videos", nargs="*", default=None, help="Limitar a estos videos (ej. video1 video2)")
    args = ap.parse_args()

    t0 = time.time()
    raiz = Path(args.raiz).resolve()
    ds = (raiz / args.dataset_original) if not Path(args.dataset_original).is_absolute() else Path(args.dataset_original)
    sal = (raiz / args.salida) if not Path(args.salida).is_absolute() else Path(args.salida)
    cfg_path = raiz / args.config
    config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    with open(ds / "correspondencia.csv", encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    if args.videos:
        filas = [r for r in filas if r["video"] in set(args.videos)]
    print(f"dataset_original: {ds}  ({len(filas)} imagenes)")

    (sal / "images").mkdir(parents=True, exist_ok=True)
    (sal / "labels").mkdir(parents=True, exist_ok=True)
    for extra in ("clases.yaml",):
        if (ds / extra).exists():
            shutil.copyfile(ds / extra, sal / extra)

    tareas = []
    carpetas_usadas = {}
    for video in sorted({r["video"] for r in filas}):
        filas_v = [r for r in filas if r["video"] == video]
        necesarios = {int(r["frame"]) for r in filas_v}
        print(f"\n[{video}] {len(filas_v)} imagenes")
        print("  frames_originales:")
        c_orig, idx_orig = elegir_carpeta(candidatos_carpeta(raiz, config, video, "frames_originales"), necesarios) \
            if not args.sin_verificar else (None, {})
        print("  frames_sin_hud:")
        c_sin, idx_sin = elegir_carpeta(candidatos_carpeta(raiz, config, video, "frames_sin_hud"), necesarios)
        carpetas_usadas[video] = {"frames_originales": str(c_orig) if c_orig else None,
                                  "frames_sin_hud": str(c_sin) if c_sin else None}
        if c_sin is None:
            print(f"  !! No se encontro carpeta sin HUD para {video}")
        for r in filas_v:
            stem = Path(r["imagen"]).stem
            # a cada tarea solo se le pasan su frame y sus vecinos (no el indice completo)
            n = int(r["frame"])
            vent = range(n - args.vecinos, n + args.vecinos + 1)
            idx_orig_red = {k: idx_orig[k] for k in vent if k in idx_orig}
            idx_sin_red = {k: idx_sin[k] for k in vent if k in idx_sin}
            img_o = ds / "images" / r["imagen"]
            h = w = None
            if args.igualar_resolucion:
                im = cv2.imread(str(img_o), cv2.IMREAD_UNCHANGED)
                h, w = im.shape[:2]
            tareas.append(dict(fila=r, idx_orig=idx_orig_red, idx_sin=idx_sin_red, salida=str(sal),
                               img_original=str(img_o), label=str(ds / "labels" / f"{stem}.txt"),
                               verificar=not args.sin_verificar, umbral_psnr=args.umbral_psnr,
                               vecinos=args.vecinos, copiar_no_verificadas=args.copiar_no_verificadas,
                               igualar_resolucion=args.igualar_resolucion, res_original=(h, w)))

    print(f"\nProcesando {len(tareas)} imagenes con {args.workers} procesos...")
    resultados = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(procesar, tareas, chunksize=8), 1):
            resultados.append(res)
            if i % 100 == 0 or i == len(tareas):
                print(f"  {i}/{len(tareas)}  ({time.time() - t0:.0f} s)", flush=True)

    columnas = list(filas[0].keys()) + ["frame_usado", "psnr_vs_frames_originales",
                                        "archivo_frames_originales_usado", "archivo_sin_hud_usado",
                                        "resolucion_sin_hud", "estado"]
    with open(sal / "correspondencia.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        w.writerows(resultados)
    malos = [r for r in resultados if not r["estado"].startswith("OK")]
    if malos:
        with open(sal / "faltantes.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
            w.writeheader()
            w.writerows(malos)

    conteo: dict[str, int] = {}
    for r in resultados:
        conteo[r["estado"]] = conteo.get(r["estado"], 0) + 1
    resumen = {"dataset_original": str(ds), "dataset_sin_hud": str(sal), "imagenes_entrada": len(filas),
               "imagenes_generadas": sum(1 for r in resultados if r["estado"].startswith("OK")),
               "estados": conteo, "carpetas_usadas": carpetas_usadas,
               "igualar_resolucion": args.igualar_resolucion, "verificado": not args.sin_verificar,
               "umbral_psnr": args.umbral_psnr, "segundos": round(time.time() - t0, 1)}
    (sal / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(resumen, indent=2, ensure_ascii=False))
    if malos:
        print(f"\n!! {len(malos)} imagenes con problema -> {sal / 'faltantes.csv'}")


if __name__ == "__main__":
    main()
