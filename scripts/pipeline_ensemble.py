#!/usr/bin/env python3
"""
Modulo de Ensamble y Fusion de Modelos de Denoising para video FLIR.
Combina predicciones de modelos espaciales (Blind2Unblind/StructN2V) y temporales (Frames2Residual/FastDVDnet)
mediante:
1. Fusion Wavelet Multi-escala (bajas frecuencias termicas temporales + altas frecuencias espaciales).
2. Promedio Adaptativo Ponderado.
3. Mediana Robusta Anti-outliers.
"""

import argparse
import fcntl
import gc
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
RUTA_JSON = RAIZ / "config" / "videos.json"


def argumentos():
    p = argparse.ArgumentParser(description="Ensamble y Fusion de Modelos de Denoising.")
    p.add_argument("--video", required=True)
    p.add_argument("--modo", choices=("original", "sin_hud"), required=True)
    p.add_argument("--metodo-fusion", choices=("wavelet", "promedio", "mediana"), default="wavelet")
    p.add_argument("--modelos-entrada", nargs="+", required=True, help="Carpetas o IDs de modelos a fusionar.")
    p.add_argument("--pesos", nargs="+", type=float, help="Pesos para promedio ponderado.")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--id-experimento", required=True, help="Nombre del experimento de salida.")
    return p.parse_args()


def cargar_json():
    with RUTA_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def natural(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def listar(carpeta, limite=None):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    rutas = sorted((p for p in Path(carpeta).iterdir() if p.is_file() and p.suffix.lower() in exts), key=natural)
    return rutas[:limite] if limite else rutas


def fusion_wavelet_2d(img_temporal, img_espacial):
    """
    Fusion en dominio frecuencial:
    - Bajas frecuencias (Luminancia y gradiente termico estable): provienen del modelo temporal.
    - Altas frecuencias (Bordes finos, copas de arboles, senderos): provienen del modelo espacial.
    """
    f_t = img_temporal.astype(np.float32)
    f_s = img_espacial.astype(np.float32)

    # Filtro Gaussiano como aproximacion multi-escala separable
    blur_t = cv2.GaussianBlur(f_t, (7, 7), 1.5)
    blur_s = cv2.GaussianBlur(f_s, (7, 7), 1.5)

    # Detalle de alta frecuencia espacial: S - blur(S)
    detalle_espacial = f_s - blur_s

    # Fusion: Base temporal suave + Detalles nítidos espaciales
    fusion = blur_t + detalle_espacial
    return np.clip(fusion, 0.0, 255.0).round().astype(np.uint8)


def actualizar_registro(a, salida_dir, inicio, cantidad):
    ruta_lock = RUTA_JSON.with_suffix(".lock")
    registro = {
        "nombre": a.id_experimento,
        "estado": "completada",
        "modo": a.modo,
        "carpeta_salida": str(salida_dir.relative_to(RAIZ)),
        "metodo_fusion": a.metodo_fusion,
        "modelos_entrada": a.modelos_entrada,
        "frames_procesados": cantidad,
        "fecha_ejecucion": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tiempo_total_minutos": (time.monotonic() - inicio) / 60.0,
    }

    with ruta_lock.open("w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        configuracion = cargar_json()
        configuracion[a.video].setdefault("modelos", {})
        configuracion[a.video]["modelos"][a.id_experimento] = registro
        temporal = RUTA_JSON.with_suffix(".json.tmp")
        temporal.write_text(json.dumps(configuracion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporal.replace(RUTA_JSON)
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def main():
    a = argumentos()
    inicio = time.monotonic()
    cfg = cargar_json()
    v = cfg[a.video]
    base_video = Path(v["ruta"]).resolve().parent.parent

    # Resolver carpetas de entrada de los modelos
    carpetas_entrada = []
    for mod in a.modelos_entrada:
        p = Path(mod)
        if p.is_dir():
            carpetas_entrada.append(p)
        elif (base_video / mod).is_dir():
            carpetas_entrada.append(base_video / mod)
        else:
            raise FileNotFoundError(f"No se encontro la carpeta del modelo: {mod}")

    salida_dir = base_video / a.id_experimento
    shutil.rmtree(salida_dir, ignore_errors=True)
    salida_dir.mkdir(parents=True, exist_ok=True)

    # Listar frames comunes
    listas_frames = [listar(c, a.max_frames) for c in carpetas_entrada]
    n_frames = min(len(l) for l in listas_frames)

    print(f"Ejecutando Ensamble ({a.metodo_fusion}) sobre {len(carpetas_entrada)} modelos ({n_frames} frames)...")

    for i in tqdm(range(n_frames), desc=f"Ensamble {a.id_experimento}", dynamic_ncols=True):
        nombre_frame = listas_frames[0][i].name
        imgs = [cv2.imread(str(listas_frames[m][i]), cv2.IMREAD_COLOR) for m in range(len(carpetas_entrada))]

        if a.metodo_fusion == "wavelet" and len(imgs) >= 2:
            # imgs[0]: Temporal (ej. Frames2Residual), imgs[1]: Espacial (ej. StructN2V / Blind2Unblind)
            out_img = fusion_wavelet_2d(imgs[0], imgs[1])
        elif a.metodo_fusion == "mediana":
            stack = np.stack(imgs, axis=0)
            out_img = np.median(stack, axis=0).round().astype(np.uint8)
        else:  # Promedio
            pesos = a.pesos or [1.0 / len(imgs)] * len(imgs)
            norm_pesos = [p / sum(pesos) for p in pesos]
            stack = sum(img.astype(np.float32) * w for img, w in zip(imgs, norm_pesos))
            out_img = np.clip(stack, 0.0, 255.0).round().astype(np.uint8)

        destino = salida_dir / nombre_frame
        cv2.imwrite(str(destino), out_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    actualizar_registro(a, salida_dir, inicio, n_frames)
    print(f"Ensamble {a.id_experimento} finalizado: {n_frames} frames guardados en {salida_dir}.")


if __name__ == "__main__":
    main()
