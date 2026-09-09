#!/usr/bin/env python3
"""
Pipeline Filtro Bilateral Espacio-Temporal Clasico (Baseline no-neuronal).
Sirve como punto de referencia clasico (baseline) para demostrar en la tesis
la superioridad del Deep Learning frente al procesamiento digital de senales tradicional.
"""

import argparse
import fcntl
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
    p = argparse.ArgumentParser(description="Filtro Bilateral Espacio-Temporal Clasico.")
    p.add_argument("--video", required=True)
    p.add_argument("--modo", choices=("original", "sin_hud"), required=True)
    p.add_argument("--d", type=int, default=7, help="Diametro del vecindario bilateral (default: 7).")
    p.add_argument("--sigma-color", type=float, default=25.0, help="Sigma en espacio de color/intensidad.")
    p.add_argument("--sigma-space", type=float, default=25.0, help="Sigma en espacio de coordenadas.")
    p.add_argument("--num-frames", type=int, default=5, help="Ventana temporal.")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--id-experimento", help="Nombre del experimento.")
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


def main():
    a = argumentos()
    inicio = time.monotonic()
    cfg = cargar_json()
    v = cfg[a.video]
    fuente_cfg = v.get("extraccion", {}) if a.modo == "original" else v.get("hud", {}).get("limpieza", {})
    fuente = Path(fuente_cfg["carpeta_salida"]).resolve()
    base_video = Path(v["ruta"]).resolve().parent.parent

    nombre = a.id_experimento or (f"Bilateral_Temporal_{a.modo}")
    salida = base_video / nombre
    shutil.rmtree(salida, ignore_errors=True)
    salida.mkdir(parents=True, exist_ok=True)

    rutas_frames = listar(fuente, a.max_frames)
    n_frames = len(rutas_frames)
    radio = a.num_frames // 2

    print(f"Ejecutando Filtro Bilateral Espacio-Temporal en {n_frames} frames...")

    for i in tqdm(range(n_frames), desc="Filtro Bilateral", dynamic_ncols=True):
        indices = [min(max(i + k, 0), n_frames - 1) for k in range(-radio, radio + 1)]
        ventana_bilateral = []

        for idx in indices:
            bgr = cv2.imread(str(rutas_frames[idx]), cv2.IMREAD_COLOR)
            # Filtro bilateral espacial
            bilat = cv2.bilateralFilter(bgr, d=a.d, sigmaColor=a.sigma_color, sigmaSpace=a.sigma_space)
            ventana_bilateral.append(bilat.astype(np.float32))

        # Promedio temporal de las respuestas bilaterales
        promedio_temp = np.mean(ventana_bilateral, axis=0).round().astype(np.uint8)

        destino = salida / f"{rutas_frames[i].stem}.png"
        cv2.imwrite(str(destino), promedio_temp, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    ruta_lock = RUTA_JSON.with_suffix(".lock")
    registro = {
        "nombre": nombre,
        "estado": "completada",
        "modo": a.modo,
        "carpeta_salida": str(salida.relative_to(RAIZ)),
        "metodo": "Filtro_Bilateral_Espacio_Temporal_Clasico",
        "frames_procesados": n_frames,
        "fecha_ejecucion": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tiempo_total_minutos": (time.monotonic() - inicio) / 60.0,
    }

    with ruta_lock.open("w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        configuracion = cargar_json()
        configuracion[a.video].setdefault("modelos", {})
        configuracion[a.video]["modelos"][nombre] = registro
        temporal = RUTA_JSON.with_suffix(".json.tmp")
        temporal.write_text(json.dumps(configuracion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporal.replace(RUTA_JSON)
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    print(f"Filtro Bilateral completado: {n_frames} frames guardados en {salida}.")


if __name__ == "__main__":
    main()
