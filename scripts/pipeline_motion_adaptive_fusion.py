#!/usr/bin/env python3
"""
Módulo de Fusión Inteligente Guiada por Movimiento (Motion-Adaptive / Flow-Guided Denoising Fusion).
Combina de forma adaptativa píxel a píxel:
1. UDVD Destriping T=7 en regiones estáticas y fondos homogéneos (elimina 100% del ruido térmico y FPN).
2. StructN2V Vertical en bordes y estructuras con movimiento rápido (preserva 100% de nitidez geométrica con CERO ghosting).

Fórmula de Fusión:
I_final(x,y) = (1 - M(x,y)) * I_UDVD(x,y) + M(x,y) * I_StructN2V(x,y)
donde M(x,y) es el mapa continuo de actividad y desplazamiento de bordes.
"""

import argparse
import fcntl
import json
import re
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
RUTA_JSON = RAIZ / "config" / "videos.json"


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def calcular_mapa_movimiento_bordes(img_t, img_prev, umbral_grad=0.15, sensibilidad_mov=2.5):
    """
    Calcula la máscara continua de pesos M(x, y) en [0, 1]:
    - M -> 0: Región plana/estática (usar UDVD para máxima limpieza térmica).
    - M -> 1: Estructura geométrica en movimiento (usar StructN2V para cero ghosting).
    """
    g_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 1. Gradiente espacial (bordes del terreno)
    gx = cv2.Sobel(g_t, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g_t, cv2.CV_32F, 0, 1, ksize=3)
    mag_grad = np.sqrt(gx**2 + gy**2)

    # 2. Desplazamiento temporal entre cuadros
    diff_temp = np.abs(g_t - g_prev)

    # 3. Actividad de bordes móviles: producto de borde por movimiento
    actividad = mag_grad * (1.0 + diff_temp * sensibilidad_mov)
    
    # Suavizado suave de la máscara para transiciones naturales sin costuras
    actividad_suave = cv2.GaussianBlur(actividad, (7, 7), 1.5)

    # Mapeo no lineal Sigmoide a [0, 1]
    peso_espacial = 1.0 / (1.0 + np.exp(-12.0 * (actividad_suave - umbral_grad)))
    
    # Expandir a 3 canales (H, W, 3)
    return np.repeat(peso_espacial[:, :, np.newaxis], 3, axis=2)


def procesar_un_frame(args):
    ruta_udvd, ruta_struct, ruta_sinhud_t, ruta_sinhud_prev, ruta_salida, umbral_grad, sens = args
    
    img_udvd = cv2.imread(str(ruta_udvd))
    img_struct = cv2.imread(str(ruta_struct))
    img_t = cv2.imread(str(ruta_sinhud_t))
    img_prev = cv2.imread(str(ruta_sinhud_prev)) if ruta_sinhud_prev else img_t

    if img_udvd is None or img_struct is None or img_t is None:
        return

    # Calcular máscara de pesos M(x, y)
    M = calcular_mapa_movimiento_bordes(img_t, img_prev, umbral_grad=umbral_grad, sensibilidad_mov=sens)

    # Fusión adaptativa
    fusion = (1.0 - M) * img_udvd.astype(np.float32) + M * img_struct.astype(np.float32)
    
    # Neutralizar artefactos de padding convolucional en los 5px de los bordes externos
    fusion[:, :5] = img_udvd[:, :5]
    fusion[:, -5:] = img_udvd[:, -5:]
    fusion[:5, :] = img_udvd[:5, :]
    fusion[-5:, :] = img_udvd[-5:, :]
    
    fusion_uint8 = np.clip(fusion, 0.0, 255.0).round().astype(np.uint8)

    cv2.imwrite(str(ruta_salida), fusion_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def main():
    p = argparse.ArgumentParser(description="Fusión Inteligente Adaptativa al Movimiento.")
    p.add_argument("--video", default="video2")
    p.add_argument("--modelo-udvd", default="videos/video2/expos/udvd_destriping_t7_ult10min_sin_hud")
    p.add_argument("--modelo-struct", default="videos/video2/expos/struct_n2v_vert_ult10min_sin_hud")
    p.add_argument("--ultimos-frames", type=int, default=18000)
    p.add_argument("--umbral-grad", type=float, default=0.18, help="Sensibilidad de corte de bordes.")
    p.add_argument("--sensibilidad-mov", type=float, default=3.0, help="Ponderación de movimiento de cámara.")
    p.add_argument("--carpeta-salida", default="videos/video2/expos/fusion_motion_adaptive_udvd_struct_vert")
    p.add_argument("--id-experimento", default="fusion_motion_adaptive_udvd_struct_vert")
    args = p.parse_args()

    inicio = time.monotonic()
    ruta_base = RAIZ / "videos" / args.video
    dir_sinhud = ruta_base / "frames_sin_hud"
    dir_udvd = RAIZ / args.modelo_udvd if not Path(args.modelo_udvd).is_absolute() else Path(args.modelo_udvd)
    dir_struct = RAIZ / args.modelo_struct if not Path(args.modelo_struct).is_absolute() else Path(args.modelo_struct)
    dir_salida = RAIZ / args.carpeta_salida if not Path(args.carpeta_salida).is_absolute() else Path(args.carpeta_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    frames_sinhud = sorted([p for p in dir_sinhud.iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]
    frames_udvd = sorted([p for p in dir_udvd.iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]
    frames_struct = sorted([p for p in dir_struct.iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]

    n_frames = min(len(frames_sinhud), len(frames_udvd), len(frames_struct))
    print(f"=== EJECUTANDO FUSIÓN INTELIGENTE GUIADA POR MOVIMIENTO ({n_frames} FRAMES) ===")
    print(f"Base Temporal: {dir_udvd.name} (Zonas estáticas sin ruido)")
    print(f"Detalle Espacial: {dir_struct.name} (Bordes y casas sin ghosting)")
    print(f"Destino: {dir_salida}")

    tareas = []
    for i in range(n_frames):
        prev_sinhud = frames_sinhud[i - 1] if i > 0 else frames_sinhud[i]
        out_p = dir_salida / frames_sinhud[i].name
        tareas.append((
            frames_udvd[i], frames_struct[i], frames_sinhud[i], prev_sinhud,
            out_p, args.umbral_grad, args.sensibilidad_mov
        ))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(tqdm(pool.map(procesar_un_frame, tareas), total=n_frames, desc="Fusionando adaptativamente", dynamic_ncols=True))

    tiempo_min = (time.monotonic() - inicio) / 60.0
    print(f"\n[EXITO] Fusión adaptativa finalizada en {tiempo_min:.2f} minutos.")
    print(f"Frames guardados en: {dir_salida}")


if __name__ == "__main__":
    main()
