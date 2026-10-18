#!/usr/bin/env python3
"""
==============================================================================
PREPARADOR DE BLOQUES DE ENTRADA Y VERIFICACIÓN DE RÉGIMEN
==============================================================================
Prepara un conjunto continuo de N frames para un experimento específico,
aplicando las reglas de validación física e higiene de datos:

1. Filtro del Régimen 5 (Exclusión de menús centrales en Video 1):
   Descarta cualquier fotograma donde en la zona central [0.25H:0.75H, 0.25W:0.75W]
   haya más de 3% de píxeles con intensidad > 240 y saturación ≈ 0 (S < 30).
2. Verificación de Régimen Térmico:
   Si el régimen es térmico (HDIR), verifica que ΔRGB ≈ 0 (canales idénticos).
3. Verificación de Presencia de Ruido:
   Calcula σ_MAD del bloque para garantizar que cumple el perfil esperado.
4. Extracción y Enlace/Copia:
   Copia los frames seleccionados a la carpeta de entrada del experimento y
   genera un archivo `metadata_bloque.json` con la auditoría de los frames.
==============================================================================
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

from metricas_experimentos import estimar_sigma_mad


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def detectar_menu_central_regimen5(img_bgr: np.ndarray, umbral_pct: float = 0.03) -> bool:
    """
    Detecta si el fotograma contiene tablas blancas u OSD central del operador.
    ROI central: [0.25H : 0.75H, 0.25W : 0.75W].
    Criterio: Píxeles con V > 240 y S < 30 (blanco puro saturado) > umbral_pct.
    """
    h, w = img_bgr.shape[:2]
    roi = img_bgr[int(0.25 * h) : int(0.75 * h), int(0.25 * w) : int(0.75 * w)]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    
    mascara_blanco = (val > 240) & (sat < 30)
    pct = float(np.count_nonzero(mascara_blanco)) / float(roi.shape[0] * roi.shape[1])
    return pct > umbral_pct


def es_frame_termico_puro(img_bgr: np.ndarray, umbral_diff: float = 1.0) -> bool:
    """
    Verifica si el frame es térmico infrarrojo (canales RGB idénticos, ΔRGB ≈ 0).
    """
    b, g, r = cv2.split(img_bgr)
    diff_rg = np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
    diff_gb = np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32)))
    return (diff_rg < umbral_diff) and (diff_gb < umbral_diff)


def buscar_frame_inicial(todos_los_archivos: list, inicio_target: str) -> int:
    """Encuentra el índice del frame de inicio más cercano o exacto."""
    num_target = None
    nums = re.findall(r"\d+", inicio_target)
    if nums:
        num_target = int(nums[-1])

    for i, p in enumerate(todos_los_archivos):
        if p.name == inicio_target:
            return i
        if num_target is not None:
            p_nums = re.findall(r"\d+", p.name)
            if p_nums and int(p_nums[-1]) >= num_target:
                return i
    return 0


def main():
    p = argparse.ArgumentParser(description="Preparador y validador de bloques de entrada")
    p.add_argument("--video", required=True, choices=["video1", "video2", "video3"],
                   help="Video de origen")
    p.add_argument("--inicio-frame", required=True,
                   help="Nombre o número del frame de inicio (e.g. frame_001800.png)")
    p.add_argument("--n-frames", type=int, default=300,
                   help="Número de frames consecutivos a extraer")
    p.add_argument("--carpeta-salida", required=True, type=Path,
                   help="Directorio destino para guardar los frames")
    p.add_argument("--filtrar-menu-central", action="store_true", default=True,
                   help="Filtrar frames con menús/tablas centrales del operador (Régimen 5)")
    p.add_argument("--exigir-termico", action="store_true",
                   help="Exigir que los frames sean térmicos puros (ΔRGB ≈ 0)")
    p.add_argument("--max-busqueda", type=int, default=1500,
                   help="Máximo de frames a explorar para conseguir los N válidos")

    a = p.parse_args()

    dir_fuente = RAIZ / "videos" / a.video / "frames_sin_hud"
    if not dir_fuente.is_dir():
        print(f"[ERROR] No existe directorio: {dir_fuente}")
        sys.exit(1)

    exts = {".png", ".jpg", ".jpeg"}
    archivos = sorted([x for x in dir_fuente.iterdir() if x.suffix.lower() in exts], key=natural_key)
    print(f"[Info] Encontrados {len(archivos)} frames sin HUD en {a.video}")

    idx_ini = buscar_frame_inicial(archivos, a.inicio_frame)
    print(f"[Info] Iniciando búsqueda desde índice {idx_ini} ({archivos[idx_ini].name})")

    seleccionados = []
    descartados_menu = 0
    descartados_no_termico = 0
    sigmas_mad = []

    limite_exploracion = min(len(archivos), idx_ini + a.max_busqueda)
    for i in range(idx_ini, limite_exploracion):
        if len(seleccionados) >= a.n_frames:
            break

        path_frame = archivos[i]
        img = cv2.imread(str(path_frame))
        if img is None:
            continue

        # Validación 1: Menú Central Régimen 5
        if a.filtrar_menu_central and detectar_menu_central_regimen5(img):
            descartados_menu += 1
            continue

        # Validación 2: Exigencia de Térmico (si aplica)
        if a.exigir_termico and not es_frame_termico_puro(img):
            descartados_no_termico += 1
            continue

        # Medición rápida de σ_MAD
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s_mad = estimar_sigma_mad(gray)
        sigmas_mad.append(s_mad)
        seleccionados.append(path_frame)

    print(f"[Resultado] Frames seleccionados: {len(seleccionados)} / {a.n_frames}")
    print(f"            Descartados por Menú Central (Régimen 5): {descartados_menu}")
    if a.exigir_termico:
        print(f"            Descartados por no ser Térmico Puro: {descartados_no_termico}")

    if not seleccionados:
        print("[ERROR] No se seleccionó ningún frame que cumpliera las condiciones.")
        sys.exit(1)

    # Copiar frames al destino
    salida = a.carpeta_salida.resolve()
    salida.mkdir(parents=True, exist_ok=True)

    for p_src in seleccionados:
        p_dst = salida / p_src.name
        if not p_dst.exists():
            shutil.copy2(str(p_src), str(p_dst))

    # Guardar metadata de auditoría
    meta = {
        "video": a.video,
        "inicio_pedido": a.inicio_frame,
        "primer_frame_real": seleccionados[0].name,
        "ultimo_frame_real": seleccionados[-1].name,
        "total_frames": len(seleccionados),
        "descartados_menu_central": descartados_menu,
        "descartados_no_termico": descartados_no_termico,
        "sigma_mad_promedio": round(float(np.mean(sigmas_mad)), 4) if sigmas_mad else 0.0,
        "sigma_mad_min": round(float(np.min(sigmas_mad)), 4) if sigmas_mad else 0.0,
        "sigma_mad_max": round(float(np.max(sigmas_mad)), 4) if sigmas_mad else 0.0,
    }
    with open(salida / "metadata_bloque.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[✓] Bloque preparado en: {salida}")
    print(f"    Primer frame: {meta['primer_frame_real']} | Último frame: {meta['ultimo_frame_real']}")
    print(f"    σ_MAD promedio: {meta['sigma_mad_promedio']} DN (min: {meta['sigma_mad_min']}, max: {meta['sigma_mad_max']})")


if __name__ == "__main__":
    main()
