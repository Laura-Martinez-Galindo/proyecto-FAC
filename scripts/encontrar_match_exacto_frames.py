#!/usr/bin/env python3
"""
Encuentra la correspondencia exacta (Frame Exacto y Timestamp) de cualquier imagen anotada
comparando características visuales (MAE en zona central sin HUD) contra los frames de Video 1, 2 y 3.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent


def buscar_frame_mas_cercano(img_ref_bgr, carpeta_frames, paso_coarse=60):
    """
    Búsqueda piramidal ultra-rápida Coarse-to-Fine.
    Paso 1: Muestrea 1 frame cada 2 segundos (paso=60) -> Menos de 300 lecturas.
    Paso 2: Refina únicamente en una ventana de +/- 60 frames alrededor del mínimo.
    """
    if not carpeta_frames.is_dir():
        return None, float("inf"), None

    h, w = img_ref_bgr.shape[:2]
    # Zona central (sin HUD en los bordes)
    crop_ref = img_ref_bgr[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    thumb_ref = cv2.resize(crop_ref, (48, 27), interpolation=cv2.INTER_AREA).astype(np.float32)

    archivos = sorted([p for p in carpeta_frames.iterdir() if p.suffix.lower() in [".png", ".jpg"]])
    if not archivos:
        return None, float("inf"), None

    mejor_archivo = None
    menor_mae = float("inf")
    mejor_idx = 0

    total_coarse = len(range(0, len(archivos), paso_coarse))
    print(f"    -> Escaneando {total_coarse} frames clave (paso={paso_coarse})...")

    # 1. Búsqueda Gruesa (Coarse Search)
    for idx in range(0, len(archivos), paso_coarse):
        p = archivos[idx]
        cand = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_4)
        if cand is None:
            continue
        h_c, w_c = cand.shape[:2]
        crop_cand = cand[int(h_c*0.25):int(h_c*0.75), int(w_c*0.25):int(w*0.75)]
        thumb_cand = cv2.resize(crop_cand, (48, 27), interpolation=cv2.INTER_AREA).astype(np.float32)

        mae = float(np.mean(np.abs(thumb_ref - thumb_cand)))
        if mae < menor_mae:
            menor_mae = mae
            mejor_archivo = p
            mejor_idx = idx

    # 2. Refinamiento Fino (Fine Search en ventana de +/- 60 frames)
    sub_inicio = max(0, mejor_idx - paso_coarse)
    sub_fin = min(len(archivos), mejor_idx + paso_coarse + 1)
    print(f"    -> Refinando en ventana [{sub_inicio}..{sub_fin}] ({sub_fin - sub_inicio} frames)...")

    for p in archivos[sub_inicio:sub_fin]:
        cand = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_2)
        if cand is None:
            continue
        h_c, w_c = cand.shape[:2]
        crop_cand = cand[int(h_c*0.25):int(h_c*0.75), int(w_c*0.25):int(w*0.75)]
        thumb_cand = cv2.resize(crop_cand, (48, 27), interpolation=cv2.INTER_AREA).astype(np.float32)

        mae = float(np.mean(np.abs(thumb_ref - thumb_cand)))
        if mae < menor_mae:
            menor_mae = mae
            mejor_archivo = p

    return mejor_archivo, menor_mae, mejor_idx


def main():
    parser = argparse.ArgumentParser(description="Buscar Match Exacto de Frame en Videos")
    parser.add_argument("--img-test", default="datasets/dataset_split_completo/test/images/video_11min_013.jpg", help="Imagen anotada a buscar")
    args = parser.parse_args()

    ruta_ref = (RAIZ / args.img_test).resolve()
    if not ruta_ref.is_file():
        posibles = list(RAIZ.glob(f"**/{Path(args.img_test).name}"))
        if posibles:
            ruta_ref = posibles[0]
        else:
            print(f"ERROR: No se encontró {args.img_test}")
            return

    img_ref = cv2.imread(str(ruta_ref))
    print("=" * 80)
    print(f"BUSCANDO MATCH VISUAL EXACTO PARA: {ruta_ref.name}")
    print(f"Dimensiones de referencia: {img_ref.shape}")
    print("=" * 80)

    carpetas = [
        ("Video 1 (Frames Originales)", RAIZ / "videos/video1/frames_originales"),
        ("Video 2 (Frames Originales)", RAIZ / "videos/video2/frames_originales"),
    ]

    for nombre_vid, carpeta in carpetas:
        if not carpeta.is_dir():
            print(f"  [-] {nombre_vid}: Carpeta no existe en {carpeta}")
            continue
        print(f"\n[*] Analizando {nombre_vid}...")
        match_p, mae, idx = buscar_frame_mas_cercano(img_ref, carpeta, paso_coarse=60)
        if match_p:
            print(f"  [+] {nombre_vid:<28} -> Mejor Match: {match_p.name} (Diferencia MAE: {mae:.2f})")
            if mae < 15.0:
                print(f"      *** ¡MATCH ENCONTRADO EN {nombre_vid}! Archivo: {match_p.name} ***")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
