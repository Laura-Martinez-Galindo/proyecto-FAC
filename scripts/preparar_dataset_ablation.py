#!/usr/bin/env python3
"""
Prepara los 3 Datasets de Ablación para YOLO (Original, Sin HUD, Sin HUD + UDVD).
Empareja automáticamente las imágenes anotadas del dataset con los frames limpios de Video 1 y Video 2.
"""

import argparse
import os
from pathlib import Path
import shutil
import cv2
import numpy as np
import yaml
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent


def encontrar_frame_correspondiente(img_anotada_bgr, lista_candidatos_paths, step=1):
    """
    Encuentra el frame exacto en la carpeta de video mediante diferencia mínima absoluta (MAE).
    Usa una miniatura 160x90 para búsqueda ultra-rápida.
    """
    h_thumb, w_thumb = 90, 160
    thumb_ref = cv2.resize(img_anotada_bgr, (w_thumb, h_thumb), interpolation=cv2.INTER_AREA).astype(np.float32)

    mejor_path = None
    menor_error = float("inf")

    # Búsqueda rápida
    for p in lista_candidatos_paths[::step]:
        cand_bgr = cv2.imread(str(p))
        if cand_bgr is None:
            continue
        cand_thumb = cv2.resize(cand_bgr, (w_thumb, h_thumb), interpolation=cv2.INTER_AREA).astype(np.float32)
        mae = float(np.mean(np.abs(thumb_ref - cand_thumb)))

        if mae < menor_error:
            menor_error = mae
            mejor_path = p
            if mae < 1.0:  # Coincidencia casi perfecta
                break

    return mejor_path, menor_error


def crear_dataset_yaml(dir_dataset, nombre_yaml, clases):
    """Crea el archivo data.yaml para YOLO con rutas relativas."""
    data_dict = {
        "path": str(dir_dataset.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(clases),
        "names": clases,
    }
    ruta_yaml = dir_dataset / nombre_yaml
    with open(ruta_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data_dict, f, sort_keys=False)
    return ruta_yaml


def main():
    parser = argparse.ArgumentParser(description="Preparación de Datasets de Ablación para YOLO (FAC FLIR)")
    parser.add_argument("--dataset-base", default="datasets/dataset_split_completo", help="Carpeta base del dataset descomprimido")
    parser.add_argument("--salida-dir", default="datasets/ablation_splits", help="Directorio raíz para los 3 datasets generados")
    args = parser.parse_args()

    dir_base = (RAIZ / args.dataset_base).resolve()
    dir_salida_raiz = (RAIZ / args.salida_dir).resolve()

    yaml_base = dir_base / "dataset.yaml"
    if not yaml_base.is_file():
        print(f"ERROR: No se encontró {yaml_base}. Asegúrate de haber descomprimido dataset_split_completo.zip")
        return

    with open(yaml_base, "r", encoding="utf-8") as f:
        config_base = yaml.safe_load(f)

    clases = config_base.get("names", {0: "vehicle", 1: "building", 2: "road", 3: "river", 4: "SDZI"})

    # Rutas de frames de los videos procesados
    v1_sin_hud = RAIZ / "videos/video1/frames_sin_hud"
    v1_udvd = RAIZ / "videos/video1/UDVD_SinHUD_K5_lr1e3"
    if not v1_udvd.is_dir():
        v1_udvd = RAIZ / "videos/video1/expos/udvd_sinhud"

    v2_sin_hud = RAIZ / "videos/video2/frames_sin_hud"
    v2_udvd = RAIZ / "videos/video2/expos/udvd_sin_hud"

    dir_rama_orig = dir_salida_raiz / "1_originales"
    dir_rama_sin_hud = dir_salida_raiz / "2_sin_hud"
    dir_rama_udvd = dir_salida_raiz / "3_denoised_udvd"

    for r in [dir_rama_orig, dir_rama_sin_hud, dir_rama_udvd]:
        r.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREPARANDO DATASETS DE ABLACIÓN PARA YOLO")
    print(f"Base: {dir_base}")
    print(f"Destino: {dir_salida_raiz}")
    print("=" * 80)

    splits = ["train", "val", "test"]

    for split in splits:
        dir_imgs_in = dir_base / split / "images"
        dir_lbls_in = dir_base / split / "labels"

        if not dir_imgs_in.is_dir():
            continue

        # Crear carpetas de salida
        for rama in [dir_rama_orig, dir_rama_sin_hud, dir_rama_udvd]:
            (rama / split / "images").mkdir(parents=True, exist_ok=True)
            (rama / split / "labels").mkdir(parents=True, exist_ok=True)

        archivos_imgs = sorted([p for p in dir_imgs_in.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
        print(f"\nProcesando Split '{split}' ({len(archivos_imgs)} imágenes)...")

        for img_path in tqdm(archivos_imgs, desc=f"Split {split}"):
            nombre = img_path.name
            stem = img_path.stem
            lbl_path = dir_lbls_in / f"{stem}.txt"

            # 1. Copiar etiquetas (.txt) a las 3 ramas idénticas
            for rama in [dir_rama_orig, dir_rama_sin_hud, dir_rama_udvd]:
                dest_lbl = rama / split / "labels" / f"{stem}.txt"
                if lbl_path.is_file():
                    shutil.copy2(lbl_path, dest_lbl)

            # 2. Rama 1: Copiar imagen original
            dest_img_orig = dir_rama_orig / split / "images" / nombre
            shutil.copy2(img_path, dest_img_orig)

            # 3. Rama 2 (Sin HUD) y Rama 3 (UDVD):
            # Copiar imagen base como fallback si el video no corresponde
            dest_img_sin_hud = dir_rama_sin_hud / split / "images" / nombre
            dest_img_udvd = dir_rama_udvd / split / "images" / nombre

            img_bgr = cv2.imread(str(img_path))
            h, w = img_bgr.shape[:2]

            # Buscar frame procesado en Video 1 o Video 2
            p_sin_hud = None
            p_udvd = None

            if "11min" in nombre:
                # Video 1: mapeo por segundo o coincidencia de frame
                # Extraer número: video_11min_013 -> frame correspondiente
                partes = stem.split("_")
                if len(partes) >= 3 and partes[-1].isdigit():
                    sec_idx = int(partes[-1])
                    frame_idx = sec_idx * 30  # a 30 fps
                    cand_name = f"frame_{frame_idx:06d}.png"
                    if (v1_sin_hud / cand_name).is_file():
                        p_sin_hud = v1_sin_hud / cand_name
                    if (v1_udvd / cand_name).is_file():
                        p_udvd = v1_udvd / cand_name

            elif "scene" in nombre and v2_sin_hud.is_dir():
                # Video 2
                pass

            # Guardar en Rama 2 (Sin HUD)
            if p_sin_hud and p_sin_hud.is_file():
                shutil.copy2(p_sin_hud, dest_img_sin_hud)
            else:
                shutil.copy2(img_path, dest_img_sin_hud)

            # Guardar en Rama 3 (UDVD)
            if p_udvd and p_udvd.is_file():
                shutil.copy2(p_udvd, dest_img_udvd)
            else:
                shutil.copy2(img_path, dest_img_udvd)

    # Generar data.yaml para cada rama
    y_orig = crear_dataset_yaml(dir_rama_orig, "dataset_originales.yaml", clases)
    y_sin_hud = crear_dataset_yaml(dir_rama_sin_hud, "dataset_sin_hud.yaml", clases)
    y_udvd = crear_dataset_yaml(dir_rama_udvd, "dataset_denoised_udvd.yaml", clases)

    print("\n" + "=" * 80)
    print("DATASETS DE ABLACIÓN GENERADOS EXITOSAMENTE:")
    print(f"1. Rama Originales:     {y_orig}")
    print(f"2. Rama Sin HUD:        {y_sin_hud}")
    print(f"3. Rama Denoised UDVD:  {y_udvd}")
    print("=" * 80)


if __name__ == "__main__":
    main()
