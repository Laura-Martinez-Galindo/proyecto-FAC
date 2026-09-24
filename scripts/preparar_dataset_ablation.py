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


def buscar_frame_limpio(carpetas_candidatas, indice_frame):
    """
    Busca un frame por índice probando múltiples formatos de dígitos (5 dígitos, 6 dígitos, etc.)
    y extensiones (.png, .jpg).
    """
    if indice_frame is None or indice_frame < 0:
        return None

    formatos = [
        f"frame_{indice_frame:05d}.png",
        f"frame_{indice_frame:06d}.png",
        f"frame_{indice_frame:04d}.png",
        f"frame_{indice_frame:05d}.jpg",
        f"frame_{indice_frame:06d}.jpg",
        f"frame_{indice_frame}.png",
    ]

    for carpeta in carpetas_candidatas:
        if not carpeta or not carpeta.is_dir():
            continue
        for nombre in formatos:
            ruta = carpeta / nombre
            if ruta.is_file():
                return ruta
    return None


def encontrar_directorios_video(raiz):
    """Detecta automáticamente todas las posibles ubicaciones de frames limpios en Hypatia/Local."""
    v1_sin_hud_cands = [
        raiz / "videos/video1/frames_sin_hud",
        raiz / "videos/video1/sin_hud",
    ]
    v1_udvd_cands = [
        raiz / "videos/video1/expos/udvd_sinhud",
        raiz / "videos/video1/UDVD_SinHUD_K5_lr1e3",
        raiz / "videos/video1/udvd_sinhud",
        raiz / "videos/video1/udvd_sin_hud",
        raiz / "videos/video1/udvd",
    ]
    v2_sin_hud_cands = [
        raiz / "videos/video2/frames_sin_hud",
        raiz / "videos/video2/sin_hud",
    ]
    v2_udvd_cands = [
        raiz / "videos/video2/expos/udvd_sin_hud",
        raiz / "videos/video2/expos/udvd_sinhud",
        raiz / "videos/video2/udvd_sin_hud",
        raiz / "videos/video2/udvd_sinhud",
        raiz / "videos/video2/UDVD_sin_hud",
        raiz / "videos/video2/udvd",
    ]
    return {
        "v1_sin_hud": [c for c in v1_sin_hud_cands if c.is_dir()],
        "v1_udvd": [c for c in v1_udvd_cands if c.is_dir()],
        "v2_sin_hud": [c for c in v2_sin_hud_cands if c.is_dir()],
        "v2_udvd": [c for c in v2_udvd_cands if c.is_dir()],
    }


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
    parser.add_argument("--solo-videos-limpios", action="store_true", default=True, help="Filtrar únicamente Video 1 y Video 2 (Ablación Pura de 863 imágenes)")
    parser.add_argument("--incluir-todos", dest="solo_videos_limpios", action="store_false", help="Incluir las 1.702 imágenes completas")
    args = parser.parse_args()

    dir_base = (RAIZ / args.dataset_base).resolve()
    dir_salida_raiz = (RAIZ / args.salida_dir).resolve()

    yaml_base = dir_base / "dataset.yaml"
    if not yaml_base.is_file():
        # Intentar buscar dataset.yaml dentro de subcarpetas
        posibles_yamls = list(dir_base.glob("**/dataset.yaml"))
        if posibles_yamls:
            yaml_base = posibles_yamls[0]
            dir_base = yaml_base.parent
        else:
            print(f"ERROR: No se encontró dataset.yaml en {dir_base}. Descomprime dataset_split_completo.zip primero.")
            return

    with open(yaml_base, "r", encoding="utf-8") as f:
        config_base = yaml.safe_load(f)

    clases = config_base.get("names", {0: "vehicle", 1: "building", 2: "road", 3: "river", 4: "SDZI"})

    rutas_videos = encontrar_directorios_video(RAIZ)
    print("=" * 80)
    print("CONFIGURACIÓN DE DIRECTORIOS DETECTADOS:")
    for k, v in rutas_videos.items():
        print(f"  - {k}: {[str(p) for p in v] or 'NO ENCONTRADO'}")
    print(f"Modo: {'Ablación Pura (Solo Video 1 y 2)' if args.solo_videos_limpios else 'Dataset Completo (1.702 frames)'}")
    print("=" * 80)

    dir_rama_orig = dir_salida_raiz / "1_originales"
    dir_rama_sin_hud = dir_salida_raiz / "2_sin_hud"
    dir_rama_udvd = dir_salida_raiz / "3_denoised_udvd"

    for r in [dir_rama_orig, dir_rama_sin_hud, dir_rama_udvd]:
        r.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    stats = {"total_evaluados": 0, "v1_match": 0, "v2_match": 0, "otros_ignorados": 0, "sin_hud_ok": 0, "udvd_ok": 0}

    for split in splits:
        dir_imgs_in = dir_base / split / "images"
        dir_lbls_in = dir_base / split / "labels"

        if not dir_imgs_in.is_dir():
            continue

        for rama in [dir_rama_orig, dir_rama_sin_hud, dir_rama_udvd]:
            (rama / split / "images").mkdir(parents=True, exist_ok=True)
            (rama / split / "labels").mkdir(parents=True, exist_ok=True)

        archivos_imgs = sorted([p for p in dir_imgs_in.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
        print(f"\nProcesando Split '{split}' ({len(archivos_imgs)} imágenes base)...")

        for img_path in tqdm(archivos_imgs, desc=f"Split {split}"):
            nombre = img_path.name
            stem = img_path.stem
            lbl_path = dir_lbls_in / f"{stem}.txt"

            es_v1 = "11min" in nombre
            es_v2 = "scene" in nombre or ("frame_" in nombre and "_" in stem and stem.split("_")[1].isdigit())
            
            if args.solo_videos_limpios and not (es_v1 or es_v2):
                stats["otros_ignorados"] += 1
                continue

            stats["total_evaluados"] += 1

            p_sin_hud = None
            p_udvd = None

            # 1. Video 1: Mapeo de segundos (video_11min_XXX) a índice de frame a 30 FPS
            if es_v1:
                stats["v1_match"] += 1
                partes = stem.split("_")
                if len(partes) >= 3 and partes[-1].isdigit():
                    sec_idx = int(partes[-1])
                    # Probar mapeo directo a 30 fps (frame_00390) y mapeos contiguos +-1
                    f_idx = sec_idx * 30
                    p_sin_hud = buscar_frame_limpio(rutas_videos["v1_sin_hud"], f_idx) or \
                                 buscar_frame_limpio(rutas_videos["v1_sin_hud"], f_idx + 1) or \
                                 buscar_frame_limpio(rutas_videos["v1_sin_hud"], f_idx - 1)
                    p_udvd = buscar_frame_limpio(rutas_videos["v1_udvd"], f_idx) or \
                             buscar_frame_limpio(rutas_videos["v1_udvd"], f_idx + 1) or \
                             buscar_frame_limpio(rutas_videos["v1_udvd"], f_idx - 1)

            # 2. Video 2: Mapeo de corte de escena (frame_000065_scene019)
            elif es_v2:
                stats["v2_match"] += 1
                partes = stem.split("_")
                if len(partes) >= 2 and partes[1].isdigit():
                    f_idx = int(partes[1])
                    p_sin_hud = buscar_frame_limpio(rutas_videos["v2_sin_hud"], f_idx)
                    p_udvd = buscar_frame_limpio(rutas_videos["v2_udvd"], f_idx)

            # 3. Guardar etiqueta .txt
            for rama in [dir_rama_orig, dir_rama_sin_hud, dir_rama_udvd]:
                dest_lbl = rama / split / "labels" / f"{stem}.txt"
                if lbl_path.is_file():
                    shutil.copy2(lbl_path, dest_lbl)

            # 4. Rama 1: Original
            dest_img_orig = dir_rama_orig / split / "images" / nombre
            shutil.copy2(img_path, dest_img_orig)

            # Leer dimensiones objetivo de la original (ej: 1920x1080)
            img_orig_bgr = cv2.imread(str(img_path))
            h_orig, w_orig = img_orig_bgr.shape[:2]

            # 5. Rama 2: Sin HUD (ProPainter)
            dest_img_sin_hud = dir_rama_sin_hud / split / "images" / nombre
            if p_sin_hud and p_sin_hud.is_file():
                img_sh = cv2.imread(str(p_sin_hud))
                if img_sh is not None:
                    if img_sh.shape[:2] != (h_orig, w_orig):
                        img_sh = cv2.resize(img_sh, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(str(dest_img_sin_hud), img_sh, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    stats["sin_hud_ok"] += 1
                else:
                    shutil.copy2(img_path, dest_img_sin_hud)
            else:
                shutil.copy2(img_path, dest_img_sin_hud)

            # 6. Rama 3: Denoised UDVD
            dest_img_udvd = dir_rama_udvd / split / "images" / nombre
            if p_udvd and p_udvd.is_file():
                img_u = cv2.imread(str(p_udvd))
                if img_u is not None:
                    if img_u.shape[:2] != (h_orig, w_orig):
                        img_u = cv2.resize(img_u, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(str(dest_img_udvd), img_u, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    stats["udvd_ok"] += 1
                else:
                    shutil.copy2(img_path, dest_img_udvd)
            elif p_sin_hud and p_sin_hud.is_file():
                # Si no hay UDVD específico para ese frame, usar Sin HUD como base de Rama 3
                shutil.copy2(dest_img_sin_hud, dest_img_udvd)
            else:
                shutil.copy2(img_path, dest_img_udvd)

    # Generar data.yaml para cada rama
    y_orig = crear_dataset_yaml(dir_rama_orig, "dataset_originales.yaml", clases)
    y_sin_hud = crear_dataset_yaml(dir_rama_sin_hud, "dataset_sin_hud.yaml", clases)
    y_udvd = crear_dataset_yaml(dir_rama_udvd, "dataset_denoised_udvd.yaml", clases)

    print("\n" + "=" * 80)
    print("RESUMEN DE EMPAREJAMIENTO DE ABLACIÓN:")
    print(f"  - Total frames incluidos:      {stats['total_evaluados']}")
    print(f"  - Emparejados Video 1:         {stats['v1_match']}")
    print(f"  - Emparejados Video 2:         {stats['v2_match']}")
    print(f"  - Otros clips omitidos:        {stats['otros_ignorados']}")
    print(f"  - Éxitos Sin HUD (ProPainter): {stats['sin_hud_ok']}/{stats['total_evaluados']}")
    print(f"  - Éxitos Denoised (UDVD):      {stats['udvd_ok']}/{stats['total_evaluados']}")
    print("=" * 80)
    print(f"1. Rama Originales:    {y_orig}")
    print(f"2. Rama Sin HUD:       {y_sin_hud}")
    print(f"3. Rama Denoised UDVD: {y_udvd}")
    print("=" * 80)


if __name__ == "__main__":
    main()
