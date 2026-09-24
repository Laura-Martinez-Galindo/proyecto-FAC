#!/usr/bin/env python3
"""
Caracterización Profunda y Mapeo 1-a-1 de los Datasets de la FAC:
1. Analiza los 3 datasets (dataset_split_completo, dataset_preprocesado_11gb, dataset_yolo_7gb).
2. Determina el origen exacto de video (Video 1 vs Video 2 vs Otros) de cada archivo anotado.
3. Analiza la distribución de clases y splits (Train/Val/Test).
4. Explica y previene el Data Leakage identificando qué frames estuvieron en Train vs Test.
"""

from collections import Counter
import json
import os
from pathlib import Path
import yaml
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

CLASES_MAP = {
    0: "Vehículo (Vehicle)",
    1: "Bodega/Edificio (Building)",
    2: "Camino/Carretera (Road)",
    3: "Río (River)",
    4: "Minería Ilegal / SDZI (Illegal Mining)",
}


def analizar_dataset(nombre_ds, ruta_raiz):
    """Analiza a fondo un dataset YOLO: imágenes, etiquetas, clases y orígenes."""
    if not ruta_raiz.is_dir():
        return None

    # Buscar dataset.yaml
    yamls = list(ruta_raiz.glob("**/dataset.yaml")) + list(ruta_raiz.glob("**/data.yaml"))
    config_yaml = {}
    if yamls:
        with open(yamls[0], "r", encoding="utf-8") as f:
            try:
                config_yaml = yaml.safe_load(f)
            except Exception:
                pass

    # Mapeo de archivos
    imgs = list(ruta_raiz.rglob("*.jpg")) + list(ruta_raiz.rglob("*.png"))
    
    conteo_splits = Counter()
    conteo_origenes = Counter()
    conteo_clases_global = Counter()
    conteo_clases_por_split = {"train": Counter(), "val": Counter(), "test": Counter()}
    
    # Detalle por video
    video_detalle = {
        "Video 1 (Misión Telembí / Nariño)": {"train": 0, "val": 0, "test": 0, "total": 0},
        "Video 2 (Misión Santander / Codefest - 13min + Scenes)": {"train": 0, "val": 0, "test": 0, "total": 0},
        "Augmentations Sintéticos (Train Balance)": {"train": 0, "val": 0, "test": 0, "total": 0},
        "Otros Clips / Tomas fijas": {"train": 0, "val": 0, "test": 0, "total": 0},
    }

    for img_p in imgs:
        # Detectar split
        parts = [p.lower() for p in img_p.parts]
        split = "train" if "train" in parts else ("val" if "val" in parts else ("test" if "test" in parts else "otro"))
        conteo_splits[split] += 1

        stem = img_p.stem
        # Clasificar origen
        if "aug_" in stem:
            origen = "Augmentations Sintéticos (Train Balance)"
        elif "11min" in stem:
            origen = "Video 1 (Misión Telembí / Nariño)"
        elif "13min" in stem or "scene" in stem:
            origen = "Video 2 (Misión Santander / Codefest - 13min + Scenes)"
        else:
            origen = "Otros Clips / Tomas fijas"

        conteo_origenes[origen] += 1
        if split in ["train", "val", "test"] and origen in video_detalle:
            video_detalle[origen][split] += 1
            video_detalle[origen]["total"] += 1

        # Leer etiquetas
        lbl_p = img_p.parent.parent / "labels" / f"{stem}.txt"
        if not lbl_p.is_file():
            # Buscar en misma carpeta
            lbl_p = img_p.parent / f"{stem}.txt"

        if lbl_p.is_file():
            with open(lbl_p, "r") as f:
                for line in f:
                    partes = line.strip().split()
                    if partes:
                        try:
                            cls_id = int(partes[0])
                            conteo_clases_global[cls_id] += 1
                            if split in conteo_clases_por_split:
                                conteo_clases_por_split[split][cls_id] += 1
                        except ValueError:
                            pass

    return {
        "nombre": nombre_ds,
        "ruta": str(ruta_raiz),
        "yaml": str(yamls[0]) if yamls else "No encontrado",
        "clases_yaml": config_yaml.get("names", []),
        "total_imagenes": len(imgs),
        "splits": conteo_splits,
        "origenes": conteo_origenes,
        "video_detalle": video_detalle,
        "clases_total": conteo_clases_global,
        "clases_por_split": conteo_clases_por_split,
    }


def imprimir_reporte_completo(resultados):
    print("\n" + "=" * 90)
    print("      REPORTE DE CARACTERIZACIÓN EXHAUSTIVA DE DATASETS FAC (TESIS)")
    print("=" * 90)

    for r in resultados:
        if not r:
            continue
        print(f"\n📦 DATASET: {r['nombre']}")
        print(f"   Ruta: {r['ruta']}")
        print(f"   Total Imágenes: {r['total_imagenes']:,}")
        print(f"   Distribución de Splits: {dict(r['splits'])}")
        print(f"   Clases Declaradas: {r['clases_yaml']}")

        print("\n   --- Desglose por Origen de Video y Splits ---")
        df_vid = pd.DataFrame(r["video_detalle"]).T
        print(df_vid.to_string())

        print("\n   --- Distribución de Cajas por Clase (Ground Truth) ---")
        filas_clases = []
        for cls_id, nombre_c in CLASES_MAP.items():
            filas_clases.append({
                "ID": cls_id,
                "Clase": nombre_c,
                "Total Cajas": r["clases_total"].get(cls_id, 0),
                "Train": r["clases_por_split"]["train"].get(cls_id, 0),
                "Val": r["clases_por_split"]["val"].get(cls_id, 0),
                "Test": r["clases_por_split"]["test"].get(cls_id, 0),
            })
        df_clases = pd.DataFrame(filas_clases)
        print(df_clases.to_string(index=False))
        print("-" * 90)


def main():
    dir_datasets = RAIZ / "datasets"

    datasets_a_revisar = [
        ("1. Dataset Base Original (1.702 frames)", dir_datasets / "dataset_split_completo"),
        ("2. Dataset YOLO Oficial del Profesor (11.7 GB / 26K frames)", dir_datasets / "dataset_preprocesado_11gb"),
        ("3. Dataset YOLO Compacto (7.07 GB / 14K frames)", dir_datasets / "dataset_yolo_7gb"),
    ]

    resultados = []
    for nombre, ruta in datasets_a_revisar:
        if ruta.is_dir():
            res = analizar_dataset(nombre, ruta)
            if res:
                resultados.append(res)

    imprimir_reporte_completo(resultados)

    # Exportar a Excel
    ruta_salida = RAIZ / "videos/caracterizacion_datasets_fac.xlsx"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(ruta_salida), engine="openpyxl") as writer:
        for r in resultados:
            if not r:
                continue
            df_v = pd.DataFrame(r["video_detalle"]).T
            df_v.to_excel(writer, sheet_name=r["nombre"][:28] + "_videos")
            
            filas = [{"Clase_ID": k, "Nombre": CLASES_MAP.get(k, k), "Total": v} for k, v in r["clases_total"].items()]
            pd.DataFrame(filas).to_excel(writer, sheet_name=r["nombre"][:28] + "_clases", index=False)

    print(f"\n[+] Caracterización guardada exitosamente en: {ruta_salida}")


if __name__ == "__main__":
    main()
