#!/usr/bin/env python3
"""
Evaluación de Ablación con YOLO (Impacto en Tarea Posterior de Detección).
Basado en el benchmark de Acosta-Bernal et al. (2025) y De la Hoz (2025) para videos FLIR (FAC).

Compara el desempeño de detección de objetos (construcciones, vías, ríos, alteraciones, vehículos)
sobre las tres ramas metodológicas:
1. Rama Base: Frames Originales (con HUD y con ruido)
2. Rama Intermedia: Frames Sin HUD (con ruido)
3. Rama Restaurada: Frames Sin HUD + Denoising (UDVD / Ensemble Wavelet)
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def evaluar_rama(modelo_yolo, data_yaml, nombre_rama, split="val", imgsz=640, device="0"):
    """Evalúa métricas estándar COCO/YOLO (mAP50, mAP50-95, Precision, Recall)."""
    print(f"\n>>> Evaluando Rama: {nombre_rama} con YOLO...")
    resultados = modelo_yolo.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        verbose=False,
        plots=True,
    )

    metrics = resultados.results_dict
    return {
        "Rama_Evaluada": nombre_rama,
        "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
        "mAP50_95": metrics.get("metrics/mAP50-95(B)", 0.0),
        "Precision": metrics.get("metrics/precision(B)", 0.0),
        "Recall": metrics.get("metrics/recall(B)", 0.0),
        "Fitness": metrics.get("fitness", 0.0),
        "Fecha": datetime.now().isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluación de Ablación con YOLO en Video FLIR")
    parser.add_argument("--pesos", default="yolo11n.pt", help="Ruta a los pesos del modelo YOLO (yolo11n.pt, yolo11s.pt, etc.)")
    parser.add_argument("--yaml-originales", help="Ruta al data.yaml apuntando a frames originales con HUD")
    parser.add_argument("--yaml-sin-hud", help="Ruta al data.yaml apuntando a frames sin HUD con ruido")
    parser.add_argument("--yaml-denoised", help="Ruta al data.yaml apuntando a frames restaurados sin HUD y sin ruido (UDVD)")
    parser.add_argument("--salida-excel", default="resumen_yolo_ablation.xlsx", help="Archivo Excel consolidado")
    parser.add_argument("--device", default="0", help="GPU device id o 'cpu'")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen de inferencia")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: La librería 'ultralytics' no está instalada. Instálala con: pip install ultralytics")
        return

    print("=" * 80)
    print("SUITE DE ABLACIÓN DE DETECCIÓN DE OBJETOS (YOLOv11 FLIR FAC)")
    print(f"Pesos base: {args.pesos} | Dispositivo: {args.device}")
    print("=" * 80)

    modelo = YOLO(args.pesos)
    filas = []

    ramas = []
    if args.yaml_originales:
        ramas.append(("1. Originales (Con HUD + Con Ruido)", args.yaml_originales))
    if args.yaml_sin_hud:
        ramas.append(("2. Sin HUD (Con Ruido)", args.yaml_sin_hud))
    if args.yaml_denoised:
        ramas.append(("3. Sin HUD + Denoised (UDVD / Ensemble)", args.yaml_denoised))

    if not ramas:
        print("AVISO: No se proporcionaron archivos YAML para evaluación directa.")
        print("Ejemplo de uso cuando el profesor comparta las anotaciones:")
        print("  python scripts/evaluar_yolo_ablation.py \\")
        print("      --pesos yolo11n.pt \\")
        print("      --yaml-originales config/yolo_originales.yaml \\")
        print("      --yaml-sin-hud config/yolo_sin_hud.yaml \\")
        print("      --yaml-denoised config/yolo_denoised_udvd.yaml")
        return

    for nombre, yaml_path in ramas:
        res = evaluar_rama(modelo, Path(yaml_path), nombre, imgsz=args.imgsz, device=args.device)
        filas.append(res)

    df = pd.DataFrame(filas)
    ruta_out = Path(args.salida_excel).resolve()
    df.to_excel(ruta_out, index=False)
    print("\n" + "=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"Resultados consolidados exportados a: {ruta_out}")


if __name__ == "__main__":
    main()
