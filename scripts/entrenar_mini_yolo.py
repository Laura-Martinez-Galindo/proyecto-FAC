#!/usr/bin/env python3
"""
Entrenamiento y Evaluación de Mini-YOLO (YOLOv11 nano) para Estudio de Ablación.
Compara el entrenamiento de un modelo ligero sobre imágenes Originales vs Limpias (UDVD).
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

RAIZ = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Entrenar y Evaluar Mini-YOLO para Detección de Minería Ilegal")
    parser.add_argument("--modelo-base", default="yolo11n.pt", help="Pesos base de YOLO (yolo11n.pt, yolo11s.pt)")
    parser.add_argument("--yaml-train", required=True, help="Ruta al data.yaml del dataset a entrenar")
    parser.add_argument("--nombre-exp", required=True, help="Nombre del experimento (ej. mini_yolo_originales o mini_yolo_udvd)")
    parser.add_argument("--epocas", type=int, default=30, help="Cantidad de épocas de entrenamiento (default: 30)")
    parser.add_argument("--batch", type=int, default=16, help="Tamaño de batch")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución de entrenamiento")
    parser.add_argument("--device", default="0", help="GPU device ID")
    parser.add_argument("--salida-excel", default="resumen_mini_yolo.xlsx", help="Archivo Excel con métricas")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: Instala ultralytics con: pip install ultralytics")
        return

    print("=" * 80)
    print(f"ENTRENANDO MINI-YOLO: {args.nombre_exp}")
    print(f"Base: {args.modelo_base} | Épocas: {args.epocas} | Batch: {args.batch} | GPU: {args.device}")
    print(f"Dataset: {args.yaml_train}")
    print("=" * 80)

    # 1. Cargar modelo base
    modelo = YOLO(args.modelo_base)

    # 2. Entrenar
    resultados_train = modelo.train(
        data=str(Path(args.yaml_train).resolve()),
        epochs=args.epocas,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.nombre_exp,
        project="runs/detect",
        plots=True,
        save=True,
    )

    # 3. Evaluar sobre el split de Test
    print("\n>>> Evaluando sobre Test Split...")
    resultados_test = modelo.val(
        data=str(Path(args.yaml_train).resolve()),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
    )

    metrics = resultados_test.results_dict
    fila_resumen = {
        "Experimento": args.nombre_exp,
        "Modelo_Base": args.modelo_base,
        "Epocas": args.epocas,
        "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
        "mAP50_95": metrics.get("metrics/mAP50-95(B)", 0.0),
        "Precision": metrics.get("metrics/precision(B)", 0.0),
        "Recall": metrics.get("metrics/recall(B)", 0.0),
        "Fitness": metrics.get("fitness", 0.0),
        "Fecha": datetime.now().isoformat(timespec="seconds"),
    }

    df = pd.DataFrame([fila_resumen])
    ruta_excel = (RAIZ / args.salida_excel).resolve()

    if ruta_excel.is_file():
        df_existente = pd.read_excel(ruta_excel)
        df = pd.concat([df_existente, df], ignore_index=True)

    df.to_excel(ruta_excel, index=False)

    print("\n" + "=" * 80)
    print("RESULTADOS DE EVALUACIÓN EN TEST:")
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"Resultados consolidados en: {ruta_excel}")


if __name__ == "__main__":
    main()
