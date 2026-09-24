#!/usr/bin/env python3
"""
Entrenamiento de YOLOv11 en Datasets de Ablación (Sin HUD / UDVD Estándar / UDVD Mejorado):
Entrena YOLOv11 con los hiperparámetros oficiales de la tesis (100 épocas, batch 32/64, imgsz 640).
Guarda los pesos resultantes en 'pesos/' y genera las curvas de pérdida y matrices de confusión.
"""

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import yaml

RAIZ = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Entrenar YOLOv11 en Ramas de Ablación")
    parser.add_argument("--yaml-dataset", required=True, help="Ruta al dataset.yaml de la rama (ej. dataset_2_sin_hud.yaml)")
    parser.add_argument("--nombre-exp", required=True, help="Nombre del experimento (ej. yolo11_sin_hud, yolo11_udvd_std, yolo11_udvd_mejorado)")
    parser.add_argument("--modelo-base", default="yolo11n.pt", help="Pesos base iniciales (yolo11n.pt, yolo11s.pt o yolov26_best100.pt)")
    parser.add_argument("--epocas", type=int, default=100, help="Número de épocas (default: 100)")
    parser.add_argument("--batch", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución de imagen (default: 640)")
    parser.add_argument("--device", default="0", help="GPU ID (ej. 0 o 0,1)")
    parser.add_argument("--patience", type=int, default=25, help="Paciencia para Early Stopping")
    args = parser.parse_args()

    from ultralytics import YOLO

    yaml_path = Path(args.yaml_dataset).resolve()
    if not yaml_path.is_file():
        cands = list(RAIZ.glob(f"**/{args.yaml_dataset}"))
        if cands:
            yaml_path = cands[0]
        else:
            raise FileNotFoundError(f"No se encontró el archivo {args.yaml_dataset}")

    # Asegurar path absoluto en yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(yaml_path.parent.resolve())
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print("=" * 85)
    print(f"      ENTRENAMIENTO YOLOv11 - RAMA DE ABLACIÓN: {args.nombre_exp}")
    print(f"Dataset:      {yaml_path}")
    print(f"Modelo Base:  {args.modelo_base}")
    print(f"Épocas:       {args.epocas} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print(f"Dispositivo:  CUDA {args.device}")
    print("=" * 85)

    modelo = YOLO(args.modelo_base)

    # Entrenar
    resultados = modelo.train(
        data=str(yaml_path),
        epochs=args.epocas,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        name=args.nombre_exp,
        project=str((RAIZ / "runs/detect").resolve()),
        plots=True,
        save=True,
        workers=8,
    )

    # Copiar best.pt a pesos/ con nombre descriptivo
    dir_pesos = (RAIZ / "pesos").resolve()
    dir_pesos.mkdir(parents=True, exist_ok=True)
    
    best_weights = RAIZ / f"runs/detect/{args.nombre_exp}/weights/best.pt"
    if best_weights.is_file():
        dest_pesos = dir_pesos / f"{args.nombre_exp}_best.pt"
        shutil.copy2(best_weights, dest_pesos)
        print(f"\n[+] Pesos óptimos guardados exitosamente en: {dest_pesos}")

    # Validar en Test Split
    print("\n>>> Evaluando métricas finales en Test Split...")
    test_metrics = modelo.val(
        data=str(yaml_path),
        split="test",
        device=args.device,
        plots=True,
    )

    print("\n" + "=" * 85)
    print(f"ENTRENAMIENTO Y VALIDACIÓN FINAL COMPLETADOS: {args.nombre_exp}")
    print(f"mAP@50:     {test_metrics.results_dict.get('metrics/mAP50(B)', 0.0):.4f}")
    print(f"mAP@50-95:  {test_metrics.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")
    print(f"Precision:  {test_metrics.results_dict.get('metrics/precision(B)', 0.0):.4f}")
    print(f"Recall:     {test_metrics.results_dict.get('metrics/recall(B)', 0.0):.4f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
