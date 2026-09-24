#!/usr/bin/env python3
"""
Ejecuta el Benchmark Oficial de YOLOv11 en GPU para la Tesis:
1. Evalúa los pesos oficiales del profesor (yolov11_best100.pt) sobre el split de Test.
2. Desglosa métricas globales y por cada una de las 5 clases (mAP50, mAP50-95, Precision, Recall).
3. Exporta la tabla final a Excel y la imprime en consola lista para presentar.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import yaml
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parent.parent


def asegurar_yaml_valido(ruta_yaml):
    """Garantiza que la ruta base 'path' dentro de dataset.yaml sea absoluta y válida."""
    if not ruta_yaml.is_file():
        return None
    with open(ruta_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    dir_padre = ruta_yaml.parent.resolve()
    config["path"] = str(dir_padre)
    
    # Escribir yaml corregido
    with open(ruta_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)
    return ruta_yaml


def main():
    parser = argparse.ArgumentParser(description="Benchmark Oficial YOLOv11 FAC FLIR")
    parser.add_argument("--pesos", default="yolov11_best100.pt", help="Ruta a los pesos del modelo")
    parser.add_argument("--device", default="0", help="ID de GPU (0) o cpu")
    parser.add_argument("--salida-excel", default="videos/resumen_benchmark_oficial_yolo.xlsx", help="Ruta de salida Excel")
    args = parser.parse_args()

    ruta_pesos = (RAIZ / args.pesos).resolve()
    if not ruta_pesos.is_file():
        # Buscar en subcarpetas
        posibles = list(RAIZ.glob(f"**/{args.pesos}"))
        if posibles:
            ruta_pesos = posibles[0]
        else:
            print(f"ERROR: No se encontraron los pesos {args.pesos}")
            return

    from ultralytics import YOLO

    print("=" * 85)
    print("       BENCHMARK OFICIAL DE DETECCIÓN YOLOv11 (FAC FLIR - TESIS)")
    print(f"Pesos Modelo: {ruta_pesos.name} ({ruta_pesos.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Dispositivo:  CUDA {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    modelo = YOLO(str(ruta_pesos))

    # Datasets a evaluar
    datasets_test = [
        ("Dataset Preprocesado Oficial (11.7 GB - 26K frames)", RAIZ / "datasets/dataset_preprocesado_11gb/modelo_yolov11_dataset_completo_preprocesado/dataset.yaml"),
        ("Dataset YOLO Compacto (7.07 GB - 14K frames)", RAIZ / "datasets/dataset_yolo_7gb/yolo_dataset/dataset.yaml"),
        ("Dataset Base Original (1.702 frames)", RAIZ / "datasets/dataset_split_completo/dataset.yaml"),
    ]

    resumen_global = []
    resumen_clases = []

    for nombre_ds, p_yaml in datasets_test:
        if not p_yaml.is_file():
            # Buscar en subcarpetas
            cands = list(p_yaml.parent.glob("**/dataset.yaml"))
            if cands:
                p_yaml = cands[0]
            else:
                continue

        p_yaml = asegurar_yaml_valido(p_yaml)
        print(f"\n[+] Evaluando Split de TEST en: {nombre_ds}...")
        
        try:
            metrics = modelo.val(data=str(p_yaml), split="test", device=args.device, verbose=False)
            
            # 1. Métricas Globales
            m_dict = metrics.results_dict
            map50 = m_dict.get("metrics/mAP50(B)", 0.0)
            map50_95 = m_dict.get("metrics/mAP50-95(B)", 0.0)
            prec = m_dict.get("metrics/precision(B)", 0.0)
            rec = m_dict.get("metrics/recall(B)", 0.0)
            fitness = m_dict.get("fitness", 0.0)

            resumen_global.append({
                "Dataset": nombre_ds,
                "mAP@50": round(map50, 4),
                "mAP@50-95": round(map50_95, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "Fitness": round(fitness, 4),
            })

            # 2. Desglose por Clases
            nombres_c = metrics.names
            # Extraer AP por clase si está disponible
            for i, c_name in nombres_c.items():
                ap50_c = metrics.box.ap50[i] if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > i else 0.0
                ap_c = metrics.box.ap[i] if hasattr(metrics.box, "ap") and len(metrics.box.ap) > i else 0.0
                p_c = metrics.box.p[i] if hasattr(metrics.box, "p") and len(metrics.box.p) > i else 0.0
                r_c = metrics.box.r[i] if hasattr(metrics.box, "r") and len(metrics.box.r) > i else 0.0

                resumen_clases.append({
                    "Dataset": nombre_ds[:30],
                    "ID": i,
                    "Clase": c_name,
                    "mAP@50": round(float(ap50_c), 4),
                    "mAP@50-95": round(float(ap_c), 4),
                    "Precision": round(float(p_c), 4),
                    "Recall": round(float(r_c), 4),
                })

        except Exception as e:
            print(f"[-] Error evaluando {nombre_ds}: {e}")

    # Mostrar Resultados en Consola
    df_global = pd.DataFrame(resumen_global)
    df_clases = pd.DataFrame(resumen_clases)

    print("\n" + "=" * 85)
    print("                     TABLA DE RESULTADOS GLOBALES (TEST SET)")
    print("=" * 85)
    print(df_global.to_string(index=False))
    print("=" * 85)

    if not df_clases.empty:
        print("\n" + "=" * 85)
        print("                     DESGLOSE POR CLASE (GROUND TRUTH)")
        print("=" * 85)
        print(df_clases.to_string(index=False))
        print("=" * 85)

    # Exportar a Excel
    ruta_out = (RAIZ / args.salida_excel).resolve()
    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(ruta_out), engine="openpyxl") as writer:
        df_global.to_excel(writer, sheet_name="Metricas_Globales", index=False)
        if not df_clases.empty:
            df_clases.to_excel(writer, sheet_name="Desglose_Clases", index=False)

    print(f"\n[+] Resultados oficiales exportados a Excel: {ruta_out}\n")


if __name__ == "__main__":
    main()
