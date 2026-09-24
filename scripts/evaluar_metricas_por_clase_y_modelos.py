#!/usr/bin/env python3
"""
Evaluación Detallada por Clase y Búsqueda de Modelos (.pt) para la Tesis:
1. Encuentra todos los checkpoints .pt en el repositorio y datasets descargados.
2. Evalúa en detalle el Test Set oficial (474 imágenes).
3. Extrae métricas exactas por clase:
   - Vehículos
   - Bodegas
   - Caminos
   - Ríos
   - Zonas de minería ilegal (SDZI)
4. Exporta tabla detallada a Excel y consola.
"""

import argparse
from pathlib import Path
import yaml
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parent.parent


def asegurar_yaml(ruta_yaml):
    with open(ruta_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["path"] = str(ruta_yaml.parent.resolve())
    with open(ruta_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)
    return ruta_yaml


def main():
    parser = argparse.ArgumentParser(description="Métricas por Clase YOLOv11 FAC FLIR")
    parser.add_argument("--device", default="cpu", help="ID GPU (0) o cpu")
    args = parser.parse_args()

    from ultralytics import YOLO

    print("=" * 85)
    print("   BÚSQUEDA DE MODELOS .PT Y EVALUACIÓN DETALLADA POR CLASE (TESIS FAC)")
    print("=" * 85)

    # 1. Buscar todos los archivos .pt
    pesos_encontrados = list(RAIZ.glob("**/*.pt"))
    print(f"Modelos .pt encontrados ({len(pesos_encontrados)}):")
    for p in pesos_encontrados:
        print(f"  * {p.relative_to(RAIZ)} ({p.stat().st_size / (1024*1024):.1f} MB)")

    # 2. Dataset oficial de 474 imágenes
    p_yaml = RAIZ / "datasets/dataset_preprocesado_11gb/modelo_yolov11_dataset_completo_preprocesado/dataset.yaml"
    if not p_yaml.is_file():
        cands = list(RAIZ.glob("**/modelo_yolov11_dataset_completo_preprocesado/**/dataset.yaml"))
        if cands:
            p_yaml = cands[0]
        else:
            print(f"ERROR: No se encontró dataset.yaml en {p_yaml}")
            return

    asegurar_yaml(p_yaml)
    print(f"\nEvaluando Test Set oficial en: {p_yaml.relative_to(RAIZ)}")

    # 3. Evaluar modelo principal
    ruta_modelo_main = RAIZ / "yolov11_best100.pt"
    if not ruta_modelo_main.is_file():
        ruta_modelo_main = pesos_encontrados[0]

    print(f"Cargando modelo: {ruta_modelo_main.name}...")
    modelo = YOLO(str(ruta_modelo_main))

    # Ejecutar validación detallada en test
    metrics = modelo.val(data=str(p_yaml), split="test", device=args.device, verbose=True)

    # 4. Extraer desglose exacto por clase
    nombres = metrics.names
    filas_clases = []

    for idx, c_name in nombres.items():
        # Extraer métricas por clase de ultralytics box metrics
        ap50 = metrics.box.ap50[idx] if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > idx else 0.0
        ap = metrics.box.ap[idx] if hasattr(metrics.box, "ap") and len(metrics.box.ap) > idx else 0.0
        p = metrics.box.p[idx] if hasattr(metrics.box, "p") and len(metrics.box.p) > idx else 0.0
        r = metrics.box.r[idx] if hasattr(metrics.box, "r") and len(metrics.box.r) > idx else 0.0

        filas_clases.append({
            "ID": idx,
            "Clase": c_name,
            "Precision": round(float(p), 4),
            "Recall": round(float(r), 4),
            "mAP@50": round(float(ap50), 4),
            "mAP@50-95": round(float(ap), 4),
        })

    # Resumen Global
    m_dict = metrics.results_dict
    resumen_global = [{
        "Métrica": "PROMEDIO GLOBAL (ALL CLASSES)",
        "Precision": round(m_dict.get("metrics/precision(B)", 0.0), 4),
        "Recall": round(m_dict.get("metrics/recall(B)", 0.0), 4),
        "mAP@50": round(m_dict.get("metrics/mAP50(B)", 0.0), 4),
        "mAP@50-95": round(m_dict.get("metrics/mAP50-95(B)", 0.0), 4),
    }]

    df_clases = pd.DataFrame(filas_clases)
    df_global = pd.DataFrame(resumen_global)

    print("\n" + "=" * 85)
    print("           TABLA DE RESULTADOS POR CLASE (TEST SET - 474 IMÁGENES)")
    print("=" * 85)
    print(df_clases.to_string(index=False))
    print("-" * 85)
    print(df_global.to_string(index=False))
    print("=" * 85)

    # Exportar a Excel
    ruta_salida = RAIZ / "videos/resumen_metricas_por_clase_yolo.xlsx"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(ruta_salida), engine="openpyxl") as writer:
        df_clases.to_excel(writer, sheet_name="Por_Clase", index=False)
        df_global.to_excel(writer, sheet_name="Global", index=False)

    print(f"\n[+] Resultados exportados a Excel: {ruta_salida}\n")


if __name__ == "__main__":
    main()
