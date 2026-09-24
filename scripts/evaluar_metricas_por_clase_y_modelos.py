#!/usr/bin/env python3
"""
Matriz de Evaluación Completa de Todos los Modelos .pt vs Todos los Datasets:
1. Evalúa cada modelo (.pt) encontrado:
   - best.pt
   - bes_base_heavyaug.pt
   - yolov11_best100.pt
   - yolov26_best100.pt
2. En cada dataset disponible:
   - Dataset Preprocesado (11.7 GB - 474 test)
   - Dataset YOLO (7.07 GB - 236 test)
3. Extrae métricas globales y por cada una de las 5 clases (Vehículos, Bodegas, Caminos, Ríos, Minería).
4. Exporta tabla consolidada a Excel y genera el reporte de reproducibilidad oficial.
"""

import argparse
from datetime import datetime
from pathlib import Path
import yaml
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parent.parent

CLASES_MAP = {
    0: "Vehículo (Vehicle)",
    1: "Bodega/Edificio (Building)",
    2: "Camino/Carretera (Road)",
    3: "Río (River)",
    4: "Minería Ilegal / SDZI (Illegal Mining)",
}


def asegurar_yaml(ruta_yaml):
    if not ruta_yaml.is_file():
        return None
    with open(ruta_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["path"] = str(ruta_yaml.parent.resolve())
    with open(ruta_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)
    return ruta_yaml


def main():
    parser = argparse.ArgumentParser(description="Matriz de Evaluación Completa de Modelos y Datasets")
    parser.add_argument("--device", default="0", help="ID GPU (0) o cpu")
    parser.add_argument("--salida-excel", default="videos/matriz_reproducibilidad_modelos_yolo.xlsx", help="Ruta de salida")
    args = parser.parse_args()

    from ultralytics import YOLO

    print("=" * 90)
    print("    MATRIZ DE REPRODUCIBILIDAD OFICIAL: TODOS LOS MODELOS .PT VS TODOS LOS DATASETS")
    print(f"Dispositivo: CUDA {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device != 'cpu' else 'CPU'})")
    print("=" * 90)

    # 1. Encontrar todos los modelos .pt en la raíz y subcarpetas
    todos_pesos = sorted(list(RAIZ.glob("*.pt")) + list((RAIZ / "pesos").glob("*.pt")) if (RAIZ / "pesos").is_dir() else list(RAIZ.glob("*.pt")))
    # Eliminar duplicados
    pesos_dict = {p.name: p for p in todos_pesos}
    print(f"\nModelos .pt detectados ({len(pesos_dict)}):")
    for nombre, p in pesos_dict.items():
        print(f"  * {nombre:<25} ({p.stat().st_size / (1024*1024):.1f} MB)")

    # 2. Datasets disponibles
    datasets = [
        ("Dataset Preprocesado (11.7 GB - 474 test)", RAIZ / "datasets/dataset_preprocesado_11gb/modelo_yolov11_dataset_completo_preprocesado/dataset.yaml"),
        ("Dataset YOLO Compacto (7.07 GB - 236 test)", RAIZ / "datasets/dataset_yolo_7gb/yolo_dataset/dataset.yaml"),
        ("Dataset Base Original (1.702 frames)", RAIZ / "datasets/dataset_split_completo/dataset.yaml"),
    ]

    datasets_validos = []
    for d_nombre, p_yaml in datasets:
        if not p_yaml.is_file():
            cands = list(p_yaml.parent.glob("**/dataset.yaml"))
            if cands:
                p_yaml = cands[0]
            else:
                continue
        p_yaml = asegurar_yaml(p_yaml)
        datasets_validos.append((d_nombre, p_yaml))

    print(f"\nDatasets de evaluación disponibles ({len(datasets_validos)}):")
    for d_nombre, p_yaml in datasets_validos:
        print(f"  * {d_nombre}")

    filas_globales = []
    filas_por_clase = []

    # 3. Evaluación exhaustiva cruzada
    for nombre_m, ruta_m in pesos_dict.items():
        print("\n" + "#" * 90)
        print(f"📦 MODELO: {nombre_m}")
        print("#" * 90)
        modelo = YOLO(str(ruta_m))

        for nombre_ds, p_yaml in datasets_validos:
            print(f"\n  [+] Evaluando en: {nombre_ds}...")
            try:
                metrics = modelo.val(data=str(p_yaml), split="test", device=args.device, verbose=False)
                
                # Métricas globales
                m_dict = metrics.results_dict
                map50 = m_dict.get("metrics/mAP50(B)", 0.0)
                map50_95 = m_dict.get("metrics/mAP50-95(B)", 0.0)
                prec = m_dict.get("metrics/precision(B)", 0.0)
                rec = m_dict.get("metrics/recall(B)", 0.0)
                fitness = m_dict.get("fitness", 0.0)

                filas_globales.append({
                    "Modelo": nombre_m,
                    "Dataset": nombre_ds,
                    "mAP@50": round(map50, 4),
                    "mAP@50-95": round(map50_95, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "Fitness": round(fitness, 4),
                })

                # Desglose por clase
                nombres_c = metrics.names
                for i, c_name in nombres_c.items():
                    ap50_c = metrics.box.ap50[i] if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > i else 0.0
                    ap_c = metrics.box.ap[i] if hasattr(metrics.box, "ap") and len(metrics.box.ap) > i else 0.0
                    p_c = metrics.box.p[i] if hasattr(metrics.box, "p") and len(metrics.box.p) > i else 0.0
                    r_c = metrics.box.r[i] if hasattr(metrics.box, "r") and len(metrics.box.r) > i else 0.0

                    filas_por_clase.append({
                        "Modelo": nombre_m,
                        "Dataset": nombre_ds[:25],
                        "ID": i,
                        "Clase": c_name,
                        "Precision": round(float(p_c), 4),
                        "Recall": round(float(r_c), 4),
                        "mAP@50": round(float(ap50_c), 4),
                        "mAP@50-95": round(float(ap_c), 4),
                    })

            except Exception as e:
                print(f"    [-] Error evaluando {nombre_m} en {nombre_ds}: {e}")

    # 4. Tablas consolidadas
    df_global = pd.DataFrame(filas_globales)
    df_clases = pd.DataFrame(filas_por_clase)

    print("\n" + "=" * 90)
    print("                     TABLA COMPARATIVA GLOBAL (TEST SET)")
    print("=" * 90)
    print(df_global.to_string(index=False))
    print("=" * 90)

    if not df_clases.empty:
        print("\n" + "=" * 90)
        print("                     DESGLOSE COMPLETO POR CLASE")
        print("=" * 90)
        print(df_clases.to_string(index=False))
        print("=" * 90)

    # 5. Exportar a Excel
    ruta_out = (RAIZ / args.salida_excel).resolve()
    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(ruta_out), engine="openpyxl") as writer:
        df_global.to_excel(writer, sheet_name="Comparativa_Global", index=False)
        df_clases.to_excel(writer, sheet_name="Desglose_Clases", index=False)

    print(f"\n[+] Matriz de reproducibilidad exportada exitosamente a: {ruta_out}\n")


if __name__ == "__main__":
    main()
