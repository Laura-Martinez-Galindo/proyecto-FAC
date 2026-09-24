#!/usr/bin/env python3
"""
Evaluación Integral de Ablación YOLOv11 (Rendimiento + Conteo Exacto de Instancias):
Calcula para cada modelo (.pt) y cada rama de ablación (Original, Sin HUD, UDVD Estándar, UDVD Mejorado):
1. Métricas estándar: mAP50, mAP50-95, Precisión, Recall, Fitness.
2. Métricas de Instancias Físicas por Clase:
   - N_GT: Número real de instancias anotadas en Ground Truth.
   - N_Pred: Número total de detecciones predichas por YOLO.
   - TP (Verdaderos Positivos): Instancias correctamente detectadas (IoU >= 0.50).
   - FP (Falsos Positivos / Alarmas Fantasma): Detecciones sobre ruido térmico, texto de HUD o fondos.
   - FN (Falsos Negativos): Objetos reales no detectados por el modelo.
3. Exportación a Excel y visualización de reducción de alarmas falsas.
"""

import argparse
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import yaml

RAIZ = Path(__file__).resolve().parent.parent

CLASES_MAP = {
    0: "vehicles",
    1: "buildings",
    2: "roads",
    3: "rivers",
    4: "SDZI",
}


def contar_instancias_gt(dir_labels, num_clases=5):
    """Cuenta el número exacto de cajas de Ground Truth por clase en los archivos .txt."""
    gt_counts = {i: 0 for i in range(num_clases)}
    if not dir_labels.is_dir():
        return gt_counts

    for f_txt in dir_labels.glob("*.txt"):
        try:
            with open(f_txt, "r", encoding="utf-8") as f:
                for line in f:
                    partes = line.strip().split()
                    if partes:
                        c_id = int(float(partes[0]))
                        if c_id in gt_counts:
                            gt_counts[c_id] += 1
        except Exception:
            pass
    return gt_counts


def asegurar_yaml(ruta_yaml):
    if not ruta_yaml.is_file():
        return None
    with open(ruta_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["path"] = str(ruta_yaml.parent.resolve())
    with open(ruta_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)
    return ruta_yaml


def evaluar_modelo_en_dataset(modelo_yolo, ruta_yaml, split="test", conf=0.25, iou=0.50, device="0"):
    """
    Ejecuta validación en YOLO y extrae métricas continuas e instancias discretas (TP, FP, FN, GT, Pred).
    """
    res = modelo_yolo.val(
        data=str(ruta_yaml),
        split=split,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    # 1. Métricas Globales
    m_dict = res.results_dict
    map50_glob = float(m_dict.get("metrics/mAP50(B)", 0.0))
    map50_95_glob = float(m_dict.get("metrics/mAP50-95(B)", 0.0))
    prec_glob = float(m_dict.get("metrics/precision(B)", 0.0))
    rec_glob = float(m_dict.get("metrics/recall(B)", 0.0))
    fitness_glob = float(m_dict.get("fitness", 0.0))

    # 2. Conteo de GT real desde archivos de etiquetas
    yaml_dir = ruta_yaml.parent
    with open(ruta_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    split_lbl_rel = cfg.get(split, "test/images").replace("images", "labels")
    dir_test_lbls = (yaml_dir / split_lbl_rel).resolve()
    gt_counts_file = contar_instancias_gt(dir_test_lbls, num_clases=len(res.names))

    # 3. Matriz de Confusión para extraer TP, FP, FN
    cm = res.confusion_matrix.matrix  # shape (nc+1, nc+1)
    num_clases = len(res.names)

    clases_stats = []
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    tot_gt = 0
    tot_pred = 0

    for i in range(num_clases):
        c_name = res.names[i]
        tp_i = int(cm[i, i]) if cm.shape[0] > i and cm.shape[1] > i else 0
        # Falsos positivos: todo lo predicho como clase i que NO era clase i (incluyendo fondo/ruido)
        fp_i = int(np.sum(cm[:, i])) - tp_i if cm.shape[1] > i else 0
        # Falsos negativos: toda clase i que NO fue predicha como clase i (incluyendo fondo no detectado)
        fn_i = int(np.sum(cm[i, :])) - tp_i if cm.shape[0] > i else 0
        
        gt_i = gt_counts_file.get(i, tp_i + fn_i)
        pred_i = tp_i + fp_i

        tot_tp += tp_i
        tot_fp += fp_i
        tot_fn += fn_i
        tot_gt += gt_i
        tot_pred += pred_i

        ap50_c = float(res.box.ap50[i]) if hasattr(res.box, "ap50") and len(res.box.ap50) > i else 0.0
        ap_c = float(res.box.ap[i]) if hasattr(res.box, "ap") and len(res.box.ap) > i else 0.0
        p_c = float(res.box.p[i]) if hasattr(res.box, "p") and len(res.box.p) > i else 0.0
        r_c = float(res.box.r[i]) if hasattr(res.box, "r") and len(res.box.r) > i else 0.0

        clases_stats.append({
            "Clase_ID": i,
            "Clase": c_name,
            "N_GT (Etiquetas)": gt_i,
            "N_Pred (Detectadas)": pred_i,
            "TP (Aciertos)": tp_i,
            "FP (Alarmas Falsas/Ruido)": fp_i,
            "FN (Omitidas)": fn_i,
            "Precision": round(p_c, 4),
            "Recall": round(r_c, 4),
            "mAP@50": round(ap50_c, 4),
            "mAP@50-95": round(ap_c, 4),
        })

    res_global = {
        "mAP@50": round(map50_glob, 4),
        "mAP@50-95": round(map50_95_glob, 4),
        "Precision": round(prec_glob, 4),
        "Recall": round(rec_glob, 4),
        "Fitness": round(fitness_glob, 4),
        "Total_GT": tot_gt,
        "Total_Pred": tot_pred,
        "Total_TP": tot_tp,
        "Total_FP_Alarmas": tot_fp,
        "Total_FN_Omitidos": tot_fn,
    }

    return res_global, clases_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluación de Instancias y Métricas de Ablación YOLO")
    parser.add_argument("--device", default="0", help="ID GPU o 'cpu'")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    parser.add_argument("--iou", type=float, default=0.50, help="Umbral IoU para TP")
    parser.add_argument("--salida-excel", default="videos/evaluacion_ablation_yolo_instancias.xlsx", help="Ruta de exportación")
    args = parser.parse_args()

    from ultralytics import YOLO

    print("=" * 95)
    print("      EVALUACIÓN DE ABLACIÓN YOLOv11 CON CONTEO DE INSTANCIAS (GT vs TP vs FP vs FN)")
    print(f"Dispositivo: CUDA {args.device} | Conf: {args.conf} | IoU: {args.iou}")
    print("=" * 95)

    # 1. Definir Modelos a Evaluar
    modelos_candidatos = [
        ("yolov26_best100.pt", RAIZ / "yolov26_best100.pt"),
        ("yolov11_best100.pt", RAIZ / "yolov11_best100.pt"),
        ("best.pt", RAIZ / "best.pt"),
        ("bes_base_heavyaug.pt", RAIZ / "bes_base_heavyaug.pt"),
        ("yolov11_sin_hud_best.pt", RAIZ / "pesos/yolov11_sin_hud_best.pt"),
        ("yolov11_udvd_std_best.pt", RAIZ / "pesos/yolov11_udvd_std_best.pt"),
        ("yolov11_udvd_mejorado_best.pt", RAIZ / "pesos/yolov11_udvd_mejorado_best.pt"),
    ]

    modelos_disponibles = [(n, p) for n, p in modelos_candidatos if p.is_file()]
    print(f"\nModelos disponibles para evaluación ({len(modelos_disponibles)}):")
    for n, p in modelos_disponibles:
        print(f"  * {n:<32} ({p.stat().st_size / (1024*1024):.1f} MB)")

    # 2. Definir Datasets / Ramas de Ablación
    datasets_candidatos = [
        ("1. Original Base (11.7 GB)", RAIZ / "datasets/dataset_preprocesado_11gb/modelo_yolov11_dataset_completo_preprocesado/dataset.yaml"),
        ("1. Original Ablation", RAIZ / "datasets/dataset_ablation_final/1_originales/dataset_1_original.yaml"),
        ("2. Sin HUD Ablation", RAIZ / "datasets/dataset_ablation_final/2_sin_hud/dataset_2_sin_hud.yaml"),
        ("3. UDVD Standard Ablation", RAIZ / "datasets/dataset_ablation_final/3_denoised_udvd_standard/dataset_3_udvd_standard.yaml"),
        ("4. UDVD Mejorado Ablation", RAIZ / "datasets/dataset_ablation_final/4_denoised_udvd_mejorado/dataset_4_udvd_mejorado.yaml"),
    ]

    datasets_disponibles = []
    for d_nombre, p_yaml in datasets_candidatos:
        if p_yaml.is_file():
            p_yaml = asegurar_yaml(p_yaml)
            datasets_disponibles.append((d_nombre, p_yaml))

    print(f"\nDatasets / Ramas disponibles ({len(datasets_disponibles)}):")
    for d_nombre, p_yaml in datasets_disponibles:
        print(f"  * {d_nombre}")

    filas_globales = []
    filas_clases = []

    for nom_m, ruta_m in modelos_disponibles:
        print("\n" + "#" * 95)
        print(f"📦 EVALUANDO MODELO: {nom_m}")
        print("#" * 95)
        modelo = YOLO(str(ruta_m))

        for nom_ds, p_yaml in datasets_disponibles:
            print(f"  >> Procesando rama: {nom_ds}...")
            try:
                res_glob, stats_c = evaluar_modelo_en_dataset(
                    modelo_yolo=modelo,
                    ruta_yaml=p_yaml,
                    split="test",
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                )

                fila_g = {"Modelo": nom_m, "Rama_Dataset": nom_ds}
                fila_g.update(res_glob)
                filas_globales.append(fila_g)

                for item in stats_c:
                    fila_c = {"Modelo": nom_m, "Rama_Dataset": nom_ds}
                    fila_c.update(item)
                    filas_clases.append(fila_c)

            except Exception as e:
                print(f"    [-] Error evaluando en {nom_ds}: {e}")

    df_glob = pd.DataFrame(filas_globales)
    df_cla = pd.DataFrame(filas_clases)

    print("\n" + "=" * 95)
    print("                    RESUMEN GLOBAL DE MÉTRICAS E INSTANCIAS")
    print("=" * 95)
    if not df_glob.empty:
        cols_print = ["Modelo", "Rama_Dataset", "mAP@50", "Precision", "Recall", "Total_GT", "Total_Pred", "Total_TP", "Total_FP_Alarmas", "Total_FN_Omitidos"]
        print(df_glob[[c for c in cols_print if c in df_glob.columns]].to_string(index=False))
    print("=" * 95)

    print("\n" + "=" * 95)
    print("               DESGLOSE DE INSTANCIAS POR CLASE (GT vs TP vs FP vs FN)")
    print("=" * 95)
    if not df_cla.empty:
        cols_print_c = ["Modelo", "Rama_Dataset", "Clase", "N_GT (Etiquetas)", "N_Pred (Detectadas)", "TP (Aciertos)", "FP (Alarmas Falsas/Ruido)", "FN (Omitidas)", "mAP@50"]
        print(df_cla[[c for c in cols_print_c if c in df_cla.columns]].to_string(index=False))
    print("=" * 95)

    ruta_salida = (RAIZ / args.salida_excel).resolve()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(ruta_salida), engine="openpyxl") as writer:
        df_glob.to_excel(writer, sheet_name="Global_Instancias", index=False)
        df_cla.to_excel(writer, sheet_name="Clases_Instancias", index=False)

    print(f"\n[+] Tabla consolidada de métricas e instancias guardada en: {ruta_salida}\n")


if __name__ == "__main__":
    main()
