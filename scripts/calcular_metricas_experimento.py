#!/usr/bin/env python3
"""
==============================================================================
CÁLCULO DEL TABLERO DE 6 MÉTRICAS CIENTÍFICAS POR EXPERIMENTO
==============================================================================
Calcula de manera exhaustiva para cada modelo ejecutado en un bloque:
1. Reducción de σ_MAD Wavelet (%)
2. Kurtosis del Residuo y FFT Flatness (Blancura espectral)
3. SLI (Structural Leakage Index) — Auditoría de no-fuga de bordes
4. EPR (Edge & Peak Preservation Ratio) — Conservación de picos calientes
5. Generación de salidas duales:
   - residual/frame_XXXXX.png (raw residual centrado en 128)
   - residual_visual/frame_XXXXX.png (|r| × 5 en mapa inferno)
6. Exportación de `metricas_frame.csv` por modelo y resumen comparativo
   en CSV y Excel.
==============================================================================
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

from metricas_experimentos import (
    calcular_metricas_completas,
    guardar_residuo_visual,
)


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def main():
    p = argparse.ArgumentParser(description="Calculador de métricas para un bloque experimental")
    p.add_argument("--dir-input", required=True, type=Path,
                   help="Directorio con frames ruidosos de entrada")
    p.add_argument("--dir-exp", required=True, type=Path,
                   help="Directorio base del bloque experimental")
    p.add_argument("--exp-id", required=True,
                   help="Identificador del experimento (e.g. EXP-1, EXP-2)")
    p.add_argument("--etiqueta", required=True,
                   help="Etiqueta del bloque (e.g. v3_001800-002099)")
    p.add_argument("--modelos", required=True,
                   help="Lista de modelos separados por comas (e.g. udvd,nb2nb,b2u,f2r)")
    p.add_argument("--calcular-niqe", action="store_true",
                   help="Calcular también métricas NIQE y BRISQUE de referencia")

    a = p.parse_args()

    dir_input = a.dir_input.resolve()
    dir_exp = a.dir_exp.resolve()
    modelos = [m.strip() for m in a.modelos.split(",") if m.strip()]

    exts = {".png", ".jpg", ".jpeg"}
    frames_in = sorted([x for x in dir_input.iterdir() if x.suffix.lower() in exts], key=natural_key)
    if not frames_in:
        print(f"[ERROR] No se encontraron frames de entrada en: {dir_input}")
        sys.exit(1)

    print(f"=== Evaluando Métricas para {a.exp_id} ({a.etiqueta}) ===")
    print(f"Frames de entrada: {len(frames_in)}")
    print(f"Modelos a procesar: {modelos}")

    resumen_filas = []

    for modelo in modelos:
        dir_denoised = dir_exp / modelo / "denoised"
        if not dir_denoised.is_dir():
            print(f"  [!] Saltando {modelo}: no existe {dir_denoised}")
            continue

        frames_den = {x.name: x for x in dir_denoised.iterdir() if x.suffix.lower() in exts}
        if not frames_den:
            print(f"  [!] Saltando {modelo}: no hay frames denoised en {dir_denoised}")
            continue

        dir_residual_raw = dir_exp / modelo / "residual"
        dir_residual_visual = dir_exp / modelo / "residual_visual"
        dir_residual_raw.mkdir(parents=True, exist_ok=True)
        dir_residual_visual.mkdir(parents=True, exist_ok=True)

        registros_modelo = []

        for p_in in frames_in:
            p_den = frames_den.get(p_in.name)
            if p_den is None:
                continue

            y_bgr = cv2.imread(str(p_in))
            yhat_bgr = cv2.imread(str(p_den))
            if y_bgr is None or yhat_bgr is None:
                continue

            if y_bgr.shape[:2] != yhat_bgr.shape[:2]:
                yhat_bgr = cv2.resize(yhat_bgr, (y_bgr.shape[1], y_bgr.shape[0]))

            # 1. Métricas completas
            metricas = calcular_metricas_completas(y_bgr, yhat_bgr, calcular_niqe=a.calcular_niqe)
            metricas["frame"] = p_in.name
            metricas["modelo"] = modelo
            metricas["exp_id"] = a.exp_id
            metricas["bloque"] = a.etiqueta
            registros_modelo.append(metricas)

            # 2. Guardar residuo raw y visual
            guardar_residuo_visual(y_bgr, yhat_bgr, dir_residual_visual / p_in.name, escala=5.0)
            
            y_gray = cv2.cvtColor(y_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            yhat_gray = cv2.cvtColor(yhat_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            residuo_raw = np.clip((y_gray - yhat_gray + 128.0), 0, 255).astype(np.uint8)
            cv2.imwrite(str(dir_residual_raw / p_in.name), residuo_raw, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        if registros_modelo:
            df_m = pd.DataFrame(registros_modelo)
            csv_modelo = dir_exp / modelo / "metricas_frame.csv"
            df_m.to_csv(str(csv_modelo), index=False)
            print(f"  [✓] {modelo}: {len(df_m)} frames procesados -> {csv_modelo.name}")

            # Calcular resumen estadístico del modelo
            resumen_m = {
                "exp_id": a.exp_id,
                "bloque": a.etiqueta,
                "modelo": modelo,
                "n_frames": len(df_m),
                "sigma_in_media": round(float(df_m["sigma_mad_input"].mean()), 3),
                "sigma_out_media": round(float(df_m["sigma_mad_output"].mean()), 3),
                "reduccion_sigma_mad_pct": round(float(df_m["sigma_mad_reduccion_pct"].mean()), 2),
                "sli_fuga_estructural": round(float(df_m["sli"].mean()), 4),
                "epr_preservacion_picos": round(float(df_m["epr"].mean()), 4),
                "kurtosis_residuo": round(float(df_m["kurtosis_residuo"].mean()), 3),
                "fft_flatness_residuo": round(float(df_m["fft_flatness_residuo"].mean()), 4),
            }
            if a.calcular_niqe and "niqe" in df_m.columns and df_m["niqe"].notnull().any():
                resumen_m["niqe_media"] = round(float(df_m["niqe"].dropna().mean()), 3)
                resumen_m["brisque_media"] = round(float(df_m["brisque"].dropna().mean()), 3)

            resumen_filas.append(resumen_m)

    if resumen_filas:
        df_resumen = pd.DataFrame(resumen_filas)
        # Ordenar modelos por menor fuga estructural y mayor reducción
        df_resumen = df_resumen.sort_values(by=["sli_fuga_estructural", "reduccion_sigma_mad_pct"], ascending=[True, False])

        csv_resumen = dir_exp / "resumen_metricas.csv"
        excel_resumen = dir_exp / "resumen_metricas.xlsx"
        df_resumen.to_csv(str(csv_resumen), index=False)
        df_resumen.to_excel(str(excel_resumen), index=False, engine="openpyxl")

        print("\n" + "=" * 80)
        print(f"📊 TABLA RESUMEN DE RESULTADOS — {a.exp_id} ({a.etiqueta})")
        print("=" * 80)
        print(df_resumen.to_string(index=False))
        print(f"\n[✓] Resumen guardado en:")
        print(f"    - {csv_resumen}")
        print(f"    - {excel_resumen}")


if __name__ == "__main__":
    main()
