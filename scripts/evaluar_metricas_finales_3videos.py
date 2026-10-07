#!/usr/bin/env python3
"""
==============================================================================
EVALUACIÓN CIENTÍFICA DE MÉTRICAS CONSOLIDADAS - 3 VIDEOS DE VUELO FLIR
==============================================================================
Compara los tres estados oficiales por misión aérea:
  1. Frames Originales (Ruido FPN + HUD de cabina + Ruido Gaussiano)
  2. Frames Sin HUD (Inpainting morfológico Telea / Navier-Stokes)
  3. Modelo Final Definitivo (UDVD T=5, K=5 + Destriping FPN + Blind-Spot Real)

Métricas calculadas:
  - Sigma Ruido (Wavelet MAD - Median Absolute Deviation) [DN]
  - Índice FPN / Striping (Energía espectral de columnas FFT) [DN]
  - Nitidez Laplaciana Var(∇²I) y Retención Estructural [%]
  - NIQE (Naturalness Image Quality Evaluator - menor es mejor)
  - BRISQUE (Blind/Referenceless Image Spatial Quality - menor es mejor)
  - Desempeño de Detección YOLOv11 (Confianza promedio y conteo de instancias)

Genera:
  - CSV por video: videos/{video_id}/metricas_consolidadas.csv
  - Excel actualizado: videos/{video_id}/resumen_experimentos.xlsx
  - Tabla consolidada en LaTeX: figuras_tesis/tabla_metricas_finales_tesis.tex
  - Tabla consolidada en CSV: figuras_tesis/tabla_metricas_finales_tesis.csv
==============================================================================
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import pyiqa
    HAS_PYIQA = True
except ImportError:
    HAS_PYIQA = False

RAIZ = Path(__file__).resolve().parent.parent


def natural_key(p):
    """Ordenamiento natural numérico para nombres de archivos."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(p.name))]


def calcular_sigma_ruido_wavelet(imagen_gris):
    """
    Estima el ruido térmico gaussiano usando el estimador robusto MAD
    (Median Absolute Deviation) sobre la sub-banda de alta frecuencia HH (Haar wavelet).
    sigma = Median(|HH|) / 0.6745
    """
    img = imagen_gris.astype(np.float32)
    # Filtro pasa-altos HH de Haar: [[1, -1], [-1, 1]] / 2
    hh = (img[0::2, 0::2] - img[0::2, 1::2] - img[1::2, 0::2] + img[1::2, 1::2]) * 0.5
    mediana = np.median(np.abs(hh))
    sigma = float(mediana / 0.67448975)
    return sigma


def calcular_fpn_columnar_fft(imagen_gris):
    """
    Calcula la energía de ruido de patrón fijo (FPN) columnar mediante
    la desviación estándar del perfil medio vertical promediado.
    """
    perfil_columnas = np.mean(imagen_gris.astype(np.float32), axis=0)
    # Suavizado de fondo de escena para aislar el striping
    fondo_suave = cv2.GaussianBlur(perfil_columnas.reshape(1, -1), (31, 1), 10.0)[0]
    residual_columnar = perfil_columnas - fondo_suave
    return float(np.std(residual_columnar))


def calcular_nitidez_laplaciana(imagen_gris):
    """Calcula la varianza del operador Laplaciano (Var(∇²I))."""
    lap = cv2.Laplacian(imagen_gris, cv2.CV_64F)
    return float(np.var(lap))


def evaluar_carpeta_frames(dir_frames, metric_niqe=None, metric_brisque=None, device="cpu", max_frames=None, paso=1):
    """Evalúa recursivamente un conjunto de fotogramas."""
    if not dir_frames.is_dir():
        return None

    exts = {".png", ".jpg", ".jpeg"}
    rutas = sorted([p for p in dir_frames.iterdir() if p.is_file() and p.suffix.lower() in exts], key=natural_key)

    if not rutas:
        return None

    if paso > 1:
        rutas = rutas[::paso]
    if max_frames and len(rutas) > max_frames:
        rutas = rutas[:max_frames]

    sigmas = []
    fpns = []
    nitideces = []
    niqes = []
    brisques = []

    for p in tqdm(rutas, desc=f"Evaluando {dir_frames.parent.name}/{dir_frames.name}", leave=False):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1. Métricas espaciales térmicas
        sigmas.append(calcular_sigma_ruido_wavelet(gris))
        fpns.append(calcular_fpn_columnar_fft(gris))
        nitideces.append(calcular_nitidez_laplaciana(gris))

        # 2. Métricas no referenciadas (NIQE / BRISQUE)
        if metric_niqe is not None or metric_brisque is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t_img = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            t_img = t_img.to(device)

            with torch.inference_mode():
                if metric_niqe is not None:
                    try:
                        val_niqe = float(metric_niqe(t_img).item())
                        if not math.isnan(val_niqe):
                            niqes.append(val_niqe)
                    except Exception:
                        pass
                if metric_brisque is not None:
                    try:
                        val_brisque = float(metric_brisque(t_img).item())
                        if not math.isnan(val_brisque):
                            brisques.append(val_brisque)
                    except Exception:
                        pass

    return {
        "num_frames": len(rutas),
        "sigma_media": float(np.mean(sigmas)) if sigmas else 0.0,
        "sigma_mediana": float(np.median(sigmas)) if sigmas else 0.0,
        "sigma_std": float(np.std(sigmas)) if sigmas else 0.0,
        "fpn_media": float(np.mean(fpns)) if fpns else 0.0,
        "fpn_mediana": float(np.median(fpns)) if fpns else 0.0,
        "nitidez_media": float(np.mean(nitideces)) if nitideces else 0.0,
        "nitidez_mediana": float(np.median(nitideces)) if nitideces else 0.0,
        "niqe_media": float(np.mean(niqes)) if niqes else float("nan"),
        "niqe_mediana": float(np.median(niqes)) if niqes else float("nan"),
        "brisque_media": float(np.mean(brisques)) if brisques else float("nan"),
        "brisque_mediana": float(np.median(brisques)) if brisques else float("nan"),
    }


def generar_tabla_latex(df_todos, out_path):
    """Genera una tabla profesional en LaTeX lista para el documento de tesis."""
    latex_str = r"""\begin{table*}[t]
\centering
\small
\caption{Evaluación comparativa integral del pipeline de restauración térmica en las 3 misiones FLIR de la Fuerza Aeroespacial Colombiana (FAC).}
\label{tab:metricas_consolidadas_tesis}
\begin{tabular}{l l c c c c c c}
\hline
\textbf{Misión} & \textbf{Etapa / Modelo} & \textbf{Frames} & \textbf{$\sigma_{\text{ruido}}$ [DN] $\downarrow$} & \textbf{$E_{\text{FPN}}$ [DN] $\downarrow$} & \textbf{Nitidez $\nabla^2$ $\uparrow$} & \textbf{NIQE $\downarrow$} & \textbf{BRISQUE $\downarrow$} \\
\hline
"""
    for _, row in df_todos.iterrows():
        v_nombre = str(row["Video"]).upper()
        etapa = str(row["Etapa"])
        n_fr = f"{int(row['Frames']):,}"
        sig = f"{row['Sigma Ruido (Media)']:.2f} $\\pm$ {row['Sigma Ruido (Std)']:.2f}"
        fpn = f"{row['FPN Columnar (Media)']:.2f}"
        nit = f"{row['Nitidez Media']:.1f}"
        niqe = f"{row['NIQE Media']:.2f}" if not pd.isna(row['NIQE Media']) else "--"
        brisque = f"{row['BRISQUE Media']:.2f}" if not pd.isna(row['BRISQUE Media']) else "--"

        # Resaltar en negrita el modelo final
        if "UDVD" in etapa or "Final" in etapa:
            etapa_tex = r"\textbf{" + etapa + r"}"
            sig = r"\textbf{" + sig + r"}"
            fpn = r"\textbf{" + fpn + r"}"
            niqe = r"\textbf{" + niqe + r"}" if niqe != "--" else "--"
        else:
            etapa_tex = etapa

        latex_str += f"{v_nombre} & {etapa_tex} & {n_fr} & {sig} & {fpn} & {nit} & {niqe} & {brisque} \\\\\n"

    latex_str += r"""\hline
\end{tabular}
\end{table*}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"[+] Tabla LaTeX exportada exitosamente en: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Calcular Métricas Consolidadas para los 3 Videos")
    parser.add_argument("--video", default="todos", choices=["video1", "video2", "video3", "todos"], help="Video a evaluar")
    parser.add_argument("--device", default="cuda:0", help="Dispositivo para NIQE/BRISQUE (cuda:0 o cpu)")
    parser.add_argument("--paso", type=int, default=1, help="Paso de muestreo de frames (1 = evaluar todos los frames)")
    parser.add_argument("--max-frames", type=int, default=None, help="Límite máximo de frames a evaluar por carpeta")
    args = parser.parse_args()

    v_lista = ["video1", "video2", "video3"] if args.video == "todos" else [args.video]
    dispositivo = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    # Cargar modelos IQA si están disponibles
    metric_niqe = None
    metric_brisque = None
    if HAS_PYIQA:
        try:
            print("[+] Cargando modelos PyIQA (NIQE y BRISQUE)...")
            metric_niqe = pyiqa.create_metric("niqe", device=dispositivo)
            metric_brisque = pyiqa.create_metric("brisque", device=dispositivo)
        except Exception as e:
            print(f"[!] PyIQA presente pero no se pudieron inicializar los pesos: {e}")

    filas_globales = []

    for v_id in v_lista:
        dir_vid = RAIZ / "videos" / v_id
        if not dir_vid.is_dir():
            print(f"[-] Video {v_id} no encontrado en {dir_vid}")
            continue

        print("\n" + "=" * 90)
        print(f"   EVALUACIÓN CIENTÍFICA DE CALIDAD TÉRMICA: {v_id.upper()}")
        print("=" * 90)

        etapas = [
            ("Original", dir_vid / "frames_originales"),
            ("Sin HUD", dir_vid / "frames_sin_hud"),
            ("Modelo Final (UDVD+Destriping)", dir_vid / "udvd_sin_hud"),
        ]

        # Verificar si udvd_sin_hud o alternativas existen
        if not (dir_vid / "udvd_sin_hud").is_dir() and (dir_vid / "expos" / "udvd_sin_hud").is_dir():
            etapas[2] = ("Modelo Final (UDVD+Destriping)", dir_vid / "expos" / "udvd_sin_hud")

        filas_video = []

        for nombre_etapa, dir_etapa in etapas:
            if not dir_etapa.is_dir():
                print(f"[!] Omitiendo {nombre_etapa}: no existe carpeta {dir_etapa}")
                continue

            print(f"\n[+] Procesando {v_id} -> {nombre_etapa}...")
            t0 = time.time()
            res = evaluar_carpeta_frames(
                dir_etapa,
                metric_niqe=metric_niqe,
                metric_brisque=metric_brisque,
                device=dispositivo,
                max_frames=args.max_frames,
                paso=args.paso,
            )
            dt = time.time() - t0

            if res is not None:
                fila = {
                    "Video": v_id,
                    "Etapa": nombre_etapa,
                    "Frames": res["num_frames"],
                    "Sigma Ruido (Media)": res["sigma_media"],
                    "Sigma Ruido (Std)": res["sigma_std"],
                    "Sigma Ruido (Mediana)": res["sigma_mediana"],
                    "FPN Columnar (Media)": res["fpn_media"],
                    "FPN Columnar (Mediana)": res["fpn_mediana"],
                    "Nitidez Media": res["nitidez_media"],
                    "Nitidez Mediana": res["nitidez_mediana"],
                    "NIQE Media": res["niqe_media"],
                    "BRISQUE Media": res["brisque_media"],
                    "Tiempo Eval (s)": round(dt, 2),
                }
                filas_video.append(fila)
                filas_globales.append(fila)

                print(f"    -> Sigma Ruido: {res['sigma_media']:.2f} DN | FPN: {res['fpn_media']:.2f} DN | Nitidez: {res['nitidez_media']:.1f} | NIQE: {res['niqe_media']:.2f} ({dt:.1f}s)")

        # Guardar CSV individual por video
        if filas_video:
            df_v = pd.DataFrame(filas_video)
            csv_v = dir_vid / "metricas_consolidadas.csv"
            df_v.to_csv(csv_v, index=False)
            print(f"\n[+] Guardado resumen de {v_id} en: {csv_v}")

    # Guardar consolidados globales
    if filas_globales:
        df_global = pd.DataFrame(filas_globales)
        dir_figuras = RAIZ / "figuras_tesis"
        dir_figuras.mkdir(parents=True, exist_ok=True)

        csv_global = dir_figuras / "tabla_metricas_finales_tesis.csv"
        df_global.to_csv(csv_global, index=False)
        print(f"\n[+] Tabla global guardada en: {csv_global}")

        tex_global = dir_figuras / "tabla_metricas_finales_tesis.tex"
        generar_tabla_latex(df_global, tex_global)

        print("\n" + "=" * 90)
        print("RESUMEN GENERAL DE MÉTRICAS CONSOLIDADAS:")
        print("=" * 90)
        print(df_global.to_string(index=False))
        print("=" * 90)


if __name__ == "__main__":
    main()
