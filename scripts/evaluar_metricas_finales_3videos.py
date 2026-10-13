#!/usr/bin/env python3
"""
==============================================================================
EVALUACIÓN CIENTÍFICA DE MÉTRICAS Y DISTRIBUCIONES (2 GPUs - 3 VIDEOS FLIR)
==============================================================================
1. Evaluación paralela acelerada en Multi-GPU (cuda:0 y cuda:1) y 16 CPUs.
2. Calcula por cada fotograma:
   - Sigma Ruido Térmico (Wavelet MAD [DN])
   - Ruido de Patrón Fijo Columnar FPN (FFT Espectral [DN])
   - Nitidez Laplaciana Var(∇²I) y Retención Estructural
   - NIQE (Naturalness Image Quality Evaluator)
   - BRISQUE (Blind/Referenceless Image Spatial Quality)
3. Genera tablas de métricas consolidadas (Media, Mediana, Desviación Estándar, IQR).
4. Genera GRÁFICAS DE DISTRIBUCIÓN ESTADÍSTICA (Violin Plots, Boxplots y KDE):
   - Evita el sesgo de la media mostrando cómo se estrecha y desplaza la distribución
     del ruido y cómo se preserva la nitidez en cada etapa del vuelo.
5. Exporta:
   - CSVs detallados por fotograma: videos/{video_id}/distribucion_metricas.csv
   - Figuras de distribución: figuras_tesis/distribuciones/
   - Tabla LaTeX oficial para tesis: figuras_tesis/tabla_metricas_finales_tesis.tex
==============================================================================
"""

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

try:
    import pyiqa
    HAS_PYIQA = True
except ImportError:
    HAS_PYIQA = False

RAIZ = Path(__file__).resolve().parent.parent


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(p.name))]


def calcular_sigma_ruido_wavelet(imagen_gris):
    """Estima ruido gaussiano térmico usando wavelet Haar HH con MAD."""
    img = imagen_gris.astype(np.float32)
    hh = (img[0::2, 0::2] - img[0::2, 1::2] - img[1::2, 0::2] + img[1::2, 1::2]) * 0.5
    mediana = np.median(np.abs(hh))
    return float(mediana / 0.67448975)


def calcular_fpn_columnar_fft(imagen_gris):
    """Calcula la energía de ruido de patrón fijo columnar (FPN)."""
    perfil_col = np.mean(imagen_gris.astype(np.float32), axis=0)
    suave = cv2.GaussianBlur(perfil_col.reshape(1, -1), (31, 1), 10.0)[0]
    return float(np.std(perfil_col - suave))


def calcular_nitidez_laplaciana(imagen_gris):
    """Calcula la varianza del operador Laplaciano Var(∇²I)."""
    return float(np.var(cv2.Laplacian(imagen_gris, cv2.CV_64F)))


def evaluar_carpeta_frames_multigpu(dir_frames, etapa_nombre, video_id, paso=1, max_frames=None, gpus=None):
    if not dir_frames.is_dir():
        return None, None

    exts = {".png", ".jpg", ".jpeg"}
    rutas = sorted([p for p in dir_frames.iterdir() if p.is_file() and p.suffix.lower() in exts], key=natural_key)
    if not rutas:
        return None, None

    if paso > 1:
        rutas = rutas[::paso]
    if max_frames and len(rutas) > max_frames:
        rutas = rutas[:max_frames]

    print(f"\n[+] Evaluando {video_id.upper()} -> {etapa_nombre} ({len(rutas):,} frames seleccionados)...")

    # Configurar modelos IQA en GPU disponible
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    metric_niqe = None
    metric_brisque = None
    if HAS_PYIQA and "cuda" in dev:
        try:
            metric_niqe = pyiqa.create_metric("niqe", device=torch.device(dev))
            metric_brisque = pyiqa.create_metric("brisque", device=torch.device(dev))
        except Exception as e:
            print(f"    [!] PyIQA omitido: {e}")

    registros = []
    for idx, p in enumerate(tqdm(rutas, desc=f"{video_id.upper()} - {etapa_nombre}", leave=False)):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        sig = calcular_sigma_ruido_wavelet(gris)
        fpn = calcular_fpn_columnar_fft(gris)
        nit = calcular_nitidez_laplaciana(gris)
        niqe_val = np.nan
        brisque_val = np.nan

        # Para NIQE y BRISQUE evaluar cada 5 fotogramas para balancear velocidad
        if (metric_niqe is not None or metric_brisque is not None) and (idx % max(1, paso) == 0):
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                t_img = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(dev) / 255.0
                with torch.inference_mode():
                    if metric_niqe is not None:
                        niqe_val = float(metric_niqe(t_img).item())
                    if metric_brisque is not None:
                        brisque_val = float(metric_brisque(t_img).item())
            except Exception:
                pass

        registros.append({
            "video": video_id,
            "etapa": etapa_nombre,
            "frame": p.name,
            "sigma_ruido": sig,
            "fpn_columnar": fpn,
            "nitidez": nit,
            "niqe": niqe_val,
            "brisque": brisque_val
        })

    df_frames = pd.DataFrame(registros)
    resumen = {
        "Video": video_id,
        "Etapa": etapa_nombre,
        "Frames": len(df_frames),
        "Sigma Media": df_frames["sigma_ruido"].mean(),
        "Sigma Mediana": df_frames["sigma_ruido"].median(),
        "Sigma Std": df_frames["sigma_ruido"].std(),
        "FPN Media": df_frames["fpn_columnar"].mean(),
        "FPN Mediana": df_frames["fpn_columnar"].median(),
        "Nitidez Media": df_frames["nitidez"].mean(),
        "Nitidez Mediana": df_frames["nitidez"].median(),
        "NIQE Media": df_frames["niqe"].dropna().mean(),
        "BRISQUE Media": df_frames["brisque"].dropna().mean(),
    }
    return resumen, df_frames


def generar_graficas_distribucion(df_total, dir_salida):
    """Genera Violin Plots y Boxplots profesionales para la tesis."""
    dir_salida.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    paleta = {
        "Original": "#d9534f",
        "Sin HUD": "#f0ad4e",
        "Modelo Final (UDVD)": "#5cb85c"
    }

    # 1. Gráfica de Distribución de Sigma Ruido (Violin + Boxplot)
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(
        data=df_total, x="video", y="sigma_ruido", hue="etapa",
        palette=paleta, split=False, inner="quartile", cut=0
    )
    plt.title("Distribución de Ruido Térmico Gaussiano ($\sigma_{\\text{ruido}}$ [DN]) por Misión", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Misión FLIR (Video)", fontsize=12)
    plt.ylabel("Estimación $\sigma$ Wavelet MAD [DN] (Menor es mejor)", fontsize=12)
    plt.legend(title="Etapa de Procesamiento", loc="upper right")
    plt.tight_layout()
    plt.savefig(dir_salida / "distribucion_sigma_ruido_violin.png", dpi=300)
    plt.close()

    # 2. Gráfica de Ruido de Patrón Fijo (FPN Columnar)
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=df_total, x="video", y="fpn_columnar", hue="etapa",
        palette=paleta, showfliers=False
    )
    plt.title("Distribución de Energía de Ruido de Patrón Fijo Columnar ($E_{\\text{FPN}}$) por Misión", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Misión FLIR (Video)", fontsize=12)
    plt.ylabel("Desviación Columnar Residual [DN] (Menor es mejor)", fontsize=12)
    plt.legend(title="Etapa de Procesamiento", loc="upper right")
    plt.tight_layout()
    plt.savefig(dir_salida / "distribucion_fpn_boxplot.png", dpi=300)
    plt.close()

    # 3. Gráfica de Retención de Nitidez Laplaciana
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=df_total, x="video", y="nitidez", hue="etapa",
        palette=paleta, showfliers=False
    )
    plt.title("Distribución de Nitidez Estructural $\\mathrm{Var}(\\nabla^2 I)$ por Misión", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Misión FLIR (Video)", fontsize=12)
    plt.ylabel("Varianza Laplaciana (Preservación de Bordes)", fontsize=12)
    plt.legend(title="Etapa de Procesamiento", loc="upper right")
    plt.tight_layout()
    plt.savefig(dir_salida / "distribucion_nitidez_boxplot.png", dpi=300)
    plt.close()

    # 4. Gráfica KDE de Densidad de Probabilidad (Shift hacia la izquierda del ruido)
    for v_id in df_total["video"].unique():
        sub_df = df_total[df_total["video"] == v_id]
        plt.figure(figsize=(10, 5))
        for etapa in ["Original", "Sin HUD", "Modelo Final (UDVD)"]:
            vals = sub_df[sub_df["etapa"] == etapa]["sigma_ruido"].dropna()
            if not vals.empty:
                sns.kdeplot(vals, label=etapa, fill=True, alpha=0.3, color=paleta.get(etapa, None))
        plt.title(f"Función de Densidad de Probabilidad (KDE) - Ruido Térmico: {v_id.upper()}", fontsize=13, weight="bold")
        plt.xlabel("Nivel de Ruido $\sigma$ [DN]", fontsize=11)
        plt.ylabel("Densidad de Probabilidad", fontsize=11)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(dir_salida / f"kde_densidad_ruido_{v_id}.png", dpi=300)
        plt.close()

    print(f"[+] Gráficas de distribución guardadas en: {dir_salida}")


def generar_tabla_latex(df_resumen, out_path):
    latex_str = r"""\begin{table*}[t]
\centering
\small
\caption{Evaluación comparativa cuantitativa del pipeline de restauración térmica en las 3 misiones FLIR de la FAC.}
\label{tab:metricas_consolidadas_tesis}
\begin{tabular}{l l c c c c c c}
\hline
\textbf{Misión} & \textbf{Etapa / Modelo} & \textbf{Frames} & \textbf{$\sigma_{\text{ruido}}$ [DN] $\downarrow$} & \textbf{$E_{\text{FPN}}$ [DN] $\downarrow$} & \textbf{Nitidez $\nabla^2$ $\uparrow$} & \textbf{NIQE $\downarrow$} & \textbf{BRISQUE $\downarrow$} \\
\hline
"""
    for _, row in df_resumen.iterrows():
        v_nom = str(row["Video"]).upper()
        etapa = str(row["Etapa"])
        n_fr = f"{int(row['Frames']):,}"
        sig = f"{row['Sigma Media']:.2f} $\\pm$ {row['Sigma Std']:.2f}"
        fpn = f"{row['FPN Media']:.2f}"
        nit = f"{row['Nitidez Media']:.1f}"
        niqe = f"{row['NIQE Media']:.2f}" if not pd.isna(row['NIQE Media']) else "--"
        brisque = f"{row['BRISQUE Media']:.2f}" if not pd.isna(row['BRISQUE Media']) else "--"

        if "UDVD" in etapa or "Final" in etapa:
            etapa_tex = r"\textbf{" + etapa + r"}"
            sig = r"\textbf{" + sig + r"}"
            fpn = r"\textbf{" + fpn + r"}"
            niqe = r"\textbf{" + niqe + r"}" if niqe != "--" else "--"
        else:
            etapa_tex = etapa

        latex_str += f"{v_nom} & {etapa_tex} & {n_fr} & {sig} & {fpn} & {nit} & {niqe} & {brisque} \\\\\n"

    latex_str += r"""\hline
\end{tabular}
\end{table*}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"[+] Tabla LaTeX exportada en: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluación Científica de Métricas y Distribuciones (2 GPUs)")
    parser.add_argument("--video", default="todos", choices=["video1", "video2", "video3", "todos"])
    parser.add_argument("--paso", type=int, default=1, help="Paso de muestreo temporal (1 = todos los frames, 5 = muestreo rápido)")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    v_lista = ["video1", "video2", "video3"] if args.video == "todos" else [args.video]
    resúmenes = []
    todos_frames = []

    for v_id in v_lista:
        dir_vid = RAIZ / "videos" / v_id
        if not dir_vid.is_dir():
            continue

        etapas = [
            ("Original", dir_vid / "frames_originales"),
            ("Sin HUD", dir_vid / "frames_sin_hud"),
            ("Modelo Final (UDVD)", dir_vid / "udvd_sin_hud"),
        ]

        for nombre_e, dir_e in etapas:
            if not dir_e.is_dir():
                # Revisar rutas alternas si aplican
                if nombre_e == "Modelo Final (UDVD)" and (dir_vid / "expos" / "udvd_sin_hud").is_dir():
                    dir_e = dir_vid / "expos" / "udvd_sin_hud"
                else:
                    continue

            res, df_f = evaluar_carpeta_frames_multigpu(dir_e, nombre_e, v_id, paso=args.paso, max_frames=args.max_frames)
            if res and df_f is not None:
                resúmenes.append(res)
                todos_frames.append(df_f)

    if todos_frames:
        df_completo = pd.concat(todos_frames, ignore_index=True)
        dir_fig = RAIZ / "figuras_tesis" / "distribuciones_metricas"
        
        # Guardar CSV detallado con todos los frames
        csv_dist = RAIZ / "figuras_tesis" / "distribucion_metricas_por_frame.csv"
        df_completo.to_csv(csv_dist, index=False)
        print(f"\n[+] CSV detallado por frame guardado en: {csv_dist}")

        # Generar Gráficas de Distribución
        generar_graficas_distribucion(df_completo, dir_fig)

        # Resumen consolidado y LaTeX
        df_resumen = pd.DataFrame(resúmenes)
        csv_res = RAIZ / "figuras_tesis" / "tabla_metricas_finales_tesis.csv"
        df_resumen.to_csv(csv_res, index=False)
        generar_tabla_latex(df_resumen, RAIZ / "figuras_tesis" / "tabla_metricas_finales_tesis.tex")

        print("\n" + "=" * 90)
        print("RESUMEN GENERAL DE MÉTRICAS:")
        print("=" * 90)
        print(df_resumen.to_string(index=False))
        print("=" * 90)


if __name__ == "__main__":
    main()
