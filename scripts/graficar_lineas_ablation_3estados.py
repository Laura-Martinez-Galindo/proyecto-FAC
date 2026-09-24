#!/usr/bin/env python3
"""
Comparativa Temporal de los 3 Estados Metodológicos (Original vs Sin HUD vs Denoised UDVD)
y Extracción de Residuos de Ruido (HUD y Ruido Térmico):

Calcula y grafica frame a frame la evolución temporal de:
1. Curva 1: Original Crudo (Con HUD + Con Ruido) [Rojo]
2. Curva 2: Sin HUD (Inpainting ProPainter, con ruido) [Naranja/Azul]
3. Curva 3: Restaurado (Sin HUD + Denoised UDVD) [Verde]

Métricas Evaluadas:
- BRISQUE (Calidad espacial perceptual - Menor es mejor)
- NIQE (Naturalidad estadística - Menor es mejor)
- Sigma MAD (Nivel de ruido estocástico - Menor es mejor)
- Varianza Laplaciana (Ritmo y nitidez estructural por escena)

Residuos de Ruido Extraídos:
- Residuo HUD = |Original - Sin_HUD| (Aislado de telemetría)
- Residuo Térmico = |Sin_HUD - Denoised| (Ruido del sensor FLIR para Jorge)
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def estimar_sigma_mad(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    return float(mad / 0.6745)


def calcular_densidad_estructural(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    energia_sobel = float(np.mean(mag))
    var_lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return energia_sobel, var_lap


def crear_metricas_pyiqa(dispositivo):
    try:
        import pyiqa
        m_niqe = pyiqa.create_metric("niqe", device=dispositivo)
        m_brisque = pyiqa.create_metric("brisque", device=dispositivo)
        return m_niqe, m_brisque
    except Exception as e:
        print(f"[!] pyiqa no disponible: {e}. Usando métricas estándar de gradiente y MAD.")
        return None, None


def graficar_comparativa_3_estados(df, salida_png, titulo="Video 2 (FLIR Mission)"):
    """Genera la figura de 4 paneles con las 3 curvas sincronizadas."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    fig.suptitle(f"Evolución Temporal de Calidad y Ruido en los 3 Estados Metodológicos\n{titulo}", fontsize=15, fontweight="bold", y=0.98)

    frames = df["frame_idx"].values
    ventana = min(30, max(5, len(df) // 100))

    col_orig = "#d62728"      # Rojo (Original con HUD + Ruido)
    col_sin_hud = "#1f77b4"   # Azul (Sin HUD, con Ruido)
    col_denoised = "#2ca02c"  # Verde (Restaurado Sin HUD + UDVD)

    # 1. BRISQUE (Calidad Espacial Perceptual - Menor es mejor)
    if "brisque_orig" in df.columns and df["brisque_orig"].sum() > 0:
        b_o_ma = df["brisque_orig"].rolling(ventana, center=True, min_periods=1).mean().values
        b_s_ma = df["brisque_sin_hud"].rolling(ventana, center=True, min_periods=1).mean().values
        b_d_ma = df["brisque_denoised"].rolling(ventana, center=True, min_periods=1).mean().values

        axs[0].plot(frames, b_o_ma, color=col_orig, lw=2.2, label=f"1. Original (Con HUD + Ruido) [Media {ventana}f]")
        axs[0].plot(frames, b_s_ma, color=col_sin_hud, lw=2.0, linestyle="--", label=f"2. Sin HUD (Con Ruido) [Media {ventana}f]")
        axs[0].plot(frames, b_d_ma, color=col_denoised, lw=2.4, label=f"3. Restaurado (Sin HUD + Denoised UDVD) [Media {ventana}f]")
        axs[0].set_ylabel("BRISQUE (↓ Mejor)", fontsize=11, fontweight="bold")
        axs[0].set_title("Calidad Espacial Perceptual (BRISQUE)", fontsize=12, fontweight="bold")
        axs[0].legend(loc="upper right", frameon=True)
        axs[0].grid(True, alpha=0.3)

    # 2. NIQE (Naturalidad de Escena - Menor es mejor)
    if "niqe_orig" in df.columns and df["niqe_orig"].sum() > 0:
        n_o_ma = df["niqe_orig"].rolling(ventana, center=True, min_periods=1).mean().values
        n_s_ma = df["niqe_sin_hud"].rolling(ventana, center=True, min_periods=1).mean().values
        n_d_ma = df["niqe_denoised"].rolling(ventana, center=True, min_periods=1).mean().values

        axs[1].plot(frames, n_o_ma, color=col_orig, lw=2.2, label="1. Original (Con HUD)")
        axs[1].plot(frames, n_s_ma, color=col_sin_hud, lw=2.0, linestyle="--", label="2. Sin HUD")
        axs[1].plot(frames, n_d_ma, color=col_denoised, lw=2.4, label="3. Restaurado (UDVD)")
        axs[1].set_ylabel("NIQE (↓ Mejor)", fontsize=11, fontweight="bold")
        axs[1].set_title("Naturalidad Estadística de la Imagen Térmica (NIQE)", fontsize=12, fontweight="bold")
        axs[1].legend(loc="upper right", frameon=True)
        axs[1].grid(True, alpha=0.3)

    # 3. Nivel de Ruido Estocástico (Sigma MAD - Menor es mejor)
    s_o_ma = df["sigma_orig"].rolling(ventana, center=True, min_periods=1).mean().values
    s_s_ma = df["sigma_sin_hud"].rolling(ventana, center=True, min_periods=1).mean().values
    s_d_ma = df["sigma_denoised"].rolling(ventana, center=True, min_periods=1).mean().values

    axs[2].plot(frames, s_o_ma, color=col_orig, lw=2.2, label="1. Sigma Original (Con HUD)")
    axs[2].plot(frames, s_s_ma, color=col_sin_hud, lw=2.0, linestyle="--", label="2. Sigma Sin HUD")
    axs[2].plot(frames, s_d_ma, color=col_denoised, lw=2.4, label="3. Sigma Restaurado (UDVD)")
    axs[2].set_ylabel("Sigma Ruido (σ MAD)", fontsize=11, fontweight="bold")
    axs[2].set_title("Nivel de Ruido de Alta Frecuencia (Estimador Robusto MAD)", fontsize=12, fontweight="bold")
    axs[2].legend(loc="upper right", frameon=True)
    axs[2].grid(True, alpha=0.3)

    # 4. Ritmo Estructural y Nitidez de Bordes (Varianza Laplaciana)
    l_o_ma = df["var_lap_orig"].rolling(ventana, center=True, min_periods=1).mean().values
    l_s_ma = df["var_lap_sin_hud"].rolling(ventana, center=True, min_periods=1).mean().values
    l_d_ma = df["var_lap_denoised"].rolling(ventana, center=True, min_periods=1).mean().values

    axs[3].plot(frames, l_o_ma, color=col_orig, lw=1.8, alpha=0.7, label="1. Original (Inflado por letras HUD)")
    axs[3].plot(frames, l_s_ma, color=col_sin_hud, lw=2.0, linestyle="--", label="2. Sin HUD (Estructura real del terreno)")
    axs[3].plot(frames, l_d_ma, color=col_denoised, lw=2.2, label="3. Restaurado UDVD (Bordes preservados sin ruido)")
    axs[3].set_ylabel("Varianza Laplaciana", fontsize=11, fontweight="bold")
    axs[3].set_xlabel("Índice de Cuadro Temporal (Frames de Vuelo)", fontsize=12, fontweight="bold")
    axs[3].set_title("Ritmo Estructural por Escena y Preservación de Detalles", fontsize=12, fontweight="bold")
    axs[3].legend(loc="upper right", frameon=True)
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    salida_png = Path(salida_png)
    salida_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida_png, dpi=300)
    plt.close()
    print(f"[+] Gráfica de 3 estados guardada exitosamente en: {salida_png}")


def main():
    parser = argparse.ArgumentParser(description="Comparativa Temporal 3 Estados y Extracción de Ruido")
    parser.add_argument("--video", default="video2", help="Identificador del video (video1 o video2)")
    parser.add_argument("--carpeta-original", default="videos/video2/frames_originales", help="Ruta a frames originales con HUD")
    parser.add_argument("--carpeta-sin-hud", default="videos/video2/frames_sin_hud", help="Ruta a frames sin HUD")
    parser.add_argument("--carpeta-denoised", default="videos/video2/expos/udvd_blindspot_corregido_ult10min", help="Ruta a frames denoised con UDVD")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Dispositivo")
    parser.add_argument("--max-frames", type=int, default=None, help="Límite opcional de frames")
    parser.add_argument("--workers", type=int, default=8, help="Hilos en paralelo")
    args = parser.parse_args()

    dir_orig = (RAIZ / args.carpeta_original).resolve()
    dir_sin_hud = (RAIZ / args.carpeta_sin_hud).resolve()
    dir_denoised = (RAIZ / args.carpeta_denoised).resolve()

    if not dir_orig.is_dir():
        cands = list(RAIZ.glob(f"videos/{args.video}/*original*"))
        if cands:
            dir_orig = cands[0]
    if not dir_sin_hud.is_dir():
        cands = list(RAIZ.glob(f"videos/{args.video}/*sin_hud*"))
        if cands:
            dir_sin_hud = cands[0]
    if not dir_denoised.is_dir():
        cands = list(RAIZ.glob(f"videos/{args.video}/expos/*udvd*"))
        if cands:
            dir_denoised = cands[0]

    print("=" * 90)
    print(f"   ANÁLISIS COMPARATIVO DE LOS 3 ESTADOS (ORIGINAL vs SIN HUD vs UDVD) - {args.video.upper()}")
    print(f"1. Original (Con HUD + Ruido): {dir_orig}")
    print(f"2. Sin HUD (Con Ruido):        {dir_sin_hud}")
    print(f"3. Restaurado (UDVD Denoised): {dir_denoised}")
    print("=" * 90)

    exts = {".png", ".jpg", ".jpeg"}
    archivos_denoised = sorted([p for p in dir_denoised.iterdir() if p.is_file() and p.suffix.lower() in exts], key=natural_key)
    if args.max_frames:
        archivos_denoised = archivos_denoised[:args.max_frames]

    print(f"Frames a evaluar en la intersección: {len(archivos_denoised)}")

    dispositivo = torch.device(args.device if torch.cuda.is_available() else "cpu")
    m_niqe, m_brisque = crear_metricas_pyiqa(dispositivo)

    dir_ruido_termico = RAIZ / f"videos/{args.video}/ruido_residual_termico"
    dir_ruido_hud = RAIZ / f"videos/{args.video}/ruido_residual_hud"
    dir_ruido_termico.mkdir(parents=True, exist_ok=True)
    dir_ruido_hud.mkdir(parents=True, exist_ok=True)

    registros = []
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = []

    with torch.inference_mode():
        for idx, f_den in enumerate(tqdm(archivos_denoised, desc=f"Evaluando {args.video}")):
            nom = f_den.name
            f_ori = dir_orig / nom
            f_sin = dir_sin_hud / nom

            if not f_ori.is_file() or not f_sin.is_file():
                continue

            im_ori = cv2.imread(str(f_ori))
            im_sin = cv2.imread(str(f_sin))
            im_den = cv2.imread(str(f_den))

            if im_ori is None or im_sin is None or im_den is None:
                continue

            # Ajustar dimensiones si difieren
            h, w = im_sin.shape[:2]
            if im_ori.shape[:2] != (h, w):
                im_ori = cv2.resize(im_ori, (w, h), interpolation=cv2.INTER_AREA)
            if im_den.shape[:2] != (h, w):
                im_den = cv2.resize(im_den, (w, h), interpolation=cv2.INTER_AREA)

            gr_ori = cv2.cvtColor(im_ori, cv2.COLOR_BGR2GRAY)
            gr_sin = cv2.cvtColor(im_sin, cv2.COLOR_BGR2GRAY)
            gr_den = cv2.cvtColor(im_den, cv2.COLOR_BGR2GRAY)

            # 1. Extracción física de residuos
            res_hud = np.abs(gr_ori.astype(np.float32) - gr_sin.astype(np.float32))
            res_termico = gr_sin.astype(np.float32) - gr_den.astype(np.float32)

            mae_hud = float(np.mean(res_hud))
            mae_termico = float(np.mean(np.abs(res_termico)))

            # Guardar mapa de ruido térmico para Jorge (centrado en 128)
            ruido_jorge = np.clip(res_termico + 128.0, 0.0, 255.0).astype(np.uint8)
            dest_termico = dir_ruido_termico / f"ruido_termico_{nom}"
            futures.append(pool.submit(cv2.imwrite, str(dest_termico), ruido_jorge, [cv2.IMWRITE_PNG_COMPRESSION, 3]))

            # 2. Métricas de Ruido y Nitidez
            s_ori = estimar_sigma_mad(gr_ori)
            s_sin = estimar_sigma_mad(gr_sin)
            s_den = estimar_sigma_mad(gr_den)

            _, l_ori = calcular_densidad_estructural(gr_ori)
            _, l_sin = calcular_densidad_estructural(gr_sin)
            _, l_den = calcular_densidad_estructural(gr_den)

            # 3. Métricas BRISQUE y NIQE
            b_ori, b_sin, b_den = 0.0, 0.0, 0.0
            n_ori, n_sin, n_den = 0.0, 0.0, 0.0

            if m_brisque is not None and m_niqe is not None:
                try:
                    t_ori = torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dispositivo)
                    t_sin = torch.from_numpy(im_sin).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dispositivo)
                    t_den = torch.from_numpy(im_den).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dispositivo)

                    b_ori = float(m_brisque(t_ori).item())
                    b_sin = float(m_brisque(t_sin).item())
                    b_den = float(m_brisque(t_den).item())

                    n_ori = float(m_niqe(t_ori).item())
                    n_sin = float(m_niqe(t_sin).item())
                    n_den = float(m_niqe(t_den).item())
                except Exception:
                    pass

            registros.append({
                "frame_idx": idx + 1,
                "nombre_archivo": nom,
                "brisque_orig": round(b_ori, 2),
                "brisque_sin_hud": round(b_sin, 2),
                "brisque_denoised": round(b_den, 2),
                "niqe_orig": round(n_ori, 3),
                "niqe_sin_hud": round(n_sin, 3),
                "niqe_denoised": round(n_den, 3),
                "sigma_orig": round(s_ori, 4),
                "sigma_sin_hud": round(s_sin, 4),
                "sigma_denoised": round(s_den, 4),
                "var_lap_orig": round(l_ori, 2),
                "var_lap_sin_hud": round(l_sin, 2),
                "var_lap_denoised": round(l_den, 2),
                "mae_hud": round(mae_hud, 4),
                "mae_termico": round(mae_termico, 4),
            })

    for f in futures:
        f.result()
    pool.shutdown()

    if not registros:
        print("[-] No se pudieron generar registros.")
        return

    df = pd.DataFrame(registros)
    csv_out = RAIZ / f"videos/{args.video}/comparativa_3_estados_{args.video}.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[+] Tabla CSV exportada a: {csv_out}")

    png_out = RAIZ / f"figuras_tesis/linea_tiempo/linea_tiempo_3_estados_{args.video}.png"
    graficar_comparativa_3_estados(df, png_out, f"{args.video.upper()} (Misión Santander 15.820 FT)")


if __name__ == "__main__":
    main()
