#!/usr/bin/env python3
"""
Comparativa Temporal Ultra-Rápida y Extracción de Residuos de Ruido:
Soporta Video 1, Video 2 y Video 3 de forma flexible y ultrarrápida:
- Si hay modelo Denoised: Grafica 3 estados (1. Original, 2. Sin HUD, 3. Restaurado Denoised).
- Si NO hay modelo Denoised (o en Video 3 interrumpido): Grafica 2 estados (1. Original vs 2. Sin HUD) evaluando ÚNICAMENTE los frames que ya están procesados.

Optimizaciones de Alta Velocidad:
- Sub-muestreo temporal inteligente (--stride 10 o 15): Evalúa curvas de alta densidad en ~1 a 2 minutos sin perder resolución.
- Guardado asíncrono de mapas de ruido en disco con 16 hilos.
- Cálculo vectorial NumPy/OpenCV para resta instantánea de imágenes.
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


def crear_metricas_pyiqa(dispositivo, habilitar=True):
    if not habilitar:
        return None, None
    try:
        import pyiqa
        m_niqe = pyiqa.create_metric("niqe", device=dispositivo)
        m_brisque = pyiqa.create_metric("brisque", device=dispositivo)
        return m_niqe, m_brisque
    except Exception as e:
        print(f"[!] pyiqa no cargado: {e}.")
        return None, None


def graficar_comparativa(df, salida_png, titulo="Video", tiene_denoised=True):
    """Genera la figura con 2 o 3 curvas sincronizadas."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    n_paneles = 4 if ("brisque_orig" in df.columns and df["brisque_orig"].sum() > 0) else 3
    fig, axs = plt.subplots(n_paneles, 1, figsize=(16, 4 * n_paneles), sharex=True)
    if n_paneles == 1:
        axs = [axs]

    fig.suptitle(f"Evolución Temporal de Calidad y Ruido - {titulo}", fontsize=15, fontweight="bold", y=0.98)

    frames = df["frame_idx"].values
    ventana = max(3, len(df) // 100)

    col_orig = "#d62728"      # Rojo
    col_sin_hud = "#1f77b4"   # Azul
    col_den = "#2ca02c"       # Verde

    idx_ax = 0

    # 1. BRISQUE
    if "brisque_orig" in df.columns and df["brisque_orig"].sum() > 0:
        ax = axs[idx_ax]
        b_o_ma = df["brisque_orig"].rolling(ventana, center=True, min_periods=1).mean().values
        b_s_ma = df["brisque_sin_hud"].rolling(ventana, center=True, min_periods=1).mean().values
        ax.plot(frames, b_o_ma, color=col_orig, lw=2.2, label=f"1. Original (Con HUD + Ruido)")
        ax.plot(frames, b_s_ma, color=col_sin_hud, lw=2.0, linestyle="--", label=f"2. Sin HUD (Con Ruido)")
        if tiene_denoised and "brisque_denoised" in df.columns:
            b_d_ma = df["brisque_denoised"].rolling(ventana, center=True, min_periods=1).mean().values
            ax.plot(frames, b_d_ma, color=col_den, lw=2.4, label=f"3. Restaurado (Sin HUD + Denoised)")
        ax.set_ylabel("BRISQUE (↓ Mejor)", fontsize=11, fontweight="bold")
        ax.set_title("Calidad Espacial Perceptual (BRISQUE)", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)
        idx_ax += 1

    # 2. NIQE
    if "niqe_orig" in df.columns and df["niqe_orig"].sum() > 0:
        ax = axs[idx_ax]
        n_o_ma = df["niqe_orig"].rolling(ventana, center=True, min_periods=1).mean().values
        n_s_ma = df["niqe_sin_hud"].rolling(ventana, center=True, min_periods=1).mean().values
        ax.plot(frames, n_o_ma, color=col_orig, lw=2.2, label="1. Original (Con HUD)")
        ax.plot(frames, n_s_ma, color=col_sin_hud, lw=2.0, linestyle="--", label="2. Sin HUD")
        if tiene_denoised and "niqe_denoised" in df.columns:
            n_d_ma = df["niqe_denoised"].rolling(ventana, center=True, min_periods=1).mean().values
            ax.plot(frames, n_d_ma, color=col_den, lw=2.4, label="3. Restaurado (Denoised)")
        ax.set_ylabel("NIQE (↓ Mejor)", fontsize=11, fontweight="bold")
        ax.set_title("Naturalidad Estadística de la Imagen Térmica (NIQE)", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)
        idx_ax += 1

    # 3. Sigma MAD (Nivel de Ruido Físico)
    ax = axs[idx_ax]
    s_o_ma = df["sigma_orig"].rolling(ventana, center=True, min_periods=1).mean().values
    s_s_ma = df["sigma_sin_hud"].rolling(ventana, center=True, min_periods=1).mean().values
    ax.plot(frames, s_o_ma, color=col_orig, lw=2.2, label="1. Sigma Original (Con HUD)")
    ax.plot(frames, s_s_ma, color=col_sin_hud, lw=2.0, linestyle="--", label="2. Sigma Sin HUD")
    if tiene_denoised and "sigma_denoised" in df.columns:
        s_d_ma = df["sigma_denoised"].rolling(ventana, center=True, min_periods=1).mean().values
        ax.plot(frames, s_d_ma, color=col_den, lw=2.4, label="3. Sigma Restaurado (UDVD)")
    ax.set_ylabel("Sigma Ruido (σ MAD)", fontsize=11, fontweight="bold")
    ax.set_title("Nivel de Ruido de Alta Frecuencia (Estimador Robusto MAD)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    idx_ax += 1

    # 4. Ritmo Estructural (Varianza Laplaciana)
    ax = axs[idx_ax]
    l_o_ma = df["var_lap_orig"].rolling(ventana, center=True, min_periods=1).mean().values
    l_s_ma = df["var_lap_sin_hud"].rolling(ventana, center=True, min_periods=1).mean().values
    ax.plot(frames, l_o_ma, color=col_orig, lw=1.8, alpha=0.7, label="1. Original (Inflado por letras HUD)")
    ax.plot(frames, l_s_ma, color=col_sin_hud, lw=2.0, linestyle="--", label="2. Sin HUD (Estructura real del terreno)")
    if tiene_denoised and "var_lap_denoised" in df.columns:
        l_d_ma = df["var_lap_denoised"].rolling(ventana, center=True, min_periods=1).mean().values
        ax.plot(frames, l_d_ma, color=col_den, lw=2.2, label="3. Restaurado Denoised (Bordes preservados)")
    ax.set_ylabel("Varianza Laplaciana", fontsize=11, fontweight="bold")
    ax.set_xlabel("Índice de Cuadro Temporal (Frames)", fontsize=12, fontweight="bold")
    ax.set_title("Ritmo Estructural por Escena y Preservación de Detalles", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    salida_png = Path(salida_png)
    salida_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida_png, dpi=300)
    plt.close()
    print(f"[+] Gráfica temporal guardada exitosamente en: {salida_png}")


def main():
    parser = argparse.ArgumentParser(description="Comparativa Temporal Rápida y Extracción de Ruido")
    parser.add_argument("--video", default="video2", help="Identificador del video (video1, video2 o video3)")
    parser.add_argument("--carpeta-original", default=None, help="Ruta a frames originales con HUD")
    parser.add_argument("--carpeta-sin-hud", default=None, help="Ruta a frames sin HUD")
    parser.add_argument("--carpeta-denoised", default=None, help="Ruta a frames denoised (opcional)")
    parser.add_argument("--stride", type=int, default=10, help="Paso de muestreo temporal (default: 10 para velocidad instantánea)")
    parser.add_argument("--calcular-iqa", action="store_true", default=True, help="Calcular BRISQUE y NIQE (en GPU)")
    parser.add_argument("--guardar-mapas-ruido", action="store_true", default=True, help="Guardar imágenes de ruido en disco")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Dispositivo")
    parser.add_argument("--workers", type=int, default=16, help="Hilos en paralelo")
    args = parser.parse_args()

    # Detección automática de rutas según el video
    if not args.carpeta_original:
        args.carpeta_original = f"videos/{args.video}/frames_originales"
    if not args.carpeta_sin_hud:
        # En video 3 buscar frames_sin_hud.tmp si no existe frames_sin_hud
        p_sh = RAIZ / f"videos/{args.video}/frames_sin_hud"
        p_tmp = RAIZ / f"videos/{args.video}/frames_sin_hud.tmp"
        args.carpeta_sin_hud = str(p_sh if p_sh.is_dir() else (p_tmp if p_tmp.is_dir() else p_sh))
    if not args.carpeta_denoised and args.video == "video2":
        args.carpeta_denoised = "videos/video2/expos/udvd_blindspot_corregido_ult10min"
    elif not args.carpeta_denoised and args.video == "video1":
        p_d1 = RAIZ / "videos/video1/expos/UDVD_SinHUD_K5_lr1e3"
        p_d2 = RAIZ / "videos/video1/expos/n2n_sin_hud"
        args.carpeta_denoised = str(p_d1 if p_d1.is_dir() else (p_d2 if p_d2.is_dir() else ""))

    dir_orig = (RAIZ / args.carpeta_original).resolve()
    dir_sin_hud = (RAIZ / args.carpeta_sin_hud).resolve()
    dir_denoised = (RAIZ / args.carpeta_denoised).resolve() if args.carpeta_denoised else None

    tiene_denoised = dir_denoised is not None and dir_denoised.is_dir()

    print("=" * 90)
    print(f"   ANÁLISIS TEMPORAL ULTRA-RÁPIDO - {args.video.upper()} (Stride = {args.stride})")
    print(f"1. Original (Con HUD):      {dir_orig}")
    print(f"2. Sin HUD (Inpainted):     {dir_sin_hud}")
    if tiene_denoised:
        print(f"3. Denoised (Restaurado):   {dir_denoised}")
    else:
        print(f"3. Denoised:                [No configurado - Evaluando 2 Estados: Original vs Sin HUD]")
    print("=" * 90)

    exts = {".png", ".jpg", ".jpeg"}
    # Intersección automática: tomar los frames que existan en la carpeta Sin HUD
    archivos_sin_hud = sorted([p for p in dir_sin_hud.iterdir() if p.is_file() and p.suffix.lower() in exts], key=natural_key)
    
    # Aplicar stride para acelerar
    if args.stride > 1:
        archivos_eval = archivos_sin_hud[::args.stride]
    else:
        archivos_eval = archivos_sin_hud

    total_eval = len(archivos_eval)
    print(f"Total frames disponibles en Sin HUD: {len(archivos_sin_hud)}")
    print(f"Frames seleccionados para evaluación con Stride {args.stride}: {total_eval}\n")

    dispositivo = torch.device(args.device if torch.cuda.is_available() else "cpu")
    m_niqe, m_brisque = crear_metricas_pyiqa(dispositivo, habilitar=args.calcular_iqa)

    dir_ruido_hud = RAIZ / f"videos/{args.video}/ruido_residual_hud"
    dir_ruido_termico = RAIZ / f"videos/{args.video}/ruido_residual_termico"
    if args.guardar_mapas_ruido:
        dir_ruido_hud.mkdir(parents=True, exist_ok=True)
        if tiene_denoised:
            dir_ruido_termico.mkdir(parents=True, exist_ok=True)

    registros = []
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = []

    with torch.inference_mode():
        for idx_rel, f_sin in enumerate(tqdm(archivos_eval, desc=f"Evaluando {args.video}")):
            nom = f_sin.name
            f_ori = dir_orig / nom
            if not f_ori.is_file():
                continue

            im_ori = cv2.imread(str(f_ori))
            im_sin = cv2.imread(str(f_sin))
            if im_ori is None or im_sin is None:
                continue

            h, w = im_sin.shape[:2]
            if im_ori.shape[:2] != (h, w):
                im_ori = cv2.resize(im_ori, (w, h), interpolation=cv2.INTER_AREA)

            im_den = None
            if tiene_denoised:
                f_den = dir_denoised / nom
                if f_den.is_file():
                    im_den = cv2.imread(str(f_den))
                    if im_den is not None and im_den.shape[:2] != (h, w):
                        im_den = cv2.resize(im_den, (w, h), interpolation=cv2.INTER_AREA)

            gr_ori = cv2.cvtColor(im_ori, cv2.COLOR_BGR2GRAY)
            gr_sin = cv2.cvtColor(im_sin, cv2.COLOR_BGR2GRAY)

            # Residuos
            res_hud = np.abs(gr_ori.astype(np.float32) - gr_sin.astype(np.float32))
            mae_hud = float(np.mean(res_hud))

            if args.guardar_mapas_ruido:
                ruido_hud_img = np.clip(res_hud, 0.0, 255.0).astype(np.uint8)
                futures.append(pool.submit(cv2.imwrite, str(dir_ruido_hud / f"ruido_hud_{nom}"), ruido_hud_img, [cv2.IMWRITE_PNG_COMPRESSION, 3]))

            mae_termico = 0.0
            s_den = 0.0
            l_den = 0.0
            b_den = 0.0
            n_den = 0.0

            if im_den is not None:
                gr_den = cv2.cvtColor(im_den, cv2.COLOR_BGR2GRAY)
                res_termico = gr_sin.astype(np.float32) - gr_den.astype(np.float32)
                mae_termico = float(np.mean(np.abs(res_termico)))
                if args.guardar_mapas_ruido:
                    ruido_termico_img = np.clip(res_termico + 128.0, 0.0, 255.0).astype(np.uint8)
                    futures.append(pool.submit(cv2.imwrite, str(dir_ruido_termico / f"ruido_termico_{nom}"), ruido_termico_img, [cv2.IMWRITE_PNG_COMPRESSION, 3]))
                s_den = estimar_sigma_mad(gr_den)
                _, l_den = calcular_densidad_estructural(gr_den)

            # Métricas Estadísticas
            s_ori = estimar_sigma_mad(gr_ori)
            s_sin = estimar_sigma_mad(gr_sin)
            _, l_ori = calcular_densidad_estructural(gr_ori)
            _, l_sin = calcular_densidad_estructural(gr_sin)

            # Métricas BRISQUE y NIQE
            b_ori, b_sin = 0.0, 0.0
            n_ori, n_sin = 0.0, 0.0

            if m_brisque is not None and m_niqe is not None:
                try:
                    t_ori = torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dispositivo)
                    t_sin = torch.from_numpy(im_sin).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dispositivo)
                    b_ori = float(m_brisque(t_ori).item())
                    b_sin = float(m_brisque(t_sin).item())
                    n_ori = float(m_niqe(t_ori).item())
                    n_sin = float(m_niqe(t_sin).item())
                    if im_den is not None:
                        t_den = torch.from_numpy(im_den).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dispositivo)
                        b_den = float(m_brisque(t_den).item())
                        n_den = float(m_niqe(t_den).item())
                except Exception:
                    pass

            item_reg = {
                "frame_idx": (idx_rel * args.stride) + 1,
                "nombre_archivo": nom,
                "brisque_orig": round(b_ori, 2),
                "brisque_sin_hud": round(b_sin, 2),
                "niqe_orig": round(n_ori, 3),
                "niqe_sin_hud": round(n_sin, 3),
                "sigma_orig": round(s_ori, 4),
                "sigma_sin_hud": round(s_sin, 4),
                "var_lap_orig": round(l_ori, 2),
                "var_lap_sin_hud": round(l_sin, 2),
                "mae_hud": round(mae_hud, 4),
            }

            if tiene_denoised:
                item_reg.update({
                    "brisque_denoised": round(b_den, 2),
                    "niqe_denoised": round(n_den, 3),
                    "sigma_denoised": round(s_den, 4),
                    "var_lap_denoised": round(l_den, 2),
                    "mae_termico": round(mae_termico, 4),
                })

            registros.append(item_reg)

    for f in futures:
        f.result()
    pool.shutdown()

    if not registros:
        print("[-] No se generaron registros.")
        return

    df = pd.DataFrame(registros)
    csv_out = RAIZ / f"videos/{args.video}/linea_tiempo_{args.video}.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[+] Tabla CSV exportada a: {csv_out}")

    png_out = RAIZ / f"figuras_tesis/linea_tiempo/linea_tiempo_{args.video}.png"
    graficar_comparativa(df, png_out, f"{args.video.upper()}", tiene_denoised=tiene_denoised)


if __name__ == "__main__":
    main()
