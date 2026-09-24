#!/usr/bin/env python3
"""
Comparativa Temporal Ultra-Rápida de Ruido y Calidad para Todos los Videos (Video 1, Video 2, Video 3):
- Video 1: Original vs Sin HUD vs Denoised (UDVD).
- Video 2: Original vs Sin HUD vs Restaurado (UDVD/Ensemble).
- Video 3: Original vs Sin HUD (sobre los 139.520 frames procesados).
- Genera la figura comparativa consolidada multi-vuelo para el artículo de tesis.

Acelerado con inferencia por lotes en GPU (Batch Size = 16) y Stride Inteligente.
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
        print(f"[!] pyiqa no disponible: {e}.")
        return None, None


def graficar_comparativa(df, salida_png, titulo="Video", tiene_denoised=True):
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

    # 3. Sigma MAD
    ax = axs[idx_ax]
    s_o_ma = df["sigma_orig"].rolling(ventana, center=True, min_periods=1).mean().values
    s_s_ma = df["sigma_sin_hud"].rolling(ventana, center=True, min_periods=1).mean().values
    ax.plot(frames, s_o_ma, color=col_orig, lw=2.2, label="1. Sigma Original (Con HUD)")
    ax.plot(frames, s_s_ma, color=col_sin_hud, lw=2.0, linestyle="--", label="2. Sigma Sin HUD")
    if tiene_denoised and "sigma_denoised" in df.columns:
        s_d_ma = df["sigma_denoised"].rolling(ventana, center=True, min_periods=1).mean().values
        ax.plot(frames, s_d_ma, color=col_den, lw=2.4, label="3. Sigma Restaurado (Denoised)")
    ax.set_ylabel("Sigma Ruido (σ MAD)", fontsize=11, fontweight="bold")
    ax.set_title("Nivel de Ruido de Alta Frecuencia (Estimador Robusto MAD)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    idx_ax += 1

    # 4. Ritmo Estructural
    ax = axs[idx_ax]
    l_o_ma = df["var_lap_orig"].rolling(ventana, center=True, min_periods=1).mean().values
    l_s_ma = df["var_lap_sin_hud"].rolling(ventana, center=True, min_periods=1).mean().values
    ax.plot(frames, l_o_ma, color=col_orig, lw=1.8, alpha=0.7, label="1. Original (Inflado por letras HUD)")
    ax.plot(frames, l_s_ma, color=col_sin_hud, lw=2.0, linestyle="--", label="2. Sin HUD (Estructura real)")
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


def graficar_comparativa_multivuelo(dfs_dict, salida_png):
    """Genera la comparativa de 3 paneles verticales para los 3 videos."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(len(dfs_dict), 1, figsize=(16, 4.5 * len(dfs_dict)), sharey=False)
    if len(dfs_dict) == 1:
        axs = [axs]

    nombres_leg = {
        "video1": "Video 1: Telembí / Patía (FAC 5748 - 10.500 FT)",
        "video2": "Video 2: Santander (FLIR Systems - 15.820 FT)",
        "video3": "Video 3: Misión de Vigilancia Continua",
    }

    fig.suptitle("Comparativa Multi-Misión: Evolución del Nivel de Ruido Térmico (Original vs Sin HUD vs Denoised)", fontsize=15, fontweight="bold", y=0.99)

    for idx, (v_id, df) in enumerate(dfs_dict.items()):
        ax = axs[idx]
        frames = df["frame_idx"].values
        ventana = max(3, len(df) // 100)

        s_o_ma = df["sigma_orig"].rolling(ventana, center=True, min_periods=1).mean().values
        s_s_ma = df["sigma_sin_hud"].rolling(ventana, center=True, min_periods=1).mean().values

        ax.plot(frames, s_o_ma, color="#d62728", lw=2.0, label="1. Original (Con HUD)")
        ax.plot(frames, s_s_ma, color="#1f77b4", lw=2.0, linestyle="--", label="2. Sin HUD")

        if "sigma_denoised" in df.columns and df["sigma_denoised"].sum() > 0:
            s_d_ma = df["sigma_denoised"].rolling(ventana, center=True, min_periods=1).mean().values
            ax.plot(frames, s_d_ma, color="#2ca02c", lw=2.2, label="3. Restaurado Denoised (UDVD)")

        ax.set_ylabel("Sigma Ruido (σ MAD)", fontsize=11, fontweight="bold")
        ax.set_title(f"{nombres_leg.get(v_id, v_id.upper())}", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("Índice de Cuadro Temporal (Frames de Vuelo)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    salida_png = Path(salida_png)
    salida_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida_png, dpi=300)
    plt.close()
    print(f"\n[+] Gráfica consolidada multi-vuelo exportada a: {salida_png}")


def procesar_un_video(v_id, stride=30, batch_size=16, device="cuda:0", m_niqe=None, m_brisque=None):
    # Carpetas según video
    dir_orig = (RAIZ / f"videos/{v_id}/frames_originales").resolve()
    
    p_sh = RAIZ / f"videos/{v_id}/frames_sin_hud"
    p_tmp = RAIZ / f"videos/{v_id}/frames_sin_hud.tmp"
    dir_sin_hud = (p_sh if p_sh.is_dir() else (p_tmp if p_tmp.is_dir() else p_sh)).resolve()

    dir_denoised = None
    if v_id == "video1":
        p_d = RAIZ / "videos/video1/expos/UDVD_SinHUD_K5_lr1e3"
        if not p_d.is_dir():
            p_d = RAIZ / "videos/video1/expos/n2n_sin_hud"
        if p_d.is_dir():
            dir_denoised = p_d.resolve()
    elif v_id == "video2":
        p_u = RAIZ / "videos/video2/expos/udvd_blindspot_corregido_ult10min"
        if not p_u.is_dir():
            p_u = RAIZ / "videos/video2/expos/udvd"
        if not p_u.is_dir():
            cands = list(RAIZ.glob("videos/video2/expos/*udvd*"))
            if cands:
                p_u = cands[0]
            else:
                cands_all = [c for c in RAIZ.glob("videos/video2/expos/*") if c.is_dir()]
                p_u = cands_all[0] if cands_all else None
        if p_u and p_u.is_dir():
            dir_denoised = p_u.resolve()

    tiene_denoised = dir_denoised is not None and dir_denoised.is_dir()

    if not dir_orig.is_dir() or not dir_sin_hud.is_dir():
        print(f"[-] Omitiendo {v_id}: carpetas no encontradas ({dir_orig} o {dir_sin_hud})")
        return None

    print("\n" + "=" * 85)
    print(f"   ANÁLISIS TEMPORAL TURBO: {v_id.upper()} (Batch Size = {batch_size} | Stride = {stride})")
    print(f"1. Original:    {dir_orig}")
    print(f"2. Sin HUD:     {dir_sin_hud}")
    print(f"3. Denoised:    {dir_denoised if tiene_denoised else '[No configurado - Evaluando 2 Estados]'}")
    print("=" * 85)

    exts = {".png", ".jpg", ".jpeg"}
    archivos_sin = sorted([p for p in dir_sin_hud.iterdir() if p.is_file() and p.suffix.lower() in exts], key=natural_key)
    archivos_eval = archivos_sin[::stride]
    total_eval = len(archivos_eval)

    print(f"Frames totales en carpeta: {len(archivos_sin)}")
    print(f"Puntos de muestreo a evaluar en GPU: {total_eval}\n")

    dispositivo = torch.device(device if torch.cuda.is_available() else "cpu")
    registros = []

    with torch.inference_mode():
        for b_idx in tqdm(range(0, total_eval, batch_size), desc=f"Procesando {v_id}"):
            batch_files = archivos_eval[b_idx : b_idx + batch_size]

            t_orig_list, t_sin_list, t_den_list = [], [], []
            info_batch = []

            for f_sin in batch_files:
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

                s_ori = estimar_sigma_mad(gr_ori)
                s_sin = estimar_sigma_mad(gr_sin)
                _, l_ori = calcular_densidad_estructural(gr_ori)
                _, l_sin = calcular_densidad_estructural(gr_sin)

                s_den, l_den = 0.0, 0.0
                if im_den is not None:
                    gr_den = cv2.cvtColor(im_den, cv2.COLOR_BGR2GRAY)
                    s_den = estimar_sigma_mad(gr_den)
                    _, l_den = calcular_densidad_estructural(gr_den)

                t_orig_list.append(torch.from_numpy(im_ori).permute(2, 0, 1).float().div(255.0))
                t_sin_list.append(torch.from_numpy(im_sin).permute(2, 0, 1).float().div(255.0))
                if im_den is not None:
                    t_den_list.append(torch.from_numpy(im_den).permute(2, 0, 1).float().div(255.0))

                info_batch.append((nom, s_ori, s_sin, l_ori, l_sin, s_den, l_den, im_den is not None))

            if not info_batch:
                continue

            # Batch GPU para BRISQUE y NIQE
            b_ori_vals, b_sin_vals, b_den_vals = [0.0] * len(info_batch), [0.0] * len(info_batch), [0.0] * len(info_batch)
            n_ori_vals, n_sin_vals, n_den_vals = [0.0] * len(info_batch), [0.0] * len(info_batch), [0.0] * len(info_batch)

            if m_brisque is not None and m_niqe is not None:
                try:
                    tensor_ori_batch = torch.stack(t_orig_list).to(dispositivo)
                    tensor_sin_batch = torch.stack(t_sin_list).to(dispositivo)

                    b_ori_out = m_brisque(tensor_ori_batch).view(-1).cpu().tolist()
                    b_sin_out = m_brisque(tensor_sin_batch).view(-1).cpu().tolist()
                    n_ori_out = m_niqe(tensor_ori_batch).view(-1).cpu().tolist()
                    n_sin_out = m_niqe(tensor_sin_batch).view(-1).cpu().tolist()

                    b_ori_vals = [b_ori_out] if isinstance(b_ori_out, float) else b_ori_out
                    b_sin_vals = [b_sin_out] if isinstance(b_sin_out, float) else b_sin_out
                    n_ori_vals = [n_ori_out] if isinstance(n_ori_out, float) else n_ori_out
                    n_sin_vals = [n_sin_out] if isinstance(n_sin_out, float) else n_sin_out

                    if len(t_den_list) == len(info_batch):
                        tensor_den_batch = torch.stack(t_den_list).to(dispositivo)
                        b_den_out = m_brisque(tensor_den_batch).view(-1).cpu().tolist()
                        n_den_out = m_niqe(tensor_den_batch).view(-1).cpu().tolist()
                        b_den_vals = [b_den_out] if isinstance(b_den_out, float) else b_den_out
                        n_den_vals = [n_den_out] if isinstance(n_den_out, float) else n_den_out
                except Exception:
                    pass

            for idx_in_b, (nom, s_ori, s_sin, l_ori, l_sin, s_den, l_den, has_den) in enumerate(info_batch):
                item = {
                    "frame_idx": (b_idx + idx_in_b) * stride + 1,
                    "nombre_archivo": nom,
                    "brisque_orig": round(float(b_ori_vals[idx_in_b]), 2),
                    "brisque_sin_hud": round(float(b_sin_vals[idx_in_b]), 2),
                    "niqe_orig": round(float(n_ori_vals[idx_in_b]), 3),
                    "niqe_sin_hud": round(float(n_sin_vals[idx_in_b]), 3),
                    "sigma_orig": round(s_ori, 4),
                    "sigma_sin_hud": round(s_sin, 4),
                    "var_lap_orig": round(l_ori, 2),
                    "var_lap_sin_hud": round(l_sin, 2),
                }
                if tiene_denoised and has_den:
                    item.update({
                        "brisque_denoised": round(float(b_den_vals[idx_in_b]), 2),
                        "niqe_denoised": round(float(n_den_vals[idx_in_b]), 3),
                        "sigma_denoised": round(s_den, 4),
                        "var_lap_denoised": round(l_den, 2),
                    })
                registros.append(item)

    if not registros:
        return None

    df = pd.DataFrame(registros)
    csv_out = RAIZ / f"videos/{v_id}/linea_tiempo_{v_id}.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"[+] Tabla CSV guardada en: {csv_out}")

    png_out = RAIZ / f"figuras_tesis/linea_tiempo/linea_tiempo_{v_id}.png"
    graficar_comparativa(df, png_out, f"{v_id.upper()}", tiene_denoised=tiene_denoised)
    return df


def main():
    parser = argparse.ArgumentParser(description="Líneas de Tiempo Turbo de Ruido para Videos 1, 2 y 3")
    parser.add_argument("--video", default="todos", help="Identificador: video1, video2, video3 o todos")
    parser.add_argument("--stride-v1", type=int, default=15, help="Stride Video 1")
    parser.add_argument("--stride-v2", type=int, default=30, help="Stride Video 2")
    parser.add_argument("--stride-v3", type=int, default=60, help="Stride Video 3")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch Size GPU")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Dispositivo")
    args = parser.parse_args()

    dispositivo = torch.device(args.device if torch.cuda.is_available() else "cpu")
    m_niqe, m_brisque = crear_metricas_pyiqa(dispositivo, habilitar=True)

    strides = {
        "video1": args.stride_v1,
        "video2": args.stride_v2,
        "video3": args.stride_v3,
    }

    dfs_dict = {}
    videos_a_procesar = ["video1", "video2", "video3"] if args.video == "todos" else [args.video]

    for v in videos_a_procesar:
        df_v = procesar_un_video(
            v_id=v,
            stride=strides.get(v, 30),
            batch_size=args.batch_size,
            device=args.device,
            m_niqe=m_niqe,
            m_brisque=m_brisque,
        )
        if df_v is not None and not df_v.empty:
            dfs_dict[v] = df_v

    if len(dfs_dict) > 1:
        comp_png = RAIZ / "figuras_tesis/linea_tiempo/comparativa_ritmo_ruido_3videos.png"
        graficar_comparativa_multivuelo(dfs_dict, comp_png)

    print("\n" + "=" * 85)
    print("ANÁLISIS DE LÍNEAS DE TIEMPO COMPLETADO EXITOSAMENTE PARA TODOS LOS VIDEOS.")
    print("=" * 85)


if __name__ == "__main__":
    main()
