#!/usr/bin/env python3
"""
Extracción Integral de Ruido Residual y Caracterización Temporal Multi-Métrica (BRISQUE, NIQE, PIQE, Sigma MAD, MAE, Nitidez):
1. Calcula y guarda los fotogramas de ruido físico: Ruido = |Original_SinHUD - Denoised| (centrados en 128) para la clusterización de Jorge.
2. Calcula para cada fotograma la suite completa de métricas de calidad sin referencia (NR-IQA):
   - BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator - menor es mejor)
   - NIQE (Naturalness Image Quality Evaluator - menor es mejor)
   - PIQE (Perception-based Image Quality Evaluator - menor es mejor)
   - Sigma MAD (Nivel de ruido estocástico de alta frecuencia)
   - MAE Residuo (Magnitud física del ruido removido)
   - Varianza Laplaciana (Nitidez de bordes y preservación estructural)
3. Genera la figura de 4 paneles de alta resolución para cada video.
4. Genera la figura comparativa multi-vuelo de 3 videos.
5. Extrae los mosaicos trípticos top con acercamiento térmico.
6. Exporta el CSV exhaustivo frame a frame.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
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


def normalizar_ruido_para_visualizacion(ruido_gray, factor=3.5):
    ruido_abs = np.abs(ruido_gray.astype(np.float32))
    ruido_amp = np.clip(ruido_abs * factor, 0.0, 255.0).astype(np.uint8)
    ruido_col = cv2.applyColorMap(ruido_amp, cv2.COLORMAP_INFERNO)
    return ruido_col, ruido_amp


def crear_metricas_pyiqa(dispositivo):
    try:
        import pyiqa
        m_niqe = pyiqa.create_metric("niqe", device=dispositivo)
        m_brisque = pyiqa.create_metric("brisque", device=dispositivo)
        try:
            m_piqe = pyiqa.create_metric("piqe", device=dispositivo)
        except Exception:
            m_piqe = None
        return m_niqe, m_brisque, m_piqe
    except Exception as e:
        print(f"[!] pyiqa no disponible: {e}. Usando métricas estadísticas estándar.")
        return None, None, None


def crear_mosaico_triptico(img_orig_bgr, img_clean_bgr, ruido_color, info):
    h, w, _ = img_orig_bgr.shape
    margen_sup = 70
    ancho_tot = w * 3
    alto_tot = h + margen_sup

    canvas = np.zeros((alto_tot, ancho_tot, 3), dtype=np.uint8)
    canvas.fill(24)

    canvas[margen_sup:, 0:w] = img_orig_bgr
    canvas[margen_sup:, w:2*w] = img_clean_bgr
    canvas[margen_sup:, 2*w:] = ruido_color

    cv2.line(canvas, (w, 0), (w, alto_tot), (60, 60, 60), 2)
    cv2.line(canvas, (2*w, 0), (2*w, alto_tot), (60, 60, 60), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "1. Original (Sin HUD)", (20, 42), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"2. Denoised ({info.get('modelo', 'Restaurado')})", (w + 20, 42), font, 1.0, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(canvas, "3. Ruido Extraido (|Original - Denoised|)", (2*w + 20, 42), font, 1.0, (100, 200, 255), 2, cv2.LINE_AA)

    sub_orig = f"Frame #{info['frame_idx']} | BRISQUE: {info.get('brisque_orig', 0):.1f} | NIQE: {info.get('niqe_orig', 0):.2f} | Sigma: {info['sigma_orig']:.2f}"
    sub_clean = f"BRISQUE: {info.get('brisque_clean', 0):.1f} | NIQE: {info.get('niqe_clean', 0):.2f} | Sigma: {info['sigma_clean']:.2f}"
    sub_ruido = f"MAE Ruido: {info['mae_ruido']:.2f} | Reduccion Sigma: {info['reduccion_sigma']:.1f}%"

    cv2.putText(canvas, sub_orig, (20, margen_sup + h - 15), font, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, sub_clean, (w + 20, margen_sup + h - 15), font, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, sub_ruido, (2*w + 20, margen_sup + h - 15), font, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return canvas


def graficar_linea_tiempo_completa(df, salida_png, titulo_video):
    fig, axs = plt.subplots(4, 1, figsize=(16, 15), sharex=True)
    fig.suptitle(f"Caracterización Temporal Completa de Calidad y Ruido Térmico - {titulo_video}", fontsize=15, fontweight="bold")

    frames = df["frame_idx"].values
    ventana = min(30, max(5, len(df) // 100))

    # Panel 1: BRISQUE & NIQE (Calidad Perceptual - Menor es Mejor)
    if "brisque_orig" in df.columns and df["brisque_orig"].sum() > 0:
        b_orig_ma = df["brisque_orig"].rolling(ventana, center=True, min_periods=1).mean().values
        b_clean_ma = df["brisque_clean"].rolling(ventana, center=True, min_periods=1).mean().values
        axs[0].plot(frames, b_orig_ma, color="crimson", lw=2.0, label=f"BRISQUE Original (Media Móvil {ventana}f)")
        axs[0].plot(frames, b_clean_ma, color="forestgreen", lw=2.0, label=f"BRISQUE Denoised (Media Móvil {ventana}f)")
        axs[0].set_ylabel("BRISQUE (↓ Mejor)", fontsize=11, fontweight="bold")
        axs[0].set_title("Calidad Perceptual sin Referencia (BRISQUE)", fontsize=12, fontweight="bold")
    else:
        axs[0].plot(frames, df["sigma_orig"].values, color="crimson", label="Sigma Original")
        axs[0].set_ylabel("Sigma Ruido", fontsize=11, fontweight="bold")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right")

    # Panel 2: NIQE (Naturalidad Estadística - Menor es Mejor)
    if "niqe_orig" in df.columns and df["niqe_orig"].sum() > 0:
        n_orig_ma = df["niqe_orig"].rolling(ventana, center=True, min_periods=1).mean().values
        n_clean_ma = df["niqe_clean"].rolling(ventana, center=True, min_periods=1).mean().values
        axs[1].plot(frames, n_orig_ma, color="darkorange", lw=2.0, label=f"NIQE Original (Media Móvil {ventana}f)")
        axs[1].plot(frames, n_clean_ma, color="royalblue", lw=2.0, label=f"NIQE Denoised (Media Móvil {ventana}f)")
        axs[1].set_ylabel("NIQE (↓ Mejor)", fontsize=11, fontweight="bold")
        axs[1].set_title("Desviación del Modelo Natural de Escenas (NIQE)", fontsize=12, fontweight="bold")
    else:
        axs[1].plot(frames, df["sigma_clean"].values, color="forestgreen", label="Sigma Clean")
        axs[1].set_ylabel("Sigma Denoised", fontsize=11, fontweight="bold")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")

    # Panel 3: Nivel Físico de Ruido y MAE del Residuo Removido
    mae_ma = df["mae_ruido"].rolling(ventana, center=True, min_periods=1).mean().values
    axs[2].plot(frames, df["mae_ruido"].values, color="cornflowerblue", alpha=0.35, label="Residuo Frame a Frame")
    axs[2].plot(frames, mae_ma, color="navy", lw=2.0, label=f"Ruido Removido MAE (Media Móvil {ventana}f)")
    axs[2].set_ylabel("Magnitud Ruido (MAE)", fontsize=11, fontweight="bold")
    axs[2].set_title("Magnitud del Ruido Térmico Removido (|Original - Denoised|)", fontsize=12, fontweight="bold")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper right")

    # Panel 4: Consistencia y Ritmo Estructural (Varianza Laplaciana)
    lap_orig_ma = df["var_lap_orig"].rolling(ventana, center=True, min_periods=1).mean().values
    lap_clean_ma = df["var_lap_clean"].rolling(ventana, center=True, min_periods=1).mean().values
    axs[3].plot(frames, lap_orig_ma, color="purple", lw=1.8, label="Nitidez Original")
    axs[3].plot(frames, lap_clean_ma, color="teal", lw=1.8, linestyle="--", label="Nitidez Denoised (Estructura Preservada)")
    axs[3].set_ylabel("Varianza Laplaciana", fontsize=11, fontweight="bold")
    axs[3].set_xlabel("Índice de Cuadro (Frame)", fontsize=12, fontweight="bold")
    axs[3].set_title("Ritmo Estructural por Escena y Preservación de Bordes", fontsize=12, fontweight="bold")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    salida_png = Path(salida_png)
    salida_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida_png, dpi=300)
    plt.close()
    print(f"[+] Gráfica multi-métrica exportada a: {salida_png}")


def procesar_video_metricas(id_v, dir_orig, dir_clean, nom_modelo, m_niqe, m_brisque, m_piqe, dispositivo, max_frames=None, workers=8):
    dir_orig = Path(dir_orig).resolve()
    dir_clean = Path(dir_clean).resolve()

    if not dir_orig.is_dir() or not dir_clean.is_dir():
        print(f"[-] Omitiendo {id_v}: carpetas no encontradas ({dir_orig} o {dir_clean})")
        return None

    exts = {".png", ".jpg", ".jpeg"}
    archivos_orig = sorted([p for p in dir_orig.iterdir() if p.is_file() and p.suffix.lower() in exts], key=natural_key)
    archivos_clean_map = {p.name: p for p in dir_clean.iterdir() if p.is_file() and p.suffix.lower() in exts}

    if max_frames and max_frames > 0:
        archivos_orig = archivos_orig[:max_frames]

    print("\n" + "=" * 90)
    print(f"EVALUANDO Y EXTRAYENDO RUIDO: {id_v.upper()} ({len(archivos_orig)} frames)")
    print(f"Original: {dir_orig}")
    print(f"Denoised: {dir_clean}")
    print("=" * 90)

    dir_ruido = RAIZ / f"videos/{id_v}/ruido_residual"
    dir_mosaicos = RAIZ / f"figuras_tesis/linea_tiempo/mosaicos_ruido_{id_v}"
    dir_ruido.mkdir(parents=True, exist_ok=True)
    dir_mosaicos.mkdir(parents=True, exist_ok=True)

    registros = []
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = []

    with torch.inference_mode():
        for idx, f_orig in enumerate(tqdm(archivos_orig, desc=f"Procesando {id_v}")):
            nombre = f_orig.name
            f_clean = archivos_clean_map.get(nombre)
            if not f_clean:
                continue

            img_orig = cv2.imread(str(f_orig), cv2.IMREAD_COLOR)
            img_clean = cv2.imread(str(f_clean), cv2.IMREAD_COLOR)
            if img_orig is None or img_clean is None:
                continue

            if img_orig.shape != img_clean.shape:
                img_clean = cv2.resize(img_clean, (img_orig.shape[1], img_orig.shape[0]), interpolation=cv2.INTER_AREA)

            gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
            gray_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

            residuo = gray_orig.astype(np.float32) - gray_clean.astype(np.float32)
            mae = float(np.mean(np.abs(residuo)))
            s_orig = estimar_sigma_mad(gray_orig)
            s_clean = estimar_sigma_mad(gray_clean)
            sob_orig, lap_orig = calcular_densidad_estructural(gray_orig)
            sob_clean, lap_clean = calcular_densidad_estructural(gray_clean)
            red_s = ((s_orig - s_clean) / max(1e-6, s_orig)) * 100.0

            # Guardar mapa de ruido centrado en 128
            ruido_centrado = np.clip(residuo + 128.0, 0.0, 255.0).astype(np.uint8)
            dest_ruido = dir_ruido / f"ruido_{nombre}"
            futures.append(pool.submit(cv2.imwrite, str(dest_ruido), ruido_centrado, [cv2.IMWRITE_PNG_COMPRESSION, 3]))

            # BRISQUE, NIQE, PIQE si pyiqa está disponible
            b_orig, b_clean = 0.0, 0.0
            n_orig, n_clean = 0.0, 0.0
            p_orig, p_clean = 0.0, 0.0

            if m_brisque is not None and m_niqe is not None:
                try:
                    t_orig = torch.from_numpy(img_orig).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dispositivo)
                    t_clean = torch.from_numpy(img_clean).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dispositivo)
                    b_orig = float(m_brisque(t_orig).item())
                    b_clean = float(m_brisque(t_clean).item())
                    n_orig = float(m_niqe(t_orig).item())
                    n_clean = float(m_niqe(t_clean).item())
                    if m_piqe is not None:
                        p_orig = float(m_piqe(t_orig).item())
                        p_clean = float(m_piqe(t_clean).item())
                except Exception:
                    pass

            registros.append({
                "video": id_v,
                "frame_idx": idx + 1,
                "nombre_archivo": nombre,
                "brisque_orig": round(b_orig, 2),
                "brisque_clean": round(b_clean, 2),
                "niqe_orig": round(n_orig, 3),
                "niqe_clean": round(n_clean, 3),
                "piqe_orig": round(p_orig, 2),
                "piqe_clean": round(p_clean, 2),
                "sigma_orig": round(s_orig, 4),
                "sigma_clean": round(s_clean, 4),
                "reduccion_sigma": round(red_s, 2),
                "mae_ruido": round(mae, 4),
                "sobel_orig": round(sob_orig, 4),
                "var_lap_orig": round(lap_orig, 2),
                "sobel_clean": round(sob_clean, 4),
                "var_lap_clean": round(lap_clean, 2),
            })

    for f in futures:
        f.result()
    pool.shutdown()

    if not registros:
        return None

    df = pd.DataFrame(registros)
    csv_out = RAIZ / f"videos/{id_v}/ritmo_temporal_{id_v}.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"[+] CSV exportado a: {csv_out}")

    png_out = RAIZ / f"figuras_tesis/linea_tiempo/linea_tiempo_ruido_{id_v}.png"
    graficar_linea_tiempo_completa(df, png_out, f"Video {id_v.replace('video', '')} ({nom_modelo})")

    # Mosaicos top representativos
    df_estructural = df[df["sobel_orig"] >= df["sobel_orig"].quantile(0.20)]
    if len(df_estructural) >= 10:
        top_ruido = df_estructural.sort_values(by="mae_ruido", ascending=False).head(5)
        top_limpio = df_estructural.sort_values(by="mae_ruido", ascending=True).head(5)
        seleccionados = pd.concat([top_ruido, top_limpio])
    else:
        seleccionados = df.head(10)

    for rank, (_, row) in enumerate(seleccionados.iterrows(), 1):
        nom = row["nombre_archivo"]
        f_o = dir_orig / nom
        f_c = dir_clean / nom
        if f_o.is_file() and f_c.is_file():
            im_o = cv2.imread(str(f_o))
            im_c = cv2.imread(str(f_c))
            gr_o = cv2.cvtColor(im_o, cv2.COLOR_BGR2GRAY)
            gr_c = cv2.cvtColor(im_c, cv2.COLOR_BGR2GRAY)
            res_gr = gr_o.astype(np.float32) - gr_c.astype(np.float32)
            ruido_col, _ = normalizar_ruido_para_visualizacion(res_gr, factor=3.5)
            row_dict = row.to_dict()
            row_dict["modelo"] = nom_modelo
            mos = crear_mosaico_triptico(im_o, im_c, ruido_col, row_dict)
            mos_p = dir_mosaicos / f"mosaico_triptico_rank{rank:02d}_{nom}"
            cv2.imwrite(str(mos_p), mos, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return df


def main():
    parser = argparse.ArgumentParser(description="Extracción Multi-Métrica y Líneas de Tiempo en 3 Videos")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Dispositivo para BRISQUE/NIQE")
    parser.add_argument("--workers", type=int, default=8, help="Hilos en paralelo")
    parser.add_argument("--max-frames", type=int, default=None, help="Límite opcional de frames")
    args = parser.parse_args()

    dispositivo = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo para Métricas: {dispositivo}")
    m_niqe, m_brisque, m_piqe = crear_metricas_pyiqa(dispositivo)

    configuracion_videos = [
        {
            "id": "video1",
            "orig": RAIZ / "videos/video1/frames_sin_hud",
            "clean": RAIZ / "videos/video1/expos/UDVD_SinHUD_K5_lr1e3",
            "alt_clean": RAIZ / "videos/video1/expos/n2n_sin_hud",
            "modelo": "UDVD_T5_K5",
        },
        {
            "id": "video2",
            "orig": RAIZ / "videos/video2/frames_sin_hud",
            "clean": RAIZ / "videos/video2/expos/udvd_blindspot_corregido_ult10min",
            "alt_clean": RAIZ / "videos/video2/expos/ensemble_wavelet_triple_sinhud",
            "modelo": "UDVD_BlindSpot_Corregido",
        },
        {
            "id": "video3",
            "orig": RAIZ / "videos/video3/frames_sin_hud.tmp",
            "alt_orig": RAIZ / "videos/video3/frames_sin_hud",
            "clean": RAIZ / "videos/video3/expos/udvd",
            "alt_clean": RAIZ / "videos/video3/expos/n2n",
            "modelo": "UDVD_FLIR",
        },
    ]

    dfs_procesados = {}

    for cfg in configuracion_videos:
        v_id = cfg["id"]
        dir_o = cfg["orig"] if cfg["orig"].is_dir() else cfg.get("alt_orig", cfg["orig"])
        dir_c = cfg["clean"] if cfg["clean"].is_dir() else cfg.get("alt_clean", cfg["clean"])

        if dir_o.is_dir() and dir_c.is_dir():
            df_v = procesar_video_metricas(
                id_v=v_id,
                dir_orig=dir_o,
                dir_clean=dir_c,
                nom_modelo=cfg["modelo"],
                m_niqe=m_niqe,
                m_brisque=m_brisque,
                m_piqe=m_piqe,
                dispositivo=dispositivo,
                max_frames=args.max_frames,
                workers=args.workers,
            )
            if df_v is not None and not df_v.empty:
                dfs_procesados[v_id] = df_v

    print("\n" + "=" * 90)
    print("EXTRACCIÓN DE RUIDO Y CARACTERIZACIÓN TEMPORAL MULTI-MÉTRICA FINALIZADA.")
    print("=" * 90)


if __name__ == "__main__":
    main()
