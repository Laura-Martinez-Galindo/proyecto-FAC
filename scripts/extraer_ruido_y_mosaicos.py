#!/usr/bin/env python3
"""
Extracción de Ruido Residual, Caracterización de Ritmo Temporal y Mosaicos Comparativos.
Diseñado para videos infrarrojos FLIR (Proyecto FAC).

Funcionalidades:
1. Calcula y extrae el residuo de ruido físico: Ruido = |Original_SinHUD - Denoised|
2. Guarda los fotogramas de ruido en disco para la etapa de clusterización de escenas de Jorge.
3. Caracteriza el ritmo temporal frame-a-frame (Sigma MAD, MAE de residuo, Nitidez Laplaciana)
   generando una gráfica con curvas crudas y medias móviles de 30 cuadros (1 segundo).
4. Filtra cuadros homogéneos y extrae automáticamente los 5 cuadros con mayor ruido
   y los 5 cuadros con menor ruido que posean contenido estructural significativo (vehículos, vías, ríos, construcciones).
5. Genera 10 mosaicos individuales trípticos de alta resolución:
   [Original Sin HUD] | [Denoised Sin Ruido] | [Ruido Extraído y Amplificado]
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def estimar_sigma_mad(gray):
    """Estima sigma de ruido con el estimador robusto MAD sobre el Laplaciano en float32."""
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    return float(mad / 0.6745)


def calcular_densidad_estructural(gray):
    """Calcula la energía de bordes por Sobel y la varianza del Laplaciano en float32."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    energia_sobel = float(np.mean(mag))
    var_lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return energia_sobel, var_lap


def normalizar_ruido_para_visualizacion(ruido_gray, factor_amplificacion=3.0):
    """
    Normaliza y amplifica el mapa de ruido para inspección visual clara.
    Centrado en 128 (gris neutro) para visualizar desviaciones positivas y negativas,
    o en escala absoluta con colormap térmico.
    """
    ruido_abs = np.abs(ruido_gray.astype(np.float32))
    ruido_amplificado = np.clip(ruido_abs * factor_amplificacion, 0.0, 255.0).astype(np.uint8)
    ruido_color = cv2.applyColorMap(ruido_amplificado, cv2.COLORMAP_INFERNO)
    return ruido_color, ruido_amplificado


def crear_mosaico_triptico(img_orig_bgr, img_clean_bgr, ruido_color, info_dict):
    """
    Crea un mosaico tríptico horizontal con encabezados y métricas:
    [Frame Sin HUD] | [Frame Denoised] | [Ruido Extraído]
    """
    h, w, _ = img_orig_bgr.shape
    margen_superior = 70
    ancho_total = w * 3
    alto_total = h + margen_superior

    canvas = np.zeros((alto_total, ancho_total, 3), dtype=np.uint8)
    canvas.fill(24)  # Fondo gris oscuro elegante

    # Insertar paneles
    canvas[margen_superior:, 0:w] = img_orig_bgr
    canvas[margen_superior:, w:2*w] = img_clean_bgr
    canvas[margen_superior:, 2*w:] = ruido_color

    # Dibujar líneas divisorias
    cv2.line(canvas, (w, 0), (w, alto_total), (60, 60, 60), 2)
    cv2.line(canvas, (2*w, 0), (2*w, alto_total), (60, 60, 60), 2)

    # Títulos y encabezados
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "1. Original (Sin HUD)", (20, 42), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"2. Denoised ({info_dict.get('modelo', 'Restaurado')})", (w + 20, 42), font, 1.0, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(canvas, "3. Ruido Extraido (|Original - Denoised|)", (2*w + 20, 42), font, 1.0, (100, 200, 255), 2, cv2.LINE_AA)

    # Subtítulos con métricas en pie de imagen
    sub_orig = f"Frame #{info_dict['frame_idx']} | Sigma MAD: {info_dict['sigma_orig']:.2f}"
    sub_clean = f"Nitidez: {info_dict['var_lap_clean']:.1f} | Sigma MAD: {info_dict['sigma_clean']:.2f}"
    sub_ruido = f"MAE Ruido: {info_dict['mae_ruido']:.2f} | Reduccion: {info_dict['reduccion_sigma']:.1f}%"

    cv2.putText(canvas, sub_orig, (20, margen_superior + h - 15), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, sub_clean, (w + 20, margen_superior + h - 15), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, sub_ruido, (2*w + 20, margen_superior + h - 15), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return canvas


def graficar_linea_tiempo(df, salida_png, titulo_video="Video 2"):
    """
    Genera un panel de series de tiempo sincronizadas con curvas cuadro-a-cuadro
    y media móvil de 30 frames (1 segundo) para capturar el ritmo estructural.
    """
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Caracterización Temporal de Ruido y Ritmo Estructural - {titulo_video}", fontsize=16, fontweight="bold")

    frames = df["frame_idx"].values
    ventana = min(30, max(5, len(df) // 100))

    # 1. Nivel de Ruido Estocástico (Sigma MAD)
    sigma_orig = df["sigma_orig"].values
    sigma_clean = df["sigma_clean"].values
    sigma_orig_ma = df["sigma_orig"].rolling(ventana, center=True, min_periods=1).mean().values
    sigma_clean_ma = df["sigma_clean"].rolling(ventana, center=True, min_periods=1).mean().values

    axs[0].plot(frames, sigma_orig, color="lightcoral", alpha=0.35, label="Original Cuadro a Cuadro")
    axs[0].plot(frames, sigma_orig_ma, color="crimson", lw=2.0, label=f"Original (Media Móvil {ventana} frames)")
    axs[0].plot(frames, sigma_clean, color="lightgreen", alpha=0.35, label="Denoised Cuadro a Cuadro")
    axs[0].plot(frames, sigma_clean_ma, color="forestgreen", lw=2.0, label=f"Denoised (Media Móvil {ventana} frames)")
    axs[0].set_ylabel("Sigma de Ruido (σ MAD)", fontsize=11, fontweight="bold")
    axs[0].set_title("Evolución Temporal del Ruido Térmico", fontsize=12, fontweight="bold")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right")

    # 2. Magnitud del Ruido Removido (MAE Residuo)
    mae_ruido = df["mae_ruido"].values
    mae_ruido_ma = df["mae_ruido"].rolling(ventana, center=True, min_periods=1).mean().values
    axs[1].plot(frames, mae_ruido, color="cornflowerblue", alpha=0.35, label="Residuo Instantáneo")
    axs[1].plot(frames, mae_ruido_ma, color="darkblue", lw=2.0, label=f"Residuo (Media Móvil {ventana} frames)")
    axs[1].set_ylabel("Magnitud del Residuo (MAE)", fontsize=11, fontweight="bold")
    axs[1].set_title("Cantidad de Ruido Removido por Cuadro (|Original - Denoised|)", fontsize=12, fontweight="bold")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")

    # 3. Preservación Estructural (Varianza del Laplaciano)
    lap_orig_ma = df["var_lap_orig"].rolling(ventana, center=True, min_periods=1).mean().values
    lap_clean_ma = df["var_lap_clean"].rolling(ventana, center=True, min_periods=1).mean().values
    axs[2].plot(frames, lap_orig_ma, color="darkorange", lw=1.8, label="Nitidez Original (Media Móvil)")
    axs[2].plot(frames, lap_clean_ma, color="purple", lw=1.8, linestyle="--", label="Nitidez Denoised (Media Móvil)")
    axs[2].set_ylabel("Varianza Laplaciana (Nitidez)", fontsize=11, fontweight="bold")
    axs[2].set_xlabel("Índice de Cuadro (Frame)", fontsize=12, fontweight="bold")
    axs[2].set_title("Consistencia del Ritmo Estructural y Preservación de Bordes", fontsize=12, fontweight="bold")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper right")

    plt.tight_layout()
    salida_png = Path(salida_png)
    salida_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida_png, dpi=300)
    plt.close()
    print(f"Gráfica temporal guardada exitosamente en: {salida_png}")


def main():
    parser = argparse.ArgumentParser(description="Extracción de Ruido Residual y Mosaicos Estructurales")
    parser.add_argument("--video", default="video2", help="Identificador del video (video1, video2, etc.)")
    parser.add_argument("--carpeta-original", required=True, help="Ruta a frames originales o sin HUD")
    parser.add_argument("--carpeta-denoised", required=True, help="Ruta a frames procesados con Denoising")
    parser.add_argument("--modelo-nombre", default="UDVD_Denoising", help="Nombre descriptivo del modelo de denoising")
    parser.add_argument("--salida-ruido", default=None, help="Directorio donde guardar los frames de ruido para Jorge")
    parser.add_argument("--salida-mosaicos", default="figuras_tesis/mosaicos_ruido", help="Directorio para los 10 mosaicos trípticos")
    parser.add_argument("--salida-grafica", default="figuras_tesis/linea_tiempo_ruido.png", help="Ruta de la figura temporal")
    parser.add_argument("--salida-csv", default="csv_logs/ritmo_temporal.csv", help="Ruta del archivo CSV de métricas cuadro a cuadro")
    parser.add_argument("--max-frames", type=int, default=None, help="Límite de frames a procesar")
    parser.add_argument("--ultimos-frames", type=int, default=None, help="Procesar exclusivamente los últimos N frames")
    args = parser.parse_args()

    dir_orig = Path(args.carpeta_original).resolve()
    dir_clean = Path(args.carpeta_denoised).resolve()

    if not dir_orig.is_dir():
        print(f"ERROR: No existe la carpeta original {dir_orig}")
        return
    if not dir_clean.is_dir():
        print(f"ERROR: No existe la carpeta denoised {dir_clean}")
        return

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    archivos_orig = sorted([p for p in dir_orig.iterdir() if p.is_file() and p.suffix.lower() in exts], key=natural_key)
    archivos_clean_map = {p.name: p for p in dir_clean.iterdir() if p.is_file() and p.suffix.lower() in exts}

    if args.ultimos_frames and args.ultimos_frames > 0:
        archivos_orig = archivos_orig[-args.ultimos_frames:]
    elif args.max_frames and args.max_frames > 0:
        archivos_orig = archivos_orig[:args.max_frames]

    print("=" * 85)
    print(f"EXTRACCIÓN DE RUIDO RESIDUAL Y CARACTERIZACIÓN TEMPORAL ({len(archivos_orig)} frames)")
    print(f"Video: {args.video} | Modelo Denoised: {args.modelo_nombre}")
    print(f"Original: {dir_orig}")
    print(f"Denoised: {dir_clean}")
    print("=" * 85)

    if args.salida_ruido:
        dir_salida_ruido = Path(args.salida_ruido).resolve()
        dir_salida_ruido.mkdir(parents=True, exist_ok=True)
    else:
        dir_salida_ruido = None

    dir_mosaicos = Path(args.salida_mosaicos).resolve()
    dir_mosaicos.mkdir(parents=True, exist_ok=True)

    registros = []

    for idx, f_orig in enumerate(archivos_orig):
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

        # 1. Calcular residuo de ruido
        residuo_float = gray_orig.astype(np.float32) - gray_clean.astype(np.float32)
        mae_ruido = float(np.mean(np.abs(residuo_float)))

        # 2. Métricas de ruido y densidad estructural
        sigma_orig = estimar_sigma_mad(gray_orig)
        sigma_clean = estimar_sigma_mad(gray_clean)
        sobel_orig, var_lap_orig = calcular_densidad_estructural(gray_orig)
        _, var_lap_clean = calcular_densidad_estructural(gray_clean)

        reduccion_sigma = ((sigma_orig - sigma_clean) / max(1e-6, sigma_orig)) * 100.0

        # 3. Guardar imagen de residuo si se solicita
        if dir_salida_ruido is not None:
            # Guardar residuo centrado en 128 niveles de gris (para análisis espectral/clusterización)
            ruido_centrado = np.clip(residuo_float + 128.0, 0.0, 255.0).astype(np.uint8)
            cv2.imwrite(str(dir_salida_ruido / f"ruido_{nombre}"), ruido_centrado, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        item = {
            "frame_idx": idx + 1,
            "nombre_archivo": nombre,
            "sigma_orig": sigma_orig,
            "sigma_clean": sigma_clean,
            "reduccion_sigma": reduccion_sigma,
            "mae_ruido": mae_ruido,
            "sobel_energia": sobel_orig,
            "var_lap_orig": var_lap_orig,
            "var_lap_clean": var_lap_clean,
        }
        registros.append(item)

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(archivos_orig):
            print(f"Progreso: {idx + 1}/{len(archivos_orig)} cuadros analizados...")

    df = pd.DataFrame(registros)
    if df.empty:
        print("ERROR: No se pudieron procesar pares de imágenes coincidentes.")
        return

    # Guardar CSV de serie de tiempo
    salida_csv = Path(args.salida_csv).resolve()
    salida_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida_csv, index=False)
    print(f"Registro CSV cuadro a cuadro guardado en: {salida_csv}")

    # Generar gráfica de línea temporal
    graficar_linea_tiempo(df, args.salida_grafica, titulo_video=f"{args.video.upper()} ({args.modelo_nombre})")

    # 4. SELECCIÓN DE 10 MOSAICOS ESTRUCTURALES (5 Alto Ruido / 5 Bajo Ruido)
    # Filtrar frames planos/homogéneos: exigir que estén por encima del percentil 35 de energía estructural Sobel
    umbral_sobel = df["sobel_energia"].quantile(0.35)
    umbral_lap = df["var_lap_orig"].quantile(0.30)
    df_estructuras = df[(df["sobel_energia"] >= umbral_sobel) & (df["var_lap_orig"] >= umbral_lap)].copy()

    if len(df_estructuras) < 10:
        df_estructuras = df.copy()

    top5_alto_ruido = df_estructuras.sort_values(by="sigma_orig", ascending=False).head(5)
    top5_bajo_ruido = df_estructuras.sort_values(by="sigma_orig", ascending=True).head(5)

    def procesar_mosaico_desde_disco(fila, prefijo_tipo, pos):
        nombre = fila["nombre_archivo"]
        p_orig = dir_orig / nombre
        p_clean = archivos_clean_map[nombre]
        img_orig_bgr = cv2.imread(str(p_orig))
        img_clean_bgr = cv2.imread(str(p_clean))
        
        orig_gray = cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2GRAY)
        clean_gray = cv2.cvtColor(img_clean_bgr, cv2.COLOR_BGR2GRAY)
        residuo = orig_gray.astype(np.float32) - clean_gray.astype(np.float32)
        
        ruido_color, _ = normalizar_ruido_para_visualizacion(residuo, factor_amplificacion=3.5)
        
        info = fila.to_dict()
        info["modelo"] = args.modelo_nombre
        mosaico = crear_mosaico_triptico(img_orig_bgr, img_clean_bgr, ruido_color, info)
        
        ruta_out = dir_mosaicos / f"mosaico_{prefijo_tipo}_pos{pos}_{nombre}"
        cv2.imwrite(str(ruta_out), mosaico, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        print(f"  [{prefijo_tipo.replace('_', ' ').title()} {pos}] Frame #{fila['frame_idx']} (σ={fila['sigma_orig']:.2f}, Sobel={fila['sobel_energia']:.1f}) -> {ruta_out.name}")

    print("\n--- GENERANDO MOSAICOS DE ALTO RUIDO CON ESTRUCTURAS ---")
    for pos, (_, fila) in enumerate(top5_alto_ruido.iterrows(), 1):
        procesar_mosaico_desde_disco(fila, "alto_ruido", pos)

    print("\n--- GENERANDO MOSAICOS DE BAJO RUIDO CON ESTRUCTURAS ---")
    for pos, (_, fila) in enumerate(top5_bajo_ruido.iterrows(), 1):
        procesar_mosaico_desde_disco(fila, "bajo_ruido", pos)

    print("=" * 85)
    print(f"PROCESO COMPLETADO EXITOSAMENTE.")
    print(f"• 10 Mosaicos individuales guardados en: {dir_mosaicos}")
    print(f"• Línea de tiempo gráfica guardada en: {args.salida_grafica}")
    if dir_salida_ruido:
        print(f"• Matrices de ruido para Jorge guardadas en: {dir_salida_ruido}")
    print("=" * 85)


if __name__ == "__main__":
    main()
