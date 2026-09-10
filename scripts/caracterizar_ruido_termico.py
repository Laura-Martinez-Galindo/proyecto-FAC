#!/usr/bin/env python3
"""
Caracterización ultra-detallada de ruido térmico en videos FLIR:
1. Ruido estocástico gaussiano (Estimador Robusto MAD / Ondículas).
2. Ruido de Patrón Fijo (FPN) en columnas del microbolómetro (varianza inter-columnar).
3. Espectro 2D de Densidad de Potencia (PSD Fourier) para detección de periodicidad espacial.
4. Parpadeo temporal térmico (Temporal Flickering).
5. Ajuste estadístico de distribución (Media, Desviación, Asimetría / Skewness, Curtosis).
6. Generación de gráficos analíticos de diagnóstico para la tesis.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent


def natural_key(p):
    import re
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def estimar_sigma_mad(gray):
    """Estima sigma de ruido con el estimador robusto MAD sobre el Laplaciano."""
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    return float(mad / 0.6745)


def caracterizar_fpn_columnas(frames_gray):
    """
    Calcula el FPN de columnas promediando en el tiempo para eliminar el terreno en movimiento
    y aislar el perfil estático de ganancia/offset de los amplificadores del sensor.
    """
    stack = np.stack(frames_gray, axis=0).astype(np.float64)  # (T, H, W)
    
    # Promedio temporal: el terreno se mueve, el FPN se mantiene fijo
    promedio_temporal = np.mean(stack, axis=0)  # (H, W)
    
    # Perfil medio por columna (eliminando variaciones verticales del terreno)
    perfil_columnas = np.mean(promedio_temporal, axis=0)  # (W,)
    perfil_suavizado = cv2.GaussianBlur(perfil_columnas.reshape(1, -1), (1, 31), 0).ravel()
    
    # El residuo de alta frecuencia entre columnas adyacentes representa el FPN
    residuo_fpn = perfil_columnas - perfil_suavizado
    sigma_fpn = float(np.std(residuo_fpn))
    
    return perfil_columnas, residuo_fpn, sigma_fpn


def calcular_psd_2d(frames_gray):
    """Calcula el promedio del espectro de potencia 2D por FFT."""
    psd_acumulado = None
    for gray in frames_gray:
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitud = np.abs(fshift) ** 2
        if psd_acumulado is None:
            psd_acumulado = magnitud
        else:
            psd_acumulado += magnitud
            
    psd_promedio = psd_acumulado / len(frames_gray)
    psd_log = np.log1p(psd_promedio)
    return psd_log


def calcular_parpadeo_temporal(frames_gray):
    """Calcula la varianza de la diferencia temporal cuadro a cuadro."""
    diffs = []
    for t in range(len(frames_gray) - 1):
        d = frames_gray[t+1].astype(np.float64) - frames_gray[t].astype(np.float64)
        diffs.append(np.std(d))
    return float(np.mean(diffs))


def generar_reporte_grafico(perfil_cols, residuo_fpn, psd_log, sigmas_mad, salida_png):
    """Genera panel visual de 4 subgráficas de diagnóstico del sensor térmico."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Caracterización Física de Ruido Térmico FLIR - Diagnóstico de Sensor", fontsize=16, fontweight="bold")
    
    # 1. Perfil FPN de columnas
    w = len(perfil_cols)
    axs[0, 0].plot(residuo_fpn, color="crimson", lw=1.0)
    axs[0, 0].set_title(f"Ruido de Patrón Fijo (FPN) por Columnas (σ = {np.std(residuo_fpn):.3f})")
    axs[0, 0].set_xlabel("Índice de Columna del Sensor")
    axs[0, 0].set_ylabel("Desviación de Intensidad")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].set_xlim(0, w)

    # 2. Espectro 2D Fourier (PSD)
    im = axs[0, 1].imshow(psd_log, cmap="inferno")
    axs[0, 1].set_title("Espectro 2D de Densidad de Potencia (PSD FFT)")
    axs[0, 1].set_xlabel("Frecuencia Espacial Horizontal (fx)")
    axs[0, 1].set_ylabel("Frecuencia Espacial Vertical (fy)")
    plt.colorbar(im, ax=axs[0, 1], label="Log(Magnitud)")

    # 3. Evolución temporal de Sigma MAD
    axs[1, 0].plot(sigmas_mad, color="teal", lw=1.2)
    axs[1, 0].axhline(np.mean(sigmas_mad), color="darkorange", linestyle="--", label=f"Media: {np.mean(sigmas_mad):.3f}")
    axs[1, 0].set_title("Evolución Temporal del Ruido Estocástico (Sigma MAD)")
    axs[1, 0].set_xlabel("Índice de Cuadro Evaluado")
    axs[1, 0].set_ylabel("Nivel de Ruido σ")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Distribución estadística del residuo
    from scipy.stats import kurtosis, skew
    axs[1, 1].hist(residuo_fpn, bins=40, density=True, color="steelblue", edgecolor="black", alpha=0.7)
    axs[1, 1].set_title(f"Distribución del Ruido de Sensor\nSkewness: {skew(residuo_fpn):.2f} | Kurtosis: {kurtosis(residuo_fpn):.2f}")
    axs[1, 1].set_xlabel("Magnitud del Error de Ruido")
    axs[1, 1].set_ylabel("Densidad de Probabilidad")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    salida_png = Path(salida_png)
    salida_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida_png, dpi=300)
    plt.close()
    print(f"Reporte visual de caracterización guardado en: {salida_png}")


def main():
    parser = argparse.ArgumentParser(description="Caracterización avanzada de ruido térmico FLIR")
    parser.add_argument("--carpeta", required=True, help="Carpeta con frames originales o sin HUD")
    parser.add_argument("--max-frames", type=int, default=300, help="Cantidad de frames a analizar")
    parser.add_argument("--salida-grafica", default="figuras_tesis/caracterizacion_ruido_termico.png")
    args = parser.parse_args()

    carpeta = Path(args.carpeta)
    if not carpeta.is_dir():
        carpeta = RAIZ / args.carpeta
    if not carpeta.is_dir():
        print(f"ERROR: No se encontró la carpeta {carpeta}")
        return

    extensiones = {".png", ".jpg", ".jpeg"}
    archivos = sorted([p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in extensiones], key=natural_key)
    archivos = archivos[:args.max_frames]

    if not archivos:
        print(f"No se encontraron imágenes en {carpeta}")
        return

    print("=" * 80)
    print(f"CARACTERIZACIÓN AVANZADA DE RUIDO TÉRMICO FLIR ({len(archivos)} frames)")
    print(f"Carpeta de análisis: {carpeta}")
    print("=" * 80)

    frames_gray = []
    sigmas_mad = []

    for idx, p in enumerate(archivos):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        frames_gray.append(img)
        sigmas_mad.append(estimar_sigma_mad(img))

    h, w = frames_gray[0].shape
    print(f"Resolución de fotogramas: {w}x{h} píxeles")

    # 1. Ruido estocástico
    sigma_mad_media = float(np.mean(sigmas_mad))
    sigma_mad_std = float(np.std(sigmas_mad))
    print(f"1. Ruido Estocástico (MAD Pasa-Altas): σ = {sigma_mad_media:.4f} ± {sigma_mad_std:.4f}")

    # 2. Ruido FPN en columnas
    perfil_cols, residuo_fpn, sigma_fpn = caracterizar_fpn_columnas(frames_gray)
    relacion_fpn = (sigma_fpn / (sigma_mad_media + 1e-6)) * 100
    print(f"2. Ruido de Patrón Fijo (FPN Columnares): σ_FPN = {sigma_fpn:.4f} ({relacion_fpn:.1f}% de la energía de ruido total)")

    # 3. Parpadeo temporal
    flicker = calcular_parpadeo_temporal(frames_gray)
    print(f"3. Inestabilidad / Parpadeo Temporal: Δ_temporal = {flicker:.4f} niveles de gris/frame")

    # 4. Densidad espectral PSD 2D
    psd_log = calcular_psd_2d(frames_gray)

    # 5. Diagnóstico de arquitectura óptima
    print("\n--- DIAGNÓSTICO METODOLÓGICO DERIVADO ---")
    if sigma_fpn > 0.8:
        print("• ALTA PRESENCIA DE FPN: Es estrictamente necesario el uso de StructN2V con máscaras verticales (5px) para evitar franjas.")
    else:
        print("• FPN MODERADO: Los kernels dinámicos y máscaras 2D regulares son suficientes.")

    if flicker > 5.0:
        print("• ALTO PARPADEO TEMPORAL: Requiere estabilización multicuadro (UDVD con KPN espacio-temporal o Fusión Wavelet).")
    else:
        print("• BUENA COHERENCIA TEMPORAL: Los modelos de ventana corta (T=5) operarán en régimen óptimo.")

    print("=" * 80)

    generar_reporte_grafico(perfil_cols, residuo_fpn, psd_log, sigmas_mad, args.salida_grafica)


if __name__ == "__main__":
    main()
