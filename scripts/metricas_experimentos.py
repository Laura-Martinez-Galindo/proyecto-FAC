#!/usr/bin/env python3
"""
==============================================================================
MÓDULO CENTRALIZADO DE MÉTRICAS CIENTÍFICAS — PROTOCOLO EXPERIMENTAL FAC
==============================================================================
Calcula las 6 métricas dicientes acordadas para el Protocolo Experimental
Definitivo de Denoising en Imágenes FLIR de Vigilancia Aérea (Minería Ilegal).

Métricas implementadas:
  GRUPO 1 — Supresión Física de Ruido:
    [1] sigma_mad_reduccion_pct  : Reducción de σ Wavelet MAD (%)
    [2] kurtosis_residuo         : Kurtosis del residuo extraído r = y - ŷ
    [3] fft_flatness_residuo     : Blancura espectral del residuo (Spectral Flatness)

  GRUPO 2 — Auditoría "No-Fuga de Señal":
    [4] sli                      : Structural Leakage Index Corr(|∇y|, |r|) → 0 es mejor
    [5] epr                      : Edge & Peak Preservation Ratio P99(ŷ)/P99(y) → 1 es mejor

  GRUPO 3 — Métricas de Tarea Final:
    [6] snr_objeto_db            : SNR figura-fondo en cajas anotadas (dB, si hay anotaciones)

Métricas clásicas de referencia (no dicientes pero incluidas para compatibilidad):
    [7] sigma_mad_input          : σ MAD de la imagen de entrada (DN)
    [8] sigma_mad_output         : σ MAD de la imagen de salida (DN)
    [9] niqe                     : NIQE (si pyiqa disponible)
    [10] brisque                 : BRISQUE (si pyiqa disponible)
==============================================================================
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import gmean


def estimar_sigma_mad(gray: np.ndarray) -> float:
    """Estimación robusta de σ de ruido gaussiano usando coeficientes Haar HH (MAD)."""
    img = gray.astype(np.float32)
    hh = (img[0::2, 0::2] - img[0::2, 1::2] - img[1::2, 0::2] + img[1::2, 1::2]) * 0.5
    med = float(np.median(np.abs(hh)))
    return med / 0.67448975


def calcular_kurtosis_residuo(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Kurtosis del mapa residual r = y - ŷ.
    
    Un residuo con kurtosis alta indica colas pesadas → el modelo extrajo bien
    el ruido impulsivo / sal-pimienta / grumos DDE+HAZE.
    Si kurtosis ≈ 3 (gaussiana), el residuo es ruido gaussiano puro → excelente.
    Si kurtosis > 8, hay estructuras reales filtradas (bordes de casas, ríos).
    """
    r = y.astype(np.float32) - yhat.astype(np.float32)
    return float(scipy_kurtosis(r.ravel(), fisher=False))  # kurtosis no-centrada


def calcular_fft_flatness(residuo_gray: np.ndarray) -> float:
    """
    Blancura Espectral del residuo (Spectral Flatness / Wiener Entropy).
    
    Un ruido blanco puro tiene flatness = 1.0.
    Si el modelo dejó estructuras en el residuo (bordes de carreteras, ríos),
    el espectro tendrá picos y la flatness bajará hacia 0.
    Rango: [0, 1]. Mejor → 1.0 (ruido blanco puro en el residuo).
    """
    r = residuo_gray.astype(np.float32)
    fft_mag = np.abs(np.fft.fft2(r))
    fft_mag_flat = fft_mag.ravel()
    fft_mag_flat = fft_mag_flat[fft_mag_flat > 0]
    geo_mean = float(gmean(fft_mag_flat))
    arith_mean = float(np.mean(fft_mag_flat))
    if arith_mean < 1e-10:
        return 0.0
    return float(np.clip(geo_mean / arith_mean, 0.0, 1.0))


def calcular_sli(y_gray: np.ndarray, yhat_gray: np.ndarray) -> float:
    """
    Structural Leakage Index (SLI) = Corr(|∇y|, |r|).
    
    Mide la correlación entre la magnitud del gradiente de la imagen ruidosa
    y la magnitud del residuo extraído r = y - ŷ.
    
    SLI ≈ 0.0 → El residuo es ruido puro, INDEPENDIENTE de los bordes. ÓPTIMO.
    SLI > 0.25 → El modelo está filtrando bordes reales al residuo (casas, ríos).
               → DEFECTO: el modelo borrosea objetos.
    """
    y = y_gray.astype(np.float32)
    yhat = yhat_gray.astype(np.float32)
    r_abs = np.abs(y - yhat)

    gx = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2).astype(np.float32)

    # Correlación de Pearson
    r_flat = r_abs.ravel()
    g_flat = grad_mag.ravel()
    if r_flat.std() < 1e-10 or g_flat.std() < 1e-10:
        return 0.0
    corr = float(np.corrcoef(r_flat, g_flat)[0, 1])
    return float(np.clip(corr, 0.0, 1.0))


def calcular_epr(y_gray: np.ndarray, yhat_gray: np.ndarray) -> float:
    """
    Edge & Peak Preservation Ratio (EPR) = P99(ŷ) / P99(y).
    
    Mide si el denoiser preserva los picos calientes (techos blancos, motores,
    campamentos de minería). Un valor < 0.90 indica que el modelo apagó objetos
    calientes reales.
    
    EPR ≈ 1.0 → Conservación perfecta de picos. ÓPTIMO.
    EPR < 0.85 → El modelo borró la firma térmica de los objetos. DEFECTO.
    """
    p99_y = float(np.percentile(y_gray, 99))
    p99_yhat = float(np.percentile(yhat_gray, 99))
    if p99_y < 1.0:
        return 1.0
    return float(np.clip(p99_yhat / p99_y, 0.0, 2.0))


def calcular_snr_objeto_db(yhat_gray: np.ndarray, bboxes_xywh: list) -> float:
    """
    SNR Figura-Fondo en Decibelios (FBD) de SAVeD 2026.
    
    SNR_objeto(dB) = 10 * log10( |μ_caja - μ_fondo|² / σ²_fondo )
    
    Args:
        yhat_gray: Imagen denoised en escala de grises (H, W).
        bboxes_xywh: Lista de (x, y, w, h) en píxeles de las anotaciones de objetos.
    
    Returns:
        SNR promedio en dB de todos los objetos anotados.
        None si no hay anotaciones disponibles.
    """
    if not bboxes_xywh:
        return None

    h, w = yhat_gray.shape
    snrs = []
    for (bx, by, bw, bh) in bboxes_xywh:
        bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)
        # Recorte de la caja
        roi = yhat_gray[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)].astype(np.float32)
        if roi.size < 4:
            continue

        # Contexto local de fondo (corona 20px alrededor de la caja)
        margin = 20
        y0_f, y1_f = max(0, by - margin), min(h, by + bh + margin)
        x0_f, x1_f = max(0, bx - margin), min(w, bx + bw + margin)
        fondo_mask = np.ones((y1_f - y0_f, x1_f - x0_f), dtype=bool)
        # Excluir la caja de la corona de fondo
        iy0 = by - y0_f
        ix0 = bx - x0_f
        fondo_mask[max(0, iy0):iy0 + bh, max(0, ix0):ix0 + bw] = False
        fondo_full = yhat_gray[y0_f:y1_f, x0_f:x1_f].astype(np.float32)
        fondo = fondo_full[fondo_mask]
        if fondo.size < 4:
            continue

        mu_obj = float(np.mean(roi))
        mu_fnd = float(np.mean(fondo))
        sigma2_fnd = float(np.var(fondo))
        if sigma2_fnd < 1e-6:
            continue
        snr = 10 * np.log10((mu_obj - mu_fnd)**2 / sigma2_fnd)
        snrs.append(snr)

    return float(np.mean(snrs)) if snrs else None


def calcular_metricas_completas(
    y_bgr: np.ndarray,
    yhat_bgr: np.ndarray,
    bboxes_xywh: list = None,
    calcular_niqe: bool = False,
    device: str = "cpu"
) -> dict:
    """
    Calcula el panel completo de 6 métricas dicientes.
    
    Args:
        y_bgr    : Frame ruidoso de entrada (BGR uint8).
        yhat_bgr : Frame denoised de salida (BGR uint8).
        bboxes_xywh: Lista de (x, y, w, h) de objetos anotados (opcional).
        calcular_niqe: Si True, intenta calcular NIQE y BRISQUE con pyiqa.
        device: "cuda:0" o "cpu".
    
    Returns:
        dict con todas las métricas.
    """
    y_gray = cv2.cvtColor(y_bgr, cv2.COLOR_BGR2GRAY)
    yhat_gray = cv2.cvtColor(yhat_bgr, cv2.COLOR_BGR2GRAY)
    residuo_gray = y_gray.astype(np.float32) - yhat_gray.astype(np.float32)

    sigma_in = estimar_sigma_mad(y_gray)
    sigma_out = estimar_sigma_mad(yhat_gray)
    reduccion_pct = (1.0 - sigma_out / max(sigma_in, 1e-3)) * 100.0

    sli = calcular_sli(y_gray, yhat_gray)
    epr = calcular_epr(y_gray, yhat_gray)
    kurt = calcular_kurtosis_residuo(y_gray, yhat_gray)
    flatness = calcular_fft_flatness((residuo_gray - residuo_gray.mean()).astype(np.float32))
    snr = calcular_snr_objeto_db(yhat_gray, bboxes_xywh or [])

    metricas = {
        # Grupo 1: Supresión física
        "sigma_mad_input":         round(sigma_in, 4),
        "sigma_mad_output":        round(sigma_out, 4),
        "sigma_mad_reduccion_pct": round(reduccion_pct, 2),
        "kurtosis_residuo":        round(kurt, 4),
        "fft_flatness_residuo":    round(flatness, 4),
        # Grupo 2: Auditoría no-fuga
        "sli":                     round(sli, 4),
        "epr":                     round(epr, 4),
        # Grupo 3: Tarea final
        "snr_objeto_db":           round(snr, 2) if snr is not None else None,
    }

    if calcular_niqe:
        try:
            import torch
            import pyiqa
            dev = torch.device(device)
            metric_niqe = pyiqa.create_metric("niqe", device=dev)
            metric_brisque = pyiqa.create_metric("brisque", device=dev)
            t = torch.from_numpy(yhat_bgr[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(dev)
            metricas["niqe"] = round(float(metric_niqe(t).item()), 4)
            metricas["brisque"] = round(float(metric_brisque(t).item()), 4)
        except Exception:
            metricas["niqe"] = None
            metricas["brisque"] = None

    return metricas


def guardar_residuo_visual(y_bgr: np.ndarray, yhat_bgr: np.ndarray, out_path: Path, escala: float = 5.0):
    """
    Guarda el mapa de residuo |r| × escala en colormap 'inferno' para auditoría visual.
    Permite detectar fugas estructurales (ríos, carreteras en el residuo).
    """
    y_gray = cv2.cvtColor(y_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    yhat_gray = cv2.cvtColor(yhat_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r_abs = np.abs(y_gray - yhat_gray)
    r_scaled = np.clip(r_abs * escala, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(r_scaled, cv2.COLORMAP_INFERNO)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), heatmap, [cv2.IMWRITE_PNG_COMPRESSION, 3])
