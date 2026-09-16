#!/usr/bin/env python3
"""
Generador Ultrarrápido de Mosaico Top 10 (Side-by-Side Horizontal).
Diseñado para ejecutarse directamente en la terminal interactiva en ~30 segundos:
- Encuentra los 10 frames con mayor reducción de ruido térmico.
- Genera UNA SOLA imagen consolidada de alta resolución (10 filas x 3 columnas):
  [ Columna 1: Original Crudo ] | [ Columna 2: Sin HUD (ProPainter) ] | [ Columna 3: UDVD Destriping T=7 ]
- Rótulos con métricas en cada panel y zoom de detalle.
"""

import argparse
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def estimar_sigma_mad(gray):
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    return float(mad / 0.6745)


def banner_texto(img, texto, color_fondo=(20, 20, 20), color_texto=(255, 255, 255)):
    h, w, _ = img.shape
    bh = 38
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bh), color_fondo, -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)
    cv2.line(img, (0, bh), (w, bh), (0, 165, 255), 2)
    cv2.putText(img, texto, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_texto, 2, cv2.LINE_AA)
    return img


def procesar_un_frame(args):
    idx, ruta_orig, ruta_sinhud, ruta_mod = args
    g_orig = cv2.imread(str(ruta_orig), cv2.IMREAD_GRAYSCALE)
    g_sinhud = cv2.imread(str(ruta_sinhud), cv2.IMREAD_GRAYSCALE)
    g_mod = cv2.imread(str(ruta_mod), cv2.IMREAD_GRAYSCALE)

    if g_orig is None or g_sinhud is None or g_mod is None:
        return None

    s_orig = estimar_sigma_mad(g_orig)
    s_sinhud = estimar_sigma_mad(g_sinhud)
    s_mod = estimar_sigma_mad(g_mod)

    caida_sigma = s_sinhud - s_mod
    mejora_pct = (caida_sigma / max(1e-4, s_sinhud)) * 100.0

    return {
        "idx": idx,
        "nombre": ruta_orig.stem,
        "ruta_orig": ruta_orig,
        "ruta_sinhud": ruta_sinhud,
        "ruta_mod": ruta_mod,
        "s_orig": s_orig,
        "s_sinhud": s_sinhud,
        "s_mod": s_mod,
        "mejora_pct": mejora_pct,
    }


def main():
    p = argparse.ArgumentParser(description="Mosaico Top 10 Side-by-Side")
    p.add_argument("--video", default="video2")
    p.add_argument("--modelo", default="videos/video2/expos/udvd_destriping_t7_ult10min_sin_hud")
    p.add_argument("--ultimos-frames", type=int, default=18000)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--salida", default="figuras_tesis/mosaico_top10_side_by_side.png")
    args = p.parse_args()

    ruta_base = RAIZ / "videos" / args.video
    dir_orig = ruta_base / "frames_originales"
    dir_sinhud = ruta_base / "frames_sin_hud"
    dir_mod = RAIZ / args.modelo if not Path(args.modelo).is_absolute() else Path(args.modelo)

    exts = {".png", ".jpg", ".jpeg"}
    frames_orig = sorted([p for p in dir_orig.iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]
    frames_sinhud = sorted([p for p in dir_sinhud.iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]
    frames_mod = sorted([p for p in dir_mod.iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]

    n_frames = min(len(frames_orig), len(frames_sinhud), len(frames_mod))
    print(f"Buscando los Top {args.top_n} casos con mayor reduccion de ruido en {n_frames} frames...")

    tareas = [
        (i, frames_orig[i], frames_sinhud[i], frames_mod[i])
        for i in range(n_frames)
    ]

    with ThreadPoolExecutor(max_workers=16) as pool:
        resultados = list(pool.map(procesar_un_frame, tareas))

    resultados = [r for r in resultados if r is not None]
    top_frames = sorted(resultados, key=lambda x: x["mejora_pct"], reverse=True)[:args.top_n]

    print(f"Construyendo mosaico comparativo consolidado ({args.top_n} filas x 3 columnas)...")

    filas_mosaico = []
    
    for rank, item in enumerate(top_frames, start=1):
        img1 = cv2.imread(str(item["ruta_orig"]))
        img2 = cv2.imread(str(item["ruta_sinhud"]))
        img3 = cv2.imread(str(item["ruta_mod"]))

        # Banners
        img1 = banner_texto(img1, f"#{rank:02d} | Original ({item['nombre']}) | Sigma={item['s_orig']:.2f}")
        img2 = banner_texto(img2, f"#{rank:02d} | Sin HUD (ProPainter) | Sigma={item['s_sinhud']:.2f}")
        img3 = banner_texto(img3, f"#{rank:02d} | UDVD Destriping T=7 | Sigma={item['s_mod']:.2f} | Mejora: +{item['mejora_pct']:.1f}%", color_texto=(0, 255, 120))

        # Redimensionar cada panel para mantener un peso ligero en el poster final (ancho 640px por columna)
        h, w, _ = img1.shape
        w_nuevo = 640
        h_nuevo = int(h * (w_nuevo / w))

        p1 = cv2.resize(img1, (w_nuevo, h_nuevo), interpolation=cv2.INTER_AREA)
        p2 = cv2.resize(img2, (w_nuevo, h_nuevo), interpolation=cv2.INTER_AREA)
        p3 = cv2.resize(img3, (w_nuevo, h_nuevo), interpolation=cv2.INTER_AREA)

        fila = np.hstack([p1, p2, p3])
        filas_mosaico.append(fila)

    mosaico_final = np.vstack(filas_mosaico)

    # Encabezado principal del póster
    encabezado = np.zeros((70, mosaico_final.shape[1], 3), dtype=np.uint8) + 25
    cv2.putText(
        encabezado,
        f"TOP {args.top_n} CASOS DE MAYOR REDUCCION DE RUIDO TERMICO FLIR - COMPARATIVA DIRECTA (VIDEO 2)",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    mosaico_completo = np.vstack([encabezado, mosaico_final])

    ruta_out = RAIZ / args.salida
    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_out), mosaico_completo, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    print(f"\n[EXITO] Mosaico Top {args.top_n} guardado en: {ruta_out}")


if __name__ == "__main__":
    main()
