#!/usr/bin/env python3
"""
Extracción Turbo de Ruido y Generación de Mosaicos Estructurales - Video 2:
- Detecta automáticamente la INTERSECCIÓN exacta de frames existentes (ej. los 18.000 frames del experimento).
- Procesa a máxima velocidad en 16 núcleos de CPU sin tiempos muertos.
- Filtra marcos homogéneos/planos mediante gradiente Sobel.
- Genera los 10 Mosaicos Cuádruples 2x2 de alta resolución (Top 5 MAX Ruido y Top 5 MIN Ruido con estructuras).
- Guarda el CSV completo de estadísticas y extrae los mapas de ruido.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
from pathlib import Path
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def estimar_sigma_mad(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    return float(mad / 0.6745)


def calcular_energia_sobel(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.mean(mag)), float(cv2.Laplacian(gray, cv2.CV_32F).var())


def procesar_lote_en_ram(datos_lote):
    lista_archivos, dir_orig_str, dir_sin_hud_str, dir_den_str = datos_lote
    
    dir_orig = Path(dir_orig_str)
    dir_sin = Path(dir_sin_hud_str)
    dir_den = Path(dir_den_str) if dir_den_str else None
    tiene_den = dir_den is not None and dir_den.is_dir()

    resultados = []

    for nom, idx in lista_archivos:
        f_ori = dir_orig / nom
        f_sin = dir_sin / nom
        if not f_ori.is_file() or not f_sin.is_file():
            continue

        im_ori = cv2.imread(str(f_ori))
        im_sin = cv2.imread(str(f_sin))
        if im_ori is None or im_sin is None:
            continue

        h, w = im_sin.shape[:2]
        if im_ori.shape[:2] != (h, w):
            im_ori = cv2.resize(im_ori, (w, h), interpolation=cv2.INTER_AREA)

        im_den = None
        if tiene_den:
            f_den = dir_den / nom
            if f_den.is_file():
                im_den = cv2.imread(str(f_den))
                if im_den is not None and im_den.shape[:2] != (h, w):
                    im_den = cv2.resize(im_den, (w, h), interpolation=cv2.INTER_AREA)

        gr_ori = cv2.cvtColor(im_ori, cv2.COLOR_BGR2GRAY)
        gr_sin = cv2.cvtColor(im_sin, cv2.COLOR_BGR2GRAY)

        res_hud = cv2.absdiff(gr_ori, gr_sin)
        mae_hud = float(np.mean(res_hud))

        mae_termico = 0.0
        s_den = 0.0
        var_lap_den = 0.0

        if im_den is not None:
            gr_den = cv2.cvtColor(im_den, cv2.COLOR_BGR2GRAY)
            res_term_float = gr_sin.astype(np.float32) - gr_den.astype(np.float32)
            mae_termico = float(np.mean(np.abs(res_term_float)))
            s_den = estimar_sigma_mad(gr_den)
            _, var_lap_den = calcular_energia_sobel(gr_den)

        s_ori = estimar_sigma_mad(gr_ori)
        s_sin = estimar_sigma_mad(gr_sin)
        sobel_sin, var_lap_sin = calcular_energia_sobel(gr_sin)

        resultados.append({
            "frame_idx": idx,
            "nombre_archivo": nom,
            "mae_hud": round(mae_hud, 4),
            "mae_termico": round(mae_termico, 4),
            "sobel_energia": round(sobel_sin, 3),
            "var_lap_sin_hud": round(var_lap_sin, 2),
            "var_lap_denoised": round(var_lap_den, 2),
            "sigma_orig": round(s_ori, 4),
            "sigma_sin_hud": round(s_sin, 4),
            "sigma_denoised": round(s_den, 4),
        })

    return resultados


def renderizar_mosaico_1x4(img_ori, img_sin, img_den, res_term, info, salida_path):
    h, w = img_sin.shape[:2]
    if img_ori.shape[:2] != (h, w):
        img_ori = cv2.resize(img_ori, (w, h))
    if img_den is None:
        img_den = img_sin.copy()
    elif img_den.shape[:2] != (h, w):
        img_den = cv2.resize(img_den, (w, h))

    # Residuo térmico de alta frecuencia (eliminando iluminación de fondo / DC offset)
    blur_offset = cv2.GaussianBlur(res_term, (61, 61), 20.0)
    ruido_alta_frec = res_term - blur_offset
    ruido_mag = np.abs(ruido_alta_frec)

    # Escala estándar absoluta de 0 a 25 Niveles Digitales (DN) para comparación uniforme
    v_min, v_max = 0.0, 25.0

    plt.close('all')
    fig, axs = plt.subplots(1, 4, figsize=(26, 6.2), gridspec_kw={'width_ratios': [1, 1, 1, 1.15]})

    rgb_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    rgb_sin = cv2.cvtColor(img_sin, cv2.COLOR_BGR2RGB)
    rgb_den = cv2.cvtColor(img_den, cv2.COLOR_BGR2RGB)

    # Panel 1: Original
    axs[0].imshow(rgb_ori)
    axs[0].set_title("(a) Original (Con HUD)", fontsize=13, fontweight="bold", pad=10, color="#2c3e50")
    axs[0].axis("off")

    # Panel 2: Sin HUD
    axs[1].imshow(rgb_sin)
    axs[1].set_title("(b) Sin HUD (Inpainting)", fontsize=13, fontweight="bold", pad=10, color="#2c3e50")
    axs[1].axis("off")

    # Panel 3: Restaurado UDVD
    axs[2].imshow(rgb_den)
    axs[2].set_title("(c) Restaurado (UDVD)", fontsize=13, fontweight="bold", pad=10, color="#2c3e50")
    axs[2].axis("off")

    # Panel 4: Ruido Térmico Extraído
    im_ruido = axs[3].imshow(ruido_mag, cmap="inferno", vmin=v_min, vmax=v_max)
    axs[3].set_title("(d) Ruido Térmico Extraído", fontsize=13, fontweight="bold", pad=10, color="#2c3e50")
    axs[3].axis("off")

    # Colorbar estándar y limpia
    cbar = fig.colorbar(im_ruido, ax=axs[3], fraction=0.046, pad=0.03)
    cbar.set_label("Amplitud de Ruido (|DN|)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(pad=1.0)
    Path(salida_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Extracción Turbo de Ruido y Mosaicos Video 2")
    parser.add_argument("--carpeta-original", default="videos/video2/frames_originales", help="Ruta originales")
    parser.add_argument("--carpeta-sin-hud", default="videos/video2/frames_sin_hud", help="Ruta sin HUD")
    parser.add_argument("--carpeta-denoised", default=None, help="Ruta UDVD denoised")
    parser.add_argument("--cores", type=int, default=16, help="Núcleos de CPU")
    args = parser.parse_args()

    dir_orig = (RAIZ / args.carpeta_original).resolve()
    dir_sin = (RAIZ / args.carpeta_sin_hud).resolve()

    if not args.carpeta_denoised:
        p_u = RAIZ / "videos/video2/expos/udvd_blindspot_corregido_ult10min"
        if not p_u.is_dir():
            p_u = RAIZ / "videos/video2/expos/udvd"
        if not p_u.is_dir():
            cands = list(RAIZ.glob("videos/video2/expos/*udvd*"))
            p_u = cands[0] if cands else None
        dir_den = p_u.resolve() if p_u else (RAIZ / "videos/video2/expos/udvd_blindspot_corregido_ult10min")
    else:
        dir_den = (RAIZ / args.carpeta_denoised).resolve()

    print("=" * 85)
    print("      EXTRACCIÓN TURBO DE RUIDO Y MOSAICOS (INTERSECCIÓN EXACTA) - VIDEO 2")
    print(f"Original:           {dir_orig}")
    print(f"Sin HUD:            {dir_sin}")
    print(f"Denoised:           {dir_den}")
    print(f"Paralelismo:        {args.cores} Núcleos CPU")
    print("=" * 85)

    dir_mosaicos = RAIZ / "figuras_tesis/mosaicos_ruido_video2"
    dir_mosaicos.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    
    # Si la carpeta denoised tiene un subconjunto (ej. ultimos 10 min = 18k frames), iterar sobre ella para máxima velocidad
    if dir_den.is_dir():
        archivos_eval_nombres = set(p.name for p in dir_den.iterdir() if p.is_file() and p.suffix.lower() in exts)
    else:
        archivos_eval_nombres = set(p.name for p in dir_sin.iterdir() if p.is_file() and p.suffix.lower() in exts)

    archivos_sin_ordenados = sorted(list(archivos_eval_nombres), key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", x)])
    total = len(archivos_sin_ordenados)

    print(f"\nTotal frames en la intersección exacta a procesar: {total}")

    tam_lote = 150
    lotes = []
    for i in range(0, total, tam_lote):
        sub_nombres = [(archivos_sin_ordenados[j], j + 1) for j in range(i, min(i + tam_lote, total))]
        lotes.append((sub_nombres, str(dir_orig), str(dir_sin), str(dir_den)))

    todos_registros = []
    print(f"Analizando {total} frames en paralelo en {args.cores} núcleos CPU...")

    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        futures = [executor.submit(procesar_lote_en_ram, lote) for lote in lotes]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Calculando Ruido en RAM"):
            res = f.result()
            todos_registros.extend(res)

    todos_registros.sort(key=lambda x: x["frame_idx"])
    df = pd.DataFrame(todos_registros)
    csv_out = RAIZ / "videos/video2/resumen_ruidos_video2.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[+] Estadísticas completas guardadas en: {csv_out}")

    # Filtrar marcos homogéneos
    umbral_sobel = df["sobel_energia"].quantile(0.35)
    df_estructural = df[df["sobel_energia"] >= umbral_sobel].copy()
    print(f"[+] Frames con contenido estructural real (vías, ríos, dragas, vegetación): {len(df_estructural)} de {len(df)}")

    top_mas_ruido = df_estructural.sort_values(by="mae_termico", ascending=False).head(5)
    top_menos_ruido = df_estructural.sort_values(by="mae_termico", ascending=True).head(5)

    print("\nGenerando Mosaicos 1x4 Horizontales de alta resolución para los casos seleccionados...")

    for rank, (_, row) in enumerate(top_mas_ruido.iterrows(), 1):
        nom = row["nombre_archivo"]
        f_o = dir_orig / nom
        f_s = dir_sin / nom
        f_d = dir_den / nom
        if f_o.is_file() and f_s.is_file() and f_d.is_file():
            im_o = cv2.imread(str(f_o))
            im_s = cv2.imread(str(f_s))
            im_d = cv2.imread(str(f_d))
            gr_s = cv2.cvtColor(im_s, cv2.COLOR_BGR2GRAY)
            gr_d = cv2.cvtColor(im_d, cv2.COLOR_BGR2GRAY)
            res_term = gr_s.astype(np.float32) - gr_d.astype(np.float32)
            out_p = dir_mosaicos / f"mosaico_MAX_RUIDO_rank{rank:02d}_{nom}"
            renderizar_mosaico_1x4(im_o, im_s, im_d, res_term, row.to_dict(), out_p)
            print(f"  * [MAX RUIDO #{rank}] Guardado: {out_p.name} (MAE: {row['mae_termico']:.2f})")

    for rank, (_, row) in enumerate(top_menos_ruido.iterrows(), 1):
        nom = row["nombre_archivo"]
        f_o = dir_orig / nom
        f_s = dir_sin / nom
        f_d = dir_den / nom
        if f_o.is_file() and f_s.is_file() and f_d.is_file():
            im_o = cv2.imread(str(f_o))
            im_s = cv2.imread(str(f_s))
            im_d = cv2.imread(str(f_d))
            gr_s = cv2.cvtColor(im_s, cv2.COLOR_BGR2GRAY)
            gr_d = cv2.cvtColor(im_d, cv2.COLOR_BGR2GRAY)
            res_term = gr_s.astype(np.float32) - gr_d.astype(np.float32)
            out_p = dir_mosaicos / f"mosaico_MIN_RUIDO_rank{rank:02d}_{nom}"
            renderizar_mosaico_1x4(im_o, im_s, im_d, res_term, row.to_dict(), out_p)
            print(f"  * [MIN RUIDO #{rank}] Guardado: {out_p.name} (MAE: {row['mae_termico']:.2f})")

    print("\n" + "=" * 85)
    print("PROCESO COMPLETADO EXITOSAMENTE.")
    print(f"Mosaicos 2x2: {dir_mosaicos}")
    print(f"Resumen CSV:  {csv_out}")
    print("=" * 85)


if __name__ == "__main__":
    main()
