#!/usr/bin/env python3
"""
Generador de Mosaicos Comparativos Multimodelo Exhaustivos (Side-by-Side):
Genera 2 imágenes panorámicas de alta resolución:
1. 'figuras_tesis/mosaico_top10_maxima_reduccion.png' (Top 10 frames de mayor reducción de ruido).
2. 'figuras_tesis/mosaico_top10_casos_dificiles.png' (Top 10 frames de menor reducción / casos difíciles).

Columnas por fila (8 vistas sincronizadas lado a lado):
[1. Original] [2. Sin HUD] [3. UDVD T=5] [4. UDVD T=7] [5. UDVD T=7 Destriping] [6. StructN2V Vert] [7. Blind2Unblind] [8. Ensemble Mediana]
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


def banner_texto(img, texto, color_fondo=(20, 20, 20), color_texto=(255, 255, 255), color_borde=(0, 165, 255)):
    h, w, _ = img.shape
    bh = 36
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bh), color_fondo, -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
    cv2.line(img, (0, bh), (w, bh), color_borde, 2)
    cv2.putText(img, texto, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color_texto, 1, cv2.LINE_AA)
    return img


def procesar_un_frame(args):
    idx, ruta_orig, ruta_sinhud, ruta_u_destrip = args
    g_orig = cv2.imread(str(ruta_orig), cv2.IMREAD_GRAYSCALE)
    g_sinhud = cv2.imread(str(ruta_sinhud), cv2.IMREAD_GRAYSCALE)
    g_mod = cv2.imread(str(ruta_u_destrip), cv2.IMREAD_GRAYSCALE)

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
        "s_orig": s_orig,
        "s_sinhud": s_sinhud,
        "s_mod": s_mod,
        "mejora_pct": mejora_pct,
    }


def construir_mosaico(lista_frames, modelos_info, titulo_poster, ruta_salida):
    filas_mosaico = []
    
    for rank, item in enumerate(lista_frames, start=1):
        paneles_fila = []
        
        for key, (carpeta, label_nombre, color_acento) in modelos_info.items():
            ruta_img = carpeta / f"{item['nombre']}.png"
            if not ruta_img.exists():
                ruta_img = carpeta / f"{item['nombre']}.jpg"

            img = cv2.imread(str(ruta_img))
            if img is None:
                # Panel negro si no existe
                img = np.zeros((480, 640, 3), dtype=np.uint8)

            # Calcular sigma especifico de este modelo en este frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            s_val = estimar_sigma_mad(gray)
            
            # Texto
            texto_label = f"#{rank:02d} | {label_nombre} (s={s_val:.1f})"
            img = banner_texto(img, texto_label, color_borde=color_acento)

            # Redimensionar al ancho estandar de columna (400 px por panel)
            h, w, _ = img.shape
            w_col = 400
            h_col = int(h * (w_col / w))
            p_res = cv2.resize(img, (w_col, h_col), interpolation=cv2.INTER_AREA)
            paneles_fila.append(p_res)

        fila_completa = np.hstack(paneles_fila)
        filas_mosaico.append(fila_completa)

    mosaico_matriz = np.vstack(filas_mosaico)

    # Encabezado superior
    encabezado = np.zeros((65, mosaico_matriz.shape[1], 3), dtype=np.uint8) + 20
    cv2.putText(
        encabezado,
        titulo_poster,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    mosaico_final = np.vstack([encabezado, mosaico_matriz])

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_salida), mosaico_final, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    print(f"-> Guardado exitosamente: {ruta_salida} ({ruta_salida.stat().st_size / (1024**2):.1f} MB)")


def main():
    p = argparse.ArgumentParser(description="Mosaicos Multimodelo Exhaustivos")
    p.add_argument("--video", default="video2")
    p.add_argument("--ultimos-frames", type=int, default=18000)
    p.add_argument("--paso-muestreo", type=int, default=5, help="Paso de escaneo (5 = escanea 3,600 frames representativos)")
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args()

    ruta_base = RAIZ / "videos" / args.video
    dir_expos = ruta_base / "expos"

    # Definir modelos a contrastar en orden prioritario
    modelos_info = {
        "original": (ruta_base / "frames_originales", "1. Original", (0, 140, 255)),
        "sin_hud": (ruta_base / "frames_sin_hud", "2. Sin HUD", (0, 200, 200)),
        "udvd_t5_global": (dir_expos / "udvd_sin_hud", "3. UDVD T=5 Global", (255, 100, 100)),
        "udvd_t3_puro": (dir_expos / "udvd_t3_ult10min_sin_hud", "4. UDVD T=3 Puro", (255, 160, 50)),
        "udvd_t5_puro": (dir_expos / "udvd_t5_ult10min_sin_hud", "5. UDVD T=5 Puro", (255, 120, 50)),
        "udvd_t7_puro": (dir_expos / "udvd_t7_ult10min_sin_hud", "6. UDVD T=7 Puro", (255, 50, 50)),
        "udvd_destrip_t3": (dir_expos / "udvd_destriping_t3_ult10min_sin_hud", "7. UDVD T=3 Destrip", (255, 220, 0)),
        "udvd_destrip_t5": (dir_expos / "udvd_destriping_t5_ult10min_sin_hud", "8. UDVD T=5 Destrip", (0, 220, 255)),
        "udvd_destrip_t7": (dir_expos / "udvd_destriping_t7_ult10min_sin_hud", "9. UDVD T=7 Destrip", (0, 255, 120)),
        "struct_n2v": (dir_expos / "struct_n2v_vert_ult10min_sin_hud", "10. StructN2V Vert", (200, 100, 255)),
        "n2v2": (dir_expos / "n2v2_ult10min_sin_hud", "11. N2V2", (180, 80, 220)),
        "blind2unblind": (dir_expos / "blind2unblind_sin_hud", "12. Blind2Unblind", (255, 150, 200)),
        "ens_wavelet_t7": (dir_expos / "ensemble_wavelet_udvd_destrip_struct_vert_ult10min", "13. Ens. Wavelet", (100, 255, 200)),
        "ens_mediana_top3": (dir_expos / "ensemble_mediana_top3_ult10min", "14. Ens. Mediana Top3", (0, 255, 255)),
        "ens_multiescala_puro": (dir_expos / "ensemble_multitemporal_sin_destrip_t1_t3_t5_t7", "15. Ens. Puro T1357", (255, 120, 200)),
        "ens_multiescala_destrip": (dir_expos / "ensemble_multitemporal_con_destrip_t1_t3_t5_t7", "16. Ens. Destrip T1357", (0, 255, 200)),
        "fusion_adapt": (dir_expos / "fusion_motion_adaptive_udvd_struct_vert", "17. Fusion Adaptativa", (0, 255, 0)),
    }

    # Filtrar modelos que existan en disco
    disponibles = {}
    for k, v in modelos_info.items():
        if v[0].is_dir():
            disponibles[k] = v

    # Auto-descubrir cualquier otra carpeta en expos no listada explícitamente
    if dir_expos.is_dir():
        for sub in sorted(dir_expos.iterdir(), key=natural_key):
            if sub.is_dir() and sub.name not in [v[0].name for v in disponibles.values()]:
                disponibles[sub.name] = (sub, sub.name[:18], (180, 180, 180))

    print(f"=== MODELOS ACTIVOS PARA EL MOSAICO ({len(disponibles)} COLUMNAS) ===")
    for k, v in disponibles.items():
        print(f"  [{k}]: {v[1]}")

    # Listar frames comunes
    exts = {".png", ".jpg", ".jpeg"}
    frames_orig = sorted([p for p in (ruta_base / "frames_originales").iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]
    frames_sinhud = sorted([p for p in (ruta_base / "frames_sin_hud").iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]
    
    dir_ref_mod = dir_expos / "udvd_destriping_t7_ult10min_sin_hud"
    if not dir_ref_mod.is_dir():
        dir_ref_mod = dir_expos / "udvd_sin_hud"
    frames_mod = sorted([p for p in dir_ref_mod.iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]

    n_frames = min(len(frames_orig), len(frames_sinhud), len(frames_mod))
    print(f"\nAnalizando {n_frames // args.paso_muestreo} frames candidatos para ordenar por efectividad de ruido...")

    indices = list(range(0, n_frames, args.paso_muestreo))
    tareas = [(i, frames_orig[i], frames_sinhud[i], frames_mod[i]) for i in indices]

    resultados = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for res in tqdm(pool.map(procesar_un_frame, tareas), total=len(tareas), desc="Escaneando ruido", dynamic_ncols=True):
            if res is not None:
                resultados.append(res)

    # 1. Top 10 Máxima Reducción de Ruido
    top_exitos = sorted(resultados, key=lambda x: x["mejora_pct"], reverse=True)[:args.top_n]

    # 2. Top 10 Casos Difíciles / Menor Reducción
    top_dificiles = sorted(resultados, key=lambda x: x["mejora_pct"])[:args.top_n]

    dir_figuras = RAIZ / "figuras_tesis"
    dir_figuras.mkdir(parents=True, exist_ok=True)

    print("\n--- Generando Mosaico 1: TOP 10 MÁXIMA REDUCCIÓN DE RUIDO ---")
    out1 = dir_figuras / "mosaico_top10_maxima_reduccion.png"
    construir_mosaico(
        top_exitos, disponibles,
        "TOP 10 CASOS DE MAYOR REDUCCION DE RUIDO TERMICO FLIR (VIDEO 2) - COMPARATIVA MULTIMODELO",
        out1
    )

    print("\n--- Generando Mosaico 2: TOP 10 CASOS DIFÍCILES / MENOR REDUCCIÓN ---")
    out2 = dir_figuras / "mosaico_top10_casos_dificiles.png"
    construir_mosaico(
        top_dificiles, disponibles,
        "TOP 10 CASOS DIFÍCILES / ESCENAS COMPLEJAS DEL SENSOR FLIR (VIDEO 2) - COMPARATIVA MULTIMODELO",
        out2
    )

    print("\n============================================================")
    print("¡AMBOS MOSAICOS MULTIMODELO FUERON GENERADOS CON ÉXITO!")
    print(f"1. {out1}")
    print(f"2. {out2}")
    print("============================================================")


if __name__ == "__main__":
    main()
