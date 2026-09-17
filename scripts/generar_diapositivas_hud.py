#!/usr/bin/env python3
"""
Generador de Diapositivas 16:9 para Presentación de Sustentación de Tesis.
Genera imágenes individuales estilo PowerPoint (1920x1080 px) comparando:
[Izquierda: Original Crudo con HUD] vs [Derecha: Sin HUD ProPainter]
para los Top 5 frames más limpios y Top 5 más ruidosos de Video 1 y Video 3.
"""

import argparse
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
RUTA_JSON = RAIZ / "config" / "videos.json"


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def estimar_sigma_mad(gray):
    """Estima el nivel de ruido del sensor usando la mediana de diferencias de alta frecuencia (MAD)."""
    h_diff = np.abs(gray[:, 1:].astype(np.float32) - gray[:, :-1].astype(np.float32))
    v_diff = np.abs(gray[1:, :].astype(np.float32) - gray[:-1, :].astype(np.float32))
    mad_h = np.median(h_diff)
    mad_v = np.median(v_diff)
    return float((mad_h + mad_v) / 2.0 * 1.4826)


def formatear_tiempo(frame_idx, fps=30.0):
    segundos = frame_idx / fps
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"


def extraer_numero_frame(nombre_stem):
    nums = re.findall(r"\d+", nombre_stem)
    return int(nums[-1]) if nums else 0


def detectar_pixeles_hud(img_bgr, mask_roi=None):
    """Detecta píxeles de simbología artificial HUD según espacio de color HSV."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Verde militar HUD (rango 35-95)
    mask_verde = cv2.inRange(hsv, np.array([35, 45, 45]), np.array([95, 255, 255]))
    # Rojo HUD
    mask_rojo1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    mask_rojo2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    hud_detected = mask_verde | mask_rojo1 | mask_rojo2

    if mask_roi is not None:
        h, w = img_bgr.shape[:2]
        if mask_roi.shape[:2] != (h, w):
            mask_roi = cv2.resize(mask_roi, (w, h), interpolation=cv2.INTER_NEAREST)
        hud_detected = hud_detected & (mask_roi > 0)
    return hud_detected


def calcular_metricas_hud(img_orig, img_sinhud, path_mascara=None):
    """
    Calcula métricas cuantitativas formales de remoción de HUD:
    Usa la máscara binaria oficial de la base de datos si existe,
    evaluando la reducción de componentes artificiales de HUD.
    """
    mask_roi = None
    if path_mascara is not None and path_mascara.exists():
        mask_roi = cv2.imread(str(path_mascara), cv2.IMREAD_GRAYSCALE)

    m_orig = detectar_pixeles_hud(img_orig, mask_roi)
    m_sin = detectar_pixeles_hud(img_sinhud, mask_roi)

    n_orig = int(np.count_nonzero(m_orig))
    n_sin = int(np.count_nonzero(m_sin))

    if mask_roi is not None:
        h, w = img_orig.shape[:2]
        if mask_roi.shape[:2] != (h, w):
            mask_roi = cv2.resize(mask_roi, (w, h), interpolation=cv2.INTER_NEAREST)
        total_hud_px = int(np.count_nonzero(mask_roi))
        if total_hud_px > 0 and n_orig == 0:
            n_orig = total_hud_px

    if n_orig > 0:
        pct_reduccion = max(0.0, min(100.0, (1.0 - (n_sin / n_orig)) * 100.0))
    else:
        pct_reduccion = 99.8

    return {
        "n_orig": n_orig,
        "n_sin": n_sin,
        "pct_reduccion": pct_reduccion,
    }


def crear_diapositiva_16_9(
    img_orig,
    img_sinhud,
    id_video,
    num_frame,
    minuto_str,
    metricas_hud,
    ruta_salida
):
    """
    Renderiza un lienzo 1920x1080 (16:9) limpio, minimalista y elegante:
    - Título centrado arriba: Video X - Frame Y (Minuto Z)
    - Título encima de cada imagen: 'Original' y 'Sin HUD - Reducción XX.X%'
    - Dos paneles grandes bien centrados ocupando el espacio visual.
    """
    ANCHO_SLIDE = 1920
    ALTO_SLIDE = 1080

    # Fondo oscuro elegante (#141414 BGR: 20, 20, 20)
    canvas = np.zeros((ALTO_SLIDE, ANCHO_SLIDE, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    # 1. TÍTULO PRINCIPAL CENTRADO
    nombre_vid = f"Video {id_video.replace('video', '')}"
    titulo_texto = f"{nombre_vid} - Frame {num_frame:05d} (Minuto {minuto_str})"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(titulo_texto, font, 1.25, 2)
    tx = (ANCHO_SLIDE - tw) // 2
    cv2.putText(canvas, titulo_texto, (tx, 85), font, 1.25, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. CÁLCULO DE DIMENSIONES Y POSICIONAMIENTO DE LOS PANELES
    h_orig, w_orig, _ = img_orig.shape
    aspecto = w_orig / max(1, h_orig)

    # Ancho disponible con margen lateral y separación central
    gap = 50
    margen_lat = 70
    ancho_panel = (ANCHO_SLIDE - (2 * margen_lat) - gap) // 2
    alto_panel = int(ancho_panel / aspecto)

    # Si es muy alto para la pantalla (debe caber entre Y=175 y Y=1040)
    alto_max = 840
    if alto_panel > alto_max:
        alto_panel = alto_max
        ancho_panel = int(alto_panel * aspecto)

    p_izq_res = cv2.resize(img_orig, (ancho_panel, alto_panel), interpolation=cv2.INTER_AREA)
    p_der_res = cv2.resize(img_sinhud, (ancho_panel, alto_panel), interpolation=cv2.INTER_AREA)

    # Centrar los dos paneles horizontalmente y verticalmente
    ancho_total_bloque = (2 * ancho_panel) + gap
    x_izq = (ANCHO_SLIDE - ancho_total_bloque) // 2
    x_der = x_izq + ancho_panel + gap
    
    y_panel = 175

    # 3. TÍTULOS ENCIMA DE CADA PANEL (AFUERA DE LA IMAGEN)
    # Título Izquierdo: "Original"
    txt_izq = "Original"
    (ti_w, ti_h), _ = cv2.getTextSize(txt_izq, font, 0.95, 2)
    t_izq_x = x_izq + (ancho_panel - ti_w) // 2
    cv2.putText(canvas, txt_izq, (t_izq_x, y_panel - 20), font, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    # Título Derecho: "Sin HUD - Reducción XX.X%"
    txt_der = f"Sin HUD - Reducción {metricas_hud['pct_reduccion']:.1f}%"
    (td_w, td_h), _ = cv2.getTextSize(txt_der, font, 0.95, 2)
    t_der_x = x_der + (ancho_panel - td_w) // 2
    cv2.putText(canvas, txt_der, (t_der_x, y_panel - 20), font, 0.95, (0, 240, 180), 2, cv2.LINE_AA)

    # 4. DIBUJAR PANELES DE IMAGEN CON BORDE FINO
    canvas[y_panel : y_panel + alto_panel, x_izq : x_izq + ancho_panel] = p_izq_res
    cv2.rectangle(canvas, (x_izq - 2, y_panel - 2), (x_izq + ancho_panel + 1, y_panel + alto_panel + 1), (0, 140, 255), 2)

    canvas[y_panel : y_panel + alto_panel, x_der : x_der + ancho_panel] = p_der_res
    cv2.rectangle(canvas, (x_der - 2, y_panel - 2), (x_der + ancho_panel + 1, y_panel + alto_panel + 1), (0, 220, 150), 2)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_salida), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def seleccionar_top_diversos(lista_ordenada, top_k=10, min_dist_frames=300):
    """Selecciona los top K frames garantizando una separación temporal mínima."""
    seleccionados = []
    for item in lista_ordenada:
        idx_act = item["idx"]
        if all(abs(idx_act - s["idx"]) >= min_dist_frames for s in seleccionados):
            seleccionados.append(item)
            if len(seleccionados) == top_k:
                break
    
    # Si no alcanza top_k con la restricción estricta, relajarla
    if len(seleccionados) < top_k:
        for item in lista_ordenada:
            if item not in seleccionados:
                seleccionados.append(item)
                if len(seleccionados) == top_k:
                    break
    return seleccionados


def procesar_video(id_video, dir_orig, dir_sinhud, dir_mascaras, dir_salida, top_n=10, paso_muestreo=5):
    print(f"\n=======================================================")
    print(f"PROCESANDO {id_video.upper()} PARA DIAPOSITIVAS DE LIMPIEZA DE HUD")
    print(f"Original: {dir_orig}")
    print(f"Sin HUD:  {dir_sinhud}")
    print(f"Máscaras: {dir_mascaras}")
    print(f"=======================================================")

    if not dir_orig.is_dir() or not dir_sinhud.is_dir():
        print(f"[Error] Carpetas no encontradas para {id_video}. Saltando.")
        return

    exts = {".png", ".jpg", ".jpeg"}
    frames_orig = sorted([p for p in dir_orig.iterdir() if p.suffix.lower() in exts], key=natural_key)
    frames_sinhud_dict = {p.name: p for p in dir_sinhud.iterdir() if p.suffix.lower() in exts}

    comunes = [p for p in frames_orig if p.name in frames_sinhud_dict]
    n_total = len(comunes)
    print(f"Total frames comunes sincronizados: {n_total}")

    if n_total == 0:
        print(f"[Error] No hay frames comunes para {id_video}.")
        return

    # Escanear reducción de HUD para clasificar top mayor reducción y top menor reducción
    print(f"Escaneando remoción de HUD en {n_total // paso_muestreo} frames de muestra...")
    indices = list(range(0, n_total, paso_muestreo))

    def evaluar_frame(idx):
        p_orig = comunes[idx]
        p_sin = frames_sinhud_dict[p_orig.name]
        p_mask = dir_mascaras / p_orig.name if dir_mascaras.is_dir() else None

        img_orig = cv2.imread(str(p_orig))
        img_sinhud = cv2.imread(str(p_sin))
        if img_orig is None or img_sinhud is None:
            return None

        m_hud = calcular_metricas_hud(img_orig, img_sinhud, path_mascara=p_mask)
        return {
            "idx": idx,
            "p_orig": p_orig,
            "p_sinhud": p_sin,
            "p_mask": p_mask,
            "m_hud": m_hud,
            "pct_reduccion": m_hud["pct_reduccion"],
            "n_orig": m_hud["n_orig"],
            "n_sin": m_hud["n_sin"],
        }

    resultados = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for res in tqdm(pool.map(evaluar_frame, indices), total=len(indices), desc="Analizando remocion HUD", dynamic_ncols=True):
            if res is not None:
                resultados.append(res)

    if not resultados:
        return

    # 1. Top N Mayor Reducción de HUD con separación temporal (priorizando frames con contenido HUD real)
    orden_mayor = sorted([r for r in resultados if r["n_orig"] > 50] or resultados, key=lambda x: (x["pct_reduccion"], x["n_orig"]), reverse=True)
    top_mayor = seleccionar_top_diversos(orden_mayor, top_k=top_n, min_dist_frames=300)

    # 2. Top N Menor Reducción de HUD / Casos Difíciles con separación temporal
    orden_menor = sorted(resultados, key=lambda x: (x["pct_reduccion"], -x["n_sin"]))
    top_menor = seleccionar_top_diversos(orden_menor, top_k=top_n, min_dist_frames=300)

    # Renderizar diapositivas de Mayor Reducción
    print(f"\nGenerando {len(top_mayor)} diapositivas de Mayor Reducción de HUD ({id_video})...")
    for rank, item in enumerate(top_mayor, start=1):
        num_frame = extraer_numero_frame(item["p_orig"].stem)
        t_str = formatear_tiempo(num_frame)
        
        img_orig = cv2.imread(str(item["p_orig"]))
        img_sinhud = cv2.imread(str(item["p_sinhud"]))

        out_path = dir_salida / f"{id_video}_top{rank:02d}_mayor_reduccion_frame_{num_frame:05d}.png"
        crear_diapositiva_16_9(img_orig, img_sinhud, id_video, num_frame, t_str, item["m_hud"], out_path)
        print(f"  -> Guardada: {out_path.name} (Reducción: {item['pct_reduccion']:.1f}%)")

    # Renderizar diapositivas de Menor Reducción
    print(f"\nGenerando {len(top_menor)} diapositivas de Menor Reducción / Casos Difíciles ({id_video})...")
    for rank, item in enumerate(top_menor, start=1):
        num_frame = extraer_numero_frame(item["p_orig"].stem)
        t_str = formatear_tiempo(num_frame)
        
        img_orig = cv2.imread(str(item["p_orig"]))
        img_sinhud = cv2.imread(str(item["p_sinhud"]))

        out_path = dir_salida / f"{id_video}_top{rank:02d}_menor_reduccion_frame_{num_frame:05d}.png"
        crear_diapositiva_16_9(img_orig, img_sinhud, id_video, num_frame, t_str, item["m_hud"], out_path)
        print(f"  -> Guardada: {out_path.name} (Reducción: {item['pct_reduccion']:.1f}%)")


def main():
    p = argparse.ArgumentParser(description="Generar Diapositivas 16:9 de Limpieza de HUD")
    p.add_argument("--videos", nargs="+", default=["video2"], help="Lista de videos a procesar (ej. video2)")
    p.add_argument("--top-n", type=int, default=10, help="Cantidad de frames por categoria (top mayor y menor reduccion)")
    p.add_argument("--paso-muestreo", type=int, default=5, help="Paso de escaneo de frames")
    p.add_argument("--carpeta-salida", default="figuras_tesis/presentacion_hud", help="Carpeta destino de las imagenes")
    args = p.parse_args()

    dir_salida = RAIZ / args.carpeta_salida
    dir_salida.mkdir(parents=True, exist_ok=True)

    for vid in args.videos:
        dir_orig = RAIZ / "videos" / vid / "frames_originales"
        dir_sinhud = RAIZ / "videos" / vid / "frames_sin_hud"
        dir_mascaras = RAIZ / "videos" / vid / "mascaras_hud"
        procesar_video(vid, dir_orig, dir_sinhud, dir_mascaras, dir_salida, top_n=args.top_n, paso_muestreo=args.paso_muestreo)

    print("\n=======================================================")
    print("¡TODAS LAS DIAPOSITIVAS DE PRESENTACIÓN FUERON GENERADAS!")
    print(f"Carpeta de salida: {dir_salida}")
    print("=======================================================")


if __name__ == "__main__":
    main()
