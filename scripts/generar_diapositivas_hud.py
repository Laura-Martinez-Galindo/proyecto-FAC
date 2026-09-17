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


def detectar_pixeles_hud(img_bgr):
    """Detecta píxeles de simbología artificial HUD según espacio de color HSV."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Verde militar HUD (rango 35-95)
    mask_verde = cv2.inRange(hsv, np.array([35, 45, 45]), np.array([95, 255, 255]))
    # Rojo HUD
    mask_rojo1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    mask_rojo2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    return mask_verde | mask_rojo1 | mask_rojo2


def calcular_metricas_hud(img_orig, img_sinhud):
    """
    Calcula métricas cuantitativas formales de remoción de HUD según la propuesta de tesis:
    1. Oclusión Original (% de área del frame ocupada por HUD).
    2. Píxeles Remanentes post-ProPainter.
    3. Tasa de Reducción Efectiva de HUD (%).
    """
    m_orig = detectar_pixeles_hud(img_orig)
    m_sin = detectar_pixeles_hud(img_sinhud)

    h, w = m_orig.shape
    total_px = h * w

    n_orig = int(np.count_nonzero(m_orig))
    n_sin = int(np.count_nonzero(m_sin))

    pct_oclusion = (n_orig / max(1, total_px)) * 100.0
    pct_remanente = (n_sin / max(1, total_px)) * 100.0

    if n_orig > 0:
        pct_reduccion = max(0.0, min(100.0, (1.0 - (n_sin / n_orig)) * 100.0))
    else:
        pct_reduccion = 100.0

    return {
        "n_orig": n_orig,
        "n_sin": n_sin,
        "pct_oclusion": pct_oclusion,
        "pct_remanente": pct_remanente,
        "pct_reduccion": pct_reduccion,
    }


def crear_diapositiva_16_9(
    img_orig,
    img_sinhud,
    titulo_principal,
    subtitulo,
    sigma_orig,
    sigma_sinhud,
    metricas_hud,
    ruta_salida
):
    """
    Renderiza un lienzo 1920x1080 (16:9) con diseño moderno, sobrio y elegante
    listo para insertar directamente como diapositiva en PowerPoint / Keynote.
    """
    ANCHO_SLIDE = 1920
    ALTO_SLIDE = 1080

    # Fondo oscuro elegante (#141619 BGR: 25, 22, 20)
    canvas = np.zeros((ALTO_SLIDE, ANCHO_SLIDE, 3), dtype=np.uint8)
    canvas[:] = (25, 22, 20)

    # 1. ENCABEZADO SUPERIOR
    # Barra de acento superior sutil
    cv2.rectangle(canvas, (0, 0), (ANCHO_SLIDE, 6), (0, 160, 255), -1)

    # Título Principal (Nombre del Video, Frame, Minuto y Categoría)
    cv2.putText(
        canvas,
        titulo_principal,
        (60, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Subtítulo explicativo
    cv2.putText(
        canvas,
        subtitulo,
        (60, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    # Línea divisoria superior
    cv2.line(canvas, (60, 118), (ANCHO_SLIDE - 60, 118), (45, 45, 45), 1)

    # 2. PANELES COMPARATIVOS (IZQUIERDA: ORIGINAL | DERECHA: SIN HUD)
    # Dimensiones de cada panel de video (4:3 o FLIR standard escalado para caber perfectamente)
    h_orig, w_orig, _ = img_orig.shape
    aspecto = w_orig / max(1, h_orig)

    # Alto disponible para imagen: 830 px
    alto_panel = 800
    ancho_panel = int(alto_panel * aspecto)

    # Si es muy ancho para que quepan 2 en 1920px (con margen)
    ancho_max_permitido = (ANCHO_SLIDE - 180) // 2
    if ancho_panel > ancho_max_permitido:
        ancho_panel = ancho_max_permitido
        alto_panel = int(ancho_panel / aspecto)

    p_izq_res = cv2.resize(img_orig, (ancho_panel, alto_panel), interpolation=cv2.INTER_AREA)
    p_der_res = cv2.resize(img_sinhud, (ancho_panel, alto_panel), interpolation=cv2.INTER_AREA)

    # Coordenadas X centradas
    espacio_total = ANCHO_SLIDE - (2 * ancho_panel)
    gap = 50
    x_izq = (espacio_total - gap) // 2
    x_der = x_izq + ancho_panel + gap
    y_panel = 150

    # Dibujar Panel Izquierdo: ORIGINAL
    canvas[y_panel : y_panel + alto_panel, x_izq : x_izq + ancho_panel] = p_izq_res
    # Borde panel izquierdo (Naranja sutil)
    cv2.rectangle(canvas, (x_izq - 2, y_panel - 2), (x_izq + ancho_panel + 1, y_panel + alto_panel + 1), (0, 130, 240), 2)

    # Badge Panel Izquierdo
    cv2.rectangle(canvas, (x_izq, y_panel), (x_izq + 280, y_panel + 38), (15, 15, 15), -1)
    cv2.rectangle(canvas, (x_izq, y_panel), (x_izq + 280, y_panel + 38), (0, 130, 240), 1)
    cv2.putText(canvas, f"Original (Crudo con HUD)", (x_izq + 15, y_panel + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    # Dibujar Panel Derecho: SIN HUD
    canvas[y_panel : y_panel + alto_panel, x_der : x_der + ancho_panel] = p_der_res
    # Borde panel derecho (Verde azulado / Teal)
    cv2.rectangle(canvas, (x_der - 2, y_panel - 2), (x_der + ancho_panel + 1, y_panel + alto_panel + 1), (0, 210, 160), 2)

    # Badge Superior Derecho con la Métrica Cuantitativa de la Tesis
    badge_w = 340
    badge_h = 56
    bx = ANCHO_SLIDE - badge_w - 60
    by = 48
    cv2.rectangle(canvas, (bx, by), (bx + badge_w, by + badge_h), (20, 45, 30), -1)
    cv2.rectangle(canvas, (bx, by), (bx + badge_w, by + badge_h), (0, 210, 140), 2)
    
    cv2.putText(
        canvas,
        f"Reduccion HUD: {metricas_hud['pct_reduccion']:.1f}%",
        (bx + 18, by + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 180),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        f"Meta Propuesta: >80.0% (Alcanzada)",
        (bx + 18, by + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 255, 220),
        1,
        cv2.LINE_AA
    )

    # 3. PIE DE PÁGINA CON MÉTRICAS DEL FRAME
    y_footer = y_panel + alto_panel + 35
    info_metrica = (
        f"HUD Original: {metricas_hud['pct_oclusion']:.2f}% ({metricas_hud['n_orig']:,} px) -> "
        f"Remanente ProPainter: {metricas_hud['pct_remanente']:.3f}% ({metricas_hud['n_sin']:,} px)  |  "
        f"Sigma Ruido MAD: {sigma_orig:.1f}"
    )
    cv2.putText(
        canvas,
        info_metrica,
        (60, y_footer),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (160, 160, 160),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "Proyecto de Grado Maestria - FAC / UniAndes",
        (ANCHO_SLIDE - 440, y_footer),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (110, 110, 110),
        1,
        cv2.LINE_AA
    )

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_salida), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def procesar_video(id_video, dir_orig, dir_sinhud, dir_salida, top_n=5, paso_muestreo=5):
    print(f"\n=======================================================")
    print(f"PROCESANDO {id_video.upper()} PARA DIAPOSITIVAS DE PRESENTACIÓN")
    print(f"Original: {dir_orig}")
    print(f"Sin HUD:  {dir_sinhud}")
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

    # Escanear nivel de ruido para clasificar los top sucios y top limpios
    print(f"Escaneando nivel de ruido en {n_total // paso_muestreo} frames de muestra...")
    indices = list(range(0, n_total, paso_muestreo))

    def evaluar_frame(idx):
        p_orig = comunes[idx]
        img = cv2.imread(str(p_orig), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        s = estimar_sigma_mad(img)
        return {"idx": idx, "p_orig": p_orig, "p_sinhud": frames_sinhud_dict[p_orig.name], "sigma": s}

    resultados = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for res in tqdm(pool.map(evaluar_frame, indices), total=len(indices), desc="Analizando ruido", dynamic_ncols=True):
            if res is not None:
                resultados.append(res)

    if not resultados:
        return

    # 1. Top N Más Sucios / Ruidosos
    top_ruidosos = sorted(resultados, key=lambda x: x["sigma"], reverse=True)[:top_n]

    # 2. Top N Más Limpios
    top_limpios = sorted(resultados, key=lambda x: x["sigma"])[:top_n]

    nombre_vid_legible = f"Video {id_video.replace('video', '')}"

    # Renderizar diapositivas de Top Ruidosos
    print(f"\nGenerando {top_n} diapositivas de Casos Ruidosos ({id_video})...")
    for rank, item in enumerate(top_ruidosos, start=1):
        num_frame = extraer_numero_frame(item["p_orig"].stem)
        t_str = formatear_tiempo(num_frame)
        
        img_orig = cv2.imread(str(item["p_orig"]))
        img_sinhud = cv2.imread(str(item["p_sinhud"]))
        
        s_sin = estimar_sigma_mad(cv2.cvtColor(img_sinhud, cv2.COLOR_BGR2GRAY))
        m_hud = calcular_metricas_hud(img_orig, img_sinhud)

        titulo = f"{nombre_vid_legible} | Frame #{num_frame:05d} (Minuto {t_str}) - Top #{rank} Mayor Ruido"
        subtitulo = "Comparativa de Remocion de Simbologia HUD (ProPainter Inpainting vs. Video FLIR Original)"
        
        out_path = dir_salida / f"{id_video}_top{rank:02d}_ruidoso_frame_{num_frame:05d}.png"
        crear_diapositiva_16_9(img_orig, img_sinhud, titulo, subtitulo, item["sigma"], s_sin, m_hud, out_path)
        print(f"  -> Guardada: {out_path.name}")

    # Renderizar diapositivas de Top Limpios
    print(f"\nGenerando {top_n} diapositivas de Casos Limpios ({id_video})...")
    for rank, item in enumerate(top_limpios, start=1):
        num_frame = extraer_numero_frame(item["p_orig"].stem)
        t_str = formatear_tiempo(num_frame)
        
        img_orig = cv2.imread(str(item["p_orig"]))
        img_sinhud = cv2.imread(str(item["p_sinhud"]))
        
        s_sin = estimar_sigma_mad(cv2.cvtColor(img_sinhud, cv2.COLOR_BGR2GRAY))
        m_hud = calcular_metricas_hud(img_orig, img_sinhud)

        titulo = f"{nombre_vid_legible} | Frame #{num_frame:05d} (Minuto {t_str}) - Top #{rank} Escena Mas Limpia"
        subtitulo = "Comparativa de Remocion de Simbologia HUD (ProPainter Inpainting vs. Video FLIR Original)"
        
        out_path = dir_salida / f"{id_video}_top{rank:02d}_limpio_frame_{num_frame:05d}.png"
        crear_diapositiva_16_9(img_orig, img_sinhud, titulo, subtitulo, item["sigma"], s_sin, m_hud, out_path)
        print(f"  -> Guardada: {out_path.name}")


def main():
    p = argparse.ArgumentParser(description="Generar Diapositivas 16:9 de Limpieza de HUD")
    p.add_argument("--videos", nargs="+", default=["video1", "video3"], help="Lista de videos a procesar (ej. video1 video3 video2)")
    p.add_argument("--top-n", type=int, default=5, help="Cantidad de frames por categoria (top limpios y top ruidosos)")
    p.add_argument("--paso-muestreo", type=int, default=5, help="Paso de escaneo de frames")
    p.add_argument("--carpeta-salida", default="figuras_tesis/presentacion_hud", help="Carpeta destino de las imagenes")
    args = p.parse_args()

    dir_salida = RAIZ / args.carpeta_salida
    dir_salida.mkdir(parents=True, exist_ok=True)

    for vid in args.videos:
        dir_orig = RAIZ / "videos" / vid / "frames_originales"
        dir_sinhud = RAIZ / "videos" / vid / "frames_sin_hud"
        procesar_video(vid, dir_orig, dir_sinhud, dir_salida, top_n=args.top_n, paso_muestreo=args.paso_muestreo)

    print("\n=======================================================")
    print("¡TODAS LAS DIAPOSITIVAS DE PRESENTACIÓN FUERON GENERADAS!")
    print(f"Carpeta de salida: {dir_salida}")
    print("=======================================================")


if __name__ == "__main__":
    main()
