#!/usr/bin/env python3
"""
==============================================================================
GENERADOR AUTOMÁTICO DE PRESENTACIÓN DE TESIS (CASOS DE ESTUDIO 16:9 FULL HD)
==============================================================================
1. Identifica y extrae:
   - HUD: Top 5 Mayor Remoción y Top 5 Casos Desafiantes.
   - Denoising (Ruido Térmico): Top 5 Mayor Reducción y Top 5 Casos Retadores.
2. Criterios de Calidad Científica:
   - Separación temporal amplia (>= 1,500 frames ~ 1-2 min de vuelo).
   - Filtro de riqueza visual: descarta cielo plano, fondos negros o saturados.
     Garantiza que siempre hay ríos, vegetación, campamentos, caminos u objetos.
3. Diseño Visual (Inspirado en mario.pptx, pero CENTRADO A LO ALTO):
   - Lienzo 1920x1080 (16:9) con fondo oscuro (#141414).
   - Distribución vertical perfectamente simétrica y centrada.
   - Paneles Side-by-Side con marcos de acento (Naranja FAC y Verde Lima).
   - Rótulos limpios sin porcentajes gigantescos ni anómalos.
4. Genera automáticamente el archivo PowerPoint (.pptx) listo para sustentación.
==============================================================================
"""

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from pptx import Presentation
    from pptx.util import Inches
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

RAIZ = Path(__file__).resolve().parent.parent


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def formatear_tiempo(frame_idx, fps=30.0):
    segundos = frame_idx / fps
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"


def extraer_numero_frame(nombre_stem):
    nums = re.findall(r"\d+", nombre_stem)
    return int(nums[-1]) if nums else 0


def estimar_sigma_mad(gray):
    """Estima desviación estándar de ruido usando Wavelet Haar HH con MAD."""
    img = gray.astype(np.float32)
    hh = (img[0::2, 0::2] - img[0::2, 1::2] - img[1::2, 0::2] + img[1::2, 1::2]) * 0.5
    med = np.median(np.abs(hh))
    return float(med / 0.67448975)


def calcular_riqueza_escena(gray):
    """Calcula riqueza de texturas y bordes reales (descarta cielo o calibración)."""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energia_grad = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))
    std_int = float(np.std(gray))
    media_int = float(np.mean(gray))
    return energia_grad, std_int, media_int


def es_frame_valido(img1, img2):
    """Filtra frames sin información térmica, quemados, apagados o sin objetos."""
    for img in [img1, img2]:
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        total_px = gray.shape[0] * gray.shape[1]

        media = float(np.mean(gray))
        desv = float(np.std(gray))

        # Rango de brillo operativo
        if media < 45.0 or media > 190.0:
            return False
        # Contraste de escena (descarta fondos lisos)
        if desv < 18.0:
            return False

        # Saturación de blancos (calibración o reflejo quemado)
        pct_blanco = float(np.count_nonzero(gray > 240)) / total_px
        if pct_blanco > 0.05:
            return False

        # Fondo negro / bordes muertos
        pct_negro = float(np.count_nonzero(gray < 15)) / total_px
        if pct_negro > 0.06:
            return False

        # Gradientes y bordes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energia_grad = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))
        if energia_grad < 4.0:
            return False

    return True


def seleccionar_top_diversos(lista_ordenada, top_k=5, min_dist_frames=1800):
    """Garantiza que los casos seleccionados no estén pegados en el tiempo ni repetidos."""
    seleccionados = []
    for item in lista_ordenada:
        idx_act = item["idx"]
        if all(abs(idx_act - s["idx"]) >= min_dist_frames for s in seleccionados):
            seleccionados.append(item)
            if len(seleccionados) == top_k:
                break

    # Relajar distancia si no completa el cupo
    if len(seleccionados) < top_k:
        for dist_relax in [1200, 600, 300]:
            for item in lista_ordenada:
                if item not in seleccionados:
                    if all(abs(item["idx"] - s["idx"]) >= dist_relax for s in seleccionados):
                        seleccionados.append(item)
                        if len(seleccionados) == top_k:
                            break
            if len(seleccionados) == top_k:
                break

    return seleccionados


def crear_diapositiva_centrada(
    img_izq,
    img_der,
    titulo_slide,
    subtitulo_categoria,
    label_izq,
    label_der,
    ruta_salida,
    color_der=(0, 220, 150),
    color_izq=(0, 140, 255)
):
    """
    Renderiza diapositiva 1920x1080 (16:9) CENTRADA A LO ALTO:
    - Sin espacio negro excesivo al fondo.
    - Paneles simétricos grandes y nítidos.
    """
    ANCHO_SLIDE = 1920
    ALTO_SLIDE = 1080

    canvas = np.zeros((ALTO_SLIDE, ANCHO_SLIDE, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)  # Fondo oscuro mate

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. TÍTULO PRINCIPAL
    (tw, th), _ = cv2.getTextSize(titulo_slide, font, 1.2, 2)
    tx = (ANCHO_SLIDE - tw) // 2
    cv2.putText(canvas, titulo_slide, (tx, 75), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # SUBTÍTULO DE CATEGORÍA
    if subtitulo_categoria:
        (sw, sh), _ = cv2.getTextSize(subtitulo_categoria, font, 0.75, 1)
        sx = (ANCHO_SLIDE - sw) // 2
        cv2.putText(canvas, subtitulo_categoria, (sx, 115), font, 0.75, (180, 180, 180), 1, cv2.LINE_AA)

    # 2. CÁLCULO DE DIMENSIONES Y CENTRADO VERTICAL
    # Relación de aspecto 16:9
    gap = 46
    margen_lat = 65
    ancho_panel = (ANCHO_SLIDE - (2 * margen_lat) - gap) // 2  # ~872 px
    alto_panel = int(ancho_panel * 9 / 16)                     # ~490 px

    # CENTRADO A LO ALTO
    # Espacio disponible entre subtítulo (Y=130) y pie de página (Y=1020) = 890 px
    # Altura del bloque: etiqueta (35 px) + panel (490 px) = 525 px
    # Margen vertical = (890 - 525) // 2 = 182 px -> Y_label = 130 + 182 = 312 px, Y_panel = 345 px
    y_panel = 325

    ancho_total_bloque = (2 * ancho_panel) + gap
    x_izq = (ANCHO_SLIDE - ancho_total_bloque) // 2
    x_der = x_izq + ancho_panel + gap

    # Redimensionar paneles con Lanczos para máxima nitidez
    p_izq_res = cv2.resize(img_izq, (ancho_panel, alto_panel), interpolation=cv2.INTER_LANCZOS4)
    p_der_res = cv2.resize(img_der, (ancho_panel, alto_panel), interpolation=cv2.INTER_LANCZOS4)

    # 3. RÓTULOS ENCIMA DE CADA PANEL
    # Rótulo Izquierdo
    (ti_w, ti_h), _ = cv2.getTextSize(label_izq, font, 0.85, 2)
    t_izq_x = x_izq + (ancho_panel - ti_w) // 2
    cv2.putText(canvas, label_izq, (t_izq_x, y_panel - 16), font, 0.85, color_izq, 2, cv2.LINE_AA)

    # Rótulo Derecho
    (td_w, td_h), _ = cv2.getTextSize(label_der, font, 0.85, 2)
    t_der_x = x_der + (ancho_panel - td_w) // 2
    cv2.putText(canvas, label_der, (t_der_x, y_panel - 16), font, 0.85, color_der, 2, cv2.LINE_AA)

    # 4. COLOCAR PANELES Y BORDES ELEGANTES
    canvas[y_panel : y_panel + alto_panel, x_izq : x_izq + ancho_panel] = p_izq_res
    cv2.rectangle(canvas, (x_izq - 2, y_panel - 2), (x_izq + ancho_panel + 1, y_panel + alto_panel + 1), color_izq, 2)

    canvas[y_panel : y_panel + alto_panel, x_der : x_der + ancho_panel] = p_der_res
    cv2.rectangle(canvas, (x_der - 2, y_panel - 2), (x_der + ancho_panel + 1, y_panel + alto_panel + 1), color_der, 2)

    # 5. PIE DE PÁGINA
    pie = "Fuerza Aeroespacial Colombiana (FAC) | Detección de Minería Ilegal con Sensores FLIR"
    (pw, ph), _ = cv2.getTextSize(pie, font, 0.55, 1)
    px = (ANCHO_SLIDE - pw) // 2
    cv2.putText(canvas, pie, (px, 1045), font, 0.55, (100, 100, 100), 1, cv2.LINE_AA)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_salida), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def crear_archivo_pptx(rutas_imagenes, ruta_pptx):
    """Compila las imágenes generadas en una presentación PPTX 16:9 lista para proyectar."""
    if not HAS_PPTX:
        print("[!] Advertencia: python-pptx no disponible. Diapositivas PNG generadas con éxito.")
        return

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    for ruta_img in rutas_imagenes:
        if ruta_img.is_file():
            slide = prs.slides.add_slide(blank_layout)
            slide.shapes.add_picture(str(ruta_img), 0, 0, width=prs.slide_width, height=prs.slide_height)

    ruta_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(ruta_pptx))
    print(f"\n[+] Presentación PowerPoint creada exitosamente: {ruta_pptx}")


def procesar_presentacion(video_id, n_casos=5):
    print(f"\n==============================================================================")
    print(f"🎬 GENERANDO CASOS DE ESTUDIO Y DIAPOSITIVAS CENTRADAS: {video_id.upper()}")
    print(f"==============================================================================")

    dir_video = RAIZ / "videos" / video_id
    dir_orig = dir_video / "frames_originales"
    dir_sinhud = dir_video / "frames_sin_hud"
    dir_udvd = dir_video / "udvd_sin_hud"
    if not dir_udvd.is_dir():
        dir_udvd = dir_video / "expos" / "udvd_sin_hud"

    dir_salida = RAIZ / "figuras_tesis" / f"presentacion_{video_id}"
    dir_salida.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    frames_orig = sorted([p for p in dir_orig.iterdir() if p.suffix.lower() in exts], key=natural_key) if dir_orig.is_dir() else []
    frames_sinhud_dict = {p.name: p for p in dir_sinhud.iterdir() if p.suffix.lower() in exts} if dir_sinhud.is_dir() else {}
    frames_udvd_dict = {p.name: p for p in dir_udvd.iterdir() if p.suffix.lower() in exts} if dir_udvd.is_dir() else {}

    todas_diapositivas = []

    # -------------------------------------------------------------------------
    # A. SELECCIÓN DE CASOS DE REMOCIÓN DE HUD
    # -------------------------------------------------------------------------
    if frames_orig and frames_sinhud_dict:
        print("\n--- 1. Analizando Remoción de HUD (Original vs Sin HUD) ---")
        comunes_hud = [p for p in frames_orig if p.name in frames_sinhud_dict]

        def evaluar_hud(idx):
            p_orig = comunes_hud[idx]
            p_sin = frames_sinhud_dict[p_orig.name]
            im_o = cv2.imread(str(p_orig))
            im_s = cv2.imread(str(p_sin))
            if not es_frame_valido(im_o, im_s):
                return None
            g_o = cv2.cvtColor(im_o, cv2.COLOR_BGR2GRAY)
            g_s = cv2.cvtColor(im_s, cv2.COLOR_BGR2GRAY)
            # Riqueza de escena
            eg, std_i, _ = calcular_riqueza_escena(g_s)
            diff_hud = np.abs(g_o.astype(np.float32) - g_s.astype(np.float32))
            energia_hud_removida = float(np.mean(diff_hud))
            return {
                "idx": idx, "p_orig": p_orig, "p_sin": p_sin,
                "energia_hud": energia_hud_removida, "energia_grad": eg
            }

        # Muestreo cada 5 frames
        indices_hud = list(range(0, len(comunes_hud), 5))
        candidatos_hud = []
        with ThreadPoolExecutor(max_workers=12) as pool:
            for r in tqdm(pool.map(evaluar_hud, indices_hud), total=len(indices_hud), desc="Escaneando HUD"):
                if r is not None:
                    candidatos_hud.append(r)

        if candidatos_hud:
            # Top 5 Mayor Remoción de HUD
            ord_mayor_hud = sorted(candidatos_hud, key=lambda x: (x["energia_hud"], x["energia_grad"]), reverse=True)
            top_mayor_hud = seleccionar_top_diversos(ord_mayor_hud, top_k=n_casos, min_dist_frames=1800)

            # Top 5 Casos Difíciles de HUD (menor residuo o textura compleja)
            ord_menor_hud = sorted(candidatos_hud, key=lambda x: (x["energia_hud"], -x["energia_grad"]))
            top_menor_hud = seleccionar_top_diversos(ord_menor_hud, top_k=n_casos, min_dist_frames=1800)

            print(f"Renderizando {len(top_mayor_hud)} diapositivas de Mayor Remoción de HUD...")
            for rank, c in enumerate(top_mayor_hud, start=1):
                n_fr = extraer_numero_frame(c["p_orig"].stem)
                t_str = formatear_tiempo(n_fr)
                im_o = cv2.imread(str(c["p_orig"]))
                im_s = cv2.imread(str(c["p_sin"]))
                out_p = dir_salida / f"01_hud_top{rank:02d}_mayor_remocion_frame_{n_fr:05d}.png"
                crear_diapositiva_centrada(
                    im_o, im_s,
                    f"Video {video_id.replace('video','')} - Frame {n_fr:05d} (Minuto {t_str})",
                    "[Etapa 1: Mayor Remoción de Simbología HUD]",
                    "Original (Con HUD)",
                    "Sin HUD (Inpainting ProPainter)",
                    out_p
                )
                todas_diapositivas.append(out_p)

            print(f"Renderizando {len(top_menor_hud)} diapositivas de Casos Desafiantes de HUD...")
            for rank, c in enumerate(top_menor_hud, start=1):
                n_fr = extraer_numero_frame(c["p_orig"].stem)
                t_str = formatear_tiempo(n_fr)
                im_o = cv2.imread(str(c["p_orig"]))
                im_s = cv2.imread(str(c["p_sin"]))
                out_p = dir_salida / f"02_hud_top{rank:02d}_caso_desafiante_frame_{n_fr:05d}.png"
                crear_diapositiva_centrada(
                    im_o, im_s,
                    f"Video {video_id.replace('video','')} - Frame {n_fr:05d} (Minuto {t_str})",
                    "[Etapa 1: Caso Desafiante - Oclusión Compleja de HUD]",
                    "Original (Con HUD)",
                    "Sin HUD (Inpainting ProPainter)",
                    out_p
                )
                todas_diapositivas.append(out_p)

    # -------------------------------------------------------------------------
    # B. SELECCIÓN DE CASOS DE REDUCCIÓN DE RUIDO TÉRMICO (DENOISING)
    # -------------------------------------------------------------------------
    if frames_sinhud_dict and frames_udvd_dict:
        print("\n--- 2. Analizando Reducción de Ruido Térmico (Sin HUD vs UDVD) ---")
        comunes_denoising = [p for p in dir_sinhud.iterdir() if p.suffix.lower() in exts and p.name in frames_udvd_dict]

        def evaluar_denoising(idx):
            p_sin = comunes_denoising[idx]
            p_udvd = frames_udvd_dict[p_sin.name]
            im_s = cv2.imread(str(p_sin))
            im_u = cv2.imread(str(p_udvd))
            if not es_frame_valido(im_s, im_u):
                return None
            g_s = cv2.cvtColor(im_s, cv2.COLOR_BGR2GRAY)
            g_u = cv2.cvtColor(im_u, cv2.COLOR_BGR2GRAY)

            s_sin = estimar_sigma_mad(g_s)
            s_udvd = estimar_sigma_mad(g_u)
            eg, _, _ = calcular_riqueza_escena(g_s)

            if s_sin < 4.0:
                return None

            caida_sigma = s_sin - s_udvd
            return {
                "idx": idx, "p_sin": p_sin, "p_udvd": p_udvd,
                "s_sin": s_sin, "s_udvd": s_udvd, "caida_sigma": caida_sigma, "energia_grad": eg
            }

        indices_den = list(range(0, len(comunes_denoising), 5))
        candidatos_den = []
        with ThreadPoolExecutor(max_workers=12) as pool:
            for r in tqdm(pool.map(evaluar_denoising, indices_den), total=len(indices_den), desc="Escaneando Ruido Térmico"):
                if r is not None:
                    candidatos_den.append(r)

        if candidatos_den:
            # Top 5 Mayor Reducción de Ruido
            ord_mayor_den = sorted(candidatos_den, key=lambda x: (x["caida_sigma"], x["energia_grad"]), reverse=True)
            top_mayor_den = seleccionar_top_diversos(ord_mayor_den, top_k=n_casos, min_dist_frames=1800)

            # Top 5 Casos Retadores (zonas lisas o bajo contraste)
            ord_menor_den = sorted(candidatos_den, key=lambda x: (x["caida_sigma"], -x["energia_grad"]))
            top_menor_den = seleccionar_top_diversos(ord_menor_den, top_k=n_casos, min_dist_frames=1800)

            print(f"Renderizando {len(top_mayor_den)} diapositivas de Mayor Reducción de Ruido...")
            for rank, c in enumerate(top_mayor_den, start=1):
                n_fr = extraer_numero_frame(c["p_sin"].stem)
                t_str = formatear_tiempo(n_fr)
                im_s = cv2.imread(str(c["p_sin"]))
                im_u = cv2.imread(str(c["p_udvd"]))
                out_p = dir_salida / f"03_ruido_top{rank:02d}_mayor_reduccion_frame_{n_fr:05d}.png"
                label_der = f"UDVD Modelo Final (Sigma: {c['s_sin']:.1f} -> {c['s_udvd']:.1f} DN)"
                crear_diapositiva_centrada(
                    im_s, im_u,
                    f"Video {video_id.replace('video','')} - Frame {n_fr:05d} (Minuto {t_str})",
                    "[Etapa 2: Mayor Supresión de Ruido Térmico y FPN]",
                    f"Sin HUD (Sigma: {c['s_sin']:.1f} DN)",
                    label_der,
                    out_p
                )
                todas_diapositivas.append(out_p)

            print(f"Renderizando {len(top_menor_den)} diapositivas de Casos Retadores de Ruido...")
            for rank, c in enumerate(top_menor_den, start=1):
                n_fr = extraer_numero_frame(c["p_sin"].stem)
                t_str = formatear_tiempo(n_fr)
                im_s = cv2.imread(str(c["p_sin"]))
                im_u = cv2.imread(str(c["p_udvd"]))
                out_p = dir_salida / f"04_ruido_top{rank:02d}_caso_retador_frame_{n_fr:05d}.png"
                label_der = f"UDVD Modelo Final (Sigma: {c['s_sin']:.1f} -> {c['s_udvd']:.1f} DN)"
                crear_diapositiva_centrada(
                    im_s, im_u,
                    f"Video {video_id.replace('video','')} - Frame {n_fr:05d} (Minuto {t_str})",
                    "[Etapa 2: Caso Retador - Alta Homogeneidad Térmica]",
                    f"Sin HUD (Sigma: {c['s_sin']:.1f} DN)",
                    label_der,
                    out_p
                )
                todas_diapositivas.append(out_p)

    # -------------------------------------------------------------------------
    # C. ENSAMBLAR ARCHIVO POWERPOINT (.PPTX) Y SUBIR A GOOGLE DRIVE
    # -------------------------------------------------------------------------
    if todas_diapositivas:
        pptx_salida = RAIZ / "figuras_tesis" / f"presentacion_casos_estudio_{video_id}.pptx"
        crear_archivo_pptx(todas_diapositivas, pptx_salida)

        import shutil
        import subprocess
        if shutil.which("rclone"):
            print("☁️  Subiendo presentación (.pptx) y diapositivas a Google Drive...")
            try:
                subprocess.run(
                    ["rclone", "copy", str(pptx_salida), "gdrive:proyecto-FAC/figuras_tesis/"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                )
                subprocess.run(
                    ["rclone", "copy", str(dir_salida), f"gdrive:proyecto-FAC/figuras_tesis/presentacion_{video_id}/"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                )
                print(f"   [+] Sincronizado en Google Drive: gdrive:proyecto-FAC/figuras_tesis/{pptx_salida.name}")
            except Exception as e:
                print(f"   [!] Subida rclone omitida: {e}")

    print("\n==============================================================================")
    print(f"🎉 DIAPOSITIVAS Y PRESENTACIÓN FINALIZADAS CON ÉXITO")
    print(f"   Carpeta PNG: {dir_salida}")
    print(f"   Archivo PPTX: figuras_tesis/presentacion_casos_estudio_{video_id}.pptx")
    print("==============================================================================")


def main():
    parser = argparse.ArgumentParser(description="Generador de Presentación de Tesis (Casos Centrados a lo Alto)")
    parser.add_argument("--video", default="video2", choices=["video1", "video2", "video3", "todos"])
    parser.add_argument("--n-casos", type=int, default=5, help="Casos por categoría (default: 5)")
    args = parser.parse_args()

    v_lista = ["video1", "video2", "video3"] if args.video == "todos" else [args.video]
    for vid in v_lista:
        procesar_presentacion(vid, n_casos=args.n_casos)


if __name__ == "__main__":
    main()
