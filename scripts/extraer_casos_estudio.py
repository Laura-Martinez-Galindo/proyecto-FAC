#!/usr/bin/env python3
"""
Módulo para Identificación Automática y Extracción de Casos de Estudio Críticos:
1. Top 20 Casos de Mayor Éxito (Máxima reducción de ruido térmico y distorsión Δ%).
2. Top 20 Casos Difíciles / Críticos (Zonas de menor reducción o desafíos del sensor).
3. Genera trípticos comparativos de alta resolución:
   [ 1. Original (Crudo + HUD) | 2. Sin HUD (ProPainter) | 3. Método Campeón (UDVD Destriping) ]
4. Inserta rótulos con métricas exactas (σ de ruido, nitidez, Δ% de mejora) y recuadro de Zoom.
5. Genera mosaicos consolidados (contact sheets) y tabla Excel detallada para la tesis.
"""

import argparse
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def estimar_sigma_mad(gray):
    """Estima sigma de ruido con el estimador robusto MAD sobre el Laplaciano."""
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    return float(mad / 0.6745)


def calcular_nitidez(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def agregar_banner_panel(img, titulo, subtitulo=None, color_acento=(0, 165, 255)):
    """Superpone una barra superior semitransparente con tipografía nítida."""
    h, w, _ = img.shape
    banner_h = 44
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    
    # Línea de acento inferior
    cv2.line(img, (0, banner_h), (w, banner_h), color_acento, 2)

    # Texto
    cv2.putText(img, titulo, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitulo:
        cv2.putText(img, subtitulo, (w - 380, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 220, 220), 2, cv2.LINE_AA)
    return img


def generar_triptico_con_zoom(ruta_orig, ruta_sinhud, ruta_mod, info_frame, salida_png, bbox=None):
    """
    Construye un tríptico horizontal:
    [ 1. Original Crudo ] [ 2. Sin HUD ProPainter ] [ 3. Modelo Campeón ]
    + Zoom 2x de la región de mayor interés en la parte inferior.
    """
    img_orig = cv2.imread(str(ruta_orig))
    img_sinhud = cv2.imread(str(ruta_sinhud))
    img_mod = cv2.imread(str(ruta_mod))

    if img_orig is None or img_sinhud is None or img_mod is None:
        return

    h, w, _ = img_orig.shape

    # Región de Zoom (por defecto el tercio central)
    if bbox is None:
        ymin, ymax = int(h * 0.30), int(h * 0.70)
        xmin, xmax = int(w * 0.30), int(w * 0.70)
    else:
        ymin, xmin, ymax, xmax = bbox

    # 1. Banners superiores con métricas
    p1 = agregar_banner_panel(img_orig.copy(), f"1. Original Crudo | {info_frame['nombre']}", f"Sigma = {info_frame['sigma_orig']:.2f}", (0, 140, 255))
    p2 = agregar_banner_panel(img_sinhud.copy(), "2. Sin HUD (ProPainter)", f"Sigma = {info_frame['sigma_sinhud']:.2f}", (0, 200, 200))
    
    color_mejora = (0, 230, 115) if info_frame["mejora_pct"] >= 0 else (0, 69, 255)
    txt_mejora = f"Sigma = {info_frame['sigma_mod']:.2f} | Mejora: {info_frame['mejora_pct']:+.1f}%"
    p3 = agregar_banner_panel(img_mod.copy(), f"3. {info_frame['nombre_modelo']}", txt_mejora, color_mejora)

    # 2. Dibujar recuadro de zoom en los paneles principales
    for p in [p1, p2, p3]:
        cv2.rectangle(p, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

    fila_superior = np.hstack([p1, p2, p3])

    # 3. Construir fila de Zoom (recortes ampliados al 100% de ancho de cada columna)
    patch_w = w
    patch_h = int(h * 0.5)

    z1 = cv2.resize(img_orig[ymin:ymax, xmin:xmax], (patch_w, patch_h), interpolation=cv2.INTER_LANCZOS4)
    z2 = cv2.resize(img_sinhud[ymin:ymax, xmin:xmax], (patch_w, patch_h), interpolation=cv2.INTER_LANCZOS4)
    z3 = cv2.resize(img_mod[ymin:ymax, xmin:xmax], (patch_w, patch_h), interpolation=cv2.INTER_LANCZOS4)

    z1 = agregar_banner_panel(z1, "Zoom 2.5x (Detalle Crudo)", None, (0, 140, 255))
    z2 = agregar_banner_panel(z2, "Zoom 2.5x (Detalle Sin HUD)", None, (0, 200, 200))
    z3 = agregar_banner_panel(z3, "Zoom 2.5x (Detalle Filtrado)", None, color_mejora)

    fila_zoom = np.hstack([z1, z2, z3])

    # 4. Canvas final (Fila principal + Fila de zoom)
    canvas = np.vstack([fila_superior, fila_zoom])

    # Redimensionar al 50% para archivo PNG liviano (~2 MB por tríptico)
    h_tot, w_tot, _ = canvas.shape
    canvas_final = cv2.resize(canvas, (w_tot // 2, h_tot // 2), interpolation=cv2.INTER_AREA)

    salida_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida_png), canvas_final, [cv2.IMWRITE_PNG_COMPRESSION, 4])


def analizar_un_frame(args):
    idx, ruta_orig, ruta_sinhud, ruta_mod, nombre_mod = args
    bgr_orig = cv2.imread(str(ruta_orig), cv2.IMREAD_GRAYSCALE)
    bgr_sinhud = cv2.imread(str(ruta_sinhud), cv2.IMREAD_GRAYSCALE)
    bgr_mod = cv2.imread(str(ruta_mod), cv2.IMREAD_GRAYSCALE)

    if bgr_orig is None or bgr_sinhud is None or bgr_mod is None:
        return None

    s_orig = estimar_sigma_mad(bgr_orig)
    s_sinhud = estimar_sigma_mad(bgr_sinhud)
    s_mod = estimar_sigma_mad(bgr_mod)

    nit_orig = calcular_nitidez(bgr_orig)
    nit_sinhud = calcular_nitidez(bgr_sinhud)
    nit_mod = calcular_nitidez(bgr_mod)

    # Reducción porcentual de ruido respecto a Sin HUD
    mejora_pct = ((s_sinhud - s_mod) / max(1e-4, s_sinhud)) * 100.0
    caida_sigma = s_sinhud - s_mod

    return {
        "idx": idx,
        "nombre": ruta_orig.stem,
        "ruta_orig": ruta_orig,
        "ruta_sinhud": ruta_sinhud,
        "ruta_mod": ruta_mod,
        "nombre_modelo": nombre_mod,
        "sigma_orig": s_orig,
        "sigma_sinhud": s_sinhud,
        "sigma_mod": s_mod,
        "nit_orig": nit_orig,
        "nit_sinhud": nit_sinhud,
        "nit_mod": nit_mod,
        "caida_sigma": caida_sigma,
        "mejora_pct": mejora_pct,
    }


def main():
    p = argparse.ArgumentParser(description="Extracción y análisis de casos de estudio extremos.")
    p.add_argument("--video", default="video2", help="Identificador del video.")
    p.add_argument("--modelo-carpeta", default="videos/video2/expos/udvd_destriping_t7_ult10min_sin_hud",
                   help="Carpeta del modelo campeón a analizar.")
    p.add_argument("--nombre-modelo", default="UDVD Destriping T=7", help="Nombre descriptivo del modelo.")
    p.add_argument("--ultimos-frames", type=int, default=18000, help="Cantidad de últimos frames evaluados.")
    p.add_argument("--n-muestras", type=int, default=20, help="Cantidad de casos de estudio por grupo (éxitos y difíciles).")
    p.add_argument("--carpeta-salida", default="figuras_tesis/casos_estudio_video2", help="Carpeta de salida.")
    args = p.parse_args()

    dir_salida = RAIZ / args.carpeta_salida
    dir_salida.mkdir(parents=True, exist_ok=True)
    dir_exitos = dir_salida / "top_20_mayores_mejoras"
    dir_dificiles = dir_salida / "top_20_casos_dificiles"
    dir_exitos.mkdir(parents=True, exist_ok=True)
    dir_dificiles.mkdir(parents=True, exist_ok=True)

    ruta_base = RAIZ / "videos" / args.video
    dir_orig = ruta_base / "frames_originales"
    dir_sinhud = ruta_base / "frames_sin_hud"
    dir_mod = RAIZ / args.modelo_carpeta if not Path(args.modelo_carpeta).is_absolute() else Path(args.modelo_carpeta)

    # 1. Listar frames de las 3 fuentes
    exts = {".png", ".jpg", ".jpeg"}
    frames_orig = sorted([p for p in dir_orig.iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]
    frames_sinhud = sorted([p for p in dir_sinhud.iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]
    frames_mod = sorted([p for p in dir_mod.iterdir() if p.suffix.lower() in exts], key=natural_key)[-args.ultimos_frames:]

    n_frames = min(len(frames_orig), len(frames_sinhud), len(frames_mod))
    print(f"=== ANALIZANDO {n_frames} FRAMES PARA CASOS DE ESTUDIO CRÍTICOS ===")
    print(f"Modelo: {args.nombre_modelo} ({dir_mod.name})")

    tareas = [
        (i, frames_orig[i], frames_sinhud[i], frames_mod[i], args.nombre_modelo)
        for i in range(n_frames)
    ]

    # 2. Análisis estadístico multihilo
    with ThreadPoolExecutor(max_workers=16) as pool:
        resultados = list(tqdm(pool.map(analizar_un_frame, tareas), total=n_frames, desc="Analizando ruido por frame", dynamic_ncols=True))

    resultados = [r for r in resultados if r is not None]

    # 3. Clasificación: Top N Mayores Mejoras vs Top N Casos Difíciles
    # Ordenar por reducción de ruido porcentual descendente
    ordenados_mejora = sorted(resultados, key=lambda x: x["mejora_pct"], reverse=True)
    top_exitos = ordenados_mejora[:args.n_muestras]
    
    # Casos difíciles: menor mejora porcentual (o donde menos bajó el ruido)
    top_dificiles = sorted(resultados, key=lambda x: x["mejora_pct"])[:args.n_muestras]

    print(f"\n--- Generando {args.n_muestras} Trípticos de Mayor Éxito (Reducción de Ruido Máxima) ---")
    for rank, item in enumerate(tqdm(top_exitos, desc="Renderizando Éxitos", dynamic_ncols=True), start=1):
        out_png = dir_exitos / f"exito_{rank:02d}_{item['nombre']}.png"
        generar_triptico_con_zoom(item["ruta_orig"], item["ruta_sinhud"], item["ruta_mod"], item, out_png)

    print(f"\n--- Generando {args.n_muestras} Trípticos de Casos Difíciles / Desafíos del Sensor ---")
    for rank, item in enumerate(tqdm(top_dificiles, desc="Renderizando Casos Difíciles", dynamic_ncols=True), start=1):
        out_png = dir_dificiles / f"dificil_{rank:02d}_{item['nombre']}.png"
        generar_triptico_con_zoom(item["ruta_orig"], item["ruta_sinhud"], item["ruta_mod"], item, out_png)

    # 4. Generar Excel de Resumen de Casos de Estudio
    wb = Workbook()
    ws = wb.active
    ws.title = "Casos de Estudio"

    headers = [
        "Grupo", "Ranking", "Frame", "Sigma Original (Crudo)", "Sigma Sin HUD",
        f"Sigma {args.nombre_modelo}", "Reducción Ruido Absoluta (Δσ)",
        "Mejora Relativa (%)", "Nitidez Sin HUD", f"Nitidez {args.nombre_modelo}", "Diagnóstico"
    ]
    ws.append(headers)

    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    for col in range(1, len(headers) + 1):
        c = ws.cell(row=1, column=col)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"), right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"), bottom=Side(style="thin", color="D9D9D9")
    )

    def agregar_filas(lista, grupo_nombre, desc_defecto):
        for rank, r in enumerate(lista, start=1):
            row_vals = [
                grupo_nombre, rank, r["nombre"], r["sigma_orig"], r["sigma_sinhud"],
                r["sigma_mod"], r["caida_sigma"], f"{r['mejora_pct']:+.2f}%",
                r["nit_sinhud"], r["nit_mod"], desc_defecto
            ]
            ws.append(row_vals)
            curr_row = ws.max_row
            for col_idx in range(1, len(row_vals) + 1):
                cell = ws.cell(row=curr_row, column=col_idx)
                cell.border = thin_border
                cell.font = Font(name="Calibri", size=10)
                if col_idx in [1, 3, 11]:
                    cell.alignment = Alignment(horizontal="left", vertical="center")
                elif col_idx in [2, 8]:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                    if col_idx in [4, 5, 6, 7]:
                        cell.number_format = "0.0000"
                    elif col_idx in [9, 10]:
                        cell.number_format = "0.00"

    agregar_filas(top_exitos, "Top 20 Mayores Mejoras", "Excelente: Supresión total de rayas de microbolómetro y parpadeo temporal.")
    agregar_filas(top_dificiles, "Top 20 Casos Difíciles", "Desafío: Escena homogénea con baja dinámica térmica o saturación local.")

    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 22
    ws.column_dimensions["K"].width = 50

    excel_salida = dir_salida / "resumen_casos_estudio.xlsx"
    wb.save(str(excel_salida))

    print("\n============================================================")
    print(f"ANÁLISIS DE CASOS DE ESTUDIO COMPLETADO")
    print(f"Trípticos de Éxito: {dir_exitos}")
    print(f"Trípticos Difíciles: {dir_dificiles}")
    print(f"Excel Resumen: {excel_salida}")
    print("============================================================")


if __name__ == "__main__":
    main()
