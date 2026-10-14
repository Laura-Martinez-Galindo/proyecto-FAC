#!/usr/bin/env python3
"""
Módulo para Reconstrucción de Videos FLIR y Generación de Videos Comparativos Side-by-Side / Grid 2x2.
Soporta:
1. Reconstrucción de videos individuales a partir de carpetas de frames PNG (H.264 / mp4v).
2. Generación de videos comparativos sincronizados:
   - Dual Side-by-Side (1x2)
   - Triple Comparativo (1x3)
   - Cuadrícula Sincronizada (2x2)
3. Rótulos y títulos profesionales integrados sobre cada panel para presentaciones y sustentación.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
RUTA_JSON = RAIZ / "config" / "videos.json"


def argumentos():
    p = argparse.ArgumentParser(description="Reconstruccion y comparacion de videos FLIR.")
    p.add_argument("--video", required=True, help="Identificador del video (video1, video2, video3).")
    p.add_argument("--tipo", choices=("individual", "comparativo_dual", "comparativo_triple", "comparativo_2x2", "todos"), default="todos",
                   help="Tipo de renderizado a ejecutar.")
    p.add_argument("--fps", type=float, default=30.0, help="Fotogramas por segundo (FPS).")
    p.add_argument("--max-frames", type=int, help="Limite maximo de frames a renderizar.")
    p.add_argument("--ultimos-frames", type=int, help="Renderizar exclusivamente los ultimos N frames.")
    p.add_argument("--desde-frame", type=int, help="Indice inicial de frame.")
    p.add_argument("--hasta-frame", type=int, help="Indice final de frame.")
    p.add_argument("--modelos", nargs="+", help="Lista de carpetas o IDs de modelos para el comparativo.")
    p.add_argument("--carpeta-salida", help="Carpeta destino para guardar los videos renderizados.")
    p.add_argument("--forzar", action="store_true", help="Sobrescribir videos si ya existen previamente.")
    return p.parse_args()


def cargar_json():
    with RUTA_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def natural(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def listar_frames(carpeta, max_frames=None, ultimos=None, desde=None, hasta=None):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    rutas = sorted((p for p in Path(carpeta).iterdir() if p.is_file() and p.suffix.lower() in exts), key=natural)
    
    if desde is not None or hasta is not None:
        ini = desde or 0
        fin = hasta or len(rutas)
        return rutas[ini:fin]
    if ultimos and ultimos > 0:
        return rutas[-ultimos:]
    if max_frames and max_frames > 0:
        return rutas[:max_frames]
    return rutas


def agregar_titulo_panel(img, titulo, subtitulo=None):
    """Superpone una barra superior semitransparente con titulo elegante."""
    h, w, _ = img.shape
    banner_h = max(28, int(h * 0.08))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    # Linea inferior de acento naranja FAC
    cv2.line(img, (0, banner_h), (w, banner_h), (0, 165, 255), 2)

    # Texto
    texto_completo = f"{titulo}" + (f" | {subtitulo}" if subtitulo else "")
    font_scale = max(0.42, banner_h / 58.0)
    thickness = 1 if font_scale < 0.6 else 2
    cv2.putText(img, texto_completo, (10, int(banner_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def crear_proceso_ffmpeg(ruta_mp4, ancho=1920, alto=1080, fps=30.0):
    """Crea un proceso ffmpeg para recibir frames crudos por stdin y codificar a MP4 H.264."""
    ffmpeg = shutil.which("ffmpeg")
    ruta_mp4.parent.mkdir(parents=True, exist_ok=True)
    if ffmpeg:
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{ancho}x{alto}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "17",
            "-pix_fmt", "yuv420p",
            str(ruta_mp4)
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), True
    else:
        # Fallback a VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(ruta_mp4), fourcc, fps, (ancho, alto))
        return writer, False


def renderizar_video_individual(rutas_frames, ruta_mp4, fps=30.0, titulo=None, target_w=1920, target_h=1080, forzar=False):
    """Renderiza una secuencia de frames escalando a Full HD 1080p de alta nitidez."""
    if not rutas_frames:
        print(f"Advertencia: No hay frames para renderizar en {ruta_mp4}")
        return

    if not forzar and ruta_mp4.is_file() and ruta_mp4.stat().st_size > 10 * 1024 * 1024:
        print(f"-> [YA EXISTE] {ruta_mp4.name} ({ruta_mp4.stat().st_size / (1024**2):.1f} MB) - Omitiendo.")
        return

    proc, es_ffmpeg = crear_proceso_ffmpeg(ruta_mp4, ancho=target_w, alto=target_h, fps=fps)
    print(f"Renderizando {len(rutas_frames):,} frames a Full HD 1080p -> {ruta_mp4.name}...")

    try:
        for p in tqdm(rutas_frames, desc=f"Video {ruta_mp4.stem}", dynamic_ncols=True):
            f = cv2.imread(str(p))
            if f is None:
                continue

            # Escalar con Lanczos para máxima nitidez a 1080p
            if f.shape[1] != target_w or f.shape[0] != target_h:
                f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            if titulo:
                f = agregar_titulo_panel(f, titulo)

            if es_ffmpeg:
                proc.stdin.write(f.tobytes())
            else:
                proc.write(f)
    finally:
        if es_ffmpeg:
            proc.stdin.close()
            proc.wait()
        else:
            proc.release()
    print(f"-> Video 1080p completado exitosamente: {ruta_mp4.name}")


def renderizar_comparativo_dual(rutas_izq, rutas_der, ruta_mp4, titulo_izq, titulo_der, fps=30.0, forzar=False):
    """Renderiza video Side-by-Side (1x2) nítido sin distorsión de relación de aspecto."""
    n_frames = min(len(rutas_izq), len(rutas_der))
    if n_frames == 0:
        return

    if not forzar and ruta_mp4.is_file() and ruta_mp4.stat().st_size > 10 * 1024 * 1024:
        print(f"-> [YA EXISTE] {ruta_mp4.name} ({ruta_mp4.stat().st_size / (1024**2):.1f} MB) - Omitiendo.")
        return

    # Cada panel a 960x540 (relación 16:9 exacta de los frames FLIR)
    ancho_panel = 960
    alto_panel = 540
    ancho_total = 1920
    alto_total = 540

    proc, es_ffmpeg = crear_proceso_ffmpeg(ruta_mp4, ancho=ancho_total, alto=alto_total, fps=fps)
    print(f"Renderizando Comparativo Dual ({n_frames:,} frames) -> {ruta_mp4.name}...")

    try:
        for i in tqdm(range(n_frames), desc="Dual Side-by-Side", dynamic_ncols=True):
            img_izq = cv2.imread(str(rutas_izq[i]))
            img_der = cv2.imread(str(rutas_der[i]))
            if img_izq is None or img_der is None:
                continue

            p_izq = cv2.resize(img_izq, (ancho_panel, alto_panel), interpolation=cv2.INTER_LANCZOS4)
            p_der = cv2.resize(img_der, (ancho_panel, alto_panel), interpolation=cv2.INTER_LANCZOS4)

            p_izq = agregar_titulo_panel(p_izq, titulo_izq)
            p_der = agregar_titulo_panel(p_der, titulo_der)

            canvas = np.hstack([p_izq, p_der])
            if es_ffmpeg:
                proc.stdin.write(canvas.tobytes())
            else:
                proc.write(canvas)
    finally:
        if es_ffmpeg:
            proc.stdin.close()
            proc.wait()
        else:
            proc.release()
    print(f"-> Comparativo Dual completado: {ruta_mp4.name}")


def renderizar_comparativo_triple(rutas_1, rutas_2, rutas_3, ruta_mp4, titulo_1, titulo_2, titulo_3, fps=30.0, forzar=False):
    """Renderiza video Tri-Panel 1x3 sincronizado: [Original | Sin HUD | UDVD Final]."""
    n_frames = min(len(rutas_1), len(rutas_2), len(rutas_3))
    if n_frames == 0:
        return

    if not forzar and ruta_mp4.is_file() and ruta_mp4.stat().st_size > 10 * 1024 * 1024:
        print(f"-> [YA EXISTE] {ruta_mp4.name} ({ruta_mp4.stat().st_size / (1024**2):.1f} MB) - Omitiendo.")
        return

    # Cada panel a 640x360 (relación 16:9 exacta de los frames FLIR)
    ancho_panel = 640
    alto_panel = 360
    ancho_total = 1920
    alto_total = 360

    proc, es_ffmpeg = crear_proceso_ffmpeg(ruta_mp4, ancho=ancho_total, alto=alto_total, fps=fps)
    print(f"Renderizando Tri-Panel 1x3 ({n_frames:,} frames) -> {ruta_mp4.name}...")

    try:
        for i in tqdm(range(n_frames), desc="Triple 1x3 Side-by-Side", dynamic_ncols=True):
            img_1 = cv2.imread(str(rutas_1[i]))
            img_2 = cv2.imread(str(rutas_2[i]))
            img_3 = cv2.imread(str(rutas_3[i]))
            if img_1 is None or img_2 is None or img_3 is None:
                continue

            p1 = cv2.resize(img_1, (ancho_panel, alto_panel), interpolation=cv2.INTER_LANCZOS4)
            p2 = cv2.resize(img_2, (ancho_panel, alto_panel), interpolation=cv2.INTER_LANCZOS4)
            p3 = cv2.resize(img_3, (ancho_panel, alto_panel), interpolation=cv2.INTER_LANCZOS4)

            p1 = agregar_titulo_panel(p1, titulo_1)
            p2 = agregar_titulo_panel(p2, titulo_2)
            p3 = agregar_titulo_panel(p3, titulo_3)

            canvas = np.hstack([p1, p2, p3])
            if es_ffmpeg:
                proc.stdin.write(canvas.tobytes())
            else:
                proc.write(canvas)
    finally:
        if es_ffmpeg:
            proc.stdin.close()
            proc.wait()
        else:
            proc.release()
    print(f"-> Tri-Panel 1x3 completado: {ruta_mp4.name}")


def renderizar_comparativo_2x2(listas_rutas, titulos, ruta_mp4, fps=30.0, forzar=False):
    """Renderiza cuadrícula 2x2 Full HD 1080p."""
    n_frames = min(len(l) for l in listas_rutas)
    if n_frames == 0 or len(listas_rutas) < 4:
        return

    if not forzar and ruta_mp4.is_file() and ruta_mp4.stat().st_size > 10 * 1024 * 1024:
        print(f"-> [YA EXISTE] {ruta_mp4.name} ({ruta_mp4.stat().st_size / (1024**2):.1f} MB) - Omitiendo.")
        return

    ancho_p = 960
    alto_p = 540
    ancho_total = 1920
    alto_total = 1080

    proc, es_ffmpeg = crear_proceso_ffmpeg(ruta_mp4, ancho=ancho_total, alto=alto_total, fps=fps)
    print(f"Renderizando Cuadrícula 2x2 Full HD ({n_frames:,} frames) -> {ruta_mp4.name}...")

    try:
        for i in tqdm(range(n_frames), desc="Comparativo 2x2 1080p", dynamic_ncols=True):
            paneles = []
            for idx in range(4):
                img = cv2.imread(str(listas_rutas[idx][i]))
                if img is None:
                    img = np.zeros((alto_p, ancho_p, 3), dtype=np.uint8)
                else:
                    img = cv2.resize(img, (ancho_p, alto_p), interpolation=cv2.INTER_LANCZOS4)
                    img = agregar_titulo_panel(img, titulos[idx])
                paneles.append(img)

            fila_sup = np.hstack([paneles[0], paneles[1]])
            fila_inf = np.hstack([paneles[2], paneles[3]])
            canvas_2x2 = np.vstack([fila_sup, fila_inf])

            if es_ffmpeg:
                proc.stdin.write(canvas_2x2.tobytes())
            else:
                proc.write(canvas_2x2)
    finally:
        if es_ffmpeg:
            proc.stdin.close()
            proc.wait()
        else:
            proc.release()
    print(f"-> Cuadrícula 2x2 1080p completada: {ruta_mp4.name}")


def main():
    a = argumentos()
    cfg = cargar_json()
    v = cfg[a.video]
    base_video = Path(v["ruta"]).resolve().parent.parent
    dir_salida = Path(a.carpeta_salida).resolve() if a.carpeta_salida else (base_video / "videos_renderizados")
    dir_salida.mkdir(parents=True, exist_ok=True)

    # Identificar carpetas clave
    dir_orig = base_video / "frames_originales"
    dir_sin_hud = base_video / "frames_sin_hud"
    dir_expos = base_video / "expos"

    dir_udvd_oficial = base_video / "udvd_sin_hud"
    if not dir_udvd_oficial.is_dir():
        dir_udvd_oficial = dir_expos / "udvd_sin_hud"
    if not dir_udvd_oficial.is_dir():
        dir_udvd_oficial = dir_expos / "udvd_blindspot_corregido_ult10min"

    candidatos = {
        "original": (dir_orig, "Original (Crudo con HUD)"),
        "sin_hud": (dir_sin_hud, "Sin HUD (ProPainter Inpainted)"),
        "restaurado_final_udvd": (dir_udvd_oficial, "Restaurado Oficial (UDVD + Destriping)"),
        "ensemble": (dir_expos / "ensemble_wavelet_udvd_b2u_sin_hud", "Ensemble Wavelet (UDVD + B2U)"),
    }

    # Revisar carpetas que existan físicamente
    disponibles = {}
    for k, (carpeta, label) in candidatos.items():
        if carpeta.is_dir():
            frames = listar_frames(carpeta, a.max_frames, a.ultimos_frames, a.desde_frame, a.hasta_frame)
            if frames:
                disponibles[k] = (frames, label)

    print(f"=== RECONSTRUCCIÓN DE VIDEOS: {a.video.upper()} ===")
    print(f"Modelos detectados: {list(disponibles.keys())}")
    print(f"Carpeta de salida: {dir_salida}")
    print(f"Sobrescribir forzado: {a.forzar}")

    k_rest = "restaurado_final_udvd" if "restaurado_final_udvd" in disponibles else ("udvd" if "udvd" in disponibles else None)

    # A. Renderizar Videos Individuales Clave (Full HD 1080p Lanczos)
    # Solo renderizamos sin_hud y restaurado_final_udvd para evitar videos redundantes de 20 GB
    if a.tipo in ("individual", "todos"):
        for k in ("sin_hud", "restaurado_final_udvd"):
            if k in disponibles:
                frames, label = disponibles[k]
                out_file = dir_salida / f"{a.video}_{k}.mp4"
                renderizar_video_individual(frames, out_file, fps=a.fps, titulo=label, forzar=a.forzar)

    # B. Renderizar Tri-Panel 1x3 [Original | Sin HUD | UDVD Final]
    if a.tipo in ("comparativo_triple", "todos"):
        if "original" in disponibles and "sin_hud" in disponibles and k_rest:
            out_triple = dir_salida / f"{a.video}_comparativo_triple_1x3.mp4"
            renderizar_comparativo_triple(
                disponibles["original"][0], disponibles["sin_hud"][0], disponibles[k_rest][0],
                out_triple,
                disponibles["original"][1], disponibles["sin_hud"][1], disponibles[k_rest][1],
                fps=a.fps, forzar=a.forzar
            )

    # C. Renderizar Comparativo Dual (Side-by-Side: Original vs Restaurado Final)
    if a.tipo in ("comparativo_dual", "todos"):
        if "original" in disponibles and k_rest:
            out_dual_final = dir_salida / f"{a.video}_comparativo_side_by_side_original_vs_restaurado.mp4"
            renderizar_comparativo_dual(
                disponibles["original"][0], disponibles[k_rest][0],
                out_dual_final, disponibles["original"][1], disponibles[k_rest][1],
                fps=a.fps, forzar=a.forzar
            )

    # D. Renderizar Cuadrícula 2x2 (Solo si se pide explícitamente con --tipo comparativo_2x2)
    if a.tipo == "comparativo_2x2":
        keys_2x2 = ["original", "sin_hud", "udvd"]
        if "ensemble" in disponibles:
            keys_2x2.append("ensemble")
        elif "blind2unblind" in disponibles:
            keys_2x2.append("blind2unblind")

        if len(keys_2x2) == 4 and all(k in disponibles for k in keys_2x2):
            out_2x2 = dir_salida / f"{a.video}_comparativo_cuadricula_2x2.mp4"
            listas = [disponibles[k][0] for k in keys_2x2]
            labels = [disponibles[k][1] for k in keys_2x2]
            renderizar_comparativo_2x2(listas, labels, out_2x2, fps=a.fps, forzar=a.forzar)

    print("\n============================================================")
    print(f"VIDEOS FINALIZADOS CON ÉXITO EN: {dir_salida}")
    print("============================================================")


if __name__ == "__main__":
    main()

