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
    p.add_argument("--video", required=True, help="Identificador del video (video1, video2).")
    p.add_argument("--tipo", choices=("individual", "comparativo_dual", "comparativo_2x2", "todos"), default="todos",
                   help="Tipo de renderizado a ejecutar.")
    p.add_argument("--fps", type=float, default=30.0, help="Fotogramas por segundo (FPS).")
    p.add_argument("--max-frames", type=int, help="Limite maximo de frames a renderizar.")
    p.add_argument("--ultimos-frames", type=int, help="Renderizar exclusivamente los ultimos N frames.")
    p.add_argument("--desde-frame", type=int, help="Indice inicial de frame.")
    p.add_argument("--hasta-frame", type=int, help="Indice final de frame.")
    p.add_argument("--modelos", nargs="+", help="Lista de carpetas o IDs de modelos para el comparativo.")
    p.add_argument("--carpeta-salida", help="Carpeta destino para guardar los videos renderizados.")
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
    banner_h = 42
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    # Linea inferior de acento
    cv2.line(img, (0, banner_h), (w, banner_h), (0, 165, 255), 2)

    # Texto
    texto_completo = f"{titulo}" + (f" | {subtitulo}" if subtitulo else "")
    cv2.putText(img, texto_completo, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def renderizar_video_individual(rutas_frames, ruta_mp4, fps=30.0, titulo=None):
    """Renderiza una secuencia de frames en un archivo MP4."""
    if not rutas_frames:
        print(f"Advertencia: No hay frames para renderizar en {ruta_mp4}")
        return

    primero = cv2.imread(str(rutas_frames[0]))
    h, w, _ = primero.shape
    
    ruta_mp4.parent.mkdir(parents=True, exist_ok=True)
    temp_avi = ruta_mp4.with_suffix(".temp.avi")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(temp_avi), fourcc, fps, (w, h))

    print(f"Renderizando {len(rutas_frames)} frames -> {ruta_mp4.name}...")
    for p in tqdm(rutas_frames, desc=f"Video {ruta_mp4.stem}", dynamic_ncols=True):
        f = cv2.imread(str(p))
        if f is None:
            continue
        if titulo:
            f = agregar_titulo_panel(f, titulo)
        writer.write(f)

    writer.release()

    # Convertir a MP4 H.264 optimizado con FFmpeg si esta disponible
    convertir_h264(temp_avi, ruta_mp4)


def renderizar_comparativo_dual(rutas_izq, rutas_der, ruta_mp4, titulo_izq, titulo_der, fps=30.0):
    """Renderiza video Side-by-Side (1x2) sincronizado."""
    n_frames = min(len(rutas_izq), len(rutas_der))
    if n_frames == 0:
        return

    f_izq = cv2.imread(str(rutas_izq[0]))
    h, w, _ = f_izq.shape

    # Redimensionar al 50% de ancho cada uno para mantener relacion de aspecto 16:9 global
    ancho_panel = w // 2
    alto_panel = h // 2

    ancho_total = ancho_panel * 2
    alto_total = alto_panel

    ruta_mp4.parent.mkdir(parents=True, exist_ok=True)
    temp_avi = ruta_mp4.with_suffix(".temp.avi")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(temp_avi), fourcc, fps, (ancho_total, alto_total))

    print(f"Renderizando Comparativo Dual ({n_frames} frames) -> {ruta_mp4.name}...")
    for i in tqdm(range(n_frames), desc="Dual Side-by-Side", dynamic_ncols=True):
        img_izq = cv2.imread(str(rutas_izq[i]))
        img_der = cv2.imread(str(rutas_der[i]))
        if img_izq is None or img_der is None:
            continue

        img_izq = agregar_titulo_panel(img_izq, titulo_izq)
        img_der = agregar_titulo_panel(img_der, titulo_der)

        p_izq = cv2.resize(img_izq, (ancho_panel, alto_panel), interpolation=cv2.INTER_AREA)
        p_der = cv2.resize(img_der, (ancho_panel, alto_panel), interpolation=cv2.INTER_AREA)

        canvas = np.hstack([p_izq, p_der])
        writer.write(canvas)

    writer.release()
    convertir_h264(temp_avi, ruta_mp4)


def renderizar_comparativo_2x2(listas_rutas, titulos, ruta_mp4, fps=30.0):
    """
    Renderiza cuadricula 2x2 sincronizada:
    [0: Superior Izquierda]  [1: Superior Derecha]
    [2: Inferior Izquierda]  [3: Inferior Derecha]
    """
    n_frames = min(len(l) for l in listas_rutas)
    if n_frames == 0 or len(listas_rutas) < 4:
        return

    f0 = cv2.imread(str(listas_rutas[0][0]))
    h, w, _ = f0.shape

    # Cada panel al 50% de resolucion
    ancho_p = w // 2
    alto_p = h // 2
    ancho_total = ancho_p * 2
    alto_total = alto_p * 2

    ruta_mp4.parent.mkdir(parents=True, exist_ok=True)
    temp_avi = ruta_mp4.with_suffix(".temp.avi")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(temp_avi), fourcc, fps, (ancho_total, alto_total))

    print(f"Renderizando Comparativo 2x2 ({n_frames} frames) -> {ruta_mp4.name}...")
    for i in tqdm(range(n_frames), desc="Comparativo 2x2", dynamic_ncols=True):
        paneles = []
        for idx in range(4):
            img = cv2.imread(str(listas_rutas[idx][i]))
            img = agregar_titulo_panel(img, titulos[idx])
            p_res = cv2.resize(img, (ancho_p, alto_p), interpolation=cv2.INTER_AREA)
            paneles.append(p_res)

        fila_sup = np.hstack([paneles[0], paneles[1]])
        fila_inf = np.hstack([paneles[2], paneles[3]])
        canvas_2x2 = np.vstack([fila_sup, fila_inf])

        writer.write(canvas_2x2)

    writer.release()
    convertir_h264(temp_avi, ruta_mp4)


def convertir_h264(temp_avi, ruta_mp4):
    """Convierte el AVI temporal a MP4 H.264 de alta calidad compatible universalmente."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        comando = [
            ffmpeg, "-y", "-i", str(temp_avi),
            "-c:v", "libx264", "-crf", "19", "-preset", "fast",
            "-pix_fmt", "yuv420p", str(ruta_mp4)
        ]
        try:
            subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            temp_avi.unlink(missing_ok=True)
            print(f"-> Video final guardado exitosamente: {ruta_mp4}")
            return
        except Exception:
            pass
    
    # Fallback si ffmpeg no esta instalado
    if temp_avi.is_file():
        temp_avi.replace(ruta_mp4)
        print(f"-> Video guardado como: {ruta_mp4}")


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

    # 1. Buscar modelos disponibles
    candidatos = {
        "original": (dir_orig, "Original (Crudo con HUD)"),
        "sin_hud": (dir_sin_hud, "Sin HUD (ProPainter Inpainted)"),
        "udvd": (dir_expos / "udvd_sin_hud", "UDVD (Dynamic Kernels) [Campeon]"),
        "blind2unblind": (dir_expos / "blind2unblind_sin_hud", "Blind2Unblind (Espacial 2D)"),
        "ensemble": (dir_expos / "ensemble_wavelet_udvd_b2u_sin_hud", "Ensemble Wavelet (UDVD + B2U)"),
        "struct_n2v": (dir_expos / "struct_n2v_vert_sinhud", "StructN2V Vertical (Anti-FPN)"),
    }

    # Revisar carpetas que existan fisicamente
    disponibles = {}
    for k, (carpeta, label) in candidatos.items():
        if carpeta.is_dir():
            frames = listar_frames(carpeta, a.max_frames, a.ultimos_frames, a.desde_frame, a.hasta_frame)
            if frames:
                disponibles[k] = (frames, label)

    print(f"=== RECONSTRUCCION DE VIDEOS: {a.video.upper()} ===")
    print(f"Modelos detectados: {list(disponibles.keys())}")
    print(f"Carpeta de salida: {dir_salida}")

    # A. Renderizar Videos Individuales
    if a.tipo in ("individual", "todos"):
        for k, (frames, label) in disponibles.items():
            out_file = dir_salida / f"{a.video}_{k}.mp4"
            renderizar_video_individual(frames, out_file, fps=a.fps, titulo=label)

    # B. Renderizar Comparativo Dual (Original vs Sin HUD, y Sin HUD vs UDVD)
    if a.tipo in ("comparativo_dual", "todos"):
        if "original" in disponibles and "sin_hud" in disponibles:
            out_dual_1 = dir_salida / f"{a.video}_comparativo_original_vs_sin_hud.mp4"
            renderizar_comparativo_dual(
                disponibles["original"][0], disponibles["sin_hud"][0],
                out_dual_1, disponibles["original"][1], disponibles["sin_hud"][1], fps=a.fps
            )

        if "sin_hud" in disponibles and "udvd" in disponibles:
            out_dual_2 = dir_salida / f"{a.video}_comparativo_sin_hud_vs_udvd.mp4"
            renderizar_comparativo_dual(
                disponibles["sin_hud"][0], disponibles["udvd"][0],
                out_dual_2, disponibles["sin_hud"][1], disponibles["udvd"][1], fps=a.fps
            )

    # C. Renderizar Cuadricula 2x2 (Original | Sin HUD | UDVD | Ensamble o B2U)
    if a.tipo in ("comparativo_2x2", "todos"):
        keys_2x2 = ["original", "sin_hud", "udvd"]
        if "ensemble" in disponibles:
            keys_2x2.append("ensemble")
        elif "blind2unblind" in disponibles:
            keys_2x2.append("blind2unblind")
        elif "struct_n2v" in disponibles:
            keys_2x2.append("struct_n2v")

        if len(keys_2x2) == 4 and all(k in disponibles for k in keys_2x2):
            out_2x2 = dir_salida / f"{a.video}_comparativo_cuadricula_2x2.mp4"
            listas = [disponibles[k][0] for k in keys_2x2]
            labels = [disponibles[k][1] for k in keys_2x2]
            renderizar_comparativo_2x2(listas, labels, out_2x2, fps=a.fps)

    print("\n============================================================")
    print(f"TODOS LOS VIDEOS RENDERIZADOS CON EXITO EN: {dir_salida}")
    print("============================================================")


if __name__ == "__main__":
    main()
