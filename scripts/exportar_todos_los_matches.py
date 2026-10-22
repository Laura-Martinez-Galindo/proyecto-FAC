#!/usr/bin/env python3
"""
==============================================================================
EXPORTADOR MAESTRO DE MATCHES: DATASETS vs. VIDEOS CRUDOS DE LA FAC (30 FPS)
==============================================================================
Mapea el 100% de los fotogramas anotados del Dataset (1,701 frames)
hacia los fotogramas originales a 30 FPS de Video 1, Video 2 y Video 3.

Organización de subcarpetas en visual_matches/ solicitada:
  1. video1        -> Fotogramas normales de Video 1 (video_11min_*.jpg)
  2. video1_scenes -> Escenas continuas en 1080p (frame_XXXXXX_sceneXXX.jpg)
  3. video2        -> Fotogramas normales de Video 2 (video_13min_*.jpg)
  4. video2_scenes -> Escenas continuas en 720p (frame_XXXXXX_sceneXXX.jpg)
  5. video3        -> Fotogramas normales de Video 3 (frame_XXXX.jpg)
  6. video3_scenes -> Escenas continuas de Video 3 (si existen)

Soporte dual de extracción:
  - Lee fotogramas PNG individuales si existen en disco.
  - Si no existen como PNG suelto (ej. Video 3 donde solo hay 100 PNGs en Drive
    porque el paquete completo pesa 114 GB), extrae al vuelo el fotograma
    directamente desde el video MP4 renderizado (video3_original.mp4).

NO REQUIERE GPU: Se ejecuta en CPU con multihilo y bajo consumo de memoria.
==============================================================================
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import threading
import zipfile

# Asegurar codificacion UTF-8 segura en Windows y Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import pandas as pd

# Directorio raiz del proyecto
RAIZ = Path(__file__).resolve().parent.parent
if not (RAIZ / "datasets").is_dir() and (RAIZ.parent / "datasets").is_dir():
    RAIZ = RAIZ.parent

# Mapeo de Clases Oficiales del Reto FLIR / FAC
CLASES_MAP = {
    0: "Vehiculo",
    1: "Edificio_Bodega",
    2: "Camino_Carretera",
    3: "Rio",
    4: "Mineria_Ilegal_SDZI",
}

# Paleta de colores BGR para bounding boxes por clase
COLORES_BBOX = {
    0: (0, 165, 255),    # Naranja: Vehículo
    1: (255, 100, 0),    # Azul: Edificio
    2: (0, 255, 255),    # Amarillo: Camino
    3: (255, 200, 0),    # Cyan/Celeste: Río
    4: (0, 0, 255),      # Rojo: Minería Ilegal
}


def formatear_tiempo(segundos):
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"


def resolver_origen_y_frame(nombre_archivo, ancho_img=None, alto_img=None):
    """
    Determina de forma determinística el video fuente, segundo de vuelo,
    índice de fotograma crudo (30 FPS), subcarpeta de salida y nombre esperado.
    """
    stem = Path(nombre_archivo).stem

    # Caso A: Video 1 Normal (Dataset Codefest Tolima - FAC 5749, 13/06/2021, 1080p)
    if "video_11min" in stem:
        partes = stem.split("_")
        sec_idx = int(partes[-1]) if partes[-1].isdigit() else 0
        frame_idx = sec_idx * 30
        return {
            "video_origen": "video1",
            "subcarpeta_salida": "video1",
            "v_busqueda": "video1",
            "mision": "Tolima (FAC 5749 - 13/06/2021)",
            "segundo_vuelo": sec_idx,
            "tiempo_mm_ss": formatear_tiempo(sec_idx),
            "frame_crudo_30fps": frame_idx,
            "indice_frame_0based": frame_idx,
            "nombre_frame_esperado": f"frame_{frame_idx:05d}.png",
            "resolucion_esperada": "1920x1080",
            "tipo_corte": "normal_1fps",
            "pad_digitos": 5,
            "nota": "Vuelo Codefest Tolima (FAC 5749). Distinto al video30min-11to22 (FAC 5748) en Hypatia/Drive."
        }

    # Caso B: Video 2 Normal (Últimos 13 min de 46 min - Santander, FLIR Systems, 720p)
    if "video_13min" in stem:
        partes = stem.split("_")
        sec_idx = int(partes[-1]) if partes[-1].isdigit() else 0
        # Offset verificado: Minuto 32:56 = segundo 1,976 = frame 59,280 (100% exacto, MAE = 0.00)
        frame_idx = 59280 + (sec_idx * 30)
        segundo_absoluto = 1976 + sec_idx
        return {
            "video_origen": "video2",
            "subcarpeta_salida": "video2",
            "v_busqueda": "video2",
            "mision": "Santander (FLIR Systems)",
            "segundo_vuelo": segundo_absoluto,
            "tiempo_mm_ss": formatear_tiempo(segundo_absoluto),
            "frame_crudo_30fps": frame_idx,
            "indice_frame_0based": frame_idx,
            "nombre_frame_esperado": f"frame_{frame_idx:05d}.png",
            "resolucion_esperada": "1280x720",
            "tipo_corte": "normal_1fps",
            "pad_digitos": 5,
            "nota": "Match 100% verificado contra VideoCodefest_008-46min.avi (MAE = 0.00)"
        }

    # Caso C: Video 3 Normal (Río Telembí, Nariño, 19/10/2021, 1080p, formato 4 dígitos frame_XXXX)
    if re.match(r"^frame_\d{4}$", stem):
        num_str = stem.replace("frame_", "")
        idx_num = int(num_str) if num_str.isdigit() else 0
        # Coincidencia matemática confirmada por matching visual exacto:
        # Segmento A (idx <= 13): frame_0based = 3781 + (idx * 60) -> 2s por frame
        # Segmento B (idx >= 69): frame_0based = 5749 + (idx * 60) -> Salto constante de +1968 frames
        if idx_num <= 13:
            f0 = 3781 + (idx_num * 60)
        else:
            f0 = 5749 + (idx_num * 60)
        
        f_1based = f0 + 1
        seg_abs = f0 / 30.0
        return {
            "video_origen": "video3",
            "subcarpeta_salida": "video3",
            "v_busqueda": "video3",
            "mision": "Narino Telembi (19/10/2021)",
            "segundo_vuelo": round(seg_abs, 2),
            "tiempo_mm_ss": formatear_tiempo(seg_abs),
            "frame_crudo_30fps": f_1based,
            "indice_frame_0based": f0,
            "nombre_frame_esperado": f"frame_{f_1based:06d}.png",
            "resolucion_esperada": "1920x1080",
            "tipo_corte": "normal_muestreo_2s",
            "pad_digitos": 6,
            "nota": "Match comprobado en telemetria (ancla f=3782 a las 15:44:11 Z, MAE ~ 1.5, PSNR 35.4 dB)"
        }

    # Caso D: Cortes de Escena Continua (frame_XXXXXX_sceneXXX o v1_frame_XXXXXX_sceneXXX)
    if "scene" in stem:
        nums = re.findall(r"\d+", stem)
        sec_corte = int(nums[0]) if len(nums) >= 1 else 0
        sc_id = nums[1] if len(nums) >= 2 else "000"
        # El número inicial del archivo es el segundo del video nativo a 30 FPS:
        frame_idx = sec_corte * 30

        # Discriminación por resolución conocida
        es_1080 = (ancho_img == 1920 and alto_img == 1080) if ancho_img else ("v1_" in stem)
        if es_1080:
            v_orig = "video1_scenes"
            sub_salida = "video1_scenes"
            v_busq = "video1"
            mision = "Narino Telembi (FAC 5748 - 14/05/2021)"
            pad = 5
            res_str = "1920x1080"
            nota_s = "Match comprobado contra video30min-11to22.mp4 (MAE < 1.0)"
        else:
            v_orig = "video2_scenes"
            sub_salida = "video2_scenes"
            v_busq = "video2"
            mision = "Santander (FLIR Systems - Escena)"
            pad = 5
            res_str = "1280x720"
            nota_s = "Escena continua Santander (720p)"

        return {
            "video_origen": v_orig,
            "subcarpeta_salida": sub_salida,
            "v_busqueda": v_busq,
            "mision": mision,
            "segundo_vuelo": sec_corte,
            "tiempo_mm_ss": formatear_tiempo(sec_corte),
            "frame_crudo_30fps": frame_idx,
            "indice_frame_0based": frame_idx,
            "nombre_frame_esperado": f"frame_{frame_idx:0{pad}d}.png",
            "resolucion_esperada": res_str,
            "tipo_corte": f"corte_escena_{sc_id}",
            "pad_digitos": pad,
            "nota": nota_s
        }

    # Caso E: Genérico
    nums = re.findall(r"\d+", stem)
    idx = int(nums[-1]) if nums else 0
    return {
        "video_origen": "desconocido",
        "subcarpeta_salida": "desconocido",
        "v_busqueda": "desconocido",
        "mision": "No identificada",
        "segundo_vuelo": idx,
        "tiempo_mm_ss": formatear_tiempo(idx),
        "frame_crudo_30fps": idx,
        "indice_frame_0based": idx,
        "nombre_frame_esperado": f"frame_{idx:05d}.png",
        "resolucion_esperada": "desconocida",
        "tipo_corte": "generico",
        "pad_digitos": 5,
        "nota": ""
    }


def parsear_etiqueta_yolo(txt_bytes_or_str, ancho, alto):
    """Parsea bounding boxes de formato YOLO a coordenadas absolutas de píxel."""
    bboxes = []
    lineas = txt_bytes_or_str.splitlines()
    for linea in lineas:
        if isinstance(linea, bytes):
            linea = linea.decode("utf-8", errors="ignore")
        partes = linea.strip().split()
        if len(partes) >= 5:
            try:
                cls_id = int(partes[0])
                xc = float(partes[1]) * ancho
                yc = float(partes[2]) * alto
                w = float(partes[3]) * ancho
                h = float(partes[4]) * alto
                x1 = int(round(xc - w / 2))
                y1 = int(round(yc - h / 2))
                x2 = int(round(xc + w / 2))
                y2 = int(round(yc + h / 2))
                bboxes.append((cls_id, x1, y1, x2, y2))
            except ValueError:
                continue
    return bboxes


def dibujar_anotaciones(img_bgr, bboxes):
    """Dibuja bounding boxes y nombres de clases sobre la imagen."""
    canvas = img_bgr.copy()
    for cls_id, x1, y1, x2, y2 in bboxes:
        color = COLORES_BBOX.get(cls_id, (0, 255, 0))
        nom_cls = CLASES_MAP.get(cls_id, f"Clase_{cls_id}")
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(canvas, (x1, max(0, y1 - 22)), (x1 + len(nom_cls) * 11 + 10, y1), color, -1)
        cv2.putText(canvas, nom_cls, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def construir_triptico_match(img_dataset, img_crudo, info):
    """
    Construye un tríptico visual de verificación:
    [ Panel 1: Dataset con Anotaciones | Panel 2: Frame Crudo (Drive/Hypatia) | Panel 3: Mapa MAE ]
    """
    h1, w1 = img_dataset.shape[:2]
    h2, w2 = img_crudo.shape[:2]

    # Igualar resolución para calcular diferencia pixel a pixel
    if (w1, h1) != (w2, h2):
        img_crudo_res = cv2.resize(img_crudo, (w1, h1), interpolation=cv2.INTER_LINEAR)
    else:
        img_crudo_res = img_crudo

    diff = cv2.absdiff(img_dataset, img_crudo_res)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mae = float(np.mean(diff_gray))

    # Mapa de error con amplificación
    diff_amp = np.clip(diff_gray.astype(np.float32) * 5.0, 0, 255).astype(np.uint8)
    diff_col = cv2.applyColorMap(diff_amp, cv2.COLORMAP_INFERNO)

    margen_sup = 65
    ancho_tot = w1 * 3
    alto_tot = h1 + margen_sup

    canvas = np.zeros((alto_tot, ancho_tot, 3), dtype=np.uint8)
    canvas.fill(22)

    canvas[margen_sup:, 0:w1] = img_dataset
    canvas[margen_sup:, w1:2*w1] = img_crudo_res
    canvas[margen_sup:, 2*w1:] = diff_col

    cv2.line(canvas, (w1, 0), (w1, alto_tot), (70, 70, 70), 2)
    cv2.line(canvas, (2*w1, 0), (2*w1, alto_tot), (70, 70, 70), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"1. DATASET: {info['archivo_dataset']} ({info['split'].upper()})", (16, 38), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"2. VIDEO CRUDO: {info['video_origen'].upper()} | #{info['frame_crudo_30fps']} ({info['tiempo_mm_ss']})", (w1 + 16, 38), font, 0.75, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"3. ERROR DIFERENCIAL (MAE = {mae:.2f})", (2*w1 + 16, 38), font, 0.75, (100, 200, 255), 2, cv2.LINE_AA)

    sub_txt = f"Mision: {info['mision']} | Fuente: {info.get('fuente', 'Disco')} | BBoxes: {info['num_anotaciones']}"
    cv2.putText(canvas, sub_txt, (16, alto_tot - 12), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas, mae


def cargar_dataset(ruta_ds):
    """Carga todas las rutas de imágenes y etiquetas desde un .zip o carpeta."""
    items = []
    p = Path(ruta_ds).resolve()

    if p.is_file() and p.suffix.lower() == ".zip":
        print(f"[PAQUETE] Leyendo archivo ZIP comprimido: {p.name}")
        z = zipfile.ZipFile(p, "r")
        archivos = z.namelist()
        imgs = [n for n in archivos if n.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        for img_p in imgs:
            partes = img_p.split("/")
            split = "train" if "train" in partes else ("val" if "val" in partes else ("test" if "test" in partes else "general"))
            lbl_candidate = img_p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            
            lbl_content = ""
            if lbl_candidate in archivos:
                try:
                    lbl_content = z.read(lbl_candidate).decode("utf-8", errors="ignore")
                except Exception:
                    pass

            items.append({
                "nombre": Path(img_p).name,
                "split": split,
                "es_zip": True,
                "zip_obj": z,
                "ruta_zip": img_p,
                "ruta_disco": None,
                "lbl_content": lbl_content
            })
    elif p.is_dir():
        print(f"[DIR] Escaneando directorio de dataset: {p}")
        imgs = list(p.rglob("*.jpg")) + list(p.rglob("*.png"))
        for img_p in imgs:
            partes = [x.lower() for x in img_p.parts]
            split = "train" if "train" in partes else ("val" if "val" in partes else ("test" if "test" in partes else "general"))
            lbl_p = img_p.parent.parent / "labels" / f"{img_p.stem}.txt"
            if not lbl_p.is_file():
                lbl_p = img_p.parent / f"{img_p.stem}.txt"

            lbl_content = ""
            if lbl_p.is_file():
                try:
                    lbl_content = lbl_p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass

            items.append({
                "nombre": img_p.name,
                "split": split,
                "es_zip": False,
                "zip_obj": None,
                "ruta_zip": None,
                "ruta_disco": img_p,
                "lbl_content": lbl_content
            })

    return items


def main():
    parser = argparse.ArgumentParser(
        description="Mapeo 1-a-1 y Exportación Visual de Matches: Datasets FLIR vs Videos 30 FPS"
    )
    parser.add_argument("--dataset", default=None, help="Ruta al ZIP o carpeta del dataset (default: auto-detectar en datasets/)")
    parser.add_argument("--videos-dir", default=None, help="Ruta a la carpeta con videos/video{1,2,3}/ (default: auto-detectar)")
    parser.add_argument("--salida", default="resultados_matching", help="Directorio de salida para CSV, Excel e imágenes")
    parser.add_argument("--exportar-visual", action="store_true", default=True, help="Genera trípticos visuales si los frames crudos existen")
    parser.add_argument("--max-visual-por-video", type=int, default=50, help="Máximo de imágenes visuales por subcarpeta (0 = todas)")
    parser.add_argument("--cpus", type=int, default=min(16, os.cpu_count() or 4), help="Número de hilos para procesamiento multihilo")
    args = parser.parse_args()

    print("=" * 90)
    print("      MAPEO Y EXPORTACIÓN MAESTRA DE MATCHES: DATASETS -> VIDEOS CRUDOS FAC")
    print(f"      Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Hilos CPU: {args.cpus}")
    print("=" * 90)

    # 1. Localizar dataset
    ruta_ds = args.dataset
    if not ruta_ds:
        candidatos_ds = [
            RAIZ / "datasets/Dataset_Balanceado.zip",
            RAIZ / "datasets/Dataset_Balanceado",
            RAIZ / "datasets/dataset_split_completo.zip",
            RAIZ / "datasets/dataset_split_completo",
        ]
        for c in candidatos_ds:
            if c.exists():
                ruta_ds = c
                break

    if not ruta_ds or not Path(ruta_ds).exists():
        print(f"[ERROR] No se encontró ningún dataset en {RAIZ / 'datasets'}. Especifica --dataset <ruta>.")
        sys.exit(1)

    # 2. Localizar carpetas de frames y videos MP4
    dir_videos = Path(args.videos_dir).resolve() if args.videos_dir else (RAIZ / "videos")
    carpetas_frames = {}
    videos_mp4 = {}

    for v_k in ["video1", "video2", "video3"]:
        # Carpetas de imágenes sueltas (PNG / JPG)
        cands_dir = [
            dir_videos / v_k / "frames_originales",
            dir_videos / v_k / "frames_sin_hud",
            dir_videos / v_k,
        ]
        p_elegida = next((c for c in cands_dir if c.is_dir() and any(p.suffix.lower() in [".png", ".jpg", ".jpeg"] for p in c.iterdir())), None)
        if not p_elegida:
            p_elegida = dir_videos / v_k / "frames_originales"
        carpetas_frames[v_k] = p_elegida

        # Videos MP4 renderizados (para extracción al vuelo si no hay PNGs sueltos)
        cands_mp4 = [
            dir_videos / v_k / "videos_renderizados" / f"{v_k}_original.mp4",
            dir_videos / v_k / "videos_renderizados" / f"{v_k}.mp4",
            dir_videos / v_k / f"{v_k}_original.mp4",
            dir_videos / v_k / f"{v_k}.mp4",
            dir_videos / v_k / "video30min-11to22.mp4",
            dir_videos / v_k / "VideoCodefest_008-46min.avi",
        ]
        # Búsqueda adicional en subdirectorios
        if (dir_videos / v_k).is_dir():
            for f in (dir_videos / v_k).glob("**/*.mp4"):
                if f.is_file() and f not in cands_mp4:
                    cands_mp4.append(f)
            for f in (dir_videos / v_k).glob("**/*.avi"):
                if f.is_file() and f not in cands_mp4:
                    cands_mp4.append(f)

        p_mp4_elegido = next((c for c in cands_mp4 if c.is_file()), None)
        if p_mp4_elegido:
            videos_mp4[v_k] = p_mp4_elegido

    print("\n[INFO] Estado de recursos de video crudo detectados en el entorno:")
    for v_id in ["video1", "video2", "video3"]:
        p_frames = carpetas_frames.get(v_id)
        p_mp4 = videos_mp4.get(v_id)
        print(f"  • {v_id.upper()}:")
        if p_frames and p_frames.is_dir():
            ejemplos = [p.name for i, p in enumerate(p_frames.iterdir()) if i < 3 and p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            if ejemplos:
                print(f"       -> Carpeta PNGs: {p_frames} (Muestra: {ejemplos})")
            else:
                print(f"       -> Carpeta PNGs: {p_frames} (sin imágenes sueltas)")
        else:
            print(f"       -> Carpeta PNGs: No encontrada ({p_frames})")

        if p_mp4:
            print(f"       -> Video renderizado: {p_mp4.name} (soporte extracción directa al vuelo ACTIVO)")
        else:
            print(f"       -> Video renderizado: No detectado")

    # 3. Cargar imágenes del dataset
    items_ds = cargar_dataset(ruta_ds)
    print(f"\n[LISTA] Total de imágenes registradas en el dataset: {len(items_ds):,}")

    # 4. Crear directorios de salida organizados por subcarpeta
    dir_salida = (RAIZ / args.salida).resolve()
    dir_salida.mkdir(parents=True, exist_ok=True)
    dir_vis = dir_salida / "visual_matches"
    dir_vis.mkdir(parents=True, exist_ok=True)

    subcarpetas_oficiales = [
        "video1",
        "video1_scenes",
        "video2",
        "video2_scenes",
        "video3",
        "video3_scenes",
    ]
    for sub in subcarpetas_oficiales:
        (dir_vis / sub).mkdir(exist_ok=True)

    conteo_vis = {sub: 0 for sub in subcarpetas_oficiales}
    lock_vis = threading.Lock()

    print("\n[RUN] Ejecutando mapeo matemático y generación de trípticos...")

    def procesar_item(item):
        nom = item["nombre"]
        split = item["split"]
        lbl_str = item["lbl_content"]

        # Leer imagen de dataset
        img_bgr = None
        if item["es_zip"]:
            try:
                data = item["zip_obj"].read(item["ruta_zip"])
                arr = np.frombuffer(data, np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                pass
        else:
            img_bgr = cv2.imread(str(item["ruta_disco"]))

        h, w = (img_bgr.shape[0], img_bgr.shape[1]) if img_bgr is not None else (None, None)

        # Resolver fórmula de match determinística
        info_match = resolver_origen_y_frame(nom, ancho_img=w, alto_img=h)
        v_orig = info_match["video_origen"]
        sub_salida = info_match["subcarpeta_salida"]
        v_busq = info_match["v_busqueda"]
        f_1based = info_match["frame_crudo_30fps"]
        f_0based = info_match["indice_frame_0based"]
        pad = info_match["pad_digitos"]

        # Parsear bboxes de anotaciones
        bboxes = parsear_etiqueta_yolo(lbl_str, w or 1920, h or 1080)
        clases_detectadas = sorted(list(set(CLASES_MAP.get(b[0], str(b[0])) for b in bboxes)))

        # Intentar leer el fotograma crudo:
        # Paso 1: Buscar imagen suelta en disco (PNG/JPG)
        img_crudo = None
        fuente_cruda = None
        existe_en_disco = False

        if v_busq in carpetas_frames and carpetas_frames[v_busq].is_dir():
            candidatos = [
                info_match["nombre_frame_esperado"],
                f"frame_{f_1based:0{pad}d}.png",
                f"frame_{f_1based:06d}.png",
                f"frame_{f_1based:05d}.png",
                f"frame_{f_1based}.png",
                f"frame_{f_1based:0{pad}d}.jpg",
                f"frame_{f_1based:06d}.jpg",
                f"frame_{f_1based:05d}.jpg",
                f"frame_{f_1based}.jpg",
                f"frame_{f_0based:0{pad}d}.png",
                f"frame_{f_0based:06d}.png",
                f"frame_{f_0based:05d}.png",
                f"frame_{f_0based}.png",
            ]
            for c in candidatos:
                cand_p = carpetas_frames[v_busq] / c
                if cand_p.is_file():
                    img_crudo = cv2.imread(str(cand_p))
                    if img_crudo is not None:
                        fuente_cruda = f"PNG: {cand_p.name}"
                        existe_en_disco = True
                        break

        # Paso 2: Si no hay PNG suelto (ej. Video 3 en Drive), extraer al vuelo del MP4
        if not existe_en_disco and v_busq in videos_mp4 and videos_mp4[v_busq].is_file():
            try:
                cap = cv2.VideoCapture(str(videos_mp4[v_busq]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_0based)
                ret, fr = cap.read()
                cap.release()
                if ret and fr is not None:
                    img_crudo = fr
                    fuente_cruda = f"MP4: {videos_mp4[v_busq].name}#f{f_0based}"
                    existe_en_disco = True
            except Exception:
                pass

        mae_resultado = None

        # Exportación visual si está habilitada
        if args.exportar_visual and existe_en_disco and img_bgr is not None and img_crudo is not None:
            limite = args.max_visual_por_video
            with lock_vis:
                puede_exportar = (limite == 0 or conteo_vis.get(sub_salida, 0) < limite)
                if puede_exportar:
                    conteo_vis[sub_salida] = conteo_vis.get(sub_salida, 0) + 1

            if puede_exportar:
                img_con_bboxes = dibujar_anotaciones(img_bgr, bboxes)
                info_panel = {
                    "archivo_dataset": nom,
                    "split": split,
                    "video_origen": v_orig,
                    "frame_crudo_30fps": f_1based,
                    "tiempo_mm_ss": info_match["tiempo_mm_ss"],
                    "mision": info_match["mision"],
                    "archivo_frame_esperado": info_match["nombre_frame_esperado"],
                    "num_anotaciones": len(bboxes),
                    "fuente": fuente_cruda
                }
                triptico, mae = construir_triptico_match(img_con_bboxes, img_crudo, info_panel)
                mae_resultado = mae

                out_png = dir_vis / sub_salida / f"match_{nom.replace('.jpg', '')}_f{f_1based}.jpg"
                cv2.imwrite(str(out_png), triptico, [cv2.IMWRITE_JPEG_QUALITY, 88])

        return {
            "archivo_dataset": nom,
            "split": split,
            "video_origen": v_orig,
            "subcarpeta_salida": sub_salida,
            "mision": info_match["mision"],
            "segundo_vuelo": info_match["segundo_vuelo"],
            "tiempo_mm_ss": info_match["tiempo_mm_ss"],
            "frame_crudo_30fps": f_1based,
            "indice_frame_0based": f_0based,
            "archivo_frame_esperado": info_match["nombre_frame_esperado"],
            "resolucion_esperada": info_match["resolucion_esperada"],
            "tipo_corte": info_match["tipo_corte"],
            "num_anotaciones": len(bboxes),
            "clases_presentes": ", ".join(clases_detectadas) if clases_detectadas else "Ninguna",
            "frame_crudo_en_disco": existe_en_disco,
            "fuente_frame_crudo": fuente_cruda or "No_encontrado",
            "mae_diferencia": round(mae_resultado, 2) if mae_resultado is not None else None,
            "nota_auditoria": info_match.get("nota", "")
        }

    with ThreadPoolExecutor(max_workers=args.cpus) as pool:
        registros = list(pool.map(procesar_item, items_ds))

    # 5. Construir DataFrame y Guardar CSV / Excel
    df = pd.DataFrame(registros)
    
    csv_salida = dir_salida / "mapeo_completo_matches.csv"
    df.to_csv(csv_salida, index=False, encoding="utf-8-sig")
    print(f"\n[GUARDADO] CSV de Mapeo Exhaustivo guardado en:\n   -> {csv_salida}")

    xlsx_salida = dir_salida / "resumen_matches.xlsx"
    with pd.ExcelWriter(xlsx_salida, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Todos_los_Matches", index=False)
        
        # Hoja Resumen por Video y Subcarpeta
        resumen_vid = df.groupby(["subcarpeta_salida", "video_origen", "mision"]).agg(
            total_imagenes=("archivo_dataset", "count"),
            train=("split", lambda s: (s == "train").sum()),
            val=("split", lambda s: (s == "val").sum()),
            test=("split", lambda s: (s == "test").sum()),
            frames_encontrados=("frame_crudo_en_disco", "sum"),
            anotaciones_totales=("num_anotaciones", "sum")
        ).reset_index()
        resumen_vid.to_excel(writer, sheet_name="Resumen_por_Subcarpeta", index=False)

    print(f"[EXCEL] Libro Excel resumen guardado en:\n   -> {xlsx_salida}")

    # 6. Resumen de resultados en consola
    print("\n" + "=" * 90)
    print("                    RESUMEN DE RESULTADOS POR SUBCARPETAS")
    print("=" * 90)
    for sub, grp in df.groupby("subcarpeta_salida"):
        print(f"[CARPETA] visual_matches/{sub}/ -> {len(grp):,} imágenes anotadas")
        print(f"   • Misión: {grp['mision'].iloc[0]}")
        print(f"   • Train: {(grp['split'] == 'train').sum():,} | Val: {(grp['split'] == 'val').sum():,} | Test: {(grp['split'] == 'test').sum():,}")
        print(f"   • Encontrados en disco (30 FPS): {grp['frame_crudo_en_disco'].sum():,} / {len(grp):,}")
        if conteo_vis.get(sub, 0) > 0:
            print(f"   • Trípticos visuales exportados: {conteo_vis[sub]} imágenes en {dir_vis / sub}")
        else:
            print(f"   • Trípticos visuales exportados: 0")

    print("\n" + "=" * 90)
    print("[OK] Proceso finalizado exitosamente.")
    print(f"Carpetas generadas en: {dir_salida}")
    print("=" * 90)


if __name__ == "__main__":
    main()
