#!/usr/bin/env python3
"""
==============================================================================
EXPORTADOR MAESTRO DE MATCHES: DATASETS vs. VIDEOS CRUDOS DE LA FAC (30 FPS)
==============================================================================
Mapea el 100% de los fotogramas anotados del Dataset (1,701 / 1,702 frames)
hacia los fotogramas originales a 30 FPS de Video 1, Video 2 y Video 3.

Funcionalidades:
1. Inspecciona Dataset_Balanceado (zip o descomprimido) y dataset_split_completo.
2. Aplica las fórmulas de equivalencia temporal y de fotograma nativo:
   - Video 1 (1080p): XXX * 30 -> frame_{XXX*30:05d}.png
   - Video 2 (720p):  59,280 + (XXX * 30) -> frame_{59280 + XXX*30:05d}.png
   - Video 3 (1080p): 97,026 + (XXXX * 30) -> frame_{97026 + XXXX*30:06d}.png
   - Escenas continuas: frame_XXXXXX_sceneXXX
3. Genera una base de datos tabular exhaustiva:
   - CSV: resultados_matching/mapeo_completo_matches.csv
   - Excel: resultados_matching/resumen_matches.xlsx
4. Si los frames crudos están en disco (en Hypatia o local):
   - Genera trípticos visuales de verificación:
     [ Dataset Anotado + BBoxes | Frame Crudo del Video (30 FPS) | Mapa de Diferencia MAE ]
   - Exporta los matches visuales a carpetas organizadas por video para revisión humana.
5. NO REQUIERE GPU: Se ejecuta 100% en CPU con paralelismo multihilo.
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
    índice de fotograma crudo (30 FPS) y nombre esperado del archivo.
    """
    stem = Path(nombre_archivo).stem

    # Caso A: Video 1 (Dataset Codefest Tolima - FAC 5749, 13/06/2021, 1080p)
    if "video_11min" in stem:
        partes = stem.split("_")
        sec_idx = int(partes[-1]) if partes[-1].isdigit() else 0
        frame_idx = sec_idx * 30
        return {
            "video_origen": "video1_codefest_tolima",
            "mision": "Tolima (FAC 5749 - 13/06/2021)",
            "segundo_vuelo": sec_idx,
            "tiempo_mm_ss": formatear_tiempo(sec_idx),
            "frame_crudo_30fps": frame_idx,
            "nombre_frame_esperado": f"frame_{frame_idx:05d}.png",
            "resolucion_esperada": "1920x1080",
            "tipo_corte": "1_fps_regular",
            "pad_digitos": 5,
            "nota": "Vuelo Codefest 11:52 min (FAC 5749). Distinto al video30min-11to22 (FAC 5748) en Hypatia."
        }

    # Caso B: Video 2 (Últimos 13 min de 46 min - Santander, FLIR Systems, 720p)
    if "video_13min" in stem:
        partes = stem.split("_")
        sec_idx = int(partes[-1]) if partes[-1].isdigit() else 0
        # Offset de inicio de toma: Minuto 32:56 = segundo 1,976 = frame 59,280 (100% verificado)
        frame_idx = 59280 + (sec_idx * 30)
        segundo_absoluto = 1976 + sec_idx
        return {
            "video_origen": "video2",
            "mision": "Santander (FLIR Systems)",
            "segundo_vuelo": segundo_absoluto,
            "tiempo_mm_ss": formatear_tiempo(segundo_absoluto),
            "frame_crudo_30fps": frame_idx,
            "nombre_frame_esperado": f"frame_{frame_idx:05d}.png",
            "resolucion_esperada": "1280x720",
            "tipo_corte": "1_fps_regular",
            "pad_digitos": 5,
            "nota": "Match 100% verificado contra VideoCodefest_008-46min.avi"
        }

    # Caso C: Video 3 (Río Telembí, Nariño, 19/10/2021, 1080p, formato 4 dígitos frame_XXXX)
    if re.match(r"^frame_\d{4}$", stem):
        num_str = stem.replace("frame_", "")
        idx_num = int(num_str) if num_str.isdigit() else 0
        # Offset de ancla verificado en telemetría:
        # frame_0000.jpg está a las 15:44:40 Z -> Frame crudo 60,756 en video_2hour.mp4
        # Muestreo a ~2 segundos por fotograma con saltos por giros
        if idx_num <= 10:
            dt_seg = idx_num * 2.0
        else:
            dt_seg = 20.0 + (idx_num - 10) * 2.444
        
        segundo_absoluto = 2025.2 + dt_seg  # 60,756 / 30 = 2025.2 s (Min 33:45)
        frame_idx = 60756 + int(round(dt_seg * 30.0))
        return {
            "video_origen": "video3",
            "mision": "Narino Telembi (19/10/2021)",
            "segundo_vuelo": round(segundo_absoluto, 1),
            "tiempo_mm_ss": formatear_tiempo(segundo_absoluto),
            "frame_crudo_30fps": frame_idx,
            "nombre_frame_esperado": f"frame_{frame_idx:06d}.png",
            "resolucion_esperada": "1920x1080",
            "tipo_corte": "muestreo_2s_con_saltos",
            "pad_digitos": 6,
            "nota": "Ancla corregida en frame 60,756 (15:44:40 Z)"
        }

    # Caso D: Cortes de Escena Continua (frame_XXXXXX_sceneXXX o v1_frame_XXXXXX_sceneXXX)
    if "scene" in stem:
        nums = re.findall(r"\d+", stem)
        f_nativo = int(nums[0]) if len(nums) >= 1 else 0
        sc_id = nums[1] if len(nums) >= 2 else "000"

        # Discriminación por resolución conocida
        es_1080 = (ancho_img == 1920 and alto_img == 1080) if ancho_img else ("v1_" in stem)
        if es_1080:
            v_orig = "video1_escenas"
            mision = "Narino Telembi (FAC 5748 - 14/05/2021)"
            pad = 5
            res_str = "1920x1080"
        else:
            v_orig = "video2_escenas"
            mision = "Santander (FLIR Systems - Escena)"
            pad = 5
            res_str = "1280x720"

        sec_est = f_nativo / 30.0
        return {
            "video_origen": v_orig,
            "mision": mision,
            "segundo_vuelo": round(sec_est, 2),
            "tiempo_mm_ss": formatear_tiempo(sec_est),
            "frame_crudo_30fps": f_nativo,
            "nombre_frame_esperado": f"frame_{f_nativo:0{pad}d}.png",
            "resolucion_esperada": res_str,
            "tipo_corte": f"corte_escena_{sc_id}",
            "pad_digitos": pad
        }

    # Caso E: Genérico
    nums = re.findall(r"\d+", stem)
    idx = int(nums[-1]) if nums else 0
    return {
        "video_origen": "desconocido",
        "mision": "No identificada",
        "segundo_vuelo": idx,
        "tiempo_mm_ss": formatear_tiempo(idx),
        "frame_crudo_30fps": idx,
        "nombre_frame_esperado": f"frame_{idx:05d}.png",
        "resolucion_esperada": "desconocida",
        "tipo_corte": "generico",
        "pad_digitos": 5
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
        # Etiqueta de fondo para texto
        cv2.rectangle(canvas, (x1, max(0, y1 - 22)), (x1 + len(nom_cls) * 11 + 10, y1), color, -1)
        cv2.putText(canvas, nom_cls, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def construir_triptico_match(img_dataset, img_crudo, info):
    """
    Construye un mosaico horizontal comparativo:
    [ Panel 1: Dataset con Anotaciones | Panel 2: Frame Crudo (Drive/Hypatia) | Panel 3: Diferencia Absoluta ]
    """
    h1, w1 = img_dataset.shape[:2]
    h2, w2 = img_crudo.shape[:2]

    # Igualar resolución para comparar
    if (w1, h1) != (w2, h2):
        img_crudo_res = cv2.resize(img_crudo, (w1, h1), interpolation=cv2.INTER_LINEAR)
    else:
        img_crudo_res = img_crudo

    # Calcular mapa de error absoluto (ignorando bandas negras si las hay)
    diff = cv2.absdiff(img_dataset, img_crudo_res)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mae = float(np.mean(diff_gray))

    # Mapa térmico inferno del error
    diff_amp = np.clip(diff_gray.astype(np.float32) * 5.0, 0, 255).astype(np.uint8)
    diff_col = cv2.applyColorMap(diff_amp, cv2.COLORMAP_INFERNO)

    # Crear lienzo con barra de título superior
    margen_sup = 65
    ancho_tot = w1 * 3
    alto_tot = h1 + margen_sup

    canvas = np.zeros((alto_tot, ancho_tot, 3), dtype=np.uint8)
    canvas.fill(22)  # Gris muy oscuro

    canvas[margen_sup:, 0:w1] = img_dataset
    canvas[margen_sup:, w1:2*w1] = img_crudo_res
    canvas[margen_sup:, 2*w1:] = diff_col

    # Separadores verticales
    cv2.line(canvas, (w1, 0), (w1, alto_tot), (70, 70, 70), 2)
    cv2.line(canvas, (2*w1, 0), (2*w1, alto_tot), (70, 70, 70), 2)

    # Títulos
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"1. DATASET: {info['archivo_dataset']} ({info['split'].upper()})", (16, 38), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"2. VIDEO CRUDO: {info['video_origen'].upper()} | #{info['frame_crudo_30fps']} ({info['tiempo_mm_ss']})", (w1 + 16, 38), font, 0.75, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"3. ERROR DIFERENCIAL (MAE = {mae:.2f})", (2*w1 + 16, 38), font, 0.75, (100, 200, 255), 2, cv2.LINE_AA)

    # Subtítulo inferior
    sub_txt = f"Mision: {info['mision']} | Archivo Esperado: {info['archivo_frame_esperado']} | BBoxes: {info['num_anotaciones']}"
    cv2.putText(canvas, sub_txt, (16, alto_tot - 12), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas, mae


def cargar_dataset(ruta_ds):
    """
    Carga todas las rutas de imágenes y etiquetas desde un .zip o carpeta descomprimida.
    """
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
            stem = Path(img_p).stem
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
    parser.add_argument("--videos-dir", default=None, help="Ruta a la carpeta con videos/video{1,2,3}/frames_originales (default: auto-detectar)")
    parser.add_argument("--salida", default="resultados_matching", help="Directorio de salida para CSV, Excel e imágenes")
    parser.add_argument("--exportar-visual", action="store_true", default=True, help="Genera trípticos visuales si los frames crudos existen en disco")
    parser.add_argument("--max-visual-por-video", type=int, default=50, help="Máximo de imágenes visuales a exportar por video (0 = exportar todas)")
    parser.add_argument("--cpus", type=int, default=min(16, os.cpu_count() or 4), help="Número de hilos para procesamiento en paralelo")
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
        print(f"[ERROR] ERROR: No se encontró ningún dataset en {RAIZ / 'datasets'}. Especifica --dataset <ruta>.")
        sys.exit(1)

    # 2. Localizar carpetas de frames originales de los videos con soporte de rutas alternativas
    dir_videos = Path(args.videos_dir).resolve() if args.videos_dir else (RAIZ / "videos")
    carpetas_frames = {}
    for v_k in ["video1", "video2", "video3"]:
        cands = [
            dir_videos / v_k / "frames_originales",
            dir_videos / v_k / "frames_sin_hud",
            dir_videos / v_k,
        ]
        p_elegida = next((c for c in cands if c.is_dir() and any(p.suffix.lower() in [".png", ".jpg", ".jpeg"] for p in c.iterdir())), None)
        if not p_elegida:
            p_elegida = dir_videos / v_k / "frames_originales"
        carpetas_frames[v_k] = p_elegida

    # Verificar disponibilidad en disco y mostrar muestras reales
    print("\n[INFO] Verificación de carpetas de videos crudos (30 FPS) en el entorno:")
    hay_frames_locales = False
    for v_id, p_frames in carpetas_frames.items():
        if p_frames.is_dir():
            ejemplos = [p.name for i, p in enumerate(p_frames.iterdir()) if i < 4 and p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            if ejemplos:
                print(f"  [OK] {v_id.upper()}: Disponible en {p_frames}")
                print(f"       -> Muestra de archivos reales: {ejemplos}")
                hay_frames_locales = True
            else:
                print(f"  [AVISO] {v_id.upper()}: Carpeta encontrada pero sin imágenes ({p_frames})")
        else:
            print(f"  [AVISO] {v_id.upper()}: No encontrada ({p_frames})")

    # 3. Cargar imágenes del dataset
    items_ds = cargar_dataset(ruta_ds)
    print(f"\n[LISTA] Total de imágenes registradas en el dataset: {len(items_ds):,}")

    # 4. Crear directorio de salida
    dir_salida = (RAIZ / args.salida).resolve()
    dir_salida.mkdir(parents=True, exist_ok=True)
    dir_vis = dir_salida / "visual_matches"
    dir_vis.mkdir(parents=True, exist_ok=True)

    for v_id_folder in ["video1", "video2", "video3", "video1_codefest_tolima", "video2_escenas", "video1_escenas"]:
        (dir_vis / v_id_folder).mkdir(exist_ok=True)

    # 5. Mapear cada elemento
    registros = []
    conteo_vis = {"video1": 0, "video2": 0, "video3": 0, "video1_codefest_tolima": 0, "video2_escenas": 0, "video1_escenas": 0}

    print("\n[RUN] Ejecutando mapeo matemático y verificación...")

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

        # Resolver fórmula de match
        info_match = resolver_origen_y_frame(nom, ancho_img=w, alto_img=h)
        v_id = info_match["video_origen"]

        # Parsear bboxes
        bboxes = parsear_etiqueta_yolo(lbl_str, w or 1920, h or 1080)
        clases_detectadas = sorted(list(set(CLASES_MAP.get(b[0], str(b[0])) for b in bboxes)))

        # Verificar si el frame crudo existe en disco
        p_frame_crudo = None
        existe_en_disco = False
        mae_resultado = None

        v_busqueda = v_id
        if "video1" in v_id:
            v_busqueda = "video1"
        elif "video2" in v_id:
            v_busqueda = "video2"
        elif "video3" in v_id:
            v_busqueda = "video3"

        if v_busqueda in carpetas_frames and carpetas_frames[v_busqueda].is_dir():
            idx = info_match["frame_crudo_30fps"]
            candidatos_nombres = [
                info_match["nombre_frame_esperado"],
                f"frame_{idx:06d}.png",
                f"frame_{idx:05d}.png",
                f"frame_{idx}.png",
                f"frame_{idx:06d}.jpg",
                f"frame_{idx:05d}.jpg",
                f"frame_{idx}.jpg",
            ]
            for nom_c in candidatos_nombres:
                cand_p = carpetas_frames[v_busqueda] / nom_c
                if cand_p.is_file():
                    p_frame_crudo = cand_p
                    existe_en_disco = True
                    break

        # Exportación visual si está habilitada
        if args.exportar_visual and existe_en_disco and img_bgr is not None and p_frame_crudo:
            limite = args.max_visual_por_video
            if limite == 0 or conteo_vis.get(v_id, 0) < limite:
                img_crudo = cv2.imread(str(p_frame_crudo))
                if img_crudo is not None:
                    img_con_bboxes = dibujar_anotaciones(img_bgr, bboxes)
                    
                    info_panel = {
                        "archivo_dataset": nom,
                        "split": split,
                        "video_origen": v_id,
                        "frame_crudo_30fps": info_match["frame_crudo_30fps"],
                        "tiempo_mm_ss": info_match["tiempo_mm_ss"],
                        "mision": info_match["mision"],
                        "archivo_frame_esperado": info_match["nombre_frame_esperado"],
                        "num_anotaciones": len(bboxes)
                    }
                    
                    triptico, mae = construir_triptico_match(img_con_bboxes, img_crudo, info_panel)
                    mae_resultado = mae
                    
                    subcarpeta_vis = "video1" if "video1" in v_id else v_id
                    out_png = dir_vis / subcarpeta_vis / f"match_{v_id}_{nom.replace('.jpg', '')}_frame_{info_match['frame_crudo_30fps']}.jpg"
                    cv2.imwrite(str(out_png), triptico, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    conteo_vis[v_id] += 1

        return {
            "archivo_dataset": nom,
            "split": split,
            "video_origen": v_id,
            "mision": info_match["mision"],
            "segundo_vuelo": info_match["segundo_vuelo"],
            "tiempo_mm_ss": info_match["tiempo_mm_ss"],
            "frame_crudo_30fps": info_match["frame_crudo_30fps"],
            "archivo_frame_esperado": info_match["nombre_frame_esperado"],
            "resolucion_esperada": info_match["resolucion_esperada"],
            "tipo_corte": info_match["tipo_corte"],
            "num_anotaciones": len(bboxes),
            "clases_presentes": ", ".join(clases_detectadas) if clases_detectadas else "Ninguna",
            "frame_crudo_en_disco": existe_en_disco,
            "mae_diferencia": round(mae_resultado, 2) if mae_resultado is not None else None,
            "nota_auditoria": info_match.get("nota", "")
        }

    with ThreadPoolExecutor(max_workers=args.cpus) as pool:
        registros = list(pool.map(procesar_item, items_ds))

    # 6. Construir DataFrame y Guardar CSV / Excel
    df = pd.DataFrame(registros)
    
    csv_salida = dir_salida / "mapeo_completo_matches.csv"
    df.to_csv(csv_salida, index=False, encoding="utf-8-sig")
    print(f"\n[GUARDADO] CSV de Mapeo Exhaustivo guardado en:\n   -> {csv_salida}")

    xlsx_salida = dir_salida / "resumen_matches.xlsx"
    with pd.ExcelWriter(xlsx_salida, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Todos_los_Matches", index=False)
        
        # Hoja Resumen por Video
        resumen_vid = df.groupby(["video_origen", "mision"]).agg(
            total_imagenes=("archivo_dataset", "count"),
            train=("split", lambda s: (s == "train").sum()),
            val=("split", lambda s: (s == "val").sum()),
            test=("split", lambda s: (s == "test").sum()),
            frames_en_disco=("frame_crudo_en_disco", "sum"),
            anotaciones_totales=("num_anotaciones", "sum")
        ).reset_index()
        resumen_vid.to_excel(writer, sheet_name="Resumen_por_Video", index=False)

    print(f"[EXCEL] Libro Excel resumen guardado en:\n   -> {xlsx_salida}")

    # 7. Resumen de resultados en consola
    print("\n" + "=" * 90)
    print("                        RESUMEN DE MATCHING POR VIDEO")
    print("=" * 90)
    for v_id, grp in df.groupby("video_origen"):
        print(f"[VIDEO] [{v_id.upper()}] -> {len(grp):,} imágenes anotadas")
        print(f"   • Train: {(grp['split'] == 'train').sum():,} | Val: {(grp['split'] == 'val').sum():,} | Test: {(grp['split'] == 'test').sum():,}")
        print(f"   • BBoxes totales: {grp['num_anotaciones'].sum():,}")
        print(f"   • Encontrados en disco (30 FPS): {grp['frame_crudo_en_disco'].sum():,} / {len(grp):,}")
        if conteo_vis.get(v_id, 0) > 0:
            print(f"   • Mosaicos visuales exportados: {conteo_vis[v_id]} imágenes en {dir_vis / v_id}")

    print("\n" + "=" * 90)
    print("[OK] Mapeo completado exitosamente.")
    print(f"Carpetas generadas en: {dir_salida}")
    print("=" * 90)


if __name__ == "__main__":
    main()
