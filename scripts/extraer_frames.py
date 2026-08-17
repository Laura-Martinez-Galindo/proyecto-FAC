# %% 0. Imports y configuración
import math
import os
import re
import threading

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm

# Ruta principal del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parent.parent

# Ruta del video que se desea procesar
RUTA_VIDEO = RUTA_PROYECTO / "Videos" / "video30min-11to22.mp4"

# Cantidad de imágenes que se extraerán por cada segundo de video
FRAMES_POR_SEGUNDO = 5.0

# Intervalo opcional del video
# None significa comenzar desde el inicio o terminar al final
SEGUNDO_INICIO = None
SEGUNDO_FIN = None

# Compresión PNG entre 0 y 9
COMPRESION_PNG = 3

# Carpeta donde se guardarán los frames
CARPETA_FRAMES = RUTA_VIDEO.parent / RUTA_VIDEO.stem / "frames"

# Archivo donde se registra qué frame original fue elegido
RUTA_SELECCION_FRAMES = RUTA_VIDEO.parent / RUTA_VIDEO.stem / "seleccion_frames_nitidos.csv"

# Cantidad de trabajadores
# En SLURM toma automáticamente --cpus-per-task
NUM_TRABAJADORES = int(
    os.environ.get("SLURM_CPUS_PER_TASK")
    or os.environ.get("SLURM_CPUS_ON_NODE")
    or min(8, os.cpu_count() or 1)
)

# Búsqueda del frame más nítido alrededor de cada instante
RADIO_BUSQUEDA_NITIDEZ = 0.08
NUM_CANDIDATOS_NITIDEZ = 5
LADO_ANALISIS_NITIDEZ = 640
MARGEN_ROI_NITIDEZ = 0.10
PERCENTIL_NITIDEZ = 50

# Control de archivos existentes
SOBRESCRIBIR_FRAMES = True
ELIMINAR_FRAMES_ANTERIORES = True


# %% 1. Extracción de frames
if not RUTA_VIDEO.exists():
    raise FileNotFoundError(f"No se encontró el video: {RUTA_VIDEO}")

if NUM_CANDIDATOS_NITIDEZ < 1 or NUM_CANDIDATOS_NITIDEZ % 2 == 0:
    raise ValueError("NUM_CANDIDATOS_NITIDEZ debe ser un número impar mayor o igual que 1")

captura_metadatos = cv2.VideoCapture(str(RUTA_VIDEO))  # Abre el video para consultar sus metadatos

if not captura_metadatos.isOpened():
    raise RuntimeError(f"No se pudo abrir el video: {RUTA_VIDEO}")

fps_video = captura_metadatos.get(cv2.CAP_PROP_FPS)  # Obtiene los FPS originales
total_frames = int(captura_metadatos.get(cv2.CAP_PROP_FRAME_COUNT))  # Obtiene el total de frames
captura_metadatos.release()

if not np.isfinite(fps_video) or fps_video <= 0 or total_frames <= 0:
    raise RuntimeError(f"No se pudieron leer los metadatos de: {RUTA_VIDEO}")

duracion = total_frames / fps_video  # Calcula la duración en segundos
inicio = 0.0 if SEGUNDO_INICIO is None else max(0.0, float(SEGUNDO_INICIO))
fin = duracion if SEGUNDO_FIN is None else min(float(SEGUNDO_FIN), duracion)

if fin <= inicio:
    raise ValueError("SEGUNDO_FIN debe ser mayor que SEGUNDO_INICIO")

if not 0 < FRAMES_POR_SEGUNDO <= fps_video:
    raise ValueError(f"FRAMES_POR_SEGUNDO debe estar entre 0 y {fps_video:.3f}")

intervalo = 1 / FRAMES_POR_SEGUNDO  # Calcula los segundos entre muestras

if RADIO_BUSQUEDA_NITIDEZ < 0:
    raise ValueError("RADIO_BUSQUEDA_NITIDEZ no puede ser negativo")

if RADIO_BUSQUEDA_NITIDEZ >= intervalo / 2:
    raise ValueError(
        f"RADIO_BUSQUEDA_NITIDEZ debe ser menor que {intervalo / 2:.3f} "
        "para evitar que las ventanas temporales se superpongan"
    )

cantidad_salida = math.ceil((fin - inicio) * FRAMES_POR_SEGUNDO)
cantidad_digitos = max(1, len(str(cantidad_salida - 1)))
desplazamientos = np.linspace(
    -RADIO_BUSQUEDA_NITIDEZ,
    RADIO_BUSQUEDA_NITIDEZ,
    NUM_CANDIDATOS_NITIDEZ,
)

CARPETA_FRAMES.mkdir(parents=True, exist_ok=True)  # Crea la carpeta de salida
RUTA_SELECCION_FRAMES.parent.mkdir(parents=True, exist_ok=True)

if ELIMINAR_FRAMES_ANTERIORES:
    for ruta_anterior in CARPETA_FRAMES.glob("*.png"):
        ruta_anterior.unlink()

cv2.setNumThreads(1)  # Evita que OpenCV cree hilos adicionales dentro de cada trabajador
datos_hilo = threading.local()  # Mantiene un VideoCapture independiente por trabajador


def obtener_captura():
    if not hasattr(datos_hilo, "captura"):
        datos_hilo.captura = cv2.VideoCapture(str(RUTA_VIDEO))

        if not datos_hilo.captura.isOpened():
            raise RuntimeError(f"No se pudo abrir el video desde un trabajador: {RUTA_VIDEO}")

    return datos_hilo.captura


def preparar_roi_nitidez(frame):
    alto, ancho = frame.shape[:2]
    escala = min(1.0, LADO_ANALISIS_NITIDEZ / max(alto, ancho))

    if escala < 1.0:
        frame = cv2.resize(
            frame,
            (round(ancho * escala), round(alto * escala)),
            interpolation=cv2.INTER_AREA,
        )

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    alto, ancho = gris.shape
    margen_y = round(alto * MARGEN_ROI_NITIDEZ)
    margen_x = round(ancho * MARGEN_ROI_NITIDEZ)

    return gris[margen_y:alto - margen_y, margen_x:ancho - margen_x]


def calcular_nitidez(frame):
    roi = preparar_roi_nitidez(frame)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    laplaciano = cv2.Laplacian(roi, cv2.CV_32F, ksize=3)
    energia_local = cv2.blur(laplaciano * laplaciano, (31, 31))

    return float(np.percentile(energia_local, PERCENTIL_NITIDEZ))


def extraer_mejor_frame(indice_salida):
    captura = obtener_captura()
    segundo_objetivo = inicio + indice_salida * intervalo
    indices_candidatos = []

    for desplazamiento in desplazamientos:
        segundo_candidato = min(
            max(inicio, segundo_objetivo + desplazamiento),
            max(inicio, fin - 1 / fps_video),
        )

        indice_video = min(
            max(round(segundo_candidato * fps_video), 0),
            total_frames - 1,
        )

        indices_candidatos.append(indice_video)

    indices_candidatos = sorted(set(indices_candidatos))
    primer_indice = indices_candidatos[0]
    ultimo_indice = indices_candidatos[-1]
    conjunto_candidatos = set(indices_candidatos)

    captura.set(cv2.CAP_PROP_POS_FRAMES, primer_indice)
    mejor_frame = None
    mejor_indice = None
    mejor_nitidez = -np.inf
    candidatos_evaluados = 0

    for indice_video in range(primer_indice, ultimo_indice + 1):
        lectura_correcta, frame = captura.read()

        if not lectura_correcta or frame is None:
            raise RuntimeError(
                f"No se pudo leer el frame {indice_video} "
                f"cerca del segundo {segundo_objetivo:.3f}"
            )

        if indice_video not in conjunto_candidatos:
            continue

        nitidez = calcular_nitidez(frame)
        candidatos_evaluados += 1

        if nitidez > mejor_nitidez:
            mejor_nitidez = nitidez
            mejor_indice = indice_video
            mejor_frame = frame.copy()

    if mejor_frame is None or mejor_indice is None:
        raise RuntimeError(f"No se encontró un frame válido cerca del segundo {segundo_objetivo:.3f}")

    nombre = f"{indice_salida:0{cantidad_digitos}d}.png"
    ruta_salida = CARPETA_FRAMES / nombre

    if SOBRESCRIBIR_FRAMES or not ruta_salida.exists():
        escritura_correcta = cv2.imwrite(
            str(ruta_salida),
            mejor_frame,
            [cv2.IMWRITE_PNG_COMPRESSION, COMPRESION_PNG],
        )

        if not escritura_correcta:
            raise RuntimeError(f"No se pudo guardar: {ruta_salida}")

    segundo_elegido = mejor_indice / fps_video

    return {
        "indice_salida": indice_salida,
        "frame_salida": nombre,
        "segundo_objetivo": segundo_objetivo,
        "indice_video_elegido": mejor_indice,
        "segundo_elegido": segundo_elegido,
        "desplazamiento_segundos": segundo_elegido - segundo_objetivo,
        "nitidez": mejor_nitidez,
        "candidatos_evaluados": candidatos_evaluados,
    }


with ThreadPoolExecutor(max_workers=NUM_TRABAJADORES) as ejecutor:
    seleccion_frames = list(
        tqdm(
            ejecutor.map(extraer_mejor_frame, range(cantidad_salida)),
            total=cantidad_salida,
            desc=f"Extrayendo {RUTA_VIDEO.name}",
            unit="frame",
            mininterval=2.0,
        )
    )

df_seleccion_frames = pd.DataFrame(seleccion_frames).sort_values("indice_salida")
df_seleccion_frames.to_csv(RUTA_SELECCION_FRAMES, index=False)

cantidad_archivos = len(list(CARPETA_FRAMES.glob("*.png")))

if cantidad_archivos != cantidad_salida:
    raise RuntimeError(
        f"Se esperaban {cantidad_salida} PNG, pero se encontraron {cantidad_archivos}"
    )

print()
print(f"Extracción terminada: {cantidad_salida} frames guardados en {CARPETA_FRAMES}")
print(f"Selección de nitidez guardada en: {RUTA_SELECCION_FRAMES}")
print(f"FPS del video: {fps_video:.3f}")
print(f"Duración procesada: {fin - inicio:.2f} segundos")
print(f"Trabajadores utilizados: {NUM_TRABAJADORES}")
print(f"Nitidez promedio: {df_seleccion_frames['nitidez'].mean():.2f}")
print(f"Nitidez mínima: {df_seleccion_frames['nitidez'].min():.2f}")
print(f"Nitidez máxima: {df_seleccion_frames['nitidez'].max():.2f}")
print(f"Desplazamiento temporal promedio: {df_seleccion_frames['desplazamiento_segundos'].abs().mean():.4f} segundos")
print(f"Desplazamiento temporal máximo: {df_seleccion_frames['desplazamiento_segundos'].abs().max():.4f} segundos")
