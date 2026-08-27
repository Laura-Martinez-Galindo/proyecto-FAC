#!/usr/bin/env python3
"""Extrae frames de un video registrado en config/videos.json."""

# 0. Imports
import argparse
import json
import math
import shutil
import sys
import time

from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm


# 1. Configuración general
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CONFIGURACION = RUTA_PROYECTO / "config" / "videos.json"


# 2. Argumentos
def obtener_argumentos():
    """Define y obtiene los argumentos del programa."""
    parser = argparse.ArgumentParser(description="Extrae los frames de un video registrado en config/videos.json.")
    parser.add_argument("--video", required=True, help="Identificador registrado en config/videos.json, por ejemplo: video1.")
    return parser.parse_args()


# 3. Configuración de videos
def cargar_configuracion():
    """Carga y valida config/videos.json."""
    if not RUTA_CONFIGURACION.is_file():
        raise FileNotFoundError(f"No se encontró la configuración: {RUTA_CONFIGURACION}")

    try:
        with RUTA_CONFIGURACION.open("r", encoding="utf-8") as archivo:
            configuracion = json.load(archivo)
    except json.JSONDecodeError as error:
        raise ValueError(f"El archivo {RUTA_CONFIGURACION} no contiene JSON válido: {error}") from error

    if not isinstance(configuracion, dict):
        raise ValueError("config/videos.json debe contener un objeto JSON.")

    if not configuracion:
        raise ValueError("config/videos.json no contiene videos registrados.")

    return configuracion


def guardar_configuracion(configuracion):
    """Guarda config/videos.json mediante un archivo temporal."""
    ruta_temporal = RUTA_CONFIGURACION.with_suffix(".json.tmp")

    try:
        with ruta_temporal.open("w", encoding="utf-8") as archivo:
            json.dump(configuracion, archivo, indent=2, ensure_ascii=False)
            archivo.write("\n")

        ruta_temporal.replace(RUTA_CONFIGURACION)
    except BaseException:
        ruta_temporal.unlink(missing_ok=True)
        raise


def obtener_datos_video(video_id, configuracion):
    """Obtiene y valida la configuración de un video."""
    if video_id not in configuracion:
        disponibles = ", ".join(sorted(configuracion))
        raise ValueError(f"El video '{video_id}' no está registrado. Videos disponibles: {disponibles}")

    datos_video = configuracion[video_id]

    if not isinstance(datos_video, dict):
        raise ValueError(f"La configuración de '{video_id}' debe ser un objeto JSON.")

    if "ruta" not in datos_video:
        raise ValueError(f"La configuración de '{video_id}' debe contener el campo 'ruta'.")

    if "extraccion" not in datos_video:
        datos_video["extraccion"] = {}

    if not isinstance(datos_video["extraccion"], dict):
        raise ValueError(f"El campo 'extraccion' de '{video_id}' debe ser un objeto JSON.")

    return datos_video


# 4. Rutas
def resolver_ruta_video(datos_video):
    """Obtiene la ruta absoluta del video."""
    ruta_video = Path(datos_video["ruta"]).expanduser()

    if not ruta_video.is_absolute():
        ruta_video = RUTA_PROYECTO / ruta_video

    ruta_video = ruta_video.resolve()

    if not ruta_video.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de video: {ruta_video}")

    return ruta_video


def obtener_rutas_salida(ruta_video):
    """Define las carpetas definitiva, temporal y de respaldo."""
    if ruta_video.parent.name != "original":
        raise ValueError(f"El video debe estar dentro de una carpeta llamada 'original': {ruta_video}")

    carpeta_video = ruta_video.parent.parent
    carpeta_frames = carpeta_video / "frames_originales"
    carpeta_temporal = carpeta_video / "frames_originales.tmp"
    carpeta_respaldo = carpeta_video / "frames_originales.backup"

    return carpeta_frames, carpeta_temporal, carpeta_respaldo


def obtener_ruta_relativa(ruta):
    """Convierte una ruta del proyecto en una ruta relativa."""
    try:
        return str(ruta.relative_to(RUTA_PROYECTO))
    except ValueError:
        return str(ruta)


def preparar_carpeta_temporal(carpeta_temporal):
    """Crea una carpeta temporal vacía."""
    if carpeta_temporal.exists():
        shutil.rmtree(carpeta_temporal)

    carpeta_temporal.mkdir(parents=True, exist_ok=False)


def reemplazar_carpeta(carpeta_temporal, carpeta_definitiva, carpeta_respaldo):
    """Reemplaza la carpeta anterior sin perderla si ocurre un error."""
    if carpeta_respaldo.exists():
        shutil.rmtree(carpeta_respaldo)

    habia_resultado_anterior = carpeta_definitiva.exists()

    try:
        if habia_resultado_anterior:
            carpeta_definitiva.replace(carpeta_respaldo)

        carpeta_temporal.replace(carpeta_definitiva)
    except BaseException:
        if carpeta_definitiva.exists():
            shutil.rmtree(carpeta_definitiva)

        if carpeta_respaldo.exists():
            carpeta_respaldo.replace(carpeta_definitiva)

        raise

    if carpeta_respaldo.exists():
        shutil.rmtree(carpeta_respaldo)


# 5. Lectura del video
def abrir_video(ruta_video):
    """Abre el video mediante OpenCV."""
    captura = cv2.VideoCapture(str(ruta_video))

    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {ruta_video}. Revise el formato y el códec.")

    return captura


def decodificar_fourcc(codigo_fourcc):
    """Convierte el código numérico del códec en texto."""
    caracteres = [chr((codigo_fourcc >> desplazamiento) & 0xFF) for desplazamiento in range(0, 32, 8)]
    return "".join(caracteres).strip("\x00")


def leer_metadatos(captura, ruta_video):
    """Obtiene y valida los metadatos del video."""
    fps = float(captura.get(cv2.CAP_PROP_FPS))
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codigo_fourcc = int(captura.get(cv2.CAP_PROP_FOURCC))

    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"No se pudo obtener un FPS válido de: {ruta_video}")

    if total_frames <= 0:
        raise RuntimeError(f"No se pudo obtener la cantidad de frames de: {ruta_video}")

    if ancho <= 0 or alto <= 0:
        raise RuntimeError(f"No se pudo obtener la resolución de: {ruta_video}")

    return {
        "archivo": ruta_video.name,
        "ruta": obtener_ruta_relativa(ruta_video),
        "codec": decodificar_fourcc(codigo_fourcc),
        "resolucion": {
            "ancho": ancho,
            "alto": alto,
        },
        "fps_original": fps,
        "total_frames": total_frames,
        "duracion_segundos": total_frames / fps,
    }


# 6. Parámetros
def obtener_parametros(datos_video, metadatos):
    """Obtiene y valida los parámetros de extracción."""
    extraccion = datos_video["extraccion"]
    inicio = float(extraccion.get("inicio", 0.0))
    fin_configurado = extraccion.get("fin")
    fps_configurado = extraccion.get("fps")
    compresion_png = int(extraccion.get("compresion_png", 3))
    fin = metadatos["duracion_segundos"] if fin_configurado is None else float(fin_configurado)
    fps_salida = metadatos["fps_original"] if fps_configurado is None else float(fps_configurado)

    if not math.isfinite(inicio) or inicio < 0:
        raise ValueError("extraccion.inicio debe ser mayor o igual que cero.")

    if inicio >= metadatos["duracion_segundos"]:
        raise ValueError(f"extraccion.inicio debe ser menor que {metadatos['duracion_segundos']:.3f} segundos.")

    if not math.isfinite(fin):
        raise ValueError("extraccion.fin debe ser un número válido o null.")

    fin = min(fin, metadatos["duracion_segundos"])

    if fin <= inicio:
        raise ValueError("extraccion.fin debe ser mayor que extraccion.inicio.")

    if not math.isfinite(fps_salida) or fps_salida <= 0:
        raise ValueError("extraccion.fps debe ser mayor que cero o null.")

    if fps_salida > metadatos["fps_original"]:
        raise ValueError(f"extraccion.fps no puede superar el FPS original: {metadatos['fps_original']:.6f}.")

    if not 0 <= compresion_png <= 9:
        raise ValueError("extraccion.compresion_png debe estar entre 0 y 9.")

    return {
        "inicio": inicio,
        "fin": fin,
        "fps_salida": fps_salida,
        "compresion_png": compresion_png,
    }


def calcular_indices_objetivo(metadatos, parametros):
    """Calcula los índices originales que deben extraerse."""
    fps_original = metadatos["fps_original"]
    fps_salida = parametros["fps_salida"]
    total_frames = metadatos["total_frames"]
    inicio = parametros["inicio"]
    fin = parametros["fin"]
    indice_inicio = max(0, math.ceil(inicio * fps_original))
    indice_fin = min(total_frames, math.ceil(fin * fps_original))
    conserva_fps_original = math.isclose(fps_salida, fps_original, rel_tol=0.0, abs_tol=1e-9)

    if conserva_fps_original:
        return list(range(indice_inicio, indice_fin))

    cantidad_salida = math.ceil((fin - inicio) * fps_salida - 1e-9)
    indices_objetivo = []

    for posicion in range(cantidad_salida):
        segundo_objetivo = inicio + posicion / fps_salida
        indice_objetivo = math.floor(segundo_objetivo * fps_original + 0.5)
        indice_objetivo = min(max(indice_objetivo, indice_inicio), indice_fin - 1)

        if not indices_objetivo or indice_objetivo != indices_objetivo[-1]:
            indices_objetivo.append(indice_objetivo)

    return indices_objetivo


# 7. Escritura de frames
def guardar_frame(ruta_frame, frame, compresion_png):
    """Guarda un frame PNG mediante un archivo temporal."""
    ruta_temporal = ruta_frame.with_suffix(".png.tmp")
    parametros_png = [cv2.IMWRITE_PNG_COMPRESSION, compresion_png]
    escritura_correcta, imagen_codificada = cv2.imencode(".png", frame, parametros_png)

    if not escritura_correcta:
        raise RuntimeError(f"No se pudo codificar el frame: {ruta_frame}")

    imagen_codificada.tofile(ruta_temporal)
    ruta_temporal.replace(ruta_frame)


# 8. Extracción
def extraer_frames(argumentos):
    """Extrae los frames y actualiza config/videos.json."""
    configuracion = cargar_configuracion()
    datos_video = obtener_datos_video(argumentos.video, configuracion)
    ruta_video = resolver_ruta_video(datos_video)
    carpeta_frames, carpeta_temporal, carpeta_respaldo = obtener_rutas_salida(ruta_video)
    captura = abrir_video(ruta_video)
    tiempo_inicio = time.monotonic()

    try:
        metadatos = leer_metadatos(captura, ruta_video)
        parametros = obtener_parametros(datos_video, metadatos)
        indices_objetivo = calcular_indices_objetivo(metadatos, parametros)

        if not indices_objetivo:
            raise RuntimeError("El intervalo configurado no produjo ningún frame.")

        total_salida = len(indices_objetivo)
        cantidad_digitos = max(4, len(str(total_salida)))
        preparar_carpeta_temporal(carpeta_temporal)
        cv2.setNumThreads(1)
        indice_salida = 0
        indice_video = indices_objetivo[0]

        captura.set(cv2.CAP_PROP_POS_FRAMES, indice_video)

        with tqdm(total=total_salida, desc=f"Extrayendo {argumentos.video}", unit="frame", dynamic_ncols=True, mininterval=1.0) as barra:
            while indice_salida < total_salida:
                lectura_correcta, frame = captura.read()

                if not lectura_correcta or frame is None:
                    break

                indice_objetivo = indices_objetivo[indice_salida]

                if indice_video == indice_objetivo:
                    numero_frame = indice_salida + 1
                    nombre_frame = f"frame_{numero_frame:0{cantidad_digitos}d}.png"
                    guardar_frame(carpeta_temporal / nombre_frame, frame, parametros["compresion_png"])
                    indice_salida += 1
                    barra.update(1)

                indice_video += 1

        if indice_salida != total_salida:
            raise RuntimeError(f"Se esperaban {total_salida} frames, pero se extrajeron {indice_salida}.")

        reemplazar_carpeta(carpeta_temporal, carpeta_frames, carpeta_respaldo)
        tiempo_total = time.monotonic() - tiempo_inicio

        configuracion[argumentos.video]["archivo_original"] = ruta_video.name
        configuracion[argumentos.video]["metadatos"] = metadatos
        configuracion[argumentos.video]["extraccion"] = {
            "inicio": parametros["inicio"],
            "fin": parametros["fin"],
            "fps": parametros["fps_salida"],
            "compresion_png": parametros["compresion_png"],
            "estado": "completada",
            "carpeta_salida": obtener_ruta_relativa(carpeta_frames),
            "frames_extraidos": total_salida,
            "fecha_ejecucion": datetime.now().astimezone().isoformat(timespec="seconds"),
            "tiempo_ejecucion_minutos": tiempo_total / 60.0,
        }

        guardar_configuracion(configuracion)
    except BaseException:
        if carpeta_temporal.exists():
            shutil.rmtree(carpeta_temporal)

        raise
    finally:
        captura.release()

    print()
    print("Extracción terminada correctamente.")
    print(f"Video: {argumentos.video}")
    print(f"Resolución: {metadatos['resolucion']['ancho']}x{metadatos['resolucion']['alto']}")
    print(f"FPS original: {metadatos['fps_original']:.6f}")
    print(f"FPS de extracción: {parametros['fps_salida']:.6f}")
    print(f"Frames extraídos: {total_salida}")
    print(f"Carpeta de salida: {carpeta_frames}")
    print(f"Tiempo total: {tiempo_total / 60.0:.2f} minutos")


# 9. Punto de entrada
def main():
    """Ejecuta el programa y presenta los errores."""
    try:
        argumentos = obtener_argumentos()
        extraer_frames(argumentos)
        return 0
    except KeyboardInterrupt:
        print("\nExtracción interrumpida por el usuario.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())