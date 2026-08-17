#!/usr/bin/env python3
"""Extrae frames de un video registrado en config/videos.json."""

# 0. Imports
import argparse
import csv
import json
import math
import sys
import time

from pathlib import Path

import cv2
from tqdm import tqdm


# 1. Configuración general
# La raíz del proyecto se calcula a partir de la ubicación de este script.
RUTA_PROYECTO = Path(__file__).resolve().parent.parent

# Esta es la única ruta fija del programa.
RUTA_CONFIGURACION = RUTA_PROYECTO / "config" / "videos.json"


# 2. Argumentos de línea de comandos
def obtener_argumentos():
    """Define y obtiene los parámetros de la extracción."""
    parser = argparse.ArgumentParser(description="Extrae frames de un video registrado en config/videos.json.")

    parser.add_argument("--video", required=True, help="Identificador del video registrado en config/videos.json, por ejemplo: video1.")
    parser.add_argument("--inicio", type=float, default=0.0, help="Segundo inicial del intervalo. Valor predeterminado: 0.")
    parser.add_argument("--fin", type=float, default=None, help="Segundo final exclusivo. Si se omite, procesa hasta el final.")
    parser.add_argument("--fps", type=float, default=None, help="FPS de salida. Si se omite, conserva el FPS original.")
    parser.add_argument("--compresion-png", type=int, choices=range(10), default=3, metavar="[0-9]", help="Compresión PNG entre 0 y 9. Valor predeterminado: 3.")

    return parser.parse_args()


# 3. Configuración de videos
def cargar_configuracion():
    """Carga el archivo JSON que relaciona cada identificador con un video."""
    if not RUTA_CONFIGURACION.is_file():
        raise FileNotFoundError(f"No se encontró la configuración: {RUTA_CONFIGURACION}")

    with RUTA_CONFIGURACION.open("r", encoding="utf-8") as archivo:
        configuracion = json.load(archivo)

    if not isinstance(configuracion, dict):
        raise ValueError("config/videos.json debe contener un objeto JSON.")

    if not configuracion:
        raise ValueError("config/videos.json no contiene videos registrados.")

    return configuracion


def resolver_ruta_video(video_id, configuracion):
    """Obtiene y valida la ruta asociada al identificador solicitado."""
    if video_id not in configuracion:
        disponibles = ", ".join(sorted(configuracion))
        raise ValueError(f"El video '{video_id}' no está registrado. Videos disponibles: {disponibles}")

    datos_video = configuracion[video_id]

    if not isinstance(datos_video, dict):
        raise ValueError(f"La configuración de '{video_id}' debe ser un objeto JSON.")

    if "ruta" not in datos_video:
        raise ValueError(f"La configuración de '{video_id}' debe contener el campo 'ruta'.")

    ruta_video = Path(datos_video["ruta"]).expanduser()

    # Las rutas relativas del JSON se interpretan desde la raíz del proyecto.
    if not ruta_video.is_absolute():
        ruta_video = RUTA_PROYECTO / ruta_video

    ruta_video = ruta_video.resolve()

    if not ruta_video.is_file():
        raise FileNotFoundError(f"No se encontró el archivo asociado a '{video_id}': {ruta_video}")

    return ruta_video


# 4. Rutas de salida
def obtener_rutas_salida(ruta_video):
    """Define la carpeta de frames y los registros de la extracción."""
    # Se espera que el video esté dentro de videos/videoN/original/.
    if ruta_video.parent.name != "original":
        raise ValueError(f"El video debe estar dentro de una carpeta llamada 'original': {ruta_video}")

    carpeta_video = ruta_video.parent.parent
    carpeta_frames = carpeta_video / "frames_originales"
    ruta_csv = carpeta_video / "extraccion.csv"
    ruta_json = carpeta_video / "extraccion.json"

    carpeta_frames.mkdir(parents=True, exist_ok=True)

    return carpeta_frames, ruta_csv, ruta_json


def eliminar_extraccion_anterior(carpeta_frames, ruta_csv, ruta_json):
    """Elimina únicamente los productos generados por una extracción anterior."""
    cantidad_eliminada = 0

    # Se eliminan únicamente frames generados por este script.
    for patron in ("frame_*.png", "frame_*.png.tmp"):
        for ruta_archivo in carpeta_frames.glob(patron):
            ruta_archivo.unlink()
            cantidad_eliminada += 1

    # Se eliminan los registros anteriores y posibles archivos temporales.
    rutas_registro = (ruta_csv, ruta_csv.with_suffix(".csv.tmp"), ruta_json)

    for ruta_archivo in rutas_registro:
        if ruta_archivo.exists():
            ruta_archivo.unlink()
            cantidad_eliminada += 1

    return cantidad_eliminada


# 5. Lectura del video
def abrir_video(ruta_video):
    """Abre el video y comprueba que OpenCV pueda decodificarlo."""
    captura = cv2.VideoCapture(str(ruta_video))

    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {ruta_video}. Revise que el formato y el códec estén disponibles.")

    return captura


def decodificar_fourcc(codigo_fourcc):
    """Convierte el código numérico del códec en texto legible."""
    caracteres = [chr((codigo_fourcc >> desplazamiento) & 0xFF) for desplazamiento in range(0, 32, 8)]
    return "".join(caracteres).strip("\x00")


def leer_metadatos(captura, ruta_video):
    """Obtiene y valida los metadatos necesarios del video."""
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
        "fps": fps,
        "total_frames": total_frames,
        "ancho": ancho,
        "alto": alto,
        "duracion": total_frames / fps,
        "codec": decodificar_fourcc(codigo_fourcc)
    }


# 6. Validación de parámetros
def validar_parametros(argumentos, metadatos):
    """Valida el intervalo temporal y el FPS solicitado."""
    duracion = metadatos["duracion"]
    fps_original = metadatos["fps"]
    segundo_inicio = float(argumentos.inicio)
    segundo_fin = duracion if argumentos.fin is None else float(argumentos.fin)
    fps_salida = fps_original if argumentos.fps is None else float(argumentos.fps)

    if not math.isfinite(segundo_inicio) or segundo_inicio < 0:
        raise ValueError("--inicio debe ser un número mayor o igual que cero.")

    if segundo_inicio >= duracion:
        raise ValueError(f"--inicio debe ser menor que la duración del video: {duracion:.3f} segundos.")

    if not math.isfinite(segundo_fin):
        raise ValueError("--fin debe ser un número válido.")

    # Si el final supera la duración, se limita automáticamente al final del video.
    segundo_fin = min(segundo_fin, duracion)

    if segundo_fin <= segundo_inicio:
        raise ValueError("--fin debe ser mayor que --inicio.")

    if not math.isfinite(fps_salida) or fps_salida <= 0:
        raise ValueError("--fps debe ser un número mayor que cero.")

    if fps_salida > fps_original:
        raise ValueError(f"--fps no puede superar el FPS original del video: {fps_original:.6f}.")

    return {
        "inicio": segundo_inicio,
        "fin": segundo_fin,
        "fps_salida": fps_salida
    }


# 7. Selección temporal
def calcular_indices_objetivo(metadatos, parametros):
    """Calcula los índices originales que representan el FPS solicitado."""
    fps_original = metadatos["fps"]
    fps_salida = parametros["fps_salida"]
    total_frames = metadatos["total_frames"]
    segundo_inicio = parametros["inicio"]
    segundo_fin = parametros["fin"]

    indice_inicio = max(0, math.ceil(segundo_inicio * fps_original))
    indice_fin = min(total_frames, math.ceil(segundo_fin * fps_original))
    conserva_fps_original = math.isclose(fps_salida, fps_original, rel_tol=0.0, abs_tol=1e-9)

    # Si se conserva el FPS original, se seleccionan todos los frames del intervalo.
    if conserva_fps_original:
        return list(range(indice_inicio, indice_fin))

    duracion_intervalo = segundo_fin - segundo_inicio
    cantidad_salida = math.ceil(duracion_intervalo * fps_salida - 1e-9)
    indices_objetivo = []

    # Cada frame de salida se asocia con el frame original temporalmente más cercano.
    for posicion in range(cantidad_salida):
        segundo_objetivo = segundo_inicio + posicion / fps_salida
        indice_objetivo = round(segundo_objetivo * fps_original)
        indice_objetivo = min(max(indice_objetivo, indice_inicio), indice_fin - 1)

        # Se evita guardar dos veces el mismo frame original.
        if not indices_objetivo or indice_objetivo != indices_objetivo[-1]:
            indices_objetivo.append(indice_objetivo)

    return indices_objetivo


# 8. Escritura de archivos
def guardar_frame(ruta_frame, frame, compresion_png):
    """Codifica un frame como PNG y lo guarda de forma segura."""
    ruta_temporal = ruta_frame.with_suffix(".png.tmp")
    parametros_png = [cv2.IMWRITE_PNG_COMPRESSION, compresion_png]
    escritura_correcta, imagen_codificada = cv2.imencode(".png", frame, parametros_png)

    if not escritura_correcta:
        raise RuntimeError(f"No se pudo codificar el frame: {ruta_frame}")

    # Primero se escribe un archivo temporal para evitar PNG incompletos.
    imagen_codificada.tofile(ruta_temporal)
    ruta_temporal.replace(ruta_frame)


def escribir_json(ruta_archivo, contenido):
    """Guarda el resumen de la extracción en formato JSON."""
    with ruta_archivo.open("w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, indent=2, ensure_ascii=False)
        archivo.write("\n")


# 9. Extracción principal
def extraer_frames(argumentos):
    """Coordina la configuración, lectura y escritura de los frames."""
    configuracion = cargar_configuracion()
    ruta_video = resolver_ruta_video(argumentos.video, configuracion)
    carpeta_frames, ruta_csv, ruta_json = obtener_rutas_salida(ruta_video)

    # Los parámetros se validan antes de eliminar una extracción anterior.
    captura = abrir_video(ruta_video)

    try:
        metadatos = leer_metadatos(captura, ruta_video)
        parametros = validar_parametros(argumentos, metadatos)
        indices_objetivo = calcular_indices_objetivo(metadatos, parametros)

        if not indices_objetivo:
            raise RuntimeError("El intervalo solicitado no produjo ningún frame.")

        total_salida = len(indices_objetivo)

        # Se utilizan al menos cuatro dígitos para mantener nombres ordenados.
        cantidad_digitos = max(4, len(str(total_salida)))

        # La extracción anterior se reemplaza después de validar la nueva operación.
        archivos_eliminados = eliminar_extraccion_anterior(carpeta_frames, ruta_csv, ruta_json)

        if archivos_eliminados > 0:
            print(f"Archivos de la extracción anterior eliminados: {archivos_eliminados}")

        # La versión actual realiza una lectura secuencial sin workers.
        cv2.setNumThreads(1)

        ruta_csv_temporal = ruta_csv.with_suffix(".csv.tmp")
        columnas = ["numero_frame", "nombre_frame", "indice_video", "tiempo_video_segundos", "fps_original", "fps_salida"]

        indice_salida = 0
        indice_video = indices_objetivo[0]
        lectura_activa = True
        tiempo_inicio = time.monotonic()

        # La lectura comienza en el primer frame requerido.
        captura.set(cv2.CAP_PROP_POS_FRAMES, indice_video)

        try:
            with ruta_csv_temporal.open("w", encoding="utf-8", newline="") as archivo_csv:
                escritor = csv.DictWriter(archivo_csv, fieldnames=columnas)
                escritor.writeheader()

                with tqdm(total=total_salida, desc=f"Extrayendo {argumentos.video}", unit="frame", dynamic_ncols=True, mininterval=1.0) as barra:
                    while indice_salida < total_salida and lectura_activa:
                        lectura_correcta, frame = captura.read()
                        lectura_activa = lectura_correcta and frame is not None

                        if lectura_activa:
                            indice_objetivo = indices_objetivo[indice_salida]

                            if indice_video == indice_objetivo:
                                numero_frame = indice_salida + 1
                                nombre_frame = f"frame_{numero_frame:0{cantidad_digitos}d}.png"
                                ruta_frame = carpeta_frames / nombre_frame

                                guardar_frame(ruta_frame, frame, argumentos.compresion_png)

                                escritor.writerow({
                                    "numero_frame": numero_frame,
                                    "nombre_frame": nombre_frame,
                                    "indice_video": indice_video,
                                    "tiempo_video_segundos": f"{indice_video / metadatos['fps']:.9f}",
                                    "fps_original": f"{metadatos['fps']:.9f}",
                                    "fps_salida": f"{parametros['fps_salida']:.9f}"
                                })

                                indice_salida += 1
                                barra.update(1)

                            indice_video += 1

            # El CSV se vuelve definitivo solamente si la extracción termina.
            ruta_csv_temporal.replace(ruta_csv)

        except BaseException:
            ruta_csv_temporal.unlink(missing_ok=True)
            raise

        tiempo_total = time.monotonic() - tiempo_inicio

        if indice_salida != total_salida:
            raise RuntimeError(f"Se esperaban {total_salida} frames, pero se extrajeron {indice_salida}.")

        resumen = {
            "video_id": argumentos.video,
            "ruta_video": str(ruta_video),
            "carpeta_frames": str(carpeta_frames),
            "codec": metadatos["codec"],
            "resolucion": {
                "ancho": metadatos["ancho"],
                "alto": metadatos["alto"]
            },
            "fps_original": metadatos["fps"],
            "fps_salida": parametros["fps_salida"],
            "conserva_fps_original": math.isclose(parametros["fps_salida"], metadatos["fps"], rel_tol=0.0, abs_tol=1e-9),
            "duracion_video_segundos": metadatos["duracion"],
            "total_frames_video": metadatos["total_frames"],
            "segundo_inicio": parametros["inicio"],
            "segundo_fin": parametros["fin"],
            "frames_extraidos": total_salida,
            "primer_frame": f"frame_{1:0{cantidad_digitos}d}.png",
            "ultimo_frame": f"frame_{total_salida:0{cantidad_digitos}d}.png",
            "compresion_png": argumentos.compresion_png,
            "tiempo_ejecucion_segundos": tiempo_total
        }

        escribir_json(ruta_json, resumen)

    finally:
        # El video se cierra incluso si ocurre un error.
        captura.release()

    print()
    print("Extracción terminada correctamente.")
    print(f"Video: {argumentos.video}")
    print(f"Archivo original: {ruta_video}")
    print(f"Resolución: {metadatos['ancho']}x{metadatos['alto']}")
    print(f"Códec: {metadatos['codec'] or 'no identificado'}")
    print(f"FPS original: {metadatos['fps']:.6f}")
    print(f"FPS de salida: {parametros['fps_salida']:.6f}")
    print(f"Intervalo: {parametros['inicio']:.3f} a {parametros['fin']:.3f} segundos")
    print(f"Frames extraídos: {total_salida}")
    print(f"Compresión PNG: {argumentos.compresion_png}")
    print(f"Carpeta de salida: {carpeta_frames}")
    print(f"Registro CSV: {ruta_csv}")
    print(f"Resumen JSON: {ruta_json}")
    print(f"Tiempo total: {tiempo_total:.2f} segundos")


# 10. Punto de entrada
def main():
    """Ejecuta el programa y presenta los errores de forma clara."""
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
