#!/usr/bin/env python3
"""Elimina el HUD de los frames mediante ProPainter."""

# 0. Variables de entorno
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# 1. Imports
import argparse
import json
import shutil
import subprocess
import sys
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import torch
from tqdm import tqdm


# 2. Configuración general
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CONFIGURACION = RUTA_PROYECTO / "config" / "videos.json"
RUTA_PROPAINTER = RUTA_PROYECTO / "ProPainter"
RUTA_INFERENCIA = RUTA_PROPAINTER / "inference_propainter.py"

PARAMETROS_PREDETERMINADOS = {
    "ancho_procesamiento": 960,
    "alto_procesamiento": 540,
    "neighbor_length": 10,
    "ref_stride": 10,
    "subvideo_length": 40,
    "frames_por_bloque": 80,
    "solapamiento_frames": 10,
    "gpus": [0, 1],
    "usar_fp16": True,
    "compresion_png": 3,
    "eliminar_temporales": True,
}


# 3. Argumentos
def obtener_argumentos():
    """Define y obtiene los argumentos del programa."""
    parser = argparse.ArgumentParser(description="Elimina el HUD de los frames de un video mediante ProPainter.")
    parser.add_argument("--video", required=True, help="Identificador registrado en config/videos.json, por ejemplo: video1.")
    return parser.parse_args()


# 4. Configuración
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


def combinar_diccionarios(configurado, predeterminado):
    """Combina recursivamente parámetros configurados y predeterminados."""
    resultado = {}

    for clave, valor in predeterminado.items():
        resultado[clave] = combinar_diccionarios({}, valor) if isinstance(valor, dict) else valor

    for clave, valor in configurado.items():
        if clave in resultado and isinstance(resultado[clave], dict) and isinstance(valor, dict):
            resultado[clave] = combinar_diccionarios(valor, resultado[clave])
        else:
            resultado[clave] = valor

    return resultado


def obtener_datos_video(video_id, configuracion):
    """Obtiene y valida la información necesaria del video."""
    if video_id not in configuracion:
        disponibles = ", ".join(sorted(configuracion))
        raise ValueError(f"El video '{video_id}' no está registrado. Videos disponibles: {disponibles}")

    datos_video = configuracion[video_id]

    if not isinstance(datos_video, dict):
        raise ValueError(f"La configuración de '{video_id}' debe ser un objeto JSON.")

    extraccion = datos_video.get("extraccion", {})

    if extraccion.get("estado") != "completada":
        raise RuntimeError(f"El video '{video_id}' no tiene una extracción de frames completada.")

    if "carpeta_salida" not in extraccion:
        raise ValueError(f"La extracción de '{video_id}' no contiene el campo 'carpeta_salida'.")

    hud = datos_video.get("hud", {})

    if hud.get("estado") != "completada":
        raise RuntimeError(f"El video '{video_id}' no tiene una segmentación del HUD completada.")

    if "carpeta_salida" not in hud:
        raise ValueError(f"La segmentación del HUD de '{video_id}' no contiene el campo 'carpeta_salida'.")

    limpieza = hud.get("limpieza", {})
    parametros_configurados = limpieza.get("parametros", {})

    if not isinstance(parametros_configurados, dict):
        raise ValueError(f"El campo hud.limpieza.parametros de '{video_id}' debe ser un objeto JSON.")

    parametros = combinar_diccionarios(parametros_configurados, PARAMETROS_PREDETERMINADOS)

    if "limpieza" not in datos_video["hud"]:
        datos_video["hud"]["limpieza"] = {}

    datos_video["hud"]["limpieza"]["parametros"] = parametros

    return datos_video, parametros


# 5. Validación de parámetros
def validar_parametros(parametros):
    """Valida los parámetros utilizados por ProPainter."""
    ancho = int(parametros["ancho_procesamiento"])
    alto = int(parametros["alto_procesamiento"])
    neighbor_length = int(parametros["neighbor_length"])
    ref_stride = int(parametros["ref_stride"])
    subvideo_length = int(parametros["subvideo_length"])
    frames_por_bloque = int(parametros["frames_por_bloque"])
    solapamiento = int(parametros["solapamiento_frames"])
    compresion_png = int(parametros["compresion_png"])
    gpus = parametros["gpus"]

    if ancho <= 0 or alto <= 0:
        raise ValueError("El ancho y el alto de procesamiento deben ser mayores que cero.")

    if ancho % 2 != 0 or alto % 2 != 0:
        raise ValueError("El ancho y el alto de procesamiento deben ser números pares.")

    if neighbor_length <= 0:
        raise ValueError("neighbor_length debe ser mayor que cero.")

    if ref_stride <= 0:
        raise ValueError("ref_stride debe ser mayor que cero.")

    if subvideo_length <= 0:
        raise ValueError("subvideo_length debe ser mayor que cero.")

    if frames_por_bloque <= 0:
        raise ValueError("frames_por_bloque debe ser mayor que cero.")

    if solapamiento < 0:
        raise ValueError("solapamiento_frames debe ser mayor o igual que cero.")

    if solapamiento >= frames_por_bloque:
        raise ValueError("solapamiento_frames debe ser menor que frames_por_bloque.")

    if subvideo_length > frames_por_bloque + 2 * solapamiento:
        raise ValueError("subvideo_length no puede superar la cantidad máxima de frames de un bloque con solapamiento.")

    if neighbor_length > subvideo_length:
        raise ValueError("neighbor_length no puede superar subvideo_length.")

    if not 0 <= compresion_png <= 9:
        raise ValueError("compresion_png debe estar entre 0 y 9.")

    if not isinstance(gpus, list) or not gpus:
        raise ValueError("hud.limpieza.parametros.gpus debe ser una lista con al menos una GPU.")

    if len(gpus) != len(set(gpus)):
        raise ValueError("La lista de GPU no puede contener valores repetidos.")

    if any(not isinstance(indice_gpu, int) or indice_gpu < 0 for indice_gpu in gpus):
        raise ValueError("Los identificadores de GPU deben ser números enteros mayores o iguales que cero.")

    return {
        "ancho_procesamiento": ancho,
        "alto_procesamiento": alto,
        "neighbor_length": neighbor_length,
        "ref_stride": ref_stride,
        "subvideo_length": subvideo_length,
        "frames_por_bloque": frames_por_bloque,
        "solapamiento_frames": solapamiento,
        "gpus": gpus,
        "usar_fp16": bool(parametros["usar_fp16"]),
        "compresion_png": compresion_png,
        "eliminar_temporales": bool(parametros["eliminar_temporales"]),
    }


# 6. Rutas
def resolver_ruta(ruta_configurada):
    """Resuelve una ruta absoluta o relativa al proyecto."""
    ruta = Path(ruta_configurada).expanduser()

    if not ruta.is_absolute():
        ruta = RUTA_PROYECTO / ruta

    return ruta.resolve()


def obtener_ruta_relativa(ruta):
    """Convierte una ruta del proyecto en una ruta relativa."""
    try:
        return str(ruta.relative_to(RUTA_PROYECTO))
    except ValueError:
        return str(ruta)


def obtener_rutas(datos_video):
    """Obtiene las rutas de frames, máscaras y resultados."""
    carpeta_frames = resolver_ruta(datos_video["extraccion"]["carpeta_salida"])
    carpeta_mascaras = resolver_ruta(datos_video["hud"]["carpeta_salida"])

    if not carpeta_frames.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta de frames: {carpeta_frames}")

    if not carpeta_mascaras.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta de máscaras: {carpeta_mascaras}")

    carpeta_video = carpeta_frames.parent
    carpeta_salida = carpeta_video / "frames_sin_hud"
    carpeta_salida_temporal = carpeta_video / "frames_sin_hud.tmp"
    carpeta_salida_respaldo = carpeta_video / "frames_sin_hud.backup"
    carpeta_trabajo = carpeta_video / ".propainter_temporal"

    return carpeta_frames, carpeta_mascaras, carpeta_salida, carpeta_salida_temporal, carpeta_salida_respaldo, carpeta_trabajo


def preparar_carpeta_vacia(carpeta):
    """Crea una carpeta vacía."""
    if carpeta.exists():
        shutil.rmtree(carpeta)

    carpeta.mkdir(parents=True, exist_ok=False)


def reemplazar_carpeta(carpeta_temporal, carpeta_definitiva, carpeta_respaldo):
    """Reemplaza el resultado anterior de forma segura."""
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


# 7. Validación de dependencias
def validar_propainter():
    """Comprueba que ProPainter y sus dependencias estén disponibles."""
    if not RUTA_PROPAINTER.is_dir():
        raise FileNotFoundError(f"No se encontró la instalación de ProPainter: {RUTA_PROPAINTER}")

    if not RUTA_INFERENCIA.is_file():
        raise FileNotFoundError(f"No se encontró el script de inferencia de ProPainter: {RUTA_INFERENCIA}")

    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("No está instalado imageio-ffmpeg en el entorno de ejecución.") from error

    return imageio_ffmpeg.get_ffmpeg_exe()


def validar_gpus(gpus):
    """Comprueba la disponibilidad de las GPU solicitadas."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible en PyTorch.")

    cantidad_gpus = torch.cuda.device_count()

    for indice_gpu in gpus:
        if indice_gpu >= cantidad_gpus:
            raise RuntimeError(f"Se solicitó la GPU {indice_gpu}, pero PyTorch solamente detecta {cantidad_gpus} GPU.")

    return cantidad_gpus


# 8. Frames y máscaras
def obtener_indice_frame(ruta_frame):
    """Obtiene el índice de un archivo con formato frame_0001.png."""
    partes = ruta_frame.stem.rsplit("_", maxsplit=1)

    if len(partes) == 2 and partes[1].isdigit():
        return int(partes[1])

    if ruta_frame.stem.isdigit():
        return int(ruta_frame.stem)

    raise ValueError(f"Nombre de imagen no válido: {ruta_frame.name}")


def listar_imagenes(carpeta):
    """Obtiene y ordena las imágenes PNG de una carpeta."""
    rutas = [ruta for ruta in carpeta.iterdir() if ruta.is_file() and ruta.suffix.lower() == ".png"]
    rutas.sort(key=obtener_indice_frame)
    return rutas


def validar_entradas(carpeta_frames, carpeta_mascaras):
    """Valida la cantidad, nombres y resolución de frames y máscaras."""
    rutas_frames = listar_imagenes(carpeta_frames)
    rutas_mascaras = listar_imagenes(carpeta_mascaras)

    if not rutas_frames:
        raise RuntimeError(f"No se encontraron frames PNG en: {carpeta_frames}")

    if not rutas_mascaras:
        raise RuntimeError(f"No se encontraron máscaras PNG en: {carpeta_mascaras}")

    if len(rutas_frames) != len(rutas_mascaras):
        raise RuntimeError(f"La cantidad de frames ({len(rutas_frames)}) no coincide con la cantidad de máscaras ({len(rutas_mascaras)}).")

    nombres_frames = [ruta.name for ruta in rutas_frames]
    nombres_mascaras = [ruta.name for ruta in rutas_mascaras]

    if nombres_frames != nombres_mascaras:
        diferencias = [(frame, mascara) for frame, mascara in zip(nombres_frames, nombres_mascaras) if frame != mascara][:20]
        raise RuntimeError(f"Los nombres de frames y máscaras no coinciden: {diferencias}")

    primer_frame = cv2.imread(str(rutas_frames[0]), cv2.IMREAD_COLOR)
    primera_mascara = cv2.imread(str(rutas_mascaras[0]), cv2.IMREAD_GRAYSCALE)

    if primer_frame is None:
        raise RuntimeError(f"No se pudo leer el primer frame: {rutas_frames[0]}")

    if primera_mascara is None:
        raise RuntimeError(f"No se pudo leer la primera máscara: {rutas_mascaras[0]}")

    if primer_frame.shape[:2] != primera_mascara.shape[:2]:
        raise RuntimeError(f"El frame tiene resolución {primer_frame.shape[:2]} y la máscara {primera_mascara.shape[:2]}.")

    if not np_any_nonzero(primera_mascara):
        print("ADVERTENCIA: la primera máscara no contiene píxeles de HUD.", file=sys.stderr)

    return rutas_frames, rutas_mascaras, primer_frame.shape[:2]


def np_any_nonzero(mascara):
    """Comprueba si una máscara contiene al menos un píxel distinto de cero."""
    return cv2.countNonZero(mascara) > 0


# 9. Preparación de bloques
def crear_enlace_o_copiar(origen, destino):
    """Crea un enlace simbólico y utiliza una copia como respaldo."""
    if destino.exists() or destino.is_symlink():
        destino.unlink()

    try:
        destino.symlink_to(origen.resolve())
    except OSError:
        shutil.copy2(origen, destino)


def preparar_bloque(indice_bloque, inicio_central, fin_central, rutas_frames, rutas_mascaras, carpeta_trabajo, solapamiento):
    """Prepara las entradas de un bloque con solapamiento temporal."""
    inicio_lectura = max(0, inicio_central - solapamiento)
    fin_lectura = min(len(rutas_frames), fin_central + solapamiento)
    carpeta_bloque = carpeta_trabajo / f"bloque_{indice_bloque:04d}"
    carpeta_entrada = carpeta_bloque / "entrada"
    carpeta_mascaras_bloque = carpeta_bloque / "mascaras"
    carpeta_salida = carpeta_bloque / "salida"

    if carpeta_bloque.exists():
        shutil.rmtree(carpeta_bloque)

    carpeta_entrada.mkdir(parents=True, exist_ok=False)
    carpeta_mascaras_bloque.mkdir(parents=True, exist_ok=False)
    carpeta_salida.mkdir(parents=True, exist_ok=False)

    for indice_local, indice_global in enumerate(range(inicio_lectura, fin_lectura)):
        nombre_temporal = f"{indice_local:06d}.png"
        crear_enlace_o_copiar(rutas_frames[indice_global], carpeta_entrada / nombre_temporal)
        crear_enlace_o_copiar(rutas_mascaras[indice_global], carpeta_mascaras_bloque / nombre_temporal)

    return {
        "indice_bloque": indice_bloque,
        "inicio_central": inicio_central,
        "fin_central": fin_central,
        "inicio_lectura": inicio_lectura,
        "fin_lectura": fin_lectura,
        "carpeta_bloque": carpeta_bloque,
        "carpeta_entrada": carpeta_entrada,
        "carpeta_mascaras": carpeta_mascaras_bloque,
        "carpeta_salida": carpeta_salida,
        "ruta_log": carpeta_bloque / "propainter.log",
    }


def crear_bloques(rutas_frames, rutas_mascaras, carpeta_trabajo, parametros):
    """Divide la secuencia en bloques centrales con solapamiento."""
    bloques = []
    frames_por_bloque = parametros["frames_por_bloque"]
    solapamiento = parametros["solapamiento_frames"]

    for inicio in range(0, len(rutas_frames), frames_por_bloque):
        fin = min(inicio + frames_por_bloque, len(rutas_frames))
        bloque = preparar_bloque(len(bloques), inicio, fin, rutas_frames, rutas_mascaras, carpeta_trabajo, solapamiento)
        bloques.append(bloque)

    return bloques


# 10. Resultados de ProPainter
def buscar_frames_generados(carpeta_salida, cantidad_esperada):
    """Busca la carpeta interna que contiene los frames generados."""
    extensiones = {".png", ".jpg", ".jpeg"}
    rutas_imagenes = [ruta for ruta in carpeta_salida.rglob("*") if ruta.is_file() and ruta.suffix.lower() in extensiones]
    grupos = {}

    for ruta in rutas_imagenes:
        grupos.setdefault(ruta.parent, []).append(ruta)

    candidatos = []

    for rutas in grupos.values():
        rutas.sort(key=obtener_indice_frame)

        if len(rutas) == cantidad_esperada:
            candidatos.append(rutas)

    if not candidatos:
        resumen = ", ".join(f"{carpeta}: {len(rutas)}" for carpeta, rutas in grupos.items())
        raise RuntimeError(f"No se encontró una carpeta con {cantidad_esperada} frames generados. Contenido encontrado: {resumen or 'ninguna imagen'}")

    candidatos.sort(key=lambda rutas: (0 if rutas[0].parent.name.lower() == "frames" else 1, len(str(rutas[0].parent))))

    return candidatos[0]


def guardar_frame_final(origen, destino, compresion_png):
    """Guarda un resultado como PNG conservando la codificación cuando sea posible."""
    if origen.suffix.lower() == ".png":
        shutil.copy2(origen, destino)
        return

    frame = cv2.imread(str(origen), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"No se pudo leer el resultado generado: {origen}")

    parametros_png = [cv2.IMWRITE_PNG_COMPRESSION, compresion_png]
    escritura_correcta, imagen_codificada = cv2.imencode(".png", frame, parametros_png)

    if not escritura_correcta:
        raise RuntimeError(f"No se pudo codificar el frame final: {destino}")

    ruta_temporal = destino.with_suffix(".png.tmp")
    imagen_codificada.tofile(ruta_temporal)
    ruta_temporal.replace(destino)


# 11. Inferencia por bloque
def ejecutar_bloque(bloque, indice_gpu, rutas_frames, carpeta_salida_temporal, parametros, ruta_ffmpeg):
    """Ejecuta ProPainter sobre un bloque y guarda su región central."""
    cantidad_lectura = bloque["fin_lectura"] - bloque["inicio_lectura"]

    comando = [
        sys.executable,
        "-u",
        str(RUTA_INFERENCIA),
        "--video",
        str(bloque["carpeta_entrada"]),
        "--mask",
        str(bloque["carpeta_mascaras"]),
        "--output",
        str(bloque["carpeta_salida"]),
        "--width",
        str(parametros["ancho_procesamiento"]),
        "--height",
        str(parametros["alto_procesamiento"]),
        "--save_frames",
        "--neighbor_length",
        str(parametros["neighbor_length"]),
        "--ref_stride",
        str(parametros["ref_stride"]),
        "--subvideo_length",
        str(parametros["subvideo_length"]),
    ]

    if parametros["usar_fp16"]:
        comando.append("--fp16")

    entorno = os.environ.copy()
    entorno["CUDA_VISIBLE_DEVICES"] = str(indice_gpu)
    entorno["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    entorno["IMAGEIO_FFMPEG_EXE"] = ruta_ffmpeg

    with bloque["ruta_log"].open("w", encoding="utf-8") as archivo_log:
        proceso = subprocess.run(comando, cwd=RUTA_PROPAINTER, env=entorno, stdout=archivo_log, stderr=subprocess.STDOUT, text=True)

    if proceso.returncode != 0:
        ultimas_lineas = bloque["ruta_log"].read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        detalle = "\n".join(ultimas_lineas)
        raise RuntimeError(f"ProPainter falló en el bloque {bloque['indice_bloque']} usando GPU {indice_gpu}.\nLog: {bloque['ruta_log']}\n\n{detalle}")

    frames_generados = buscar_frames_generados(bloque["carpeta_salida"], cantidad_lectura)
    inicio_local = bloque["inicio_central"] - bloque["inicio_lectura"]
    fin_local = inicio_local + bloque["fin_central"] - bloque["inicio_central"]

    for indice_local in range(inicio_local, fin_local):
        indice_global = bloque["inicio_lectura"] + indice_local
        ruta_destino = carpeta_salida_temporal / rutas_frames[indice_global].name
        guardar_frame_final(frames_generados[indice_local], ruta_destino, parametros["compresion_png"])

    return {
        "indice_bloque": bloque["indice_bloque"],
        "gpu": indice_gpu,
        "frames_guardados": bloque["fin_central"] - bloque["inicio_central"],
    }


# 12. Limpieza principal
def limpiar_hud(argumentos):
    """Ejecuta ProPainter y actualiza config/videos.json."""
    configuracion = cargar_configuracion()
    datos_video, parametros_configurados = obtener_datos_video(argumentos.video, configuracion)
    parametros = validar_parametros(parametros_configurados)
    ruta_ffmpeg = validar_propainter()
    cantidad_gpus = validar_gpus(parametros["gpus"])

    carpeta_frames, carpeta_mascaras, carpeta_salida, carpeta_salida_temporal, carpeta_salida_respaldo, carpeta_trabajo = obtener_rutas(datos_video)
    rutas_frames, rutas_mascaras, resolucion_original = validar_entradas(carpeta_frames, carpeta_mascaras)

    preparar_carpeta_vacia(carpeta_salida_temporal)
    preparar_carpeta_vacia(carpeta_trabajo)

    tiempo_inicio = time.monotonic()
    bloques = crear_bloques(rutas_frames, rutas_mascaras, carpeta_trabajo, parametros)
    gpus = parametros["gpus"]
    bloqueo_progreso = threading.Lock()
    progreso = tqdm(total=len(rutas_frames), desc="Procesando HUD con ProPainter", unit="frame", mininterval=2.0, dynamic_ncols=True)
    resultados = []
    errores = []
    bloqueo_errores = threading.Lock()

    def trabajador_gpu(indice_gpu, bloques_gpu):
        """Procesa secuencialmente los bloques asignados a una GPU."""
        resultados_gpu = []

        for bloque in bloques_gpu:
            try:
                resultado = ejecutar_bloque(bloque, indice_gpu, rutas_frames, carpeta_salida_temporal, parametros, ruta_ffmpeg)
                resultados_gpu.append(resultado)

                with bloqueo_progreso:
                    progreso.update(resultado["frames_guardados"])
            except Exception as error:
                with bloqueo_errores:
                    errores.append((bloque["indice_bloque"], indice_gpu, str(error)))

                break

        return resultados_gpu

    bloques_por_gpu = {}

    for posicion_gpu, indice_gpu in enumerate(gpus):
        bloques_por_gpu[indice_gpu] = [bloque for posicion, bloque in enumerate(bloques) if posicion % len(gpus) == posicion_gpu]

    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as ejecutor:
            futuros = [ejecutor.submit(trabajador_gpu, indice_gpu, bloques_por_gpu[indice_gpu]) for indice_gpu in gpus]

            for futuro in futuros:
                resultados.extend(futuro.result())

        progreso.close()

        if errores:
            detalle = "\n\n".join(f"Bloque {indice_bloque}, GPU {indice_gpu}:\n{mensaje}" for indice_bloque, indice_gpu, mensaje in errores)
            raise RuntimeError(f"Fallaron uno o más bloques:\n\n{detalle}")

        rutas_resultados = listar_imagenes(carpeta_salida_temporal)

        if len(rutas_resultados) != len(rutas_frames):
            raise RuntimeError(f"Se esperaban {len(rutas_frames)} frames sin HUD, pero se generaron {len(rutas_resultados)}.")

        nombres_originales = [ruta.name for ruta in rutas_frames]
        nombres_resultados = [ruta.name for ruta in rutas_resultados]

        if nombres_resultados != nombres_originales:
            raise RuntimeError("Los nombres de los resultados no coinciden con los frames originales.")

        reemplazar_carpeta(carpeta_salida_temporal, carpeta_salida, carpeta_salida_respaldo)
        tiempo_total = time.monotonic() - tiempo_inicio

        configuracion[argumentos.video]["hud"]["limpieza"] = {
            "parametros": parametros,
            "estado": "completada",
            "metodo": "ProPainter",
            "carpeta_frames": obtener_ruta_relativa(carpeta_frames),
            "carpeta_mascaras": obtener_ruta_relativa(carpeta_mascaras),
            "carpeta_salida": obtener_ruta_relativa(carpeta_salida),
            "resolucion_entrada": {
                "ancho": resolucion_original[1],
                "alto": resolucion_original[0],
            },
            "resolucion_salida": {
                "ancho": parametros["ancho_procesamiento"],
                "alto": parametros["alto_procesamiento"],
            },
            "frames_procesados": len(rutas_frames),
            "bloques_procesados": len(resultados),
            "gpu_utilizadas": gpus,
            "gpu_detectadas": cantidad_gpus,
            "fecha_hora_local": datetime.now().astimezone().isoformat(timespec="seconds"),
            "tiempo_ejecucion_segundos": tiempo_total,
        }

        guardar_configuracion(configuracion)
    except BaseException:
        progreso.close()

        if carpeta_salida_temporal.exists():
            shutil.rmtree(carpeta_salida_temporal)

        raise
    finally:
        if parametros["eliminar_temporales"] and carpeta_trabajo.exists():
            shutil.rmtree(carpeta_trabajo)

    print()
    print("Limpieza del HUD terminada correctamente.")
    print(f"Video: {argumentos.video}")
    print(f"Frames procesados: {len(rutas_frames)}")
    print(f"Bloques procesados: {len(resultados)}")
    print(f"GPU utilizadas: {gpus}")
    print(f"Resolución original: {resolucion_original[1]}x{resolucion_original[0]}")
    print(f"Resolución de salida: {parametros['ancho_procesamiento']}x{parametros['alto_procesamiento']}")
    print(f"Carpeta de salida: {carpeta_salida}")
    print(f"Tiempo total: {tiempo_total:.2f} segundos")


# 13. Punto de entrada
def main():
    """Ejecuta el programa y presenta los errores."""
    try:
        argumentos = obtener_argumentos()
        limpiar_hud(argumentos)
        return 0
    except KeyboardInterrupt:
        print("\nLimpieza del HUD interrumpida por el usuario.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())