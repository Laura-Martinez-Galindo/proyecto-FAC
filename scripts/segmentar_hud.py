#!/usr/bin/env python3
"""Segmenta el HUD en los frames extraídos de un video registrado."""

# 0. Variables de entorno
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# 1. Imports
import argparse
import json
import math
import shutil
import sys
import time

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# 2. Configuración general
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CONFIGURACION = RUTA_PROYECTO / "config" / "videos.json"

PARAMETROS_PREDETERMINADOS = {
    "frames_por_bloque": 250,
    "rangos_hsv": {
        "verde": {"h_min": 35, "h_max": 95, "s_min": 60, "v_min": 60},
        "rojo_1": {"h_min": 0, "h_max": 15, "s_min": 60, "v_min": 60},
        "rojo_2": {"h_min": 165, "h_max": 179, "s_min": 60, "v_min": 60},
    },
    "umbrales_frecuencia": {
        "estricto": 0.30,
        "hud_superior": 0.10,
        "brujula": 0.08,
        "hud_inferior": 0.10,
        "panel_lateral": 0.18,
        "crosshair": 0.22,
        "proteccion": 0.65,
    },
    "morfologia": {
        "radio_proximidad_ancla": 10,
        "margen_superior_sin_hud": 0.02,
        "radio_soporte": 2,
        "tamano_cierre": 3,
        "iteraciones_cierre": 1,
        "area_minima": 2,
        "tamano_ventana_densidad": 41,
        "umbral_densidad_local": 0.72,
        "radio_proteccion": 2,
        "tamano_dilatacion": 3,
        "iteraciones_dilatacion": 1,
    },
    "compresion_png": 3,
}


# 3. Argumentos
def obtener_argumentos():
    """Define y obtiene los argumentos del programa."""
    parser = argparse.ArgumentParser(description="Segmenta el HUD de los frames extraídos de un video registrado.")
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
    """Completa recursivamente parámetros ausentes con valores predeterminados."""
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
    """Obtiene el video, valida la extracción y carga los parámetros del HUD."""
    if video_id not in configuracion:
        disponibles = ", ".join(sorted(configuracion))
        raise ValueError(f"El video '{video_id}' no está registrado. Videos disponibles: {disponibles}")

    datos_video = configuracion[video_id]

    if not isinstance(datos_video, dict):
        raise ValueError(f"La configuración de '{video_id}' debe ser un objeto JSON.")

    extraccion = datos_video.get("extraccion", {})

    if extraccion.get("estado") != "completada":
        raise RuntimeError(f"El video '{video_id}' no tiene una extracción completada.")

    if "carpeta_salida" not in extraccion:
        raise ValueError(f"La extracción de '{video_id}' no contiene 'carpeta_salida'.")

    hud_actual = datos_video.get("hud", {})

    if not isinstance(hud_actual, dict):
        raise ValueError(f"El campo 'hud' de '{video_id}' debe ser un objeto JSON.")

    parametros_configurados = hud_actual.get("parametros", {})

    if not isinstance(parametros_configurados, dict):
        raise ValueError(f"El campo 'hud.parametros' de '{video_id}' debe ser un objeto JSON.")

    parametros = combinar_diccionarios(parametros_configurados, PARAMETROS_PREDETERMINADOS)
    return datos_video, parametros


# 5. Rutas
def resolver_ruta(ruta_configurada):
    """Resuelve una ruta absoluta o relativa al proyecto."""
    ruta = Path(ruta_configurada).expanduser()

    if not ruta.is_absolute():
        ruta = RUTA_PROYECTO / ruta

    return ruta.resolve()


def obtener_ruta_relativa(ruta):
    """Convierte una ruta del proyecto en relativa."""
    try:
        return str(ruta.relative_to(RUTA_PROYECTO))
    except ValueError:
        return str(ruta)


def obtener_rutas(datos_video):
    """Obtiene la carpeta registrada por la extracción y las rutas de salida."""
    carpeta_frames = resolver_ruta(datos_video["extraccion"]["carpeta_salida"])

    if not carpeta_frames.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta registrada por la extracción: {carpeta_frames}")

    carpeta_video = carpeta_frames.parent
    carpeta_mascaras = carpeta_video / "mascaras_hud"
    carpeta_temporal = carpeta_video / "mascaras_hud.tmp"
    carpeta_respaldo = carpeta_video / "mascaras_hud.backup"
    return carpeta_frames, carpeta_mascaras, carpeta_temporal, carpeta_respaldo


def preparar_carpeta_temporal(carpeta_temporal):
    """Crea una carpeta temporal vacía."""
    if carpeta_temporal.exists():
        shutil.rmtree(carpeta_temporal)

    carpeta_temporal.mkdir(parents=True, exist_ok=False)


def reemplazar_carpeta(carpeta_temporal, carpeta_definitiva, carpeta_respaldo):
    """Reemplaza las máscaras anteriores de forma segura."""
    if carpeta_respaldo.exists():
        shutil.rmtree(carpeta_respaldo)

    try:
        if carpeta_definitiva.exists():
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


# 6. Frames
def obtener_indice_frame(ruta_frame):
    """Obtiene el índice de un archivo con formato frame_0001.png."""
    partes = ruta_frame.stem.rsplit("_", maxsplit=1)

    if len(partes) == 2 and partes[1].isdigit():
        return int(partes[1])

    if ruta_frame.stem.isdigit():
        return int(ruta_frame.stem)

    raise ValueError(f"Nombre de frame no válido: {ruta_frame.name}")


def listar_frames(carpeta_frames):
    """Obtiene y ordena los frames PNG."""
    rutas = [ruta for ruta in carpeta_frames.iterdir() if ruta.is_file() and ruta.suffix.lower() == ".png"]
    rutas.sort(key=obtener_indice_frame)

    if not rutas:
        raise RuntimeError(f"No se encontraron frames PNG en: {carpeta_frames}")

    return rutas


# 7. Segmentación HSV
def obtener_mascara_hsv(frame, parametros):
    """Obtiene los candidatos verdes y rojos del HUD."""
    rangos = parametros["rangos_hsv"]
    verde = rangos["verde"]
    rojo_1 = rangos["rojo_1"]
    rojo_2 = rangos["rojo_2"]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara_verde = cv2.inRange(hsv, np.array([verde["h_min"], verde["s_min"], verde["v_min"]], dtype=np.uint8), np.array([verde["h_max"], 255, 255], dtype=np.uint8))
    mascara_rojo_1 = cv2.inRange(hsv, np.array([rojo_1["h_min"], rojo_1["s_min"], rojo_1["v_min"]], dtype=np.uint8), np.array([rojo_1["h_max"], 255, 255], dtype=np.uint8))
    mascara_rojo_2 = cv2.inRange(hsv, np.array([rojo_2["h_min"], rojo_2["s_min"], rojo_2["v_min"]], dtype=np.uint8), np.array([rojo_2["h_max"], 255, 255], dtype=np.uint8))
    return cv2.bitwise_or(mascara_verde, cv2.bitwise_or(mascara_rojo_1, mascara_rojo_2))


def contar_candidatos(argumentos):
    """Genera una máscara binaria de candidatos para un frame."""
    ruta_frame, parametros = argumentos
    frame = cv2.imread(str(ruta_frame), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"No se pudo leer el frame: {ruta_frame}")

    return (obtener_mascara_hsv(frame, parametros) > 0).astype(np.uint8)


# 8. Soporte temporal
def calcular_soporte_bloque(argumentos):
    """Calcula el soporte temporal de un bloque."""
    indice_bloque, rutas_bloque, forma, parametros, trabajadores = argumentos
    acumulador = np.zeros(forma, dtype=np.uint16)
    entradas = [(ruta, parametros) for ruta in rutas_bloque]

    with ThreadPoolExecutor(max_workers=trabajadores) as ejecutor:
        for mascara in ejecutor.map(contar_candidatos, entradas):
            if mascara.shape != forma:
                raise RuntimeError(f"Se encontró un frame con resolución diferente: {mascara.shape}")

            acumulador += mascara

    frecuencia = acumulador.astype(np.float32) / len(rutas_bloque)
    alto, ancho = forma
    umbrales = parametros["umbrales_frecuencia"]
    morfologia = parametros["morfologia"]
    umbrales_permisivos = np.full(forma, np.inf, dtype=np.float32)
    mascara_crosshair = np.zeros(forma, dtype=np.uint8)

    def asignar_region(x_min, x_max, y_min, y_max, umbral):
        x1 = round(ancho * x_min)
        x2 = round(ancho * x_max)
        y1 = round(alto * y_min)
        y2 = round(alto * y_max)
        umbrales_permisivos[y1:y2, x1:x2] = np.minimum(umbrales_permisivos[y1:y2, x1:x2], umbral)

    asignar_region(0.00, 0.30, 0.02, 0.15, umbrales["hud_superior"])
    asignar_region(0.37, 0.63, 0.02, 0.14, umbrales["brujula"])
    asignar_region(0.70, 1.00, 0.02, 0.15, umbrales["hud_superior"])
    asignar_region(0.00, 0.14, 0.15, 0.34, umbrales["panel_lateral"])
    asignar_region(0.86, 1.00, 0.15, 0.34, umbrales["panel_lateral"])
    asignar_region(0.00, 0.14, 0.64, 0.98, umbrales["panel_lateral"])
    asignar_region(0.86, 1.00, 0.64, 0.98, umbrales["panel_lateral"])
    asignar_region(0.00, 0.30, 0.80, 0.99, umbrales["hud_inferior"])
    asignar_region(0.36, 0.64, 0.80, 0.99, umbrales["hud_inferior"])
    asignar_region(0.70, 1.00, 0.80, 0.99, umbrales["hud_inferior"])

    x1 = round(ancho * 0.39)
    x2 = round(ancho * 0.61)
    y1 = round(alto * 0.36)
    y2 = round(alto * 0.66)
    mascara_crosshair[y1:y2, x1:x2] = 255

    soporte_estricto = np.where(frecuencia >= umbrales["estricto"], 255, 0).astype(np.uint8)
    soporte_proteccion = np.where(frecuencia >= umbrales["proteccion"], 255, 0).astype(np.uint8)
    regiones_validas = np.where(np.isfinite(umbrales_permisivos), 255, 0).astype(np.uint8)
    regiones_validas = cv2.bitwise_or(regiones_validas, mascara_crosshair)
    soporte_proteccion = cv2.bitwise_and(soporte_proteccion, regiones_validas)
    tamano_ancla = 2 * morfologia["radio_proximidad_ancla"] + 1
    kernel_ancla = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano_ancla, tamano_ancla))
    proximidad_ancla = cv2.dilate(soporte_estricto, kernel_ancla, iterations=1)
    soporte_permisivo = np.where(frecuencia >= umbrales_permisivos, 255, 0).astype(np.uint8)
    soporte_permisivo = cv2.bitwise_and(soporte_permisivo, proximidad_ancla)
    soporte_crosshair = np.where(frecuencia >= umbrales["crosshair"], 255, 0).astype(np.uint8)
    soporte_crosshair = cv2.bitwise_and(soporte_crosshair, mascara_crosshair)
    soporte_base = cv2.bitwise_or(soporte_permisivo, soporte_crosshair)

    if morfologia["radio_soporte"] > 0:
        tamano_soporte = 2 * morfologia["radio_soporte"] + 1
        kernel_soporte = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano_soporte, tamano_soporte))
        soporte = cv2.dilate(soporte_base, kernel_soporte, iterations=1)
    else:
        soporte = soporte_base.copy()

    margen_superior = round(alto * morfologia["margen_superior_sin_hud"])
    soporte[:margen_superior, :] = 0
    return indice_bloque, frecuencia, soporte, soporte_proteccion


# 9. Procesamiento por frame
def limpiar_componentes(mascara, area_minima):
    """Elimina componentes menores que el área configurada."""
    cantidad, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    mascara_limpia = np.zeros_like(mascara)

    for etiqueta in range(1, cantidad):
        area = int(estadisticas[etiqueta, cv2.CC_STAT_AREA])

        if area >= area_minima:
            mascara_limpia[etiquetas == etiqueta] = 255

    return mascara_limpia


def guardar_mascara(ruta_mascara, mascara, compresion_png):
    """Guarda una máscara PNG mediante un archivo temporal."""
    ruta_temporal = ruta_mascara.with_suffix(".png.tmp")
    parametros_png = [cv2.IMWRITE_PNG_COMPRESSION, compresion_png]
    escritura_correcta, mascara_codificada = cv2.imencode(".png", mascara, parametros_png)

    if not escritura_correcta:
        raise RuntimeError(f"No se pudo codificar la máscara: {ruta_mascara}")

    mascara_codificada.tofile(ruta_temporal)
    ruta_temporal.replace(ruta_mascara)


def procesar_frame(argumentos):
    """Genera la máscara final y devuelve su porcentaje de HUD."""
    ruta_frame, frecuencia, soporte, soporte_proteccion, carpeta_salida, parametros = argumentos
    morfologia = parametros["morfologia"]
    frame = cv2.imread(str(ruta_frame), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"No se pudo leer el frame: {ruta_frame}")

    mascara_hsv = obtener_mascara_hsv(frame, parametros)
    mascara_interseccion = cv2.bitwise_and(mascara_hsv, soporte)
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morfologia["tamano_cierre"], morfologia["tamano_cierre"]))
    mascara_cerrada = cv2.morphologyEx(mascara_interseccion, cv2.MORPH_CLOSE, kernel_cierre, iterations=morfologia["iteraciones_cierre"])
    mascara_binaria = (mascara_cerrada > 0).astype(np.float32)
    tamano_ventana = morfologia["tamano_ventana_densidad"]
    densidad_local = cv2.boxFilter(mascara_binaria, cv2.CV_32F, (tamano_ventana, tamano_ventana), normalize=True, borderType=cv2.BORDER_REPLICATE)

    if morfologia["radio_proteccion"] > 0:
        tamano_proteccion = 2 * morfologia["radio_proteccion"] + 1
        kernel_proteccion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano_proteccion, tamano_proteccion))
        proteccion_expandida = cv2.dilate(soporte_proteccion, kernel_proteccion, iterations=1)
    else:
        proteccion_expandida = soporte_proteccion

    eliminar_densidad = (densidad_local >= morfologia["umbral_densidad_local"]) & (proteccion_expandida == 0)
    mascara_refinada = mascara_cerrada.copy()
    mascara_refinada[eliminar_densidad] = 0
    mascara_limpia = limpiar_componentes(mascara_refinada, morfologia["area_minima"])

    if morfologia["iteraciones_dilatacion"] > 0:
        tamano_dilatacion = morfologia["tamano_dilatacion"]
        kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano_dilatacion, tamano_dilatacion))
        mascara_final = cv2.dilate(mascara_limpia, kernel_dilatacion, iterations=morfologia["iteraciones_dilatacion"])
    else:
        mascara_final = mascara_limpia

    guardar_mascara(carpeta_salida / ruta_frame.name, mascara_final, parametros["compresion_png"])
    return 100.0 * np.count_nonzero(mascara_final) / mascara_final.size


# 10. Segmentación principal
def segmentar_hud(argumentos):
    """Segmenta el HUD y actualiza config/videos.json."""
    configuracion = cargar_configuracion()
    datos_video, parametros = obtener_datos_video(argumentos.video, configuracion)
    carpeta_frames, carpeta_mascaras, carpeta_temporal, carpeta_respaldo = obtener_rutas(datos_video)
    rutas_frames = listar_frames(carpeta_frames)
    primer_frame = cv2.imread(str(rutas_frames[0]), cv2.IMREAD_COLOR)

    if primer_frame is None:
        raise RuntimeError(f"No se pudo leer el primer frame: {rutas_frames[0]}")

    forma = primer_frame.shape[:2]
    trabajadores = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE") or min(16, os.cpu_count() or 1))
    trabajadores = max(1, trabajadores)
    frames_por_bloque = int(parametros["frames_por_bloque"])

    if frames_por_bloque <= 0:
        raise ValueError("hud.parametros.frames_por_bloque debe ser mayor que cero.")

    bloques = [rutas_frames[inicio:inicio + frames_por_bloque] for inicio in range(0, len(rutas_frames), frames_por_bloque)]
    preparar_carpeta_temporal(carpeta_temporal)
    cv2.setNumThreads(1)
    tiempo_inicio = time.monotonic()

    try:
        argumentos_bloques = [(indice, rutas_bloque, forma, parametros, trabajadores) for indice, rutas_bloque in enumerate(bloques)]
        soportes = {}

        for argumento_bloque in tqdm(argumentos_bloques, desc="Calculando soportes", unit="bloque"):
            indice_bloque, frecuencia, soporte, soporte_proteccion = calcular_soporte_bloque(argumento_bloque)
            soportes[indice_bloque] = (frecuencia, soporte, soporte_proteccion)

        argumentos_frames = []

        for posicion, ruta_frame in enumerate(rutas_frames):
            indice_bloque = posicion // frames_por_bloque
            frecuencia, soporte, soporte_proteccion = soportes[indice_bloque]
            argumentos_frames.append((ruta_frame, frecuencia, soporte, soporte_proteccion, carpeta_temporal, parametros))

        with ThreadPoolExecutor(max_workers=trabajadores) as ejecutor:
            resultados = ejecutor.map(procesar_frame, argumentos_frames)
            porcentajes_hud = list(tqdm(resultados, total=len(argumentos_frames), desc="Segmentando HUD", unit="frame", mininterval=2.0))

        cantidad_mascaras = len([ruta for ruta in carpeta_temporal.glob("frame_*.png") if ruta.is_file()])

        if cantidad_mascaras != len(rutas_frames):
            raise RuntimeError(f"Se esperaban {len(rutas_frames)} máscaras, pero se generaron {cantidad_mascaras}.")

        reemplazar_carpeta(carpeta_temporal, carpeta_mascaras, carpeta_respaldo)
        tiempo_total = time.monotonic() - tiempo_inicio
        porcentaje_hud_promedio = float(np.mean(porcentajes_hud))

        configuracion[argumentos.video]["hud"] = {
            "parametros": parametros,
            "estado": "completada",
            "carpeta_entrada": obtener_ruta_relativa(carpeta_frames),
            "carpeta_salida": obtener_ruta_relativa(carpeta_mascaras),
            "frames_procesados": len(rutas_frames),
            "mascaras_generadas": cantidad_mascaras,
            "bloques_temporales": len(bloques),
            "porcentaje_hud_promedio": porcentaje_hud_promedio,
            "fecha_ejecucion": datetime.now().astimezone().isoformat(timespec="seconds"),
            "tiempo_ejecucion_minutos": tiempo_total / 60.0,
        }

        guardar_configuracion(configuracion)
    except BaseException:
        if carpeta_temporal.exists():
            shutil.rmtree(carpeta_temporal)

        raise

    print()
    print("Segmentación del HUD terminada correctamente.")
    print(f"Video: {argumentos.video}")
    print(f"Carpeta de entrada: {carpeta_frames}")
    print(f"Frames procesados: {len(rutas_frames)}")
    print(f"Máscaras generadas: {cantidad_mascaras}")
    print(f"Porcentaje promedio de HUD: {porcentaje_hud_promedio:.4f} %")
    print(f"Carpeta de salida: {carpeta_mascaras}")
    print(f"Tiempo total: {tiempo_total / 60.0:.2f} minutos")


# 11. Punto de entrada
def main():
    """Ejecuta el programa y presenta los errores."""
    try:
        argumentos = obtener_argumentos()
        segmentar_hud(argumentos)
        return 0
    except KeyboardInterrupt:
        print("\nSegmentación del HUD interrumpida por el usuario.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())