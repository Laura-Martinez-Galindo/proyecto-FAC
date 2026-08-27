#!/usr/bin/env python3
"""Calcula NIQE y BRISQUE para un modelo registrado y actualiza resumen.xlsx."""

# 0. Imports
import argparse
import fcntl
import json
import math
import sys
import time

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyiqa
import torch

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from tqdm import tqdm


# 1. Configuración general
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CONFIGURACION = RUTA_PROYECTO / "config" / "videos.json"

NOMBRES_MODELOS = {
    "original": "Original",
    "sin_hud": "Sin HUD",
    "n2n": "Noise2Noise",
    "n2v": "Noise2Void",
    "frames2residual": "Frames2Residual",
    "frame_to_frame": "Frame-to-Frame",
}

COLUMNAS_RESUMEN = [
    "Modelo",
    "Carpeta",
    "Frames evaluados",
    "NIQE media",
    "NIQE mediana",
    "NIQE desviación",
    "NIQE mínimo",
    "NIQE máximo",
    "BRISQUE media",
    "BRISQUE mediana",
    "BRISQUE desviación",
    "BRISQUE mínimo",
    "BRISQUE máximo",
    "Dispositivo",
    "Fecha de ejecución",
    "Tiempo de ejecución (min)",
]


# 2. Argumentos
def obtener_argumentos():
    """Define y obtiene los argumentos del programa."""
    parser = argparse.ArgumentParser(description="Calcula NIQE y BRISQUE para un modelo registrado en config/videos.json.")
    parser.add_argument("--video", required=True, help="Identificador del video, por ejemplo: video1.")
    parser.add_argument("--modelo", required=True, help="Modelo que se evaluará, por ejemplo: original, sin_hud, n2n o n2v.")
    return parser.parse_args()


# 3. Configuración
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


def normalizar_modelo(modelo):
    """Normaliza el identificador de un modelo."""
    return modelo.strip().lower().replace("-", "_").replace(" ", "_")


def obtener_nombre_modelo(modelo_id, datos_modelo=None):
    """Obtiene el nombre legible del modelo."""
    if datos_modelo and datos_modelo.get("nombre"):
        return str(datos_modelo["nombre"])

    return NOMBRES_MODELOS.get(modelo_id, modelo_id.replace("_", " ").title())


# 4. Rutas
def resolver_ruta(ruta_configurada):
    """Resuelve una ruta absoluta o relativa al proyecto."""
    if not ruta_configurada:
        raise ValueError("La ruta configurada está vacía.")

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


def obtener_carpeta_video(datos_video):
    """Obtiene la carpeta principal del video."""
    if "ruta" not in datos_video:
        raise ValueError("La configuración del video no contiene el campo 'ruta'.")

    ruta_video = resolver_ruta(datos_video["ruta"])

    if ruta_video.parent.name != "original":
        raise ValueError(f"El video debe estar dentro de una carpeta llamada 'original': {ruta_video}")

    return ruta_video.parent.parent


def obtener_carpeta_modelo(video_id, modelo_id, configuracion):
    """Obtiene la carpeta de frames correspondiente al modelo."""
    if video_id not in configuracion:
        disponibles = ", ".join(sorted(configuracion))
        raise ValueError(f"El video '{video_id}' no está registrado. Videos disponibles: {disponibles}")

    datos_video = configuracion[video_id]

    if not isinstance(datos_video, dict):
        raise ValueError(f"La configuración de '{video_id}' debe ser un objeto JSON.")

    carpeta_video = obtener_carpeta_video(datos_video)

    if modelo_id == "original":
        extraccion = datos_video.get("extraccion", {})

        if extraccion.get("estado") != "completada":
            raise RuntimeError(f"La extracción de '{video_id}' no está completada.")

        if "carpeta_salida" not in extraccion:
            raise ValueError(f"La extracción de '{video_id}' no contiene 'carpeta_salida'.")

        nombre_modelo = obtener_nombre_modelo(modelo_id)
        carpeta_entrada = resolver_ruta(extraccion["carpeta_salida"])

    elif modelo_id == "sin_hud":
        limpieza = datos_video.get("hud", {}).get("limpieza", {})

        if limpieza.get("estado") != "completada":
            raise RuntimeError(f"La limpieza del HUD de '{video_id}' no está completada.")

        if "carpeta_salida" not in limpieza:
            raise ValueError(f"La limpieza del HUD de '{video_id}' no contiene 'carpeta_salida'.")

        nombre_modelo = obtener_nombre_modelo(modelo_id)
        carpeta_entrada = resolver_ruta(limpieza["carpeta_salida"])

    else:
        modelos = datos_video.get("modelos", {})

        if not isinstance(modelos, dict):
            raise ValueError(f"El campo 'modelos' de '{video_id}' debe ser un objeto JSON.")

        if modelo_id not in modelos:
            disponibles = ", ".join(sorted(modelos)) or "ninguno"
            raise ValueError(f"El modelo '{modelo_id}' no está registrado para '{video_id}'. Modelos disponibles: {disponibles}")

        datos_modelo = modelos[modelo_id]

        if not isinstance(datos_modelo, dict):
            raise ValueError(f"La configuración de modelos.{modelo_id} debe ser un objeto JSON.")

        if datos_modelo.get("estado") != "completada":
            raise RuntimeError(f"El modelo '{modelo_id}' no tiene estado 'completada'.")

        if "carpeta_salida" not in datos_modelo:
            raise ValueError(f"El modelo '{modelo_id}' no contiene 'carpeta_salida'.")

        nombre_modelo = obtener_nombre_modelo(modelo_id, datos_modelo)
        carpeta_entrada = resolver_ruta(datos_modelo["carpeta_salida"])

    if not carpeta_entrada.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta de frames para '{modelo_id}': {carpeta_entrada}")

    ruta_resumen = carpeta_video / "resumen.xlsx"
    ruta_bloqueo = carpeta_video / ".resumen.lock"

    return nombre_modelo, carpeta_entrada, ruta_resumen, ruta_bloqueo


# 5. Lectura de frames
def obtener_indice_frame(ruta_frame):
    """Obtiene el índice numérico de un frame."""
    partes = ruta_frame.stem.rsplit("_", maxsplit=1)

    if len(partes) == 2 and partes[1].isdigit():
        return int(partes[1])

    if ruta_frame.stem.isdigit():
        return int(ruta_frame.stem)

    raise ValueError(f"Nombre de frame no válido: {ruta_frame.name}")


def listar_frames(carpeta_entrada):
    """Obtiene y ordena los frames disponibles."""
    extensiones = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    rutas_frames = [ruta for ruta in carpeta_entrada.iterdir() if ruta.is_file() and ruta.suffix.lower() in extensiones]
    rutas_frames.sort(key=obtener_indice_frame)

    if not rutas_frames:
        raise RuntimeError(f"No se encontraron imágenes en: {carpeta_entrada}")

    return rutas_frames


def cargar_imagen_tensor(ruta_imagen):
    """Carga una imagen y la convierte en tensor RGB entre cero y uno."""
    imagen = cv2.imread(str(ruta_imagen), cv2.IMREAD_UNCHANGED)

    if imagen is None:
        raise RuntimeError(f"No se pudo leer la imagen: {ruta_imagen}")

    if imagen.ndim == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
    elif imagen.ndim == 3 and imagen.shape[2] == 4:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGRA2RGB)
    elif imagen.ndim == 3 and imagen.shape[2] == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    else:
        raise RuntimeError(f"Formato de imagen no compatible en {ruta_imagen}: {imagen.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(imagen)).permute(2, 0, 1).float().div(255.0)
    return tensor.unsqueeze(0)


# 6. Cálculo de métricas
def calcular_estadisticas(valores):
    """Calcula estadísticas descriptivas."""
    arreglo = np.asarray(valores, dtype=np.float64)

    if arreglo.size == 0:
        raise RuntimeError("No se recibieron valores para calcular estadísticas.")

    if not np.all(np.isfinite(arreglo)):
        raise RuntimeError("Las métricas contienen valores no finitos.")

    return {
        "media": float(np.mean(arreglo)),
        "mediana": float(np.median(arreglo)),
        "desviacion": float(np.std(arreglo, ddof=0)),
        "minimo": float(np.min(arreglo)),
        "maximo": float(np.max(arreglo)),
    }


def crear_metricas(dispositivo):
    """Crea las métricas NIQE y BRISQUE."""
    try:
        metrica_niqe = pyiqa.create_metric("niqe", device=dispositivo)
        metrica_brisque = pyiqa.create_metric("brisque", device=dispositivo)
    except Exception as error:
        raise RuntimeError(f"No se pudieron inicializar NIQE y BRISQUE en {dispositivo}: {error}") from error

    return metrica_niqe, metrica_brisque


def calcular_metricas_frames(rutas_frames):
    """Calcula NIQE y BRISQUE frame por frame."""
    dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metrica_niqe, metrica_brisque = crear_metricas(dispositivo)
    valores_niqe = []
    valores_brisque = []

    with torch.inference_mode():
        for ruta_frame in tqdm(rutas_frames, desc="Calculando NIQE y BRISQUE", unit="frame", dynamic_ncols=True, mininterval=1.0):
            entrada = cargar_imagen_tensor(ruta_frame).to(dispositivo, non_blocking=True)

            try:
                valor_niqe = float(metrica_niqe(entrada).detach().float().cpu().item())
                valor_brisque = float(metrica_brisque(entrada).detach().float().cpu().item())
            except Exception as error:
                raise RuntimeError(f"No se pudieron calcular las métricas para {ruta_frame.name}: {error}") from error
            finally:
                del entrada

            if not math.isfinite(valor_niqe):
                raise RuntimeError(f"NIQE produjo un valor no válido en: {ruta_frame}")

            if not math.isfinite(valor_brisque):
                raise RuntimeError(f"BRISQUE produjo un valor no válido en: {ruta_frame}")

            valores_niqe.append(valor_niqe)
            valores_brisque.append(valor_brisque)

    return calcular_estadisticas(valores_niqe), calcular_estadisticas(valores_brisque), str(dispositivo)


# 7. Excel
def configurar_hoja(hoja):
    """Aplica formato a la hoja de métricas."""
    hoja.title = "Métricas"
    hoja.freeze_panes = "A2"
    hoja.sheet_view.showGridLines = False

    relleno_encabezado = PatternFill(fill_type="solid", fgColor="1F4E78")
    fuente_encabezado = Font(color="FFFFFF", bold=True)
    anchos = [24, 44, 18, 15, 15, 17, 15, 15, 16, 16, 18, 16, 16, 14, 24, 26]

    for columna, encabezado in enumerate(COLUMNAS_RESUMEN, start=1):
        celda = hoja.cell(row=1, column=columna, value=encabezado)
        celda.fill = relleno_encabezado
        celda.font = fuente_encabezado
        celda.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        hoja.column_dimensions[get_column_letter(columna)].width = anchos[columna - 1]

    hoja.row_dimensions[1].height = 34


def abrir_resumen(ruta_resumen):
    """Abre resumen.xlsx o crea un libro nuevo."""
    if ruta_resumen.is_file():
        try:
            libro = load_workbook(ruta_resumen)
        except Exception as error:
            raise RuntimeError(f"No se pudo abrir {ruta_resumen}: {error}") from error

        if "Métricas" in libro.sheetnames:
            hoja = libro["Métricas"]
        else:
            hoja = libro.active

        encabezados_actuales = [hoja.cell(row=1, column=columna).value for columna in range(1, len(COLUMNAS_RESUMEN) + 1)]

        if hoja.max_row == 1 and all(valor is None for valor in encabezados_actuales):
            configurar_hoja(hoja)
        elif encabezados_actuales != COLUMNAS_RESUMEN:
            raise ValueError(f"La estructura de {ruta_resumen} no coincide con la estructura esperada.")
    else:
        libro = Workbook()
        hoja = libro.active
        configurar_hoja(hoja)

    return libro, hoja


def buscar_fila_modelo(hoja, modelo):
    """Busca la fila del modelo o devuelve una fila nueva."""
    for fila in range(2, hoja.max_row + 1):
        valor = hoja.cell(row=fila, column=1).value

        if valor is not None and str(valor).strip().lower() == modelo.strip().lower():
            return fila

    return hoja.max_row + 1


def guardar_resumen(ruta_resumen, ruta_bloqueo, nombre_modelo, carpeta_entrada, cantidad_frames, niqe, brisque, dispositivo, fecha, tiempo_minutos):
    """Crea o actualiza la fila correspondiente al modelo."""
    ruta_bloqueo.touch(exist_ok=True)

    with ruta_bloqueo.open("r+") as archivo_bloqueo:
        fcntl.flock(archivo_bloqueo.fileno(), fcntl.LOCK_EX)

        try:
            libro, hoja = abrir_resumen(ruta_resumen)
            fila = buscar_fila_modelo(hoja, nombre_modelo)

            valores = [
                nombre_modelo,
                obtener_ruta_relativa(carpeta_entrada),
                cantidad_frames,
                niqe["media"],
                niqe["mediana"],
                niqe["desviacion"],
                niqe["minimo"],
                niqe["maximo"],
                brisque["media"],
                brisque["mediana"],
                brisque["desviacion"],
                brisque["minimo"],
                brisque["maximo"],
                dispositivo,
                fecha,
                tiempo_minutos,
            ]

            for columna, valor in enumerate(valores, start=1):
                celda = hoja.cell(row=fila, column=columna, value=valor)
                celda.alignment = Alignment(horizontal="left" if columna in {1, 2} else "center", vertical="center")

            hoja.cell(row=fila, column=1).font = Font(bold=True)

            for columna in range(4, 14):
                hoja.cell(row=fila, column=columna).number_format = "0.0000"

            hoja.cell(row=fila, column=16).number_format = "0.00"
            hoja.row_dimensions[fila].height = 22

            ruta_temporal = ruta_resumen.with_name(f"{ruta_resumen.stem}.tmp.xlsx")

            try:
                libro.save(ruta_temporal)
                ruta_temporal.replace(ruta_resumen)
            except BaseException:
                ruta_temporal.unlink(missing_ok=True)
                raise
        finally:
            fcntl.flock(archivo_bloqueo.fileno(), fcntl.LOCK_UN)


# 8. Ejecución principal
def ejecutar(argumentos):
    """Calcula las métricas y actualiza únicamente resumen.xlsx."""
    configuracion = cargar_configuracion()
    modelo_id = normalizar_modelo(argumentos.modelo)
    nombre_modelo, carpeta_entrada, ruta_resumen, ruta_bloqueo = obtener_carpeta_modelo(argumentos.video, modelo_id, configuracion)
    rutas_frames = listar_frames(carpeta_entrada)

    print("Iniciando cálculo de métricas.")
    print(f"Video: {argumentos.video}")
    print(f"Modelo: {nombre_modelo}")
    print(f"Carpeta: {carpeta_entrada}")
    print(f"Frames encontrados: {len(rutas_frames)}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    tiempo_inicio = time.monotonic()
    estadisticas_niqe, estadisticas_brisque, dispositivo = calcular_metricas_frames(rutas_frames)
    tiempo_total = time.monotonic() - tiempo_inicio
    tiempo_minutos = tiempo_total / 60.0
    fecha = datetime.now().astimezone().isoformat(timespec="seconds")

    guardar_resumen(
        ruta_resumen=ruta_resumen,
        ruta_bloqueo=ruta_bloqueo,
        nombre_modelo=nombre_modelo,
        carpeta_entrada=carpeta_entrada,
        cantidad_frames=len(rutas_frames),
        niqe=estadisticas_niqe,
        brisque=estadisticas_brisque,
        dispositivo=dispositivo,
        fecha=fecha,
        tiempo_minutos=tiempo_minutos,
    )

    print()
    print("Métricas calculadas correctamente.")
    print(f"Video: {argumentos.video}")
    print(f"Modelo: {nombre_modelo}")
    print(f"Frames evaluados: {len(rutas_frames)}")
    print(f"NIQE promedio: {estadisticas_niqe['media']:.4f}")
    print(f"NIQE mediana: {estadisticas_niqe['mediana']:.4f}")
    print(f"BRISQUE promedio: {estadisticas_brisque['media']:.4f}")
    print(f"BRISQUE mediana: {estadisticas_brisque['mediana']:.4f}")
    print(f"Dispositivo: {dispositivo}")
    print(f"Resumen: {ruta_resumen}")
    print(f"Tiempo total: {tiempo_minutos:.2f} minutos")


# 9. Punto de entrada
def main():
    """Ejecuta el programa y presenta errores."""
    try:
        argumentos = obtener_argumentos()
        ejecutar(argumentos)
        return 0
    except KeyboardInterrupt:
        print("\nCálculo de métricas interrumpido por el usuario.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())