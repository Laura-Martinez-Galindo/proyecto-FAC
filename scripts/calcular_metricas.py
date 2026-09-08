#!/usr/bin/env python3
"""Calcula metricas de calidad sin referencia (NIQE, BRISQUE, PIQE, Sigma Ruido, Nitidez) y actualiza Excel."""

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


# 1. Configuracion general
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CONFIGURACION = RUTA_PROYECTO / "config" / "videos.json"

NOMBRES_MODELOS = {
    "original": "Original",
    "sin_hud": "Sin HUD",
    "n2n": "Noise2Noise",
    "n2v": "Noise2Void",
    "neighbor2neighbor": "Neighbor2Neighbor",
    "blind2unblind": "Blind2Unblind",
    "frames2residual": "Frames2Residual",
    "frame_to_frame": "Frame-to-Frame",
}

COLUMNAS_RESUMEN = [
    "Experimento",
    "Modelo",
    "Modo",
    "Carpeta",
    "Frames evaluados",
    "NIQE media",
    "NIQE mediana",
    "NIQE desviacion",
    "BRISQUE media",
    "BRISQUE mediana",
    "BRISQUE desviacion",
    "PIQE media",
    "PIQE mediana",
    "PIQE desviacion",
    "Sigma Ruido media",
    "Sigma Ruido mediana",
    "Nitidez Laplaciana media",
    "Retencion Nitidez media",
    "Dispositivo",
    "Fecha de ejecucion",
    "Tiempo de ejecucion (min)",
    "Parametros extra",
]


# 2. Argumentos
def obtener_argumentos():
    """Define y obtiene los argumentos del programa."""
    parser = argparse.ArgumentParser(description="Calcula metricas completas sin referencia para un modelo y actualiza Excel.")
    parser.add_argument("--video", required=True, help="Identificador del video, por ejemplo: video1.")
    parser.add_argument("--modelo", required=True, help="Modelo que se evaluara (original, sin_hud, n2n, n2v, neighbor2neighbor, etc.).")
    parser.add_argument("--carpeta-directa", help="Ruta directa de frames para evaluar (opcional, sobreescribe config/videos.json).")
    parser.add_argument("--experimento", help="Nombre o identificador descriptivo del experimento para la tabla.")
    parser.add_argument("--archivo-resumen", default="resumen.xlsx", help="Nombre del archivo Excel de salida dentro de la carpeta del video (ej. resumen.xlsx o resumen_experimentos.xlsx).")
    parser.add_argument("--max-frames", type=int, help="Limite maximo de frames a evaluar.")
    parser.add_argument("--parametros-extra", default="", help="Texto o JSON con detalles de hiperparametros.")
    return parser.parse_args()


# 3. Configuracion
def cargar_configuracion():
    """Carga y valida config/videos.json."""
    if not RUTA_CONFIGURACION.is_file():
        raise FileNotFoundError(f"No se encontro la configuracion: {RUTA_CONFIGURACION}")

    try:
        with RUTA_CONFIGURACION.open("r", encoding="utf-8") as archivo:
            configuracion = json.load(archivo)
    except json.JSONDecodeError as error:
        raise ValueError(f"El archivo {RUTA_CONFIGURACION} no contiene JSON valido: {error}") from error

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
        raise ValueError("La ruta configurada esta vacia.")

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
        raise ValueError("La configuracion del video no contiene el campo 'ruta'.")

    ruta_video = resolver_ruta(datos_video["ruta"])

    if ruta_video.parent.name != "original":
        raise ValueError(f"El video debe estar dentro de una carpeta llamada 'original': {ruta_video}")

    return ruta_video.parent.parent


def resolver_carpetas(video_id, modelo_id, carpeta_directa, archivo_resumen, configuracion):
    """Resuelve carpetas de entrada, original y archivo Excel."""
    if video_id not in configuracion:
        disponibles = ", ".join(sorted(configuracion))
        raise ValueError(f"El video '{video_id}' no esta registrado. Videos disponibles: {disponibles}")

    datos_video = configuracion[video_id]
    carpeta_video = obtener_carpeta_video(datos_video)

    extraccion = datos_video.get("extraccion", {})
    carpeta_original = resolver_ruta(extraccion.get("carpeta_salida")) if extraccion.get("carpeta_salida") else None

    if carpeta_directa:
        carpeta_entrada = resolver_ruta(carpeta_directa)
        nombre_modelo = obtener_nombre_modelo(modelo_id)
        modo = "directo"
    elif modelo_id == "original":
        if extraccion.get("estado") != "completada":
            raise RuntimeError(f"La extraccion de '{video_id}' no esta completada.")
        nombre_modelo = obtener_nombre_modelo(modelo_id)
        carpeta_entrada = resolver_ruta(extraccion["carpeta_salida"])
        modo = "original"
    elif modelo_id == "sin_hud":
        limpieza = datos_video.get("hud", {}).get("limpieza", {})
        if limpieza.get("estado") != "completada":
            raise RuntimeError(f"La limpieza del HUD de '{video_id}' no esta completada.")
        nombre_modelo = obtener_nombre_modelo(modelo_id)
        carpeta_entrada = resolver_ruta(limpieza["carpeta_salida"])
        modo = "sin_hud"
    else:
        modelos = datos_video.get("modelos", {})
        
        # 1. Buscar en modelos (exacto o insensible a mayusculas)
        match_key = None
        if modelo_id in modelos:
            match_key = modelo_id
        else:
            for k in modelos:
                if k.lower() == modelo_id.lower():
                    match_key = k
                    break
        
        if match_key:
            datos_modelo = modelos[match_key]
            nombre_modelo = obtener_nombre_modelo(match_key, datos_modelo)
            carpeta_entrada = resolver_ruta(datos_modelo["carpeta_salida"])
            modo = datos_modelo.get("modo", "original" if "sin_hud" not in match_key.lower() else "sin_hud")
        else:
            # 2. Buscar carpeta fisica en el disco
            carpeta_entrada = None
            for p in carpeta_video.iterdir():
                if p.is_dir() and p.name.lower() == modelo_id.lower():
                    carpeta_entrada = p
                    break
            
            if carpeta_entrada is not None:
                nombre_modelo = modelo_id
                modo = "original" if "sin_hud" not in modelo_id.lower() else "sin_hud"
            else:
                disponibles = ", ".join(sorted(modelos)) or "ninguno"
                raise ValueError(f"El modelo/experimento '{modelo_id}' no se encontro en config/videos.json ni en {carpeta_video}. Disponibles: {disponibles}")

    if not carpeta_entrada.is_dir():
        raise FileNotFoundError(f"No se encontro la carpeta de frames: {carpeta_entrada}")

    ruta_resumen = carpeta_video / archivo_resumen
    ruta_bloqueo = carpeta_video / f".{Path(archivo_resumen).stem}.lock"

    return nombre_modelo, modo, carpeta_entrada, carpeta_original, ruta_resumen, ruta_bloqueo


# 5. Lectura de frames
def natural_key(p):
    import re
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def listar_frames(carpeta_entrada, max_frames=None):
    """Obtiene y ordena los frames disponibles."""
    extensiones = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    rutas_frames = [ruta for ruta in carpeta_entrada.iterdir() if ruta.is_file() and ruta.suffix.lower() in extensiones]
    rutas_frames.sort(key=natural_key)

    if not rutas_frames:
        raise RuntimeError(f"No se encontraron imagenes en: {carpeta_entrada}")

    return rutas_frames[:max_frames] if max_frames else rutas_frames


def cargar_imagen_rgb(ruta_imagen):
    """Carga imagen RGB como array uint8 y tensor float [0, 1]."""
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
        raise RuntimeError(f"Formato no compatible en {ruta_imagen}: {imagen.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(imagen)).permute(2, 0, 1).float().div(255.0)
    return imagen, tensor.unsqueeze(0)


# 6. Metricas adicionales (Ruido y Nitidez)
def estimar_sigma_ruido(imagen_rgb):
    """Estima la desviacion estandar del ruido mediante MAD en altas frecuencias (Wavelet/Laplacian)."""
    gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    # Filtro pasa-altas Laplaciano
    lap = cv2.Laplacian(gris, cv2.CV_64F)
    # Estimador robusto basado en Median Absolute Deviation (MAD)
    mediana = np.median(lap)
    mad = np.median(np.abs(lap - mediana))
    # Factor de escala para distribucion normal gaussiana
    sigma = mad / 0.6745
    return float(sigma)


def calcular_nitidez_laplaciana(imagen_rgb):
    """Calcula la varianza del Laplaciano como indicador de nitidez de bordes."""
    gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
    varianza = cv2.Laplacian(gris, cv2.CV_64F).var()
    return float(varianza)


# 7. Estadisticas descriptivas
def calcular_estadisticas(valores):
    """Calcula estadisticas descriptivas."""
    arreglo = np.asarray(valores, dtype=np.float64)

    if arreglo.size == 0:
        return {"media": 0.0, "mediana": 0.0, "desviacion": 0.0, "minimo": 0.0, "maximo": 0.0}

    arreglo = arreglo[np.isfinite(arreglo)]
    if arreglo.size == 0:
        return {"media": 0.0, "mediana": 0.0, "desviacion": 0.0, "minimo": 0.0, "maximo": 0.0}

    return {
        "media": float(np.mean(arreglo)),
        "mediana": float(np.median(arreglo)),
        "desviacion": float(np.std(arreglo, ddof=0)),
        "minimo": float(np.min(arreglo)),
        "maximo": float(np.max(arreglo)),
    }


def crear_metricas_iqa(dispositivo):
    """Inicializa NIQE, BRISQUE y PIQE con PyIQA."""
    metrica_niqe = pyiqa.create_metric("niqe", device=dispositivo)
    metrica_brisque = pyiqa.create_metric("brisque", device=dispositivo)
    try:
        metrica_piqe = pyiqa.create_metric("piqe", device=dispositivo)
    except Exception:
        metrica_piqe = None

    return metrica_niqe, metrica_brisque, metrica_piqe


def calcular_todas_las_metricas(rutas_frames, carpeta_original=None):
    """Calcula la suite completa de metricas sin referencia frame por frame."""
    dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metrica_niqe, metrica_brisque, metrica_piqe = crear_metricas_iqa(dispositivo)

    valores_niqe = []
    valores_brisque = []
    valores_piqe = []
    valores_sigma = []
    valores_nitidez = []
    valores_retencion = []

    tiene_original = carpeta_original is not None and carpeta_original.is_dir()

    with torch.inference_mode():
        for i, ruta_frame in enumerate(tqdm(rutas_frames, desc="Calculando metricas completas", unit="frame", dynamic_ncols=True, mininterval=1.0)):
            img_np, entrada = cargar_imagen_rgb(ruta_frame)
            entrada_gpu = entrada.to(dispositivo, non_blocking=True)

            # NIQE
            try:
                v_niqe = float(metrica_niqe(entrada_gpu).detach().float().cpu().item())
                if math.isfinite(v_niqe): valores_niqe.append(v_niqe)
            except Exception:
                pass

            # BRISQUE
            try:
                v_brisque = float(metrica_brisque(entrada_gpu).detach().float().cpu().item())
                if math.isfinite(v_brisque): valores_brisque.append(v_brisque)
            except Exception:
                pass

            # PIQE
            if metrica_piqe is not None:
                try:
                    v_piqe = float(metrica_piqe(entrada_gpu).detach().float().cpu().item())
                    if math.isfinite(v_piqe): valores_piqe.append(v_piqe)
                except Exception:
                    pass

            del entrada_gpu

            # Sigma de Ruido Estimado
            s_ruido = estimar_sigma_ruido(img_np)
            valores_sigma.append(s_ruido)

            # Nitidez Laplaciana
            nit = calcular_nitidez_laplaciana(img_np)
            valores_nitidez.append(nit)

            # Retencion de nitidez frente al original
            if tiene_original:
                ruta_orig = carpeta_original / ruta_frame.name
                if ruta_orig.is_file():
                    img_orig_np, _ = cargar_imagen_rgb(ruta_orig)
                    nit_orig = calcular_nitidez_laplaciana(img_orig_np)
                    if nit_orig > 1e-4:
                        valores_retencion.append(float(nit / nit_orig))

    del metrica_niqe, metrica_brisque, metrica_piqe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "niqe": calcular_estadisticas(valores_niqe),
        "brisque": calcular_estadisticas(valores_brisque),
        "piqe": calcular_estadisticas(valores_piqe),
        "sigma": calcular_estadisticas(valores_sigma),
        "nitidez": calcular_estadisticas(valores_nitidez),
        "retencion": calcular_estadisticas(valores_retencion) if valores_retencion else {"media": 1.0, "mediana": 1.0, "desviacion": 0.0},
        "dispositivo": str(dispositivo),
    }


# 8. Excel
def configurar_hoja(hoja):
    """Aplica formato a la hoja de metricas."""
    hoja.title = "Metricas"
    hoja.freeze_panes = "A2"
    hoja.sheet_view.showGridLines = False

    relleno_encabezado = PatternFill(fill_type="solid", fgColor="1F4E78")
    fuente_encabezado = Font(color="FFFFFF", bold=True)
    anchos = [24, 20, 14, 40, 16, 14, 14, 14, 14, 14, 14, 14, 14, 14, 16, 16, 18, 18, 14, 24, 20, 30]

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

        hoja = libro["Metricas"] if "Metricas" in libro.sheetnames else libro.active

        encabezados_actuales = [hoja.cell(row=1, column=c).value for c in range(1, len(COLUMNAS_RESUMEN) + 1)]
        if hoja.max_row == 1 and all(v is None for v in encabezados_actuales):
            configurar_hoja(hoja)
        elif encabezados_actuales != COLUMNAS_RESUMEN:
            configurar_hoja(hoja)
    else:
        libro = Workbook()
        hoja = libro.active
        configurar_hoja(hoja)

    return libro, hoja


def buscar_fila_registro(hoja, id_experimento, nombre_modelo):
    """Busca la fila del experimento/modelo para actualizar o devuelve una fila nueva."""
    clave_buscada = (id_experimento or nombre_modelo).strip().lower()
    for fila in range(2, hoja.max_row + 1):
        val_exp = hoja.cell(row=fila, column=1).value
        val_mod = hoja.cell(row=fila, column=2).value
        if val_exp and str(val_exp).strip().lower() == clave_buscada:
            return fila
        if not id_experimento and val_mod and str(val_mod).strip().lower() == clave_buscada:
            return fila

    return hoja.max_row + 1


def guardar_resumen_excel(ruta_resumen, ruta_bloqueo, datos):
    """Crea o actualiza la fila de metricas en el archivo Excel."""
    ruta_bloqueo.touch(exist_ok=True)

    with ruta_bloqueo.open("r+") as archivo_bloqueo:
        fcntl.flock(archivo_bloqueo.fileno(), fcntl.LOCK_EX)

        try:
            libro, hoja = abrir_resumen(ruta_resumen)
            fila = buscar_fila_registro(hoja, datos.get("experimento"), datos["modelo"])

            valores = [
                datos.get("experimento") or datos["modelo"],
                datos["modelo"],
                datos.get("modo", "original"),
                obtener_ruta_relativa(datos["carpeta"]),
                datos["cantidad_frames"],
                datos["niqe"]["media"],
                datos["niqe"]["mediana"],
                datos["niqe"]["desviacion"],
                datos["brisque"]["media"],
                datos["brisque"]["mediana"],
                datos["brisque"]["desviacion"],
                datos["piqe"]["media"],
                datos["piqe"]["mediana"],
                datos["piqe"]["desviacion"],
                datos["sigma"]["media"],
                datos["sigma"]["mediana"],
                datos["nitidez"]["media"],
                datos["retencion"]["media"],
                datos["dispositivo"],
                datos["fecha"],
                datos["tiempo_minutos"],
                datos.get("parametros_extra", ""),
            ]

            for columna, valor in enumerate(valores, start=1):
                celda = hoja.cell(row=fila, column=columna, value=valor)
                celda.alignment = Alignment(horizontal="left" if columna in {1, 2, 3, 4, 22} else "center", vertical="center")

            hoja.cell(row=fila, column=1).font = Font(bold=True)
            hoja.cell(row=fila, column=2).font = Font(bold=True)

            for columna in range(6, 19):
                hoja.cell(row=fila, column=columna).number_format = "0.0000"

            hoja.cell(row=fila, column=21).number_format = "0.00"
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


# 9. Ejecucion
def ejecutar(args):
    """Calcula las metricas y actualiza el archivo Excel correspondiente."""
    configuracion = cargar_configuracion()
    modelo_id = normalizar_modelo(args.modelo)

    nombre_modelo, modo, carpeta_entrada, carpeta_original, ruta_resumen, ruta_bloqueo = resolver_carpetas(
        args.video, modelo_id, args.carpeta_directa, args.archivo_resumen, configuracion
    )

    rutas_frames = listar_frames(carpeta_entrada, args.max_frames)

    id_exp = args.experimento or f"{modelo_id}_{modo}"
    print(f"Iniciando calculo de metricas para {id_exp}...")
    print(f"Video: {args.video} | Modelo: {nombre_modelo} | Modo: {modo}")
    print(f"Carpeta: {carpeta_entrada}")
    print(f"Frames a evaluar: {len(rutas_frames)}")

    tiempo_inicio = time.monotonic()
    resultados = calcular_todas_las_metricas(rutas_frames, carpeta_original)
    tiempo_total = (time.monotonic() - tiempo_inicio) / 60.0
    fecha = datetime.now().astimezone().isoformat(timespec="seconds")

    datos = {
        "experimento": id_exp,
        "modelo": nombre_modelo,
        "modo": modo,
        "carpeta": carpeta_entrada,
        "cantidad_frames": len(rutas_frames),
        "niqe": resultados["niqe"],
        "brisque": resultados["brisque"],
        "piqe": resultados["piqe"],
        "sigma": resultados["sigma"],
        "nitidez": resultados["nitidez"],
        "retencion": resultados["retencion"],
        "dispositivo": resultados["dispositivo"],
        "fecha": fecha,
        "tiempo_minutos": tiempo_total,
        "parametros_extra": args.parametros_extra,
    }

    guardar_resumen_excel(ruta_resumen, ruta_bloqueo, datos)

    print("\nMetricas calculadas con exito:")
    print(f"  NIQE media (↓): {resultados['niqe']['media']:.4f}")
    print(f"  BRISQUE media (↓): {resultados['brisque']['media']:.4f}")
    print(f"  PIQE media (↓): {resultados['piqe']['media']:.4f}")
    print(f"  Sigma Ruido media (↓): {resultados['sigma']['media']:.4f}")
    print(f"  Nitidez Laplaciana: {resultados['nitidez']['media']:.2f}")
    print(f"  Retencion de Nitidez: {resultados['retencion']['media']:.4f}")
    print(f"Guardado en: {ruta_resumen}")


def main():
    args = obtener_argumentos()
    ejecutar(args)


if __name__ == "__main__":
    main()