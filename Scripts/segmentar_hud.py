# %% 0. Imports y configuración
import os
import shutil

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm


# Ruta principal del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parent.parent

# Ruta del video asociado con los frames
RUTA_VIDEO = RUTA_PROYECTO / "Videos" / "video30min-11to22.mp4"

# Carpetas de entrada y salida
CARPETA_VIDEO = RUTA_VIDEO.parent / RUTA_VIDEO.stem
CARPETA_FRAMES = CARPETA_VIDEO / "frames"
CARPETA_MASCARAS = CARPETA_VIDEO / "masks"
CARPETA_DIAGNOSTICOS = CARPETA_VIDEO / "diagnostics"
CARPETA_DIAGNOSTICOS_ANTIGUA = CARPETA_VIDEO / "diagnosticos"

# CSV con información para evaluar posibles fallos
RUTA_CSV_DIAGNOSTICOS = CARPETA_VIDEO / "diagnostics.csv"

# Rangos HSV para el HUD verde
VERDE_H_MIN = 35
VERDE_H_MAX = 95
VERDE_S_MIN = 60
VERDE_V_MIN = 60

# Rangos HSV para el HUD rojo
ROJO_1_H_MIN = 0
ROJO_1_H_MAX = 15
ROJO_2_H_MIN = 165
ROJO_2_H_MAX = 179
ROJO_S_MIN = 60
ROJO_V_MIN = 60

# Cantidad de frames por bloque temporal
# 250 frames equivalen a 50 segundos cuando se extraen 5 FPS
FRAMES_POR_BLOQUE = 250

# Umbral estricto utilizado para construir anclas confiables
UMBRAL_FRECUENCIA_ESTRICTO = 0.30

# Umbrales permisivos dentro de las regiones conocidas del HUD
UMBRAL_HUD_SUPERIOR = 0.10
UMBRAL_BRUJULA = 0.08
UMBRAL_HUD_INFERIOR = 0.10
UMBRAL_PANEL_LATERAL = 0.18
UMBRAL_CROSSHAIR = 0.22

# Distancia máxima entre una detección permisiva y un ancla estricta
RADIO_PROXIMIDAD_ANCLA = 10

# Margen superior sin HUD
MARGEN_SUPERIOR_SIN_HUD = 0.02

# Expansión ligera del soporte antes de intersectar con HSV
RADIO_SOPORTE = 2

# Cierre posterior a la intersección para conectar huecos pequeños
TAMANO_CIERRE = 3
ITERACIONES_CIERRE = 1

# Eliminación de componentes extremadamente pequeños
AREA_MINIMA = 2

# Supresión de regiones localmente densas
# Las letras, líneas, escalas y anillos del HUD suelen ser estructuras delgadas
TAMANO_VENTANA_DENSIDAD = 41
UMBRAL_DENSIDAD_LOCAL = 0.72

# Los píxeles muy persistentes se protegen del filtro de densidad
UMBRAL_FRECUENCIA_PROTECCION = 0.65
RADIO_PROTECCION = 2

# Dilatación final suave para ampliar los trazos del HUD
TAMANO_DILATACION = 3
ITERACIONES_DILATACION = 1

# Color BGR y transparencia del HUD en los diagnósticos
COLOR_HUD = (0, 0, 255)
ALFA_HUD = 0.60

# Compresión PNG entre 0 y 9
COMPRESION_PNG = 3

# Cantidad de trabajadores
# En SLURM toma automáticamente --cpus-per-task
NUM_TRABAJADORES = int(
    os.environ.get("SLURM_CPUS_PER_TASK")
    or os.environ.get("SLURM_CPUS_ON_NODE")
    or min(16, os.cpu_count() or 1)
)

# Control de resultados existentes
SOBRESCRIBIR = True
ELIMINAR_RESULTADOS_ANTERIORES = True


# %% 1. Funciones auxiliares
def obtener_indice_frame(ruta_frame):
    try:
        return int(ruta_frame.stem)
    except ValueError:
        return ruta_frame.stem


def obtener_mascara_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara_verde = cv2.inRange(hsv, np.array([VERDE_H_MIN, VERDE_S_MIN, VERDE_V_MIN], dtype=np.uint8), np.array([VERDE_H_MAX, 255, 255], dtype=np.uint8))
    mascara_rojo_1 = cv2.inRange(hsv, np.array([ROJO_1_H_MIN, ROJO_S_MIN, ROJO_V_MIN], dtype=np.uint8), np.array([ROJO_1_H_MAX, 255, 255], dtype=np.uint8))
    mascara_rojo_2 = cv2.inRange(hsv, np.array([ROJO_2_H_MIN, ROJO_S_MIN, ROJO_V_MIN], dtype=np.uint8), np.array([ROJO_2_H_MAX, 255, 255], dtype=np.uint8))
    mascara_roja = cv2.bitwise_or(mascara_rojo_1, mascara_rojo_2)
    mascara_hsv = cv2.bitwise_or(mascara_verde, mascara_roja)

    return mascara_verde, mascara_roja, mascara_hsv


def limpiar_componentes(mascara):
    cantidad, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    mascara_limpia = np.zeros_like(mascara)
    areas = []

    for etiqueta in range(1, cantidad):
        area = int(estadisticas[etiqueta, cv2.CC_STAT_AREA])

        if area >= AREA_MINIMA:
            mascara_limpia[etiquetas == etiqueta] = 255
            areas.append(area)

    return mascara_limpia, areas


def contar_candidatos(ruta_frame):
    frame = cv2.imread(str(ruta_frame), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"No se pudo leer el frame: {ruta_frame}")

    _, _, mascara_hsv = obtener_mascara_hsv(frame)

    return (mascara_hsv > 0).astype(np.uint8)


def calcular_soporte_bloque(argumentos):
    indice_bloque, rutas_bloque, forma = argumentos
    acumulador = np.zeros(forma, dtype=np.uint16)

    with ThreadPoolExecutor(max_workers=NUM_TRABAJADORES) as ejecutor:
        for mascara in ejecutor.map(contar_candidatos, rutas_bloque):
            if mascara.shape != forma:
                raise RuntimeError(f"Se encontró un frame con resolución diferente: {mascara.shape}")

            acumulador += mascara

    frecuencia = acumulador.astype(np.float32) / len(rutas_bloque)
    alto, ancho = forma
    umbrales_permisivos = np.full(forma, np.inf, dtype=np.float32)
    mascara_crosshair = np.zeros(forma, dtype=np.uint8)

    def asignar_region(x_min, x_max, y_min, y_max, umbral):
        x1 = round(ancho * x_min)
        x2 = round(ancho * x_max)
        y1 = round(alto * y_min)
        y2 = round(alto * y_max)
        umbrales_permisivos[y1:y2, x1:x2] = np.minimum(umbrales_permisivos[y1:y2, x1:x2], umbral)

    # HUD superior con regiones ajustadas a la altura real de los textos
    asignar_region(0.00, 0.30, 0.02, 0.15, UMBRAL_HUD_SUPERIOR)
    asignar_region(0.37, 0.63, 0.02, 0.14, UMBRAL_BRUJULA)
    asignar_region(0.70, 1.00, 0.02, 0.15, UMBRAL_HUD_SUPERIOR)

    # Paneles laterales
    asignar_region(0.00, 0.14, 0.15, 0.34, UMBRAL_PANEL_LATERAL)
    asignar_region(0.86, 1.00, 0.15, 0.34, UMBRAL_PANEL_LATERAL)
    asignar_region(0.00, 0.14, 0.64, 0.98, UMBRAL_PANEL_LATERAL)
    asignar_region(0.86, 1.00, 0.64, 0.98, UMBRAL_PANEL_LATERAL)

    # HUD inferior
    asignar_region(0.00, 0.30, 0.80, 0.99, UMBRAL_HUD_INFERIOR)
    asignar_region(0.36, 0.64, 0.80, 0.99, UMBRAL_HUD_INFERIOR)
    asignar_region(0.70, 1.00, 0.80, 0.99, UMBRAL_HUD_INFERIOR)

    # Crosshair tratado de forma independiente y con mayor exigencia
    x1 = round(ancho * 0.39)
    x2 = round(ancho * 0.61)
    y1 = round(alto * 0.36)
    y2 = round(alto * 0.66)
    mascara_crosshair[y1:y2, x1:x2] = 255

    soporte_estricto = np.where(frecuencia >= UMBRAL_FRECUENCIA_ESTRICTO, 255, 0).astype(np.uint8)
    soporte_proteccion = np.where(frecuencia >= UMBRAL_FRECUENCIA_PROTECCION, 255, 0).astype(np.uint8)
    regiones_validas = np.where(np.isfinite(umbrales_permisivos), 255, 0).astype(np.uint8)
    regiones_validas = cv2.bitwise_or(regiones_validas, mascara_crosshair)
    soporte_proteccion = cv2.bitwise_and(soporte_proteccion, regiones_validas)
    tamano_ancla = 2 * RADIO_PROXIMIDAD_ANCLA + 1
    kernel_ancla = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano_ancla, tamano_ancla))
    proximidad_ancla = cv2.dilate(soporte_estricto, kernel_ancla, iterations=1)

    soporte_permisivo = np.where(frecuencia >= umbrales_permisivos, 255, 0).astype(np.uint8)
    soporte_permisivo = cv2.bitwise_and(soporte_permisivo, proximidad_ancla)
    soporte_crosshair = np.where(frecuencia >= UMBRAL_CROSSHAIR, 255, 0).astype(np.uint8)
    soporte_crosshair = cv2.bitwise_and(soporte_crosshair, mascara_crosshair)
    soporte_base = cv2.bitwise_or(soporte_permisivo, soporte_crosshair)

    if RADIO_SOPORTE > 0:
        tamano_soporte = 2 * RADIO_SOPORTE + 1
        kernel_soporte = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano_soporte, tamano_soporte))
        soporte = cv2.dilate(soporte_base, kernel_soporte, iterations=1)
    else:
        soporte = soporte_base.copy()

    margen_superior = round(alto * MARGEN_SUPERIOR_SIN_HUD)
    soporte_base[:margen_superior, :] = 0
    soporte[:margen_superior, :] = 0

    return indice_bloque, frecuencia, soporte_base, soporte, soporte_proteccion


def procesar_frame(argumentos):
    posicion, ruta_frame, indice_bloque, frecuencia, soporte_base, soporte, soporte_proteccion = argumentos
    frame = cv2.imread(str(ruta_frame), cv2.IMREAD_COLOR)

    if frame is None:
        return {"posicion": posicion, "frame": ruta_frame.name, "error": "No se pudo leer el frame"}

    mascara_verde, mascara_roja, mascara_hsv = obtener_mascara_hsv(frame)
    mascara_interseccion_base = cv2.bitwise_and(mascara_hsv, soporte_base)
    mascara_interseccion = cv2.bitwise_and(mascara_hsv, soporte)
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TAMANO_CIERRE, TAMANO_CIERRE))
    mascara_cerrada = cv2.morphologyEx(mascara_interseccion, cv2.MORPH_CLOSE, kernel_cierre, iterations=ITERACIONES_CIERRE)

    mascara_binaria = (mascara_cerrada > 0).astype(np.float32)
    densidad_local = cv2.boxFilter(
        mascara_binaria,
        cv2.CV_32F,
        (TAMANO_VENTANA_DENSIDAD, TAMANO_VENTANA_DENSIDAD),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )

    if RADIO_PROTECCION > 0:
        tamano_proteccion = 2 * RADIO_PROTECCION + 1
        kernel_proteccion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano_proteccion, tamano_proteccion))
        proteccion_expandida = cv2.dilate(soporte_proteccion, kernel_proteccion, iterations=1)
    else:
        proteccion_expandida = soporte_proteccion

    mascara_protegida = cv2.bitwise_and(mascara_cerrada, proteccion_expandida)
    eliminar_densidad = (densidad_local >= UMBRAL_DENSIDAD_LOCAL) & (proteccion_expandida == 0)
    mascara_refinada = mascara_cerrada.copy()
    mascara_refinada[eliminar_densidad] = 0
    pixeles_protegidos = int(np.count_nonzero(mascara_protegida))
    pixeles_eliminados_densidad = int(np.count_nonzero(mascara_cerrada)) - int(np.count_nonzero(mascara_refinada))
    mascara_limpia, areas_finales = limpiar_componentes(mascara_refinada)
    pixeles_antes_dilatacion = int(np.count_nonzero(mascara_limpia))

    if ITERACIONES_DILATACION > 0:
        kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TAMANO_DILATACION, TAMANO_DILATACION))
        mascara_final = cv2.dilate(mascara_limpia, kernel_dilatacion, iterations=ITERACIONES_DILATACION)
    else:
        mascara_final = mascara_limpia

    ruta_mascara = CARPETA_MASCARAS / ruta_frame.name
    ruta_diagnostico = CARPETA_DIAGNOSTICOS / ruta_frame.name
    overlay = frame.copy()
    overlay[mascara_final > 0] = COLOR_HUD
    diagnostico = cv2.addWeighted(frame, 1.0 - ALFA_HUD, overlay, ALFA_HUD, 0.0)

    if SOBRESCRIBIR or not ruta_mascara.exists():
        escritura_mascara = cv2.imwrite(str(ruta_mascara), mascara_final, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESION_PNG])
        escritura_diagnostico = cv2.imwrite(str(ruta_diagnostico), diagnostico, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESION_PNG])

        if not escritura_mascara:
            return {"posicion": posicion, "frame": ruta_frame.name, "error": "No se pudo guardar la máscara"}

        if not escritura_diagnostico:
            return {"posicion": posicion, "frame": ruta_frame.name, "error": "No se pudo guardar el diagnóstico"}

    pixeles_totales = mascara_final.size
    pixeles_verdes = int(np.count_nonzero(mascara_verde))
    pixeles_rojos = int(np.count_nonzero(mascara_roja))
    pixeles_hsv = int(np.count_nonzero(mascara_hsv))
    pixeles_interseccion_base = int(np.count_nonzero(mascara_interseccion_base))
    pixeles_interseccion = int(np.count_nonzero(mascara_interseccion))
    pixeles_cerrados = int(np.count_nonzero(mascara_cerrada))
    pixeles_finales = int(np.count_nonzero(mascara_final))
    retencion_base = 100.0 * pixeles_interseccion_base / pixeles_hsv if pixeles_hsv else 0.0
    retencion_soporte = 100.0 * pixeles_interseccion / pixeles_hsv if pixeles_hsv else 0.0
    seleccion_hsv = mascara_hsv > 0
    seleccion_final = mascara_final > 0
    frecuencia_hsv = frecuencia[seleccion_hsv]
    frecuencia_final = frecuencia[seleccion_final]

    return {
        "posicion": posicion,
        "frame": ruta_frame.name,
        "bloque_temporal": indice_bloque,
        "pixeles_totales": pixeles_totales,
        "pixeles_verdes": pixeles_verdes,
        "pixeles_rojos": pixeles_rojos,
        "pixeles_candidatos_hsv": pixeles_hsv,
        "pixeles_con_frecuencia_exacta": pixeles_interseccion_base,
        "pixeles_recuperados_expansion_soporte": pixeles_interseccion - pixeles_interseccion_base,
        "pixeles_con_soporte_expandido": pixeles_interseccion,
        "pixeles_eliminados_soporte": pixeles_hsv - pixeles_interseccion,
        "pixeles_agregados_cierre": max(0, pixeles_cerrados - pixeles_interseccion),
        "pixeles_protegidos_frecuencia": pixeles_protegidos,
        "porcentaje_protegido_frecuencia": 100.0 * pixeles_protegidos / pixeles_totales,
        "pixeles_eliminados_densidad": pixeles_eliminados_densidad,
        "porcentaje_eliminado_densidad": 100.0 * pixeles_eliminados_densidad / pixeles_totales,
        "pixeles_eliminados_area": max(0, int(np.count_nonzero(mascara_refinada)) - pixeles_antes_dilatacion),
        "pixeles_antes_dilatacion": pixeles_antes_dilatacion,
        "pixeles_agregados_dilatacion": pixeles_finales - pixeles_antes_dilatacion,
        "pixeles_hud_final": pixeles_finales,
        "porcentaje_candidato_hsv": 100.0 * pixeles_hsv / pixeles_totales,
        "porcentaje_con_soporte": 100.0 * pixeles_interseccion / pixeles_totales,
        "porcentaje_hud_final": 100.0 * pixeles_finales / pixeles_totales,
        "retencion_soporte_exacto_porcentaje": retencion_base,
        "retencion_soporte_expandido_porcentaje": retencion_soporte,
        "componentes_finales": len(areas_finales),
        "area_componente_minima": min(areas_finales, default=0),
        "area_componente_mediana": float(np.median(areas_finales)) if areas_finales else 0.0,
        "area_componente_maxima": max(areas_finales, default=0),
        "frecuencia_media_candidatos": float(frecuencia_hsv.mean()) if frecuencia_hsv.size else 0.0,
        "frecuencia_media_hud_final": float(frecuencia_final.mean()) if frecuencia_final.size else 0.0,
        "frecuencia_minima_hud_final": float(frecuencia_final.min()) if frecuencia_final.size else 0.0,
        "error": "",
    }


# %% 2. Segmentación del HUD
def main():
    if not CARPETA_FRAMES.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de frames: {CARPETA_FRAMES}")

    rutas_frames = sorted((ruta for ruta in CARPETA_FRAMES.iterdir() if ruta.suffix.lower() == ".png"), key=obtener_indice_frame)

    if not rutas_frames:
        raise RuntimeError(f"No se encontraron archivos PNG en: {CARPETA_FRAMES}")

    if CARPETA_DIAGNOSTICOS_ANTIGUA.exists():
        shutil.rmtree(CARPETA_DIAGNOSTICOS_ANTIGUA)

    CARPETA_MASCARAS.mkdir(parents=True, exist_ok=True)
    CARPETA_DIAGNOSTICOS.mkdir(parents=True, exist_ok=True)

    if ELIMINAR_RESULTADOS_ANTERIORES:
        for carpeta in (CARPETA_MASCARAS, CARPETA_DIAGNOSTICOS):
            for ruta_anterior in carpeta.glob("*.png"):
                ruta_anterior.unlink()

        if RUTA_CSV_DIAGNOSTICOS.exists():
            RUTA_CSV_DIAGNOSTICOS.unlink()

    primer_frame = cv2.imread(str(rutas_frames[0]), cv2.IMREAD_COLOR)

    if primer_frame is None:
        raise RuntimeError(f"No se pudo leer el primer frame: {rutas_frames[0]}")

    forma = primer_frame.shape[:2]
    bloques = [rutas_frames[inicio:inicio + FRAMES_POR_BLOQUE] for inicio in range(0, len(rutas_frames), FRAMES_POR_BLOQUE)]
    argumentos_bloques = [(indice, rutas_bloque, forma) for indice, rutas_bloque in enumerate(bloques)]
    cv2.setNumThreads(1)

    print(f"Frames encontrados: {len(rutas_frames)}")
    print(f"Bloques temporales: {len(bloques)}")
    print(f"Frames por bloque: {FRAMES_POR_BLOQUE}")
    print(f"Umbral estricto de anclaje: {UMBRAL_FRECUENCIA_ESTRICTO:.2f}")
    print(f"Radio de proximidad al ancla: {RADIO_PROXIMIDAD_ANCLA} píxeles")
    print(f"Umbral HUD superior: {UMBRAL_HUD_SUPERIOR:.2f}")
    print(f"Umbral brújula: {UMBRAL_BRUJULA:.2f}")
    print(f"Umbral HUD inferior: {UMBRAL_HUD_INFERIOR:.2f}")
    print(f"Umbral panel lateral: {UMBRAL_PANEL_LATERAL:.2f}")
    print(f"Umbral crosshair: {UMBRAL_CROSSHAIR:.2f}")
    print(f"Radio de expansión del soporte: {RADIO_SOPORTE} píxeles")
    print(f"Ventana de densidad local: {TAMANO_VENTANA_DENSIDAD} x {TAMANO_VENTANA_DENSIDAD}")
    print(f"Umbral de densidad local: {UMBRAL_DENSIDAD_LOCAL:.2f}")
    print(f"Umbral de protección por frecuencia: {UMBRAL_FRECUENCIA_PROTECCION:.2f}")
    print(f"Radio de protección: {RADIO_PROTECCION} píxeles")
    print(f"Trabajadores utilizados: {NUM_TRABAJADORES}")

    soportes = {}

    for argumento in tqdm(argumentos_bloques, desc="Calculando soportes", unit="bloque"):
        indice_bloque, frecuencia, soporte_base, soporte, soporte_proteccion = calcular_soporte_bloque(argumento)
        soportes[indice_bloque] = (frecuencia, soporte_base, soporte, soporte_proteccion)

    argumentos_frames = []

    for posicion, ruta_frame in enumerate(rutas_frames):
        indice_bloque = posicion // FRAMES_POR_BLOQUE
        frecuencia, soporte_base, soporte, soporte_proteccion = soportes[indice_bloque]
        argumentos_frames.append(
            (
                posicion,
                ruta_frame,
                indice_bloque,
                frecuencia,
                soporte_base,
                soporte,
                soporte_proteccion,
            )
        )

    with ThreadPoolExecutor(max_workers=NUM_TRABAJADORES) as ejecutor:
        diagnosticos = list(
            tqdm(
                ejecutor.map(procesar_frame, argumentos_frames),
                total=len(argumentos_frames),
                desc="Segmentando HUD",
                unit="frame",
                mininterval=2.0,
            )
        )

    df_diagnosticos = pd.DataFrame(diagnosticos).sort_values("posicion").reset_index(drop=True)
    df_diagnosticos.to_csv(RUTA_CSV_DIAGNOSTICOS, index=False)
    errores = df_diagnosticos["error"].fillna("").ne("")

    if errores.any():
        print(df_diagnosticos.loc[errores, ["frame", "error"]].head(20).to_string(index=False))
        raise RuntimeError(f"No se pudieron procesar {errores.sum()} frames")

    cantidad_mascaras = len(list(CARPETA_MASCARAS.glob("*.png")))
    cantidad_diagnosticos = len(list(CARPETA_DIAGNOSTICOS.glob("*.png")))

    if cantidad_mascaras != len(rutas_frames) or cantidad_diagnosticos != len(rutas_frames):
        raise RuntimeError(
            f"Se esperaban {len(rutas_frames)} resultados, pero se encontraron "
            f"{cantidad_mascaras} máscaras y {cantidad_diagnosticos} diagnósticos"
        )

    indice_minimo = df_diagnosticos["porcentaje_hud_final"].idxmin()
    indice_maximo = df_diagnosticos["porcentaje_hud_final"].idxmax()
    fila_minima = df_diagnosticos.loc[indice_minimo]
    fila_maxima = df_diagnosticos.loc[indice_maximo]
    ruta_distribucion = CARPETA_VIDEO / "distribucion_porcentaje_hud.png"

    figura, eje = plt.subplots(figsize=(7, 4))
    eje.hist(df_diagnosticos["porcentaje_hud_final"], bins=40)
    eje.axvline(df_diagnosticos["porcentaje_hud_final"].mean(), color="red", linestyle="--", label=f"Media: {df_diagnosticos['porcentaje_hud_final'].mean():.2f} %")
    eje.axvline(df_diagnosticos["porcentaje_hud_final"].median(), color="green", linestyle="--", label=f"Mediana: {df_diagnosticos['porcentaje_hud_final'].median():.2f} %")
    eje.set_title("Distribución del porcentaje de HUD")
    eje.set_xlabel("Porcentaje de píxeles clasificados como HUD")
    eje.set_ylabel("Cantidad de frames")
    eje.legend()
    figura.tight_layout()
    figura.savefig(str(ruta_distribucion), dpi=300, bbox_inches="tight")
    plt.close(figura)

    print()
    print(f"Frames procesados: {len(df_diagnosticos)}")
    print(f"Retención exacta promedio: {df_diagnosticos['retencion_soporte_exacto_porcentaje'].mean():.2f} %")
    print(f"Retención expandida promedio: {df_diagnosticos['retencion_soporte_expandido_porcentaje'].mean():.2f} %")
    print(f"HUD promedio: {df_diagnosticos['porcentaje_hud_final'].mean():.4f} %")
    print(f"HUD mediano: {df_diagnosticos['porcentaje_hud_final'].median():.4f} %")
    print(f"HUD mínimo: {fila_minima['porcentaje_hud_final']:.4f} % en {fila_minima['frame']}")
    print(f"HUD máximo: {fila_maxima['porcentaje_hud_final']:.4f} % en {fila_maxima['frame']}")
    print(f"Distribución guardada en: {ruta_distribucion}")
    print(f"Máscaras guardadas en: {CARPETA_MASCARAS}")
    print(f"Diagnósticos guardados en: {CARPETA_DIAGNOSTICOS}")
    print(f"CSV guardado en: {RUTA_CSV_DIAGNOSTICOS}")


if __name__ == "__main__":
    main()
