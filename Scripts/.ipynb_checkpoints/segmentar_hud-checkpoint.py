# %% 0. Imports y configuración
import os
import re

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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

# Carpeta que contiene los frames extraídos
CARPETA_FRAMES = RUTA_VIDEO.parent / RUTA_VIDEO.stem / "frames"

# Carpetas donde se guardarán máscaras y diagnósticos
CARPETA_MASCARAS = RUTA_VIDEO.parent / RUTA_VIDEO.stem / "masks"
CARPETA_DIAGNOSTICOS = RUTA_VIDEO.parent / RUTA_VIDEO.stem / "diagnosticos"

# Archivos de diagnóstico
RUTA_DIAGNOSTICO_CSV = RUTA_VIDEO.parent / RUTA_VIDEO.stem / "diagnostico_hud.csv"
RUTA_TONOS_VERDES_CSV = RUTA_VIDEO.parent / RUTA_VIDEO.stem / "diagnostico_tonos_verdes.csv"

# Cantidad de frames utilizados en la prueba inicial
CANTIDAD_FRAMES_PRUEBA = 50

# Cantidad de trabajadores
NUM_TRABAJADORES = int(
    os.environ.get("SLURM_CPUS_PER_TASK")
    or os.environ.get("SLURM_CPUS_ON_NODE")
    or min(8, os.cpu_count() or 1)
)

# Preparación del frame
LADO_ANALISIS = 960
LIMITE_CLAHE = 2.5
TAMANO_CUADRICULA_CLAHE = 8

# Rangos HSV de verde
# OpenCV representa H entre 0 y 179
H_VERDE_MIN = 30
H_VERDE_MAX = 95
S_VERDE_MIN = 55
V_VERDE_MIN = 60

# Detección general de color, blanco y brillo
S_COLOR_MIN = 70
V_COLOR_MIN = 80
S_BLANCO_MAX = 65
V_BLANCO_MIN = 155
V_BRILLO_MIN = 175

# Detección de bordes
UMBRAL_CANNY_INFERIOR = 45
UMBRAL_CANNY_SUPERIOR = 130

# Persistencia temporal
FRACCION_PERSISTENCIA_BORDE = 0.35
FRACCION_PERSISTENCIA_VERDE = 0.30
FRACCION_PERSISTENCIA_COLOR = 0.35
FRACCION_PERSISTENCIA_BLANCO = 0.35
UMBRAL_DESVIACION_TEMPORAL = 28.0

# Limpieza morfológica
TAMANO_CIERRE = 5
TAMANO_ZONA_CERCANA = 9
TAMANO_DILATACION = 3
ITERACIONES_DILATACION = 1
AREA_MINIMA_COMPONENTE = 15
AREA_MAXIMA_COMPONENTE = 50000

# Diagnóstico visual
COLOR_MASCARA = (0, 0, 255)
COLOR_CONTORNO = (0, 255, 255)
ALFA_MASCARA = 0.45
COMPRESION_PNG = 3
SOBRESCRIBIR_RESULTADOS = True


# %% 1. Segmentación del HUD
def obtener_indice_frame(ruta):
    partes = re.split(r"(\d+)", ruta.name)

    return [int(parte) if parte.isdigit() else parte for parte in partes]


def preparar_frame(ruta_frame):
    frame_original = cv2.imread(str(ruta_frame), cv2.IMREAD_COLOR)

    if frame_original is None:
        raise RuntimeError(f"No se pudo leer: {ruta_frame}")

    alto_original, ancho_original = frame_original.shape[:2]
    escala = min(1.0, LADO_ANALISIS / max(alto_original, ancho_original))

    if escala < 1.0:
        frame_analisis = cv2.resize(
            frame_original,
            (round(ancho_original * escala), round(alto_original * escala)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        frame_analisis = frame_original.copy()

    lab = cv2.cvtColor(frame_analisis, cv2.COLOR_BGR2LAB)
    canal_l, canal_a, canal_b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=LIMITE_CLAHE,
        tileGridSize=(TAMANO_CUADRICULA_CLAHE, TAMANO_CUADRICULA_CLAHE),
    )

    canal_l_clahe = clahe.apply(canal_l)
    frame_clahe = cv2.cvtColor(
        cv2.merge((canal_l_clahe, canal_a, canal_b)),
        cv2.COLOR_LAB2BGR,
    )

    gris_original = cv2.cvtColor(frame_analisis, cv2.COLOR_BGR2GRAY)
    gris_clahe = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2HSV)

    tono = hsv[:, :, 0]
    saturacion = hsv[:, :, 1]
    valor = hsv[:, :, 2]

    bordes = cv2.Canny(
        gris_clahe,
        UMBRAL_CANNY_INFERIOR,
        UMBRAL_CANNY_SUPERIOR,
    )

    mascara_verde = (
        (tono >= H_VERDE_MIN)
        & (tono <= H_VERDE_MAX)
        & (saturacion >= S_VERDE_MIN)
        & (valor >= V_VERDE_MIN)
    ).astype(np.uint8)

    mascara_color = (
        (saturacion >= S_COLOR_MIN)
        & (valor >= V_COLOR_MIN)
    ).astype(np.uint8)

    mascara_blanco = (
        (saturacion <= S_BLANCO_MAX)
        & (valor >= V_BLANCO_MIN)
    ).astype(np.uint8)

    mascara_brillo = (
        valor >= V_BRILLO_MIN
    ).astype(np.uint8)

    pixeles_totales = tono.size
    pixeles_verdes = int(mascara_verde.sum())
    tonos_verdes = tono[mascara_verde > 0]
    saturaciones_verdes = saturacion[mascara_verde > 0]
    valores_verdes = valor[mascara_verde > 0]

    if pixeles_verdes:
        tono_verde_mediano = float(np.median(tonos_verdes))
        tono_verde_percentil_10 = float(np.percentile(tonos_verdes, 10))
        tono_verde_percentil_90 = float(np.percentile(tonos_verdes, 90))
        saturacion_verde_media = float(saturaciones_verdes.mean())
        valor_verde_medio = float(valores_verdes.mean())
    else:
        tono_verde_mediano = np.nan
        tono_verde_percentil_10 = np.nan
        tono_verde_percentil_90 = np.nan
        saturacion_verde_media = np.nan
        valor_verde_medio = np.nan

    return {
        "ruta": ruta_frame,
        "frame_original": frame_original,
        "forma_original": (alto_original, ancho_original),
        "gris_original": gris_original,
        "gris_clahe": gris_clahe,
        "bordes": (bordes > 0).astype(np.uint8),
        "verde": mascara_verde,
        "color": mascara_color,
        "blanco": mascara_blanco,
        "brillo": mascara_brillo,
        "pixeles_verdes": pixeles_verdes,
        "porcentaje_verde": pixeles_verdes / pixeles_totales * 100,
        "tono_verde_mediano": tono_verde_mediano,
        "tono_verde_percentil_10": tono_verde_percentil_10,
        "tono_verde_percentil_90": tono_verde_percentil_90,
        "saturacion_verde_media": saturacion_verde_media,
        "valor_verde_medio": valor_verde_medio,
    }


def limpiar_componentes(mascara):
    cantidad, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(
        mascara,
        connectivity=8,
    )

    mascara_limpia = np.zeros_like(mascara)

    for etiqueta in range(1, cantidad):
        area = estadisticas[etiqueta, cv2.CC_STAT_AREA]

        if AREA_MINIMA_COMPONENTE <= area <= AREA_MAXIMA_COMPONENTE:
            mascara_limpia[etiquetas == etiqueta] = 255

    return mascara_limpia


def guardar_imagen(ruta, imagen):
    escritura_correcta = cv2.imwrite(
        str(ruta),
        imagen,
        [cv2.IMWRITE_PNG_COMPRESSION, COMPRESION_PNG],
    )

    if not escritura_correcta:
        raise RuntimeError(f"No se pudo guardar: {ruta}")


def generar_resultado_frame(argumentos):
    datos_frame, mascara_base = argumentos
    frame_original = datos_frame["frame_original"]
    alto_original, ancho_original = datos_frame["forma_original"]

    apariencia_actual = (
        (datos_frame["verde"] > 0)
        | (datos_frame["color"] > 0)
        | (datos_frame["blanco"] > 0)
        | (datos_frame["brillo"] > 0)
    ).astype(np.uint8) * 255

    apariencia_actual = cv2.dilate(
        apariencia_actual,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    mascara_frame = cv2.bitwise_and(
        mascara_base,
        apariencia_actual,
    )

    mascara_frame = cv2.morphologyEx(
        mascara_frame,
        cv2.MORPH_CLOSE,
        np.ones((TAMANO_CIERRE, TAMANO_CIERRE), dtype=np.uint8),
    )

    mascara_frame = cv2.dilate(
        mascara_frame,
        np.ones((TAMANO_DILATACION, TAMANO_DILATACION), dtype=np.uint8),
        iterations=ITERACIONES_DILATACION,
    )

    mascara_frame = limpiar_componentes(mascara_frame)

    if mascara_frame.shape != (alto_original, ancho_original):
        mascara_frame = cv2.resize(
            mascara_frame,
            (ancho_original, alto_original),
            interpolation=cv2.INTER_NEAREST,
        )

    nombre = datos_frame["ruta"].name
    ruta_mascara = CARPETA_MASCARAS / nombre
    ruta_diagnostico = CARPETA_DIAGNOSTICOS / nombre

    if SOBRESCRIBIR_RESULTADOS or not ruta_mascara.exists():
        guardar_imagen(ruta_mascara, mascara_frame)

    capa_color = np.zeros_like(frame_original)
    capa_color[mascara_frame > 0] = COLOR_MASCARA

    diagnostico = cv2.addWeighted(
        frame_original,
        1.0,
        capa_color,
        ALFA_MASCARA,
        0,
    )

    contornos, _ = cv2.findContours(
        mascara_frame,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        diagnostico,
        contornos,
        -1,
        COLOR_CONTORNO,
        1,
    )

    if SOBRESCRIBIR_RESULTADOS or not ruta_diagnostico.exists():
        guardar_imagen(ruta_diagnostico, diagnostico)

    pixeles_hud = int((mascara_frame > 0).sum())
    pixeles_totales = int(mascara_frame.size)
    fraccion_hud = pixeles_hud / pixeles_totales

    return {
        "frame": nombre,
        "pixeles_hud": pixeles_hud,
        "pixeles_totales": pixeles_totales,
        "fraccion_hud": fraccion_hud,
        "porcentaje_hud": fraccion_hud * 100,
        "componentes_hud": len(contornos),
        "pixeles_verdes": datos_frame["pixeles_verdes"],
        "porcentaje_verde": datos_frame["porcentaje_verde"],
        "tono_verde_mediano": datos_frame["tono_verde_mediano"],
        "saturacion_verde_media": datos_frame["saturacion_verde_media"],
        "valor_verde_medio": datos_frame["valor_verde_medio"],
        "ruta_mascara": str(ruta_mascara),
        "ruta_diagnostico": str(ruta_diagnostico),
    }


def main():
    if not CARPETA_FRAMES.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de frames: {CARPETA_FRAMES}")

    rutas_frames = sorted(
        (
            ruta
            for ruta in CARPETA_FRAMES.iterdir()
            if ruta.suffix.lower() == ".png"
        ),
        key=obtener_indice_frame,
    )

    if not rutas_frames:
        raise RuntimeError(f"No se encontraron archivos PNG en: {CARPETA_FRAMES}")

    rutas_frames = rutas_frames[:CANTIDAD_FRAMES_PRUEBA]

    CARPETA_MASCARAS.mkdir(parents=True, exist_ok=True)
    CARPETA_DIAGNOSTICOS.mkdir(parents=True, exist_ok=True)

    if SOBRESCRIBIR_RESULTADOS:
        for carpeta in (CARPETA_MASCARAS, CARPETA_DIAGNOSTICOS):
            for ruta_anterior in carpeta.glob("*.png"):
                ruta_anter*or.unlink()

    cv2.setNumThreads*1)

    print(f"Frames utilizados *ara la prueba: {len(rutas_frames)}*)
    print(f"Trabajadores utiliza*os: {NUM_TRABAJADORES}")
    print*f"Rango verde HSV: H={H_VERDE_MIN}*{H_VERDE_MAX}, S>={S_VERDE_MIN}, V*={V_VERDE_MIN}")

    with ThreadP*olExecutor(max_workers=NUM_TRABAJA*ORES) as ejecutor:
        datos_f*ames = list(
            tqdm(
   *            ejecutor.map(preparar_*rame, rutas_frames),
             *  total=len(rutas_frames),
       *        desc="Preparando frames",
*               unit="frame",
     *          mininterval=2.0,
       *    )
        )

    pila_grises =*np.stack(
        [datos["gris_clahe"] for datos in datos_frames],
  *     axis=0,
    ).astype(np.float*2)

    pila_bordes = np.stack(
  *     [datos["bordes"] for datos in datos_frames],
        axis=0,
   *).astype(np.float32)

    pila_ver*e = np.stack(
        [datos["verde"] for datos in datos_frames],
   *    axis=0,
    ).astype(np.float3*)

    pila_color = np.stack(
    *   [datos["color"] for datos in datos_frames],
        axis=0,
    ).*stype(np.float32)

    pila_blanco*= np.stack(
        [datos["blanco"] for datos in datos_frames],
    *   axis=0,
    ).astype(np.float32*

    pila_brillo = np.stack(
    *   [datos["brillo"] for datos in datos_frames],
        axis=0,
    )*astype(np.float32)

    frecuencia*bordes = pila_bordes.mean(axis=0)
*   frecuencia_verde = pila_verde.m*an(axis=0)
    frecuencia_color = *ila_color.mean(axis=0)
    frecuen*ia_blanco = pila_blanco.mean(axis=*)
    frecuencia_brillo = pila_bri*lo.mean(axis=0)
    desviacion_tem*oral = pila_grises.std(axis=0)

  * persistencia_borde = frecuencia_b*rdes >= FRACCION_PERSISTENCIA_BORD*
    persistencia_verde = frecuenc*a_verde >= FRACCION_PERSISTENCIA_V*RDE
    persistencia_color = frecu*ncia_color >= FRACCION_PERSISTENCI*_COLOR
    persistencia_blanco = f*ecuencia_blanco >= FRACCION_PERSIS*ENCIA_BLANCO
    persistencia_bril*o = frecuencia_brillo >= FRACCION_*ERSISTENCIA_BLANCO
    estabilidad*temporal = desviacion_temporal <= *MBRAL_DESVIACION_TEMPORAL

    mas*ara_bordes = (
        persistenci*_borde
        & estabilidad_tempo*al
        & (
            persist*ncia_verde
            | persisten*ia_color
            | persistenci*_blanco
            | persistencia*brillo
        )
    ).astype(np.u*nt8) * 255

    mascara_relleno = *
        estabilidad_temporal
    *   & (
            persistencia_ve*de
            | persistencia_colo*
            | persistencia_blanco*        )
    ).astype(np.uint8) **255

    mascara_relleno = cv2.mor*hologyEx(
        mascara_relleno,*        cv2.MORPH_OPEN,
        np*ones((3, 3), dtype=np.uint8),
    *

    zona_cercana_bordes = cv2.di*ate(
        mascara_bordes,
     *  np.ones((TAMANO_ZONA_CERCANA, TA*ANO_ZONA_CERCANA), dtype=np.uint8)*
        iterations=1,
    )

    *ascara_base = cv2.bitwise_or(
    *   mascara_bordes,
        cv2.bit*ise_and(
            mascara_relle*o,
            zona_cercana_bordes*
        ),
    )

    mascara_bas* = cv2.morphologyEx(
        masca*a_base,
        cv2.MORPH_CLOSE,
 *      np.ones((TAMANO_CIERRE, TAMA*O_CIERRE), dtype=np.uint8),
    )
*    mascara_base = cv2.dilate(
   *    mascara_base,
        np.ones(*TAMANO_DILATACION, TAMANO_DILATACI*N), dtype=np.uint8),
        itera*ions=ITERACIONES_DILATACION,
    )*
    mascara_base = limpiar_compon*ntes(mascara_base)

    argumentos*resultados = [
        (datos_frame, mascara_base)
        for datos_*rame in datos_frames
    ]

    wi*h ThreadPoolExecutor(max_workers=NUM_TRABAJADORES) as ejecutor:
        resultados = list(
            tqdm(
                ejecutor.map(generar_resultado_frame, argumentos_resultados),
                total=len(argumentos_resultados),
                desc="Generando máscaras",
                unit="frame",
                mininterval=2.0,
            )
        )

    df_diagnostico = pd.DataFrame(resultados)
    df_diagnostico.to_csv(RUTA_DIAGNOSTICO_CSV, index=False)

    df_tonos_verdes = pd.DataFrame(
        [
            {
                "frame": datos["ruta"].name,
                "pixeles_verdes": datos["pixeles_verdes"],
                "porcentaje_verde": datos["porcentaje_verde"],
                "tono_verde_mediano": datos["tono_verde_mediano"],
                "tono_verde_percentil_10": datos["tono_verde_percentil_10"],
                "tono_verde_percentil_90": datos["tono_verde_percentil_90"],
                "saturacion_verde_media": datos["saturacion_verde_media"],
                "valor_verde_medio": datos["valor_verde_medio"],
            }
            for datos in datos_frames
        ]
    )

    df_tonos_verdes.to_csv(RUTA_TONOS_VERDES_CSV, index=False)

    imagen_persistencia_bordes = np.clip(
        frecuencia_bordes * 255,
        0,
        255,
    ).astype(np.uint8)

    imagen_persistencia_verde = np.clip(
        frecuencia_verde * 255,
        0,
        255,
    ).astype(np.uint8)

    imagen_persistencia_apariencia = np.clip(
        np.maximum.reduce(
            [
                frecuencia_verde,
                frecuencia_color,
                frecuencia_blanco,
                frecuencia_brillo,
            ]
        )
        * 255,
        0,
        255,
    ).astype(np.uint8)

    imagen_estabilidad = np.clip(
        255 - desviacion_temporal * 5,
        0,
        255,
    ).astype(np.uint8)

    guardar_imagen(
        CARPETA_DIAGNOSTICOS / "_persistencia_bordes.png",
        imagen_persistencia_bordes,
    )

    guardar_imagen(
        CARPETA_DIAGNOSTICOS / "_persistencia_verde.png",
        imagen_persistencia_verde,
    )

    guardar_imagen(
        CARPETA_DIAGNOSTICOS / "_persistencia_apariencia.png",
        imagen_persistencia_apariencia,
    )

    guardar_imagen(
        CARPETA_DIAGNOSTICOS / "_estabilidad_temporal.png",
        imagen_estabilidad,
    )

    guardar_imagen(
        CARPETA_DIAGNOSTICOS / "_mascara_bordes.png",
        mascara_bordes,
    )

    guardar_imagen(
        CARPETA_DIAGNOSTICOS / "_mascara_relleno.png",
        mascara_relleno,
    )

    guardar_imagen(
        CARPETA_DIAGNOSTICOS / "_mascara_base.png",
        mascara_base,
    )

    print()
    print("Diagnóstico de segmentación:")
    print(df_diagnostico[
        [
            "frame",
            "porcentaje_hud",
            "componentes_hud",
            "porcentaje_verde",
        ]
    ].describe().to_string())

    print()
    print("Diagnóstico de tonos verdes:")
    print(df_tonos_verdes[
        [
            "porcentaje_verde",
            "tono_verde_mediano",
            "saturacion_verde_media",
            "valor_verde_medio",
        ]
    ].describe().to_string())

    print()
    print(f"Segmentación terminada: {len(resultados)} frames procesados")
    print(f"Porcentaje HUD promedio: {df_diagnostico['porcentaje_hud'].mean():.3f} %")
    print(f"Porcentaje HUD mínimo: {df_diagnostico['porcentaje_hud'].min():.3f} %")
    print(f"Porcentaje HUD máximo: {df_diagnostico['porcentaje_hud'].max():.3f} %")
    print(f"Porcentaje verde promedio: {df_tonos_verdes['porcentaje_verde'].mean():.3f} %")
    print(f"Tono verde mediano global: {df_tonos_verdes['tono_verde_mediano'].median():.2f}")
    print(f"Máscaras guardadas en: {CARPETA_MASCARAS}")
    print(f"Diagnósticos guardados en: {CARPETA_DIAGNOSTICOS}")
    print(f"Diagnóstico HUD guardado en: {RUTA_DIAGNOSTICO_CSV}")
    print(f"Diagnóstico de verdes guardado en: {RUTA_TONOS_VERDES_CSV}")


if __name__ == "__main__":
    main()