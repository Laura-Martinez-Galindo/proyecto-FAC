# %% 0. Imports y configuración
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyiqa
import seaborn as sns
import torch

from scipy import stats
from skimage.restoration import estimate_sigma
from tqdm import tqdm


# Ruta principal del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parent.parent

# Ruta del video que se desea procesar
RUTA_VIDEO = RUTA_PROYECTO / "Videos" / "video30min-11to22.mp4"

# Etapa del proyecto que se desea evaluar
ETAPA = "baseline"

# Cantidad de imágenes extraídas por segundo de video
FRAMES_POR_SEGUNDO = 5.0

# Segundo donde comenzó la extracción
SEGUNDO_INICIO = None

# Semilla para obtener el mismo muestreo en cada ejecución
SEMILLA = 42

# Cantidad de píxeles tomados de cada frame para representar las distribuciones
PIXELES_RUIDO_POR_FRAME = 10

# Carpeta que contiene los frames extraídos
CARPETA_FRAMES = RUTA_VIDEO.parent / RUTA_VIDEO.stem / "frames"

# Carpeta donde se guardarán métricas, resúmenes y gráficas
CARPETA_METRICAS = RUTA_VIDEO.parent / RUTA_VIDEO.stem / "metricas"

# Archivo Excel detallado de la etapa
RUTA_EXCEL_METRICAS = CARPETA_METRICAS / f"metricas_{ETAPA}.xlsx"

# Archivo Excel donde se compararán todas las etapas
RUTA_EXCEL_RESUMEN = CARPETA_METRICAS / "resumen.xlsx"

# Archivo CSV auxiliar para conservar resultados aunque falle la creación del Excel
RUTA_CSV_METRICAS = CARPETA_METRICAS / f"metricas_{ETAPA}.csv"

# Cantidad de trabajadores de CPU
# En SLURM toma automáticamente --cpus-per-task
NUM_TRABAJADORES_CPU = int(
    os.environ.get("SLURM_CPUS_PER_TASK")
    or os.environ.get("SLURM_CPUS_ON_NODE")
    or min(8, os.cpu_count() or 1)
)

# Cantidad de GPU que se utilizarán para NIQE y BRISQUE
NUM_GPUS = min(2, torch.cuda.device_count())

# Cantidad de procesos GPU
NUM_PROCESOS_GPU = max(1, NUM_GPUS)

# Cantidad de frames procesados antes de guardar un respaldo
INTERVALO_RESPALDO = 250


# %% 1. Cálculo de métricas
def obtener_indice_frame(ruta_frame):
    try:
        return int(ruta_frame.stem)
    except ValueError:
        return ruta_frame.stem


def calcular_metricas_cpu(argumentos):
    posicion, ruta_frame = argumentos
    frame_bgr = cv2.imread(str(ruta_frame), cv2.IMREAD_COLOR)

    if frame_bgr is None:
        return {
            "posicion": posicion,
            "frame": ruta_frame.name,
            "error_cpu": f"No se pudo leer {ruta_frame.name}",
        }

    gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ruido_estimado = float(
        estimate_sigma(
            gris / 255.0,
            channel_axis=None,
            average_sigmas=True,
        )
        * 255.0
    )

    frame_suavizado = cv2.GaussianBlur(
        gris,
        (0, 0),
        sigmaX=1.5,
        sigmaY=1.5,
    )

    ruido_aproximado = gris - frame_suavizado
    ruido_vector = ruido_aproximado.ravel()

    potencia_senal = float(np.mean(np.square(frame_suavizado)))
    potencia_ruido = float(np.mean(np.square(ruido_vector)))
    potencia_ruido = max(potencia_ruido, np.finfo(float).eps)
    snr_estimado = float(10.0 * np.log10(max(potencia_senal, np.finfo(float).eps) / potencia_ruido))

    curtosis = float(
        stats.kurtosis(
            ruido_vector,
            fisher=True,
            bias=False,
        )
    )

    intensidad_media = float(gris.mean())
    contraste = float(gris.std())
    nitidez_laplaciana = float(cv2.Laplacian(gris, cv2.CV_32F, ksize=3).var())

    cantidad_pixeles = min(PIXELES_RUIDO_POR_FRAME, gris.size)
    generador = np.random.default_rng(SEMILLA + posicion)
    indices_ruido = generador.choice(
        ruido_vector.size,
        size=cantidad_pixeles,
        replace=False,
    )
    indices_intensidad = generador.choice(
        gris.size,
        size=cantidad_pixeles,
        replace=False,
    )

    indice_frame = obtener_indice_frame(ruta_frame)
    segundo_inicial = 0.0 if SEGUNDO_INICIO is None else float(SEGUNDO_INICIO)

    segundo_video = (
        segundo_inicial + indice_frame / FRAMES_POR_SEGUNDO
        if isinstance(indice_frame, int)
        else np.nan
    )

    return {
        "posicion": posicion,
        "frame": ruta_frame.name,
        "segundo_video": segundo_video,
        "ruido_estimado_sigma": ruido_estimado,
        "snr_estimado_db": snr_estimado,
        "curtosis": curtosis,
        "intensidad_media": intensidad_media,
        "contraste": contraste,
        "nitidez_laplaciana": nitidez_laplaciana,
        "muestra_ruido": ruido_vector[indices_ruido].astype(np.float32),
        "muestra_intensidad": gris.ravel()[indices_intensidad].astype(np.float32),
        "error_cpu": "",
    }


def calcular_metricas_gpu(argumentos):
    indice_gpu, rutas_asignadas = argumentos

    if torch.cuda.is_available():
        torch.cuda.set_device(indice_gpu)
        dispositivo = f"cuda:{indice_gpu}"
    else:
        dispositivo = "cpu"

    metrica_niqe = pyiqa.create_metric("niqe", device=dispositivo)
    metrica_brisque = pyiqa.create_metric("brisque", device=dispositivo)
    resultados_gpu = []

    with torch.inference_mode():
        for posicion, ruta_frame in tqdm(
            rutas_asignadas,
            desc=f"Métricas {dispositivo}",
            unit="frame",
            mininterval=2.0,
            position=indice_gpu,
        ):
            try:
                niqe = float(metrica_niqe(str(ruta_frame)).item())
                brisque = float(metrica_brisque(str(ruta_frame)).item())

                resultados_gpu.append({
                    "posicion": posicion,
                    "niqe": niqe,
                    "brisque": brisque,
                    "dispositivo_metricas": dispositivo,
                    "error_gpu": "",
                })

            except Exception as error:
                resultados_gpu.append({
                    "posicion": posicion,
                    "niqe": np.nan,
                    "brisque": np.nan,
                    "dispositivo_metricas": dispositivo,
                    "error_gpu": str(error),
                })

    del metrica_niqe
    del metrica_brisque

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return resultados_gpu


def guardar_grafica_distribucion(df_metricas, columna, titulo, etiqueta, ruta_salida):
    figura, eje = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data=df_metricas,
        x=columna,
        bins=30,
        kde=True,
        ax=eje,
    )
    eje.set_title(titulo)
    eje.set_xlabel(etiqueta)
    eje.set_ylabel("Cantidad de frames")
    figura.tight_layout()
    figura.savefig(
        str(ruta_salida),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figura)


def main():
    if not CARPETA_FRAMES.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de frames: {CARPETA_FRAMES}")

    rutas_frames = sorted(
        (ruta for ruta in CARPETA_FRAMES.iterdir() if ruta.suffix.lower() == ".png"),
        key=lambda ruta: obtener_indice_frame(ruta),
    )

    if not rutas_frames:
        raise RuntimeError(f"No se encontraron archivos PNG en: {CARPETA_FRAMES}")

    CARPETA_METRICAS.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    print(f"Frames encontrados: {len(rutas_frames)}")
    print(f"Trabajadores CPU: {NUM_TRABAJADORES_CPU}")
    print(f"GPU visibles: {torch.cuda.device_count()}")
    print(f"Procesos GPU utilizados: {NUM_PROCESOS_GPU}")

    if torch.cuda.is_available():
        for indice_gpu in range(torch.cuda.device_count()):
            print(f"GPU {indice_gpu}: {torch.cuda.get_device_name(indice_gpu)}")
    else:
        print("CUDA no está disponible. NIQE y BRISQUE se calcularán en CPU.")

    argumentos_cpu = list(enumerate(rutas_frames))

    with ThreadPoolExecutor(max_workers=NUM_TRABAJADORES_CPU) as ejecutor:
        resultados_cpu = list(
            tqdm(
                ejecutor.map(calcular_metricas_cpu, argumentos_cpu),
                total=len(argumentos_cpu),
                desc="Métricas CPU",
                unit="frame",
                mininterval=2.0,
            )
        )

    errores_cpu = [
        resultado
        for resultado in resultados_cpu
        if resultado["error_cpu"]
    ]

    if errores_cpu:
        for resultado in errores_cpu[:20]:
            print(resultado["error_cpu"])

        raise RuntimeError(f"No se pudieron procesar {len(errores_cpu)} frames en CPU")

    grupos_gpu = [
        argumentos_cpu[indice_gpu::NUM_PROCESOS_GPU]
        for indice_gpu in range(NUM_PROCESOS_GPU)
    ]

    argumentos_gpu = [
        (indice_gpu, grupo)
        for indice_gpu, grupo in enumerate(grupos_gpu)
        if grupo
    ]

    contexto = get_context("spawn")

    with contexto.Pool(processes=len(argumentos_gpu)) as grupo_procesos:
        bloques_gpu = grupo_procesos.map(calcular_metricas_gpu, argumentos_gpu)

    resultados_gpu = [
        resultado
        for bloque in bloques_gpu
        for resultado in bloque
    ]

    df_cpu = pd.DataFrame(resultados_cpu)
    df_gpu = pd.DataFrame(resultados_gpu)

    muestras_ruido = df_cpu.pop("muestra_ruido").tolist()
    muestras_intensidad = df_cpu.pop("muestra_intensidad").tolist()

    df_metricas = (
        df_cpu.merge(
            df_gpu,
            on="posicion",
            how="left",
            validate="one_to_one",
        )
        .sort_values("posicion")
        .reset_index(drop=True)
    )

    df_metricas = df_metricas[
        [
            "frame",
            "segundo_video",
            "niqe",
            "brisque",
            "ruido_estimado_sigma",
            "snr_estimado_db",
            "curtosis",
            "intensidad_media",
            "contraste",
            "nitidez_laplaciana",
            "dispositivo_metricas",
            "error_cpu",
            "error_gpu",
        ]
    ]

    df_metricas.to_csv(RUTA_CSV_METRICAS, index=False)

    errores_gpu = df_metricas["error_gpu"].fillna("").ne("")

    if errores_gpu.any():
        print(f"Advertencia: {errores_gpu.sum()} frames presentaron errores en NIQE o BRISQUE.")
        print(df_metricas.loc[errores_gpu, ["frame", "error_gpu"]].head(20).to_string(index=False))

    columnas_metricas = [
        "niqe",
        "brisque",
        "ruido_estimado_sigma",
        "snr_estimado_db",
        "curtosis",
        "intensidad_media",
        "contraste",
        "nitidez_laplaciana",
    ]

    resumen_metricas = (
        df_metricas[columnas_metricas]
        .describe()
        .T
        .reset_index()
        .rename(
            columns={
                "index": "metrica",
                "count": "cantidad",
                "mean": "media",
                "std": "desviacion_estandar",
                "min": "minimo",
                "25%": "percentil_25",
                "50%": "mediana",
                "75%": "percentil_75",
                "max": "maximo",
            }
        )
    )

    with pd.ExcelWriter(RUTA_EXCEL_METRICAS, engine="openpyxl") as escritor:
        df_metricas.to_excel(
            escritor,
            sheet_name="metricas_por_frame",
            index=False,
        )
        resumen_metricas.to_excel(
            escritor,
            sheet_name="estadistica_descriptiva",
            index=False,
        )

    resumen_etapa = df_metricas[columnas_metricas].mean().rename(ETAPA)

    if RUTA_EXCEL_RESUMEN.exists():
        resumen_global = pd.read_excel(
            RUTA_EXCEL_RESUMEN,
            engine="openpyxl",
        ).set_index("metrica")

        resumen_global = resumen_global.reindex(
            resumen_global.index.union(
                resumen_etapa.index,
                sort=False,
            )
        )

        resumen_global[ETAPA] = resumen_etapa
    else:
        resumen_global = resumen_etapa.to_frame()

    resumen_global.index.name = "metrica"
    resumen_global.reset_index().to_excel(
        RUTA_EXCEL_RESUMEN,
        index=False,
        engine="openpyxl",
    )

    valores_ruido = np.concatenate(muestras_ruido)
    valores_intensidad = np.concatenate(muestras_intensidad)
    media_ruido = float(valores_ruido.mean())
    desviacion_ruido = float(valores_ruido.std())
    limite_ruido = float(np.percentile(np.abs(valores_ruido), 99.5))
    limite_ruido = max(limite_ruido, np.finfo(float).eps)
    eje_normal = np.linspace(-limite_ruido, limite_ruido, 300)

    if desviacion_ruido > 0:
        densidad_normal = stats.norm.pdf(
            eje_normal,
            loc=media_ruido,
            scale=desviacion_ruido,
        )
    else:
        densidad_normal = np.zeros_like(eje_normal)

    guardar_grafica_distribucion(
        df_metricas,
        "niqe",
        "Distribución de NIQE",
        "NIQE (↓)",
        CARPETA_METRICAS / f"niqe_{ETAPA}.png",
    )

    guardar_grafica_distribucion(
        df_metricas,
        "brisque",
        "Distribución de BRISQUE",
        "BRISQUE (↓)",
        CARPETA_METRICAS / f"brisque_{ETAPA}.png",
    )

    guardar_grafica_distribucion(
        df_metricas,
        "nitidez_laplaciana",
        "Distribución de nitidez",
        "Varianza del Laplaciano (↑)",
        CARPETA_METRICAS / f"nitidez_{ETAPA}.png",
    )

    figura, eje = plt.subplots(figsize=(6, 4))
    sns.histplot(
        valores_ruido,
        bins=100,
        stat="density",
        label="Ruido aproximado",
        ax=eje,
    )
    eje.plot(
        eje_normal,
        densidad_normal,
        label="Distribución normal",
    )
    eje.set_xlim(-limite_ruido, limite_ruido)
    eje.set_title("Distribución del ruido aproximado")
    eje.set_xlabel("Valor de la diferencia")
    eje.set_ylabel("Densidad")
    eje.legend()
    figura.tight_layout()
    figura.savefig(
        str(CARPETA_METRICAS / f"ruido_{ETAPA}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(6, 4))
    sns.histplot(
        valores_intensidad,
        bins=256,
        binrange=(0, 255),
        ax=eje,
    )
    eje.set_xlim(0, 255)
    eje.set_title("Distribución global de intensidades")
    eje.set_xlabel("Intensidad")
    eje.set_ylabel("Cantidad de píxeles")
    figura.tight_layout()
    figura.savefig(
        str(CARPETA_METRICAS / f"intensidad_{ETAPA}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figura)

    print()
    print("Estadística descriptiva:")
    print(resumen_metricas.to_string(index=False))

    print()
    print("Comparación global:")
    print(resumen_global.reset_index().to_string(index=False))

    print()
    print(f"CSV de respaldo guardado en: {RUTA_CSV_METRICAS}")
    print(f"Excel de métricas guardado en: {RUTA_EXCEL_METRICAS}")
    print(f"Resumen de etapas guardado en: {RUTA_EXCEL_RESUMEN}")
    print(f"Gráficas guardadas en: {CARPETA_METRICAS}")
    print(f"Frames analizados: {len(df_metricas)}")
    print(f"Píxeles usados en la gráfica de ruido: {len(valores_ruido)}")
    print(f"Píxeles usados en la gráfica de intensidad: {len(valores_intensidad)}")


if __name__ == "__main__":
    main()