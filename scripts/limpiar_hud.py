# %% 0. Imports y configuración
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import shutil
import subprocess
import threading

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch

from tqdm import tqdm


# Ruta principal del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parent.parent

# Ruta de la instalación de ProPainter
RUTA_PROPAINTER = RUTA_PROYECTO / "ProPainter"
RUTA_INFERENCIA = RUTA_PROPAINTER / "inference_propainter.py"

# Ruta del video asociado con los frames
RUTA_VIDEO = RUTA_PROYECTO / "Videos" / "video30min-11to22.mp4"

# Carpetas de entrada y salida
CARPETA_VIDEO = RUTA_VIDEO.parent / RUTA_VIDEO.stem
CARPETA_FRAMES = CARPETA_VIDEO / "frames"
CARPETA_MASCARAS = CARPETA_VIDEO / "masks"
CARPETA_FRAMES_SIN_HUD = CARPETA_VIDEO / "frames_sin_hud"

# Carpeta temporal utilizada para dividir el video entre GPU
CARPETA_TEMPORAL = CARPETA_VIDEO / ".propainter_temporal"

# Resolución utilizada durante la inferencia
# Esta configuración deja margen suficiente en GPU de 22 GB
ANCHO_PROCESAMIENTO = 960
ALTO_PROCESAMIENTO = 540

# Parámetros temporales de ProPainter
NEIGHBOR_LENGTH = 10
REF_STRIDE = 10
SUBVIDEO_LENGTH = 40

# División externa para utilizar las dos GPU
FRAMES_POR_BLOQUE = 80
SOLAPAMIENTO_FRAMES = 10

# GPU que se utilizarán
GPUS = (0, 1)

# Control de resultados existentes
SOBRESCRIBIR = True
ELIMINAR_RESULTADOS_ANTERIORES = True
ELIMINAR_TEMPORALES_AL_FINALIZAR = True

# Compresión de los PNG finales
COMPRESION_PNG = 3


# %% 1. Funciones auxiliares
def obtener_indice_frame(ruta_frame):
    try:
        return int(ruta_frame.stem)
    except ValueError:
        return ruta_frame.stem


def crear_enlace_o_copiar(origen, destino):
    if destino.exists() or destino.is_symlink():
        destino.unlink()

    try:
        destino.symlink_to(origen.resolve())
    except OSError:
        shutil.copy2(origen, destino)


def buscar_frames_generados(carpeta_salida, cantidad_esperada):
    extensiones = {".png", ".jpg", ".jpeg"}
    rutas_imagenes = sorted(
        (ruta for ruta in carpeta_salida.rglob("*") if ruta.is_file() and ruta.suffix.lower() in extensiones),
        key=lambda ruta: (str(ruta.parent), obtener_indice_frame(ruta)),
    )
    grupos = {}

    for ruta in rutas_imagenes:
        grupos.setdefault(ruta.parent, []).append(ruta)

    candidatos = [
        sorted(rutas, key=obtener_indice_frame)
        for rutas in grupos.values()
        if len(rutas) == cantidad_esperada
    ]

    if not candidatos:
        resumen = ", ".join(f"{carpeta}: {len(rutas)}" for carpeta, rutas in grupos.items())
        raise RuntimeError(
            f"No se encontró una carpeta con {cantidad_esperada} frames generados. "
            f"Contenido encontrado: {resumen or 'ninguna imagen'}"
        )

    candidatos.sort(key=lambda rutas: (0 if rutas[0].parent.name.lower() == "frames" else 1, len(str(rutas[0].parent))))

    return candidatos[0]


def guardar_frame_final(origen, destino):
    if destino.exists() and not SOBRESCRIBIR:
        return

    if origen.suffix.lower() == ".png":
        shutil.copy2(origen, destino)
        return

    frame = cv2.imread(str(origen), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"No se pudo leer el resultado generado: {origen}")

    if not cv2.imwrite(str(destino), frame, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESION_PNG]):
        raise RuntimeError(f"No se pudo guardar el frame final: {destino}")


def preparar_bloque(indice_bloque, inicio_central, fin_central, rutas_frames, rutas_mascaras):
    inicio_lectura = max(0, inicio_central - SOLAPAMIENTO_FRAMES)
    fin_lectura = min(len(rutas_frames), fin_central + SOLAPAMIENTO_FRAMES)
    carpeta_bloque = CARPETA_TEMPORAL / f"bloque_{indice_bloque:04d}"
    carpeta_entrada = carpeta_bloque / "entrada"
    carpeta_mascaras = carpeta_bloque / "mascaras"
    carpeta_salida = carpeta_bloque / "salida"

    if carpeta_bloque.exists():
        shutil.rmtree(carpeta_bloque)

    carpeta_entrada.mkdir(parents=True, exist_ok=True)
    carpeta_mascaras.mkdir(parents=True, exist_ok=True)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    for indice_local, indice_global in enumerate(range(inicio_lectura, fin_lectura)):
        nombre_temporal = f"{indice_local:06d}.png"
        crear_enlace_o_copiar(rutas_frames[indice_global], carpeta_entrada / nombre_temporal)
        crear_enlace_o_copiar(rutas_mascaras[indice_global], carpeta_mascaras / nombre_temporal)

    return {
        "indice_bloque": indice_bloque,
        "inicio_central": inicio_central,
        "fin_central": fin_central,
        "inicio_lectura": inicio_lectura,
        "fin_lectura": fin_lectura,
        "carpeta_bloque": carpeta_bloque,
        "carpeta_entrada": carpeta_entrada,
        "carpeta_mascaras": carpeta_mascaras,
        "carpeta_salida": carpeta_salida,
        "ruta_log": carpeta_bloque / "propainter.log",
    }


def ejecutar_bloque(bloque, indice_gpu, rutas_frames):
    cantidad_lectura = bloque["fin_lectura"] - bloque["inicio_lectura"]
    comando = [
        "python",
        "-u",
        str(RUTA_INFERENCIA),
        "--video",
        str(bloque["carpeta_entrada"]),
        "--mask",
        str(bloque["carpeta_mascaras"]),
        "--output",
        str(bloque["carpeta_salida"]),
        "--width",
        str(ANCHO_PROCESAMIENTO),
        "--height",
        str(ALTO_PROCESAMIENTO),
        "--fp16",
        "--save_frames",
        "--neighbor_length",
        str(NEIGHBOR_LENGTH),
        "--ref_stride",
        str(REF_STRIDE),
        "--subvideo_length",
        str(SUBVIDEO_LENGTH),
    ]
    entorno = os.environ.copy()
    entorno["CUDA_VISIBLE_DEVICES"] = str(indice_gpu)
    entorno["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        import imageio_ffmpeg

        entorno["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError(
            "No está instalado imageio-ffmpeg en el entorno de ejecución"
        ) from error

    with bloque["ruta_log"].open("w", encoding="utf-8") as archivo_log:
        proceso = subprocess.run(
            comando,
            cwd=RUTA_PROPAINTER,
            env=entorno,
            stdout=archivo_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proceso.returncode != 0:
        ultimas_lineas = bloque["ruta_log"].read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        detalle = "\n".join(ultimas_lineas)

        raise RuntimeError(
            f"ProPainter falló en el bloque {bloque['indice_bloque']} usando GPU {indice_gpu}.\n"
            f"Log: {bloque['ruta_log']}\n\n{detalle}"
        )

    frames_generados = buscar_frames_generados(bloque["carpeta_salida"], cantidad_lectura)
    inicio_local = bloque["inicio_central"] - bloque["inicio_lectura"]
    fin_local = inicio_local + bloque["fin_central"] - bloque["inicio_central"]

    for indice_local in range(inicio_local, fin_local):
        indice_global = bloque["inicio_lectura"] + indice_local
        ruta_destino = CARPETA_FRAMES_SIN_HUD / rutas_frames[indice_global].name
        guardar_frame_final(frames_generados[indice_local], ruta_destino)

    return {
        "indice_bloque": bloque["indice_bloque"],
        "gpu": indice_gpu,
        "inicio": bloque["inicio_central"],
        "fin": bloque["fin_central"],
        "frames_guardados": bloque["fin_central"] - bloque["inicio_central"],
        "log": bloque["ruta_log"],
    }


# %% 2. Limpieza del HUD
def main():
    if not RUTA_INFERENCIA.exists():
        raise FileNotFoundError(
            f"No se encontró ProPainter en: {RUTA_PROPAINTER}\n"
            "Clona el repositorio oficial antes de ejecutar este script."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible en PyTorch")

    if torch.cuda.device_count() < len(GPUS):
        raise RuntimeError(
            f"Se solicitaron {len(GPUS)} GPU, pero PyTorch solamente detecta {torch.cuda.device_count()}"
        )

    if not CARPETA_FRAMES.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de frames: {CARPETA_FRAMES}")

    if not CARPETA_MASCARAS.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de máscaras: {CARPETA_MASCARAS}")

    rutas_frames = sorted(
        (ruta for ruta in CARPETA_FRAMES.iterdir() if ruta.suffix.lower() == ".png"),
        key=obtener_indice_frame,
    )
    rutas_mascaras = sorted(
        (ruta for ruta in CARPETA_MASCARAS.iterdir() if ruta.suffix.lower() == ".png"),
        key=obtener_indice_frame,
    )

    if not rutas_frames:
        raise RuntimeError(f"No se encontraron frames PNG en: {CARPETA_FRAMES}")

    if len(rutas_frames) != len(rutas_mascaras):
        raise RuntimeError(
            f"La cantidad de frames ({len(rutas_frames)}) no coincide con "
            f"la cantidad de máscaras ({len(rutas_mascaras)})"
        )

    nombres_frames = [ruta.name for ruta in rutas_frames]
    nombres_mascaras = [ruta.name for ruta in rutas_mascaras]

    if nombres_frames != nombres_mascaras:
        diferencias = [
            (frame, mascara)
            for frame, mascara in zip(nombres_frames, nombres_mascaras)
            if frame != mascara
        ][:20]

        raise RuntimeError(f"Los nombres de frames y máscaras no coinciden: {diferencias}")

    primer_frame = cv2.imread(str(rutas_frames[0]), cv2.IMREAD_COLOR)
    primera_mascara = cv2.imread(str(rutas_mascaras[0]), cv2.IMREAD_GRAYSCALE)

    if primer_frame is None or primera_mascara is None:
        raise RuntimeError("No se pudo leer el primer frame o la primera máscara")

    if primer_frame.shape[:2] != primera_mascara.shape[:2]:
        raise RuntimeError(
            f"El frame tiene forma {primer_frame.shape[:2]} y la máscara {primera_mascara.shape[:2]}"
        )

    CARPETA_FRAMES_SIN_HUD.mkdir(parents=True, exist_ok=True)

    if ELIMINAR_RESULTADOS_ANTERIORES:
        for ruta_anterior in CARPETA_FRAMES_SIN_HUD.glob("*.png"):
            ruta_anterior.unlink()

    if CARPETA_TEMPORAL.exists():
        shutil.rmtree(CARPETA_TEMPORAL)

    CARPETA_TEMPORAL.mkdir(parents=True, exist_ok=True)
    bloques = []

    for inicio in range(0, len(rutas_frames), FRAMES_POR_BLOQUE):
        fin = min(inicio + FRAMES_POR_BLOQUE, len(rutas_frames))
        bloques.append(
            preparar_bloque(
                len(bloques),
                inicio,
                fin,
                rutas_frames,
                rutas_mascaras,
            )
        )

    print(f"Frames encontrados: {len(rutas_frames)}")
    print(f"Resolución original: {primer_frame.shape[1]} x {primer_frame.shape[0]}")
    print(f"Resolución de ProPainter: {ANCHO_PROCESAMIENTO} x {ALTO_PROCESAMIENTO}")
    print(f"Bloques: {len(bloques)}")
    print(f"Frames centrales por bloque: {FRAMES_POR_BLOQUE}")
    print(f"Solapamiento: {SOLAPAMIENTO_FRAMES} frames por extremo")
    print(f"Subvideo length: {SUBVIDEO_LENGTH}")
    print(f"Neighbor length: {NEIGHBOR_LENGTH}")
    print(f"Reference stride: {REF_STRIDE}")
    print(f"GPU utilizadas: {GPUS}")

    progreso = tqdm(
        total=len(rutas_frames),
        desc="Frames terminados con ProPainter",
        unit="frame",
        mininterval=2.0,
        dynamic_ncols=True,
    )
    bloqueo_progreso = threading.Lock()
    resultados = []
    errores = []

    def trabajador_gpu(indice_gpu, bloques_gpu):
        resultados_gpu = []

        for bloque in bloques_gpu:
            tqdm.write(
                f"GPU {indice_gpu}: iniciando bloque "
                f"{bloque['indice_bloque'] + 1}/{len(bloques)} "
                f"con frames {bloque['inicio_central']}:{bloque['fin_central']}"
            )

            try:
                resultado = ejecutar_bloque(bloque, indice_gpu, rutas_frames)
                resultados_gpu.append(resultado)

                with bloqueo_progreso:
                    progreso.update(resultado["frames_guardados"])

                tqdm.write(
                    f"GPU {indice_gpu}: bloque "
                    f"{bloque['indice_bloque'] + 1}/{len(bloques)} terminado; "
                    f"{resultado['frames_guardados']} frames guardados"
                )
            except Exception as error:
                errores.append(
                    (
                        bloque["indice_bloque"],
                        indice_gpu,
                        str(error),
                    )
                )
                tqdm.write(
                    f"GPU {indice_gpu}: error en bloque "
                    f"{bloque['indice_bloque'] + 1}/{len(bloques)}"
                )
                break

        return resultados_gpu

    bloques_por_gpu = {
        indice_gpu: [
            bloque
            for posicion, bloque in enumerate(bloques)
            if posicion % len(GPUS) == posicion_gpu
        ]
        for posicion_gpu, indice_gpu in enumerate(GPUS)
    }

    with ThreadPoolExecutor(max_workers=len(GPUS)) as ejecutor:
        futuros = [
            ejecutor.submit(trabajador_gpu, indice_gpu, bloques_por_gpu[indice_gpu])
            for indice_gpu in GPUS
        ]

        for futuro in futuros:
            resultados.extend(futuro.result())

    progreso.close()

    if errores:
        detalle = "\n\n".join(
            f"Bloque {indice_bloque}, GPU {indice_gpu}:\n{mensaje}"
            for indice_bloque, indice_gpu, mensaje in errores
        )
        raise RuntimeError(f"Fallaron uno o más bloques:\n\n{detalle}")

    cantidad_resultados = len(list(CARPETA_FRAMES_SIN_HUD.glob("*.png")))

    if cantidad_resultados != len(rutas_frames):
        raise RuntimeError(
            f"Se esperaban {len(rutas_frames)} frames limpios, "
            f"pero se encontraron {cantidad_resultados}"
        )

    nombres_resultados = sorted(
        (ruta.name for ruta in CARPETA_FRAMES_SIN_HUD.glob("*.png")),
        key=lambda nombre: obtener_indice_frame(Path(nombre)),
    )

    if nombres_resultados != nombres_frames:
        raise RuntimeError("Los nombres de los resultados no coinciden con los frames originales")

    if ELIMINAR_TEMPORALES_AL_FINALIZAR:
        shutil.rmtree(CARPETA_TEMPORAL)

    print()
    print("Limpieza del HUD terminada")
    print(f"Bloques procesados: {len(resultados)}")
    print(f"Frames limpios: {cantidad_resultados}")
    print(f"Resolución de salida: {ANCHO_PROCESAMIENTO} x {ALTO_PROCESAMIENTO}")
    print(f"Resultados guardados en: {CARPETA_FRAMES_SIN_HUD}")


if __name__ == "__main__":
    main()
