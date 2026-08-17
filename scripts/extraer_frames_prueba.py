# %% 0. Imports y configuración
import shutil
import time

from pathlib import Path

import cv2

from tqdm import tqdm


RUTA_PROYECTO = Path(__file__).resolve().parent.parent

RUTA_VIDEO = (
    RUTA_PROYECTO
    / "Videos"
    / "video30min-11to22.mp4"
)

CARPETA_SALIDA = (
    RUTA_PROYECTO
    / "Videos"
    / "video30min-11to22"
    / "frames_prueba"
)

EXTENSION_SALIDA = ".png"
COMPRESION_PNG = 3
SOBRESCRIBIR = True


# %% 1. Extracción
def main():
    if not RUTA_VIDEO.exists():
        raise FileNotFoundError(
            f"No existe el video: {RUTA_VIDEO}"
        )

    captura = cv2.VideoCapture(
        str(RUTA_VIDEO)
    )

    if not captura.isOpened():
        raise RuntimeError(
            f"No se pudo abrir el video: {RUTA_VIDEO}"
        )

    fps = captura.get(
        cv2.CAP_PROP_FPS
    )
    cantidad_declarada = int(
        captura.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )
    ancho = int(
        captura.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )
    alto = int(
        captura.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if CARPETA_SALIDA.exists():
        if not SOBRESCRIBIR:
            raise FileExistsError(
                f"La salida ya existe: {CARPETA_SALIDA}"
            )

        shutil.rmtree(
            CARPETA_SALIDA
        )

    CARPETA_SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Video: {RUTA_VIDEO}",
        flush=True,
    )
    print(
        f"FPS original: {fps:.6f}",
        flush=True,
    )
    print(
        f"Frames declarados: "
        f"{cantidad_declarada}",
        flush=True,
    )
    print(
        f"Resolución: {ancho} x {alto}",
        flush=True,
    )
    print(
        f"Salida: {CARPETA_SALIDA}",
        flush=True,
    )

    inicio = time.time()
    cantidad_extraida = 0

    barra = tqdm(
        total=(
            cantidad_declarada
            if cantidad_declarada > 0
            else None
        ),
        desc="Extrayendo frames nativos",
        unit="frame",
        mininterval=2.0,
    )

    while True:
        lectura_correcta, frame_bgr = (
            captura.read()
        )

        if not lectura_correcta:
            break

        ruta_salida = (
            CARPETA_SALIDA
            / (
                f"frame_"
                f"{cantidad_extraida:06d}"
                f"{EXTENSION_SALIDA}"
            )
        )

        guardado_correcto = cv2.imwrite(
            str(ruta_salida),
            frame_bgr,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                COMPRESION_PNG,
            ],
        )

        if not guardado_correcto:
            raise RuntimeError(
                f"No se pudo guardar: {ruta_salida}"
            )

        cantidad_extraida += 1
        barra.update(1)

    barra.close()
    captura.release()

    duracion = time.time() - inicio

    if cantidad_extraida == 0:
        raise RuntimeError(
            "No se extrajo ningún frame."
        )

    cantidad_archivos = len(
        list(
            CARPETA_SALIDA.glob(
                f"*{EXTENSION_SALIDA}"
            )
        )
    )

    if cantidad_archivos != cantidad_extraida:
        raise RuntimeError(
            f"Se extrajeron {cantidad_extraida} "
            f"frames, pero hay "
            f"{cantidad_archivos} archivos."
        )

    print()
    print(
        f"Frames extraídos: "
        f"{cantidad_extraida}",
        flush=True,
    )
    print(
        f"Tiempo: "
        f"{duracion / 60.0:.2f} minutos",
        flush=True,
    )
    print(
        f"FPS procesados: "
        f"{cantidad_extraida / max(duracion, 1e-8):.2f}",
        flush=True,
    )
    print(
        f"Carpeta: {CARPETA_SALIDA}",
        flush=True,
    )


if __name__ == "__main__":
    main()
