# %% 0. Imports y configuracion
import csv
import gc
import re

from pathlib import Path

import cv2
import numpy as np


# %% 1. Rutas y parametros
RUTA_PROYECTO = Path(__file__).resolve().parent.parent

CARPETA_VIDEO = (
    RUTA_PROYECTO
    / "Videos"
    / "video30min-11to22"
)

CARPETA_ORIGINAL = (
    CARPETA_VIDEO
    / "frames_prueba"
)

CARPETA_N2N = (
    CARPETA_VIDEO
    / "n2n_prueba"
    / "clean"
)

RUTA_METRICAS = (
    CARPETA_VIDEO
    / "n2n_prueba"
    / "metricas_frames.csv"
)

CARPETA_SALIDA = (
    CARPETA_VIDEO
    / "comparaciones_cualitativas"
)

CANTIDAD_EJEMPLOS = 5

# Tamano de cada imagen dentro de la comparacion.
ANCHO_IMAGEN = 640

# Calidad de los JPEG de salida.
CALIDAD_JPEG = 92

# Altura del encabezado de cada pareja.
ALTO_ENCABEZADO = 42


# %% 2. Funciones generales
def extraer_indice(nombre):
    numeros = re.findall(
        r"\d+",
        Path(nombre).stem,
    )

    if not numeros:
        return -1

    return int(numeros[-1])


def buscar_columna(
    columnas,
    opciones,
):
    for opcion in opciones:
        if opcion in columnas:
            return opcion

    return None


def convertir_numero(valor):
    try:
        return float(valor)
    except (
        TypeError,
        ValueError,
    ):
        return np.nan


def crear_mapa_imagenes(carpeta):
    extensiones = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }

    if not carpeta.exists():
        raise FileNotFoundError(
            f"No existe: {carpeta}"
        )

    return {
        ruta.stem: ruta
        for ruta in carpeta.iterdir()
        if ruta.suffix.lower()
        in extensiones
    }


def leer_bgr(ruta):
    imagen = cv2.imread(
        str(ruta),
        cv2.IMREAD_COLOR,
    )

    if imagen is None:
        raise RuntimeError(
            f"No se pudo leer: {ruta}"
        )

    return imagen


def redimensionar(imagen):
    alto, ancho = imagen.shape[:2]

    nuevo_alto = max(
        1,
        round(
            alto
            * ANCHO_IMAGEN
            / ancho
        ),
    )

    return cv2.resize(
        imagen,
        (
            ANCHO_IMAGEN,
            nuevo_alto,
        ),
        interpolation=cv2.INTER_AREA,
    )


def agregar_encabezado(
    imagen,
    texto,
):
    alto, ancho = imagen.shape[:2]

    encabezado = np.full(
        (
            ALTO_ENCABEZADO,
            ancho,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    cv2.putText(
        encabezado,
        texto,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )

    return np.vstack(
        [
            encabezado,
            imagen,
        ]
    )


# %% 3. Lectura de metricas
def cargar_metricas():
    if not RUTA_METRICAS.exists():
        raise FileNotFoundError(
            f"No existe: {RUTA_METRICAS}"
        )

    with RUTA_METRICAS.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as archivo:
        lector = csv.DictReader(
            archivo
        )

        columnas = (
            lector.fieldnames
            if lector.fieldnames
            else []
        )

        columna_residual = buscar_columna(
            columnas,
            (
                "residual_rms",
                "residual_RMS",
                "residual_RMS (diagnostico)",
            ),
        )

        columna_nitidez = buscar_columna(
            columnas,
            (
                "nitidez_original",
                "nitidez_laplaciana",
            ),
        )

        columna_contraste = buscar_columna(
            columnas,
            (
                "contraste_original",
                "contraste",
            ),
        )

        if columna_residual is None:
            raise RuntimeError(
                "No se encontro residual_rms "
                "en metricas_frames.csv"
            )

        registros = []

        for fila in lector:
            nombre_frame = fila.get(
                "frame",
                "",
            )

            if not nombre_frame:
                continue

            stem = Path(
                nombre_frame
            ).stem

            registros.append(
                {
                    "frame": nombre_frame,
                    "stem": stem,
                    "indice": extraer_indice(
                        nombre_frame
                    ),
                    "residual": convertir_numero(
                        fila.get(
                            columna_residual
                        )
                    ),
                    "nitidez": convertir_numero(
                        fila.get(
                            columna_nitidez
                        )
                    )
                    if columna_nitidez
                    else 0.0,
                    "contraste": convertir_numero(
                        fila.get(
                            columna_contraste
                        )
                    )
                    if columna_contraste
                    else 0.0,
                }
            )

    registros = [
        registro
        for registro in registros
        if np.isfinite(
            registro["residual"]
        )
    ]

    registros.sort(
        key=lambda registro: (
            registro["indice"]
        )
    )

    if (
        len(registros)
        < CANTIDAD_EJEMPLOS
    ):
        raise RuntimeError(
            "No hay suficientes registros "
            "para seleccionar cinco ejemplos."
        )

    return registros


# %% 4. Seleccion temporal y visual
def normalizar_valores(
    registros,
    clave,
):
    valores = np.asarray(
        [
            registro[clave]
            if np.isfinite(
                registro[clave]
            )
            else 0.0
            for registro in registros
        ],
        dtype=np.float64,
    )

    minimo = float(
        valores.min()
    )

    maximo = float(
        valores.max()
    )

    rango = (
        maximo - minimo
    )

    if rango <= 0:
        return np.zeros_like(
            valores
        )

    return (
        valores - minimo
    ) / rango


def seleccionar_ejemplos(
    registros,
    mapa_original,
    mapa_n2n,
):
    registros = [
        registro
        for registro in registros
        if (
            registro["stem"]
            in mapa_original
            and registro["stem"]
            in mapa_n2n
        )
    ]

    limites = np.linspace(
        0,
        len(registros),
        CANTIDAD_EJEMPLOS + 1,
        dtype=int,
    )

    seleccionados = []

    for numero in range(
        CANTIDAD_EJEMPLOS
    ):
        inicio = limites[numero]
        fin = limites[numero + 1]

        intervalo = registros[
            inicio:fin
        ]

        residual_normalizado = (
            normalizar_valores(
                intervalo,
                "residual",
            )
        )

        nitidez_normalizada = (
            normalizar_valores(
                intervalo,
                "nitidez",
            )
        )

        contraste_normalizado = (
            normalizar_valores(
                intervalo,
                "contraste",
            )
        )

        # Se favorecen las diferencias visibles,
        # pero tambien el detalle y contraste para
        # evitar elegir fondos homogeneos.
        puntajes = (
            0.50
            * residual_normalizado
            + 0.30
            * nitidez_normalizada
            + 0.20
            * contraste_normalizado
        )

        posicion_mejor = int(
            np.argmax(puntajes)
        )

        seleccionado = dict(
            intervalo[
                posicion_mejor
            ]
        )

        seleccionado[
            "ruta_original"
        ] = mapa_original[
            seleccionado["stem"]
        ]

        seleccionado[
            "ruta_n2n"
        ] = mapa_n2n[
            seleccionado["stem"]
        ]

        seleccionados.append(
            seleccionado
        )

    return seleccionados


# %% 5. Creacion de comparaciones
def crear_comparacion(
    seleccionado,
    numero,
):
    original = redimensionar(
        leer_bgr(
            seleccionado[
                "ruta_original"
            ]
        )
    )

    resultado = redimensionar(
        leer_bgr(
            seleccionado[
                "ruta_n2n"
            ]
        )
    )

    if (
        original.shape
        != resultado.shape
    ):
        raise RuntimeError(
            "Las dimensiones no coinciden "
            f"para {seleccionado['frame']}"
        )

    original = agregar_encabezado(
        original,
        seleccionado["frame"],
    )

    resultado = agregar_encabezado(
        resultado,
        "Resultado N2N",
    )

    separador = np.full(
        (
            original.shape[0],
            8,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    comparacion = np.hstack(
        [
            original,
            separador,
            resultado,
        ]
    )

    nombre_salida = (
        f"comparacion_"
        f"{numero:02d}_"
        f"{seleccionado['stem']}.jpg"
    )

    ruta_salida = (
        CARPETA_SALIDA
        / nombre_salida
    )

    guardada = cv2.imwrite(
        str(ruta_salida),
        comparacion,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            CALIDAD_JPEG,
        ],
    )

    if not guardada:
        raise RuntimeError(
            f"No se pudo guardar: "
            f"{ruta_salida}"
        )

    del original
    del resultado
    del separador
    del comparacion

    gc.collect()

    return ruta_salida


def crear_resumen(
    rutas_comparaciones,
):
    imagenes = [
        cv2.imread(
            str(ruta),
            cv2.IMREAD_COLOR,
        )
        for ruta in rutas_comparaciones
    ]

    if any(
        imagen is None
        for imagen in imagenes
    ):
        raise RuntimeError(
            "No se pudieron releer "
            "las comparaciones."
        )

    ancho_maximo = max(
        imagen.shape[1]
        for imagen in imagenes
    )

    imagenes_ajustadas = []

    for imagen in imagenes:
        if (
            imagen.shape[1]
            != ancho_maximo
        ):
            nuevo_alto = round(
                imagen.shape[0]
                * ancho_maximo
                / imagen.shape[1]
            )

            imagen = cv2.resize(
                imagen,
                (
                    ancho_maximo,
                    nuevo_alto,
                ),
                interpolation=cv2.INTER_AREA,
            )

        imagenes_ajustadas.append(
            imagen
        )

    separador_horizontal = np.full(
        (
            8,
            ancho_maximo,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    partes = []

    for posicion, imagen in enumerate(
        imagenes_ajustadas
    ):
        partes.append(
            imagen
        )

        if (
            posicion
            < len(
                imagenes_ajustadas
            )
            - 1
        ):
            partes.append(
                separador_horizontal
            )

    resumen = np.vstack(
        partes
    )

    ruta_resumen = (
        CARPETA_SALIDA
        / "comparacion_resumen.jpg"
    )

    guardada = cv2.imwrite(
        str(ruta_resumen),
        resumen,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            CALIDAD_JPEG,
        ],
    )

    if not guardada:
        raise RuntimeError(
            "No se pudo guardar "
            "comparacion_resumen.jpg"
        )

    del imagenes
    del imagenes_ajustadas
    del partes
    del resumen

    gc.collect()

    return ruta_resumen


# %% 6. Ejecucion
def main():
    CARPETA_SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    for ruta_anterior in (
        CARPETA_SALIDA.glob(
            "*.jpg"
        )
    ):
        ruta_anterior.unlink()

    mapa_original = crear_mapa_imagenes(
        CARPETA_ORIGINAL
    )

    mapa_n2n = crear_mapa_imagenes(
        CARPETA_N2N
    )

    registros = cargar_metricas()

    seleccionados = seleccionar_ejemplos(
        registros,
        mapa_original,
        mapa_n2n,
    )

    rutas_comparaciones = []

    print(
        "Frames seleccionados:"
    )

    for numero, seleccionado in enumerate(
        seleccionados,
        start=1,
    ):
        print(
            f"  {seleccionado['frame']}"
        )

        ruta_comparacion = (
            crear_comparacion(
                seleccionado,
                numero,
            )
        )

        rutas_comparaciones.append(
            ruta_comparacion
        )

    ruta_resumen = crear_resumen(
        rutas_comparaciones
    )

    ruta_csv = (
        CARPETA_SALIDA
        / "seleccion_frames.csv"
    )

    with ruta_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as archivo:
        columnas = [
            "frame",
            "indice",
            "ruta_original",
            "ruta_n2n",
        ]

        escritor = csv.DictWriter(
            archivo,
            fieldnames=columnas,
        )

        escritor.writeheader()

        for seleccionado in seleccionados:
            escritor.writerow(
                {
                    "frame": (
                        seleccionado[
                            "frame"
                        ]
                    ),
                    "indice": (
                        seleccionado[
                            "indice"
                        ]
                    ),
                    "ruta_original": str(
                        seleccionado[
                            "ruta_original"
                        ]
                    ),
                    "ruta_n2n": str(
                        seleccionado[
                            "ruta_n2n"
                        ]
                    ),
                }
            )

    print()
    print(
        "Comparaciones creadas:"
    )

    for ruta in rutas_comparaciones:
        print(f"  {ruta}")

    print(
        f"  {ruta_resumen}"
    )

    print()
    print(
        f"Salida: {CARPETA_SALIDA}"
    )


if __name__ == "__main__":
    main()
