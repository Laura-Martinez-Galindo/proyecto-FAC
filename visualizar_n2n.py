from pathlib import Path
import argparse
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np


EXTENSIONES_VALIDAS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def obtener_numero_frame(ruta):
    numeros = re.findall(r"\d+", ruta.stem)
    return int(numeros[-1]) if numeros else None


def listar_imagenes(carpeta):
    rutas = [
        ruta
        for ruta in Path(carpeta).iterdir()
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_VALIDAS
    ]

    rutas.sort(
        key=lambda ruta: (
            obtener_numero_frame(ruta) is None,
            obtener_numero_frame(ruta) or 0,
            ruta.name,
        )
    )
    return rutas


def emparejar_imagenes(carpeta_originales, carpeta_n2n):
    originales = listar_imagenes(carpeta_originales)
    resultados = listar_imagenes(carpeta_n2n)

    originales_por_numero = {
        obtener_numero_frame(ruta): ruta
        for ruta in originales
        if obtener_numero_frame(ruta) is not None
    }
    resultados_por_numero = {
        obtener_numero_frame(ruta): ruta
        for ruta in resultados
        if obtener_numero_frame(ruta) is not None
    }

    numeros_comunes = sorted(
        set(originales_por_numero) & set(resultados_por_numero)
    )

    if numeros_comunes:
        return [
            (
                numero,
                originales_por_numero[numero],
                resultados_por_numero[numero],
            )
            for numero in numeros_comunes
        ]

    resultados_por_nombre = {ruta.name: ruta for ruta in resultados}
    pares = []

    for indice, original in enumerate(originales):
        if original.name in resultados_por_nombre:
            pares.append(
                (
                    indice,
                    original,
                    resultados_por_nombre[original.name],
                )
            )

    if not pares:
        raise RuntimeError(
            "No se encontraron pares. Los archivos deben compartir el mismo "
            "numero de frame o el mismo nombre."
        )

    return pares


def leer_rgb(ruta):
    imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)

    if imagen is None:
        raise RuntimeError(f"No se pudo leer la imagen: {ruta}")

    return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)


def descriptor_visual(imagen_rgb):
    imagen_pequena = cv2.resize(
        imagen_rgb,
        (160, 90),
        interpolation=cv2.INTER_AREA,
    )
    hsv = cv2.cvtColor(imagen_pequena, cv2.COLOR_RGB2HSV)

    histograma = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [24, 24],
        [0, 180, 0, 256],
    )
    histograma = cv2.normalize(
        histograma,
        None,
        alpha=1.0,
        norm_type=cv2.NORM_L1,
    )

    gris = cv2.cvtColor(imagen_pequena, cv2.COLOR_RGB2GRAY)
    gradiente_x = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    energia_bordes = np.mean(
        np.sqrt(gradiente_x**2 + gradiente_y**2)
    ) / 255.0

    brillo = np.mean(gris) / 255.0
    contraste = np.std(gris) / 255.0

    return np.concatenate(
        [
            histograma.flatten(),
            np.array(
                [energia_bordes, brillo, contraste],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def distancia_visual(descriptor_a, descriptor_b):
    histograma_a = descriptor_a[:-3]
    histograma_b = descriptor_b[:-3]

    distancia_histograma = cv2.compareHist(
        histograma_a.astype(np.float32),
        histograma_b.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    )
    distancia_estadisticas = np.linalg.norm(
        descriptor_a[-3:] - descriptor_b[-3:]
    )

    return float(distancia_histograma + 0.35 * distancia_estadisticas)


def seleccionar_cinco_diversos(
    pares,
    separacion_minima_relativa=0.12,
):
    if len(pares) < 5:
        raise ValueError(
            f"Solo se encontraron {len(pares)} pares y se necesitan 5."
        )

    descriptores = [
        descriptor_visual(leer_rgb(ruta_original))
        for _, ruta_original, _ in pares
    ]

    cantidad = len(pares)
    separacion_minima = max(
        1,
        int(cantidad * separacion_minima_relativa),
    )

    candidatos_iniciales = np.linspace(
        0,
        cantidad - 1,
        num=min(cantidad, 15),
        dtype=int,
    )

    seleccionado_inicial = max(
        candidatos_iniciales,
        key[
_visual(leer_rgb(ruta_original))
][-3],
    )
    seleccionados = [int(seleccionado_inicial)]

    while len(seleccionados) < 5:
        candidatos = [
            indice
            for indice in range(cantidad)
            if indice not in seleccionados
            and all(
                abs(indice - elegido) >= separacion_minima
                for elegido in seleccionados
            )
        ]

        if not candidatos:
            separacion_minima = max(1, separacion_minima - 1)
            continue

        mejor_indice = max(
            candidatos,
            key=lambda indice: min(
                distancia_visual(
                    descriptores[indice],
                    descriptores[elegido],
                )
                for elegido in seleccionados
            ),
        )
        seleccionados.append(mejor_indice)

    return [pares[indice] for indice in sorted(seleccionados)]


def seleccionar_por_ids(pares, ids_frames):
    pares_por_id = {
        numero: (numero, original, resultado)
        for numero, original, resultado in pares
    }

    faltantes = [
        numero
        for numero in ids_frames
        if numero not in pares_por_id
    ]

    if faltantes:
        raise ValueError(
            f"No se encontraron estos frames: {faltantes}"
        )

    return [pares_por_id[numero] for numero in ids_frames]


def recorte_con_textura(imagen_rgb, proporcion=0.30):
    gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
    alto, ancho = gris.shape

    ancho_recorte = max(64, int(ancho * proporcion))
    alto_recorte = max(64, int(alto * proporcion))

    ancho_recorte = min(ancho_recorte, ancho)
    alto_recorte = min(alto_recorte, alto)

    gradiente_x = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    energia = cv2.magnitude(gradiente_x, gradiente_y)

    mapa_energia = cv2.boxFilter(
        energia,
        ddepth=-1,
        ksize=(ancho_recorte, alto_recorte),
        normalize=True,
    )

    margen_x = ancho_recorte // 2
    margen_y = alto_recorte // 2

    mapa_valido = mapa_energia.copy()
    mapa_valido[:margen_y, :] = -1
    mapa_valido[-margen_y:, :] = -1
    mapa_valido[:, :margen_x] = -1
    mapa_valido[:, -margen_x:] = -1

    _, _, _, centro = cv2.minMaxLoc(mapa_valido)
    centro_x, centro_y = centro

    x1 = np.clip(
        centro_x - ancho_recorte // 2,
        0,
        ancho - ancho_recorte,
    )
    y1 = np.clip(
        centro_y - alto_recorte // 2,
        0,
        alto - alto_recorte,
    )

    return (
        int(x1),
        int(y1),
        int(x1 + ancho_recorte),
        int(y1 + alto_recorte),
    )


def dibujar_rectangulo(imagen_rgb, coordenadas):
    x1, y1, x2, y2 = coordenadas
    imagen = imagen_rgb.copy()

    grosor = max(
        2,
        round(min(imagen.shape[:2]) / 300),
    )

    cv2.rectangle(
        imagen,
        (x1, y1),
        (x2, y2),
        color=(255, 210, 0),
        thickness=grosor,
    )
    return imagen


def guardar_comparacion_completa(seleccionados, ruta_salida):
    figura, ejes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(13, 19),
        constrained_layout=True,
    )

    for fila, (numero, ruta_original, ruta_n2n) in enumerate(
        seleccionados
    ):
        original = leer_rgb(ruta_original)
        resultado = leer_rgb(ruta_n2n)

        ejes[fila, 0].imshow(original)
        ejes[fila, 1].imshow(resultado)

        ejes[fila, 0].set_title(
            f"Frame {numero}\nOriginal",
            fontsize=13,
        )
        ejes[fila, 1].set_title(
            f"Frame {numero}\nN2N",
            fontsize=13,
        )

        ejes[fila, 0].axis("off")
        ejes[fila, 1].axis("off")

    figura.savefig(
        ruta_salida,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figura)


def guardar_comparacion_con_zoom(seleccionados, ruta_salida):
    figura, ejes = plt.subplots(
        nrows=5,
        ncols=4,
        figsize=(18, 18),
        constrained_layout=True,
    )

    for fila, (numero, ruta_original, ruta_n2n) in enumerate(
        seleccionados
    ):
        original = leer_rgb(ruta_original)
        resultado = leer_rgb(ruta_n2n)

        if resultado.shape[:2] != original.shape[:2]:
            resultado = cv2.resize(
                resultado,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        coordenadas = recorte_con_textura(original)
        x1, y1, x2, y2 = coordenadas

        original_marcado = dibujar_rectangulo(
            original,
            coordenadas,
        )
        resultado_marcado = dibujar_rectangulo(
            resultado,
            coordenadas,
        )

        zoom_original = original[y1:y2, x1:x2]
        zoom_n[y1:y2, x1:x22n = resultado]

        imagenes = [
            original_marcado,
            resultado_marcado,
            zoom_original,
            zoom_n2n,
        ]
        titulos = [
            f"Frame {numero}\nOriginal",
            f"Frame {numero}\nN2N",
            f"Frame {numero}\nOriginal ampliado",
            f"Frame {numero}\nN2N ampliado",
        ]

        for columna, (imagen, titulo) in enumerate(
            zip(imagenes, titulos)
        ):
            ejes[fila, columna].imshow(imagen)
            ejes[fila, columna].set_title(
                titulo,
                fontsize=11,
            )
            ejes[fila, columna].axis("off")

    figura.savefig(
        ruta_salida,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figura)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Muestra exactamente cinco comparaciones "
            "cualitativas entre frames originales y N2N."
        )
    )
    parser.add_argument(
        "--originales",
        required=True,
        help="Carpeta con los frames originales.",
    )
    parser.add_argument(
        "--n2n",
        required=True,
        help="Carpeta con los resultados de N2N.",
    )
    parser.add_argument(
        "--frames",
        nargs=5,
        type=int,
        default=None,
        metavar=("F1", "F2", "F3", "F4", "F5"),
        help=(
            "Cinco numeros de frame opcionales. "
            "Si se omiten, se seleccionan automáticamente."
        ),
    )
    parser.add_argument(
        "--salida",
        default="comparacion_n2n_5_frames.png",
        help="Ruta de la comparación completa.",
    )
    parser.add_argument(
        "--salida-zoom",
        default="comparacion_n2n_5_frames_zoom.png",
        help="Ruta de la comparación con recortes ampliados.",
    )
    args = parser.parse_args()

    pares = emparejar_imagenes(
        args.originales,
        args.n2n,
    )

    if args.frames is not None:
        seleccionados = seleccionar_por_ids(
            pares,
            args.frames,
        )
    else:
        seleccionados = seleccionar_cinco_diversos(pares)

    guardar_comparacion_completa(
        seleccionados,
        args.salida,
    )
    guardar_comparacion_con_zoom(
        seleccionados,
        args.salida_zoom,
    )

    print("Frames seleccionados:")
    for numero, ruta_original, ruta_n2n in seleccionados:
        print(
            f"  Frame {numero}: "
            f"{ruta_original.name} | {ruta_n2n.name}"
        )

    print(f"\nComparacion guardada en: {args.salida}")
    print(f"Comparacion con zoom guardada en: {args.salida_zoom}")


if __name__ == "__main__":
    main()
