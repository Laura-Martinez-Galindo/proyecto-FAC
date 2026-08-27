#!/usr/bin/env python3
"""Predice un lote N2V en un proceso aislado para liberar VRAM al terminar."""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUTA_PROYECTO / "scripts"))

import pipeline_n2v as pipeline


def obtener_argumentos():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", required=True)
    parser.add_argument("--salida", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trabajo", required=True)
    return parser.parse_args()


def ejecutar(argumentos):
    carpeta_entrada = Path(argumentos.entrada).resolve()
    carpeta_salida = Path(argumentos.salida).resolve()
    checkpoint = Path(argumentos.checkpoint).resolve()
    carpeta_trabajo = Path(argumentos.trabajo).resolve()

    if not carpeta_entrada.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de entrada: {carpeta_entrada}"
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No existe el checkpoint: {checkpoint}"
        )

    archivos_entrada = sorted(carpeta_entrada.glob("*.tif"))

    if not archivos_entrada:
        raise RuntimeError(
            f"No se encontraron TIFF en: {carpeta_entrada}"
        )

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    carpeta_trabajo.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    argumentos_configuracion = argparse.Namespace(
        video="video1",
        modo="original",
        epocas=1,
        pasos_por_epoca=5,
    )

    configuracion = pipeline.crear_configuracion(
        argumentos_configuracion,
        entrenamiento=False,
    )

    careamist = pipeline.CAREamist(
        config=configuracion,
        work_dir=str(carpeta_trabajo),
        enable_progress_bar=False,
    )

    resultado = careamist.predict(
        pred_data=str(carpeta_entrada),
        batch_size=1,
        tile_size=pipeline.TAMANO_TILE,
        tile_overlap=pipeline.SOLAPAMIENTO_TILE,
        axes=pipeline.EJES,
        data_type="tiff",
        num_workers=0,
        in_memory=False,
        checkpoint=str(checkpoint),
    )

    predicciones, fuentes = pipeline.extraer_predicciones(
        resultado
    )

    if len(predicciones) != len(archivos_entrada):
        raise RuntimeError(
            f"Se esperaban {len(archivos_entrada)} predicciones "
            f"y CAREamics devolvió {len(predicciones)}."
        )

    mapa_entradas = {
        archivo.stem: archivo
        for archivo in archivos_entrada
    }

    if fuentes and len(fuentes) == len(predicciones):
        pares = []

        for fuente, prediccion in zip(
            fuentes,
            predicciones,
        ):
            nombre = Path(fuente).stem

            if nombre not in mapa_entradas:
                raise RuntimeError(
                    f"CAREamics devolvió una fuente desconocida: {fuente}"
                )

            pares.append((nombre, prediccion))
    else:
        pares = [
            (archivo.stem, prediccion)
            for archivo, prediccion in zip(
                archivos_entrada,
                predicciones,
            )
        ]

    for nombre, prediccion in pares:
        if isinstance(prediccion, torch.Tensor):
            prediccion = (
                prediccion
                .detach()
                .cpu()
                .numpy()
            )

        prediccion = np.asarray(prediccion)

        pipeline.guardar_png(
            carpeta_salida / f"{nombre}.png",
            prediccion,
        )

    cantidad_guardada = sum(
        1
        for archivo in archivos_entrada
        if (
            carpeta_salida
            / f"{archivo.stem}.png"
        ).is_file()
    )

    if cantidad_guardada != len(archivos_entrada):
        raise RuntimeError(
            f"Solo se guardaron {cantidad_guardada} de "
            f"{len(archivos_entrada)} imágenes."
        )

    print(
        f"Lote N2V completado: {cantidad_guardada} frames.",
        flush=True,
    )

    del resultado
    del predicciones
    del careamist
    del configuracion

    gc.collect()
    torch.cuda.empty_cache()


def main():
    try:
        ejecutar(obtener_argumentos())
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
