#!/usr/bin/env python3
"""
Devuelve el COLOR ORIGINAL a las salidas de un metodo de denoising.

Varias redes (UDVD, NB2NB, B2U y en menor medida AP-BSN) cambian el tono de la imagen al limpiarla
(ronda 1: les queda entre el 24 % y el 83 % del color). Aqui se toma de la salida SOLO el brillo
(canal Y de YCrCb, que es donde la red quito el ruido) y el color (Cr, Cb) se toma de la imagen de
entrada, suavizado para quitarle las manchitas de color (mediana + gaussiano). El color del H.264
ya viene a media resolucion (4:2:0), asi que suavizarlo no le quita detalle real.

  python scripts/color_original.py --entrada datasets/dataset_sin_hud/images \\
      --base cache/denoising/HDEO_DDE_HAZE/apbsn_r1/salidas \\
      --salida cache/denoising/HDEO_DDE_HAZE/apbsn_r1_ycolor/salidas
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def recolorear(entrada: np.ndarray, salida: np.ndarray, mediana: int, sigma: float) -> np.ndarray:
    if salida.shape != entrada.shape:
        salida = cv2.resize(salida, (entrada.shape[1], entrada.shape[0]), interpolation=cv2.INTER_AREA)
    yo = cv2.cvtColor(salida, cv2.COLOR_BGR2YCrCb)
    yi = cv2.cvtColor(entrada, cv2.COLOR_BGR2YCrCb)
    for c in (1, 2):
        ch = yi[..., c]
        if mediana > 1:
            ch = cv2.medianBlur(ch, mediana)
        if sigma > 0:
            ch = cv2.GaussianBlur(ch, (0, 0), sigma)
        yo[..., c] = ch
    return cv2.cvtColor(yo, cv2.COLOR_YCrCb2BGR)


def _tarea(t):
    p_in, p_out, p_dest, mediana, sigma = t
    a, b = cv2.imread(p_in, cv2.IMREAD_COLOR), cv2.imread(p_out, cv2.IMREAD_COLOR)
    if a is None or b is None:
        return False
    cv2.imwrite(p_dest, recolorear(a, b, mediana, sigma), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True, help="Imagenes de entrada (ruidosas), p.ej. dataset_sin_hud/images")
    ap.add_argument("--base", required=True, help="Carpeta salidas/ del metodo")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--mediana", type=int, default=5, help="Mediana sobre Cr/Cb (impar; 1 = no)")
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussiano sobre Cr/Cb")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()
    ent, base, sal = Path(a.entrada), Path(a.base), Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    tareas = []
    for p in sorted(base.glob("*.png")):
        p_in = ent / p.name
        if p_in.exists():
            tareas.append((str(p_in), str(p), str(sal / p.name), a.mediana, a.sigma))
    with ProcessPoolExecutor(a.workers) as ex:
        ok = sum(ex.map(_tarea, tareas, chunksize=4))
    print(f"{base} -> {sal}: {ok}/{len(tareas)} imagenes con el color original")


if __name__ == "__main__":
    main()
