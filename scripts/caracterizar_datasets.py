#!/usr/bin/env python3
"""Caracteriza el ruido de CADA imagen en las tres versiones del dataset final:
original -> sin_hud -> denoised, con las mismas metricas de scripts/caracterizar_ruido.py
(sigma Immerkaer, sigma en zonas planas, grumos/autocorrelacion, rayas, impulsos, bloques 8x8...)
mas el grano visible de scripts/evaluar_denoising.py y, si se pide, NIQE y BRISQUE.

Las metricas de ruido se miden en la MISMA zona en las tres versiones: fuera del HUD (mascara de
dataset_sin_hud/mascaras_hud, dilatada) y de los bordes del frame. Asi el cambio original -> sin_hud
mide solo lo que el inpainting toca fuera de la mascara (deberia ser ~0) y sin_hud -> denoised mide el
denoising. NIQE y BRISQUE se miden sobre la imagen completa (como en la evaluacion), asi que ahi si
se ve el efecto del HUD.

Salida: resultados/caracterizacion_datasets/metricas_por_imagen.csv (version, imagen, regimen,
familia, metodo + metricas) y resultados/caracterizacion_datasets.tar
"""
from __future__ import annotations
import argparse, csv, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
from caracterizar_ruido import medir  # noqa: E402

VERSIONES = ("original", "sin_hud", "denoised")


def _uno(t):
    r = medir(t)
    if r is None:
        return None
    fila = r[0]
    img = cv2.imread(t["ruta"], cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    m = cv2.imread(t["mascara"], cv2.IMREAD_GRAYSCALE) if t["mascara"] and os.path.exists(t["mascara"]) else None
    hud = (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0) if m is not None else np.zeros((h, w), bool)
    hud = cv2.dilate(hud.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    sv = cv2.GaussianBlur(cv2.medianBlur(g.astype(np.uint8), 5).astype(np.float32), (0, 0), 1.5)
    mag = cv2.magnitude(cv2.Sobel(sv, cv2.CV_32F, 1, 0), cv2.Sobel(sv, cv2.CV_32F, 0, 1))
    excl = hud | (mag > np.quantile(mag[~hud], 0.90))
    fila["grano"] = round(float((g - cv2.GaussianBlur(g, (0, 0), 2.0))[~excl].std()), 5)
    return fila


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--salida", default="resultados/caracterizacion_datasets")
    ap.add_argument("--iqa", nargs="*", default=["niqe", "brisque"])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()
    D = Path(a.datasets)
    info = {}
    with open(D / "dataset_denoised/correspondencia.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            info[r["imagen"]] = {"video": r.get("video", ""), "regimen": r.get("regimen", ""),
                                 "familia": r.get("familia_denoising", ""), "metodo": r.get("metodo", "")}
    tareas = []
    for v in VERSIONES:
        for im in sorted(info):
            tareas.append({"ruta": str(D / f"dataset_{v}/images/{im}"),
                           "mascara": str(D / f"dataset_sin_hud/mascaras_hud/{im}"),
                           "margen": 12, "sigma_suave": 1.5,
                           "fila": {"version": v, "imagen": im, **info[im]}})
    with ProcessPoolExecutor(a.workers) as ex:
        filas = [f for f in ex.map(_uno, tareas, chunksize=8) if f is not None]
    print(len(filas), "imagenes medidas", flush=True)
    if a.iqa:
        from evaluar_denoising import iqa
        for v in VERSIONES:
            sel = [f for f in filas if f["version"] == v]
            res = iqa([str(D / f"dataset_{v}/images/{f['imagen']}") for f in sel], [f["imagen"] for f in sel], a.iqa, 4)
            for f in sel:
                f.update(res.get(f["imagen"], {}))
            print(v, "IQA listo", flush=True)
    sal = Path(a.salida); sal.mkdir(parents=True, exist_ok=True)
    campos = list(dict.fromkeys(k for f in filas for k in f))
    with open(sal / "metricas_por_imagen.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos); w.writeheader(); w.writerows(filas)
    print("->", sal / "metricas_por_imagen.csv")


if __name__ == "__main__":
    main()
