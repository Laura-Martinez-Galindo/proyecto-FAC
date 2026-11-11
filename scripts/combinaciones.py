#!/usr/bin/env python3
"""
Combinaciones de los mejores metodos de un regimen (ensembles y stacks), a partir del ranking.csv
de evaluar_denoising.py. Se crean como experimentos nuevos en cache/denoising/<FAMILIA>/ y se
evaluan igual que los demas (denoising_evaluar.sbatch vuelve a evaluar despues de crearlas).

  ens3media_r3      promedio pixel a pixel de las salidas de los 3 primeros del ranking
  ens3mediana_r3    mediana pixel a pixel de los 3 primeros (robusta si uno falla en una zona)
  stack1guiado_r3   el 1.o del ranking y despues el filtro guiado (quita el grano que quede)
  stack1medad_r3    el 1.o y despues la mediana adaptativa (quita impulsos que sobrevivan)
  stackmedad1_r3    mediana adaptativa ANTES del 1.o (le quita la sal y pimienta a la entrada); solo si
                    el 1.o es un modelo entrenado (vuelve a inferir con su checkpoint)
Tambien se arman en los frames vecinos (si existen) para las metricas temporales.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from fac_denoise.metodos import METODOS  # noqa: E402

PREFIJOS = ("ens3media_", "ens3mediana_", "stack1guiado_", "stack1medad_", "stackmedad1_")


def es_combinacion(nombre: str) -> bool:
    return nombre.startswith(PREFIJOS)


def leer(p):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None if im is None else np.ascontiguousarray(im[..., ::-1])


def guardar(p, im):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), np.ascontiguousarray(im[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--familia", required=True)
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--cache", default="cache/denoising")
    ap.add_argument("--entrada", default="datasets/dataset_sin_hud/images")
    ap.add_argument("--vecinos", default="datasets/vecinos_eval/images")
    ap.add_argument("--listas", default="resultados/denoising/listas")
    ap.add_argument("--sufijo", default="_r3")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()

    base = Path(a.cache) / a.familia
    filas = [r for r in csv.DictReader(open(a.ranking, encoding="utf-8")) if r["regimen"] == "TODOS"
             and r["posicion"] and r["estado"].startswith("OK") and not es_combinacion(r["experimento"])
             and r["experimento"].endswith((a.sufijo, a.sufijo + "_ycolor"))]
    filas.sort(key=lambda r: int(r["posicion"]))
    top = [r["experimento"] for r in filas[:a.top]]
    if len(top) < 2:
        print(f"{a.familia}: menos de 2 metodos validos en el ranking, sin combinaciones")
        return
    print(f"{a.familia}: combinaciones con {top}")
    for d in base.iterdir():  # las combinaciones se rehacen con el ranking actual
        if d.is_dir() and es_combinacion(d.name):
            shutil.rmtree(d)

    conf = Namespace(guiado_r=4, guiado_k=2.0)
    for sub in ("salidas", "vecinos"):
        dirs = [base / t / sub for t in top]
        if not all(d.is_dir() for d in dirs):
            continue
        nombres = sorted(set.intersection(*[{p.name for p in d.glob("*.png")} for d in dirs]))
        for n in nombres:
            ims = [leer(d / n) for d in dirs]
            if any(i is None for i in ims):
                continue
            st = np.stack([i.astype(np.float32) for i in ims])
            guardar(base / f"ens3media{a.sufijo}" / sub / n, np.clip(np.rint(st.mean(0)), 0, 255).astype(np.uint8))
            guardar(base / f"ens3mediana{a.sufijo}" / sub / n, np.clip(np.rint(np.median(st, 0)), 0, 255).astype(np.uint8))
            guardar(base / f"stack1guiado{a.sufijo}" / sub / n, METODOS["guiado"].clasico(ims[0], conf))
            guardar(base / f"stack1medad{a.sufijo}" / sub / n, METODOS["mediana_adaptativa"].clasico(ims[0], conf))
        for nom in ("ens3media", "ens3mediana", "stack1guiado", "stack1medad"):
            (base / f"{nom}{a.sufijo}" / sub / "inferencia.json").write_text(
                '{"combinacion": "%s", "de": %s}' % (nom, str(top).replace("'", '"')), encoding="utf-8")

    # mediana adaptativa ANTES del mejor modelo entrenado
    t1 = top[0].replace("_ycolor", "")
    ck = base / t1 / "modelo.pt"
    if ck.exists():
        tmp = base / f"stackmedad1{a.sufijo}" / "_entrada"
        for sub, ent, lista in (("salidas", a.entrada, f"{a.familia}_evaluacion.csv"),
                                ("vecinos", a.vecinos, f"{a.familia}_vecinos.csv")):
            lp = Path(a.listas) / lista
            if not lp.exists():
                continue
            filas_l = list(csv.DictReader(open(lp, encoding="utf-8-sig")))
            for r in filas_l:
                n = Path(r["imagen"]).stem + ".png"
                im = leer(Path(ent) / n)
                if im is not None:
                    guardar(tmp / sub / n, METODOS["mediana_adaptativa"].clasico(im, conf))
            subprocess.run([sys.executable, str(RAIZ / "scripts/denoising.py"), "inferir", "--checkpoint", str(ck),
                            "--lista", str(lp), "--entrada", str(tmp / sub),
                            "--salida", str(base / f"stackmedad1{a.sufijo}" / sub)], check=False)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
