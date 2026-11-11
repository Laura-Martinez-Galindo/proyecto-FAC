#!/usr/bin/env python3
"""
Frames vecinos (sin HUD) de las primeras N imagenes de evaluacion de cada familia, para las
metricas temporales de evaluar_denoising.py:

  warping_error  (Lai et al., ECCV 2018) |salida(t) - salida(t+-1) alineada a t|: si el metodo
                 "parpadea" entre frames, sube. Necesita la salida del metodo en t-1 y t+1.
  upsnr          (Marcos-Morales et al., 2023, uMSE): estima el error contra la imagen limpia SIN
                 tenerla, con 3 realizaciones ruidosas independientes alineadas a t (t+1, t-1, t+2):
                 uMSE = mean((f(t) - y_t+1)^2) - 0.5 * mean((y_t-1 - y_t+2)^2).

Escribe datasets/vecinos_eval/images/<video>_<frame>.png (construidos igual que dataset_sin_hud,
Fuente.compuesto) y resultados/denoising/listas/<FAMILIA>_vecinos.csv (los t-1 y t+1, que son los
que cada metodo debe limpiar).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from fac_denoise.datos import Fuente, guardar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listas", default="resultados/denoising/listas")
    ap.add_argument("--familias", default="", help="por defecto las de config/denoising/familias.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--salida", default="datasets/vecinos_eval/images")
    a = ap.parse_args()
    fams = a.familias.split() or list(json.load(open(RAIZ / "config/denoising/familias.json")))
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    fuentes = {}
    for f in fams:
        ruta = Path(a.listas) / f"{f}_evaluacion.csv"
        if not ruta.exists():
            print(f"!! {f}: no hay {ruta}")
            continue
        filas = list(csv.DictReader(open(ruta, encoding="utf-8-sig")))[:a.n]
        salida = []
        for r in filas:
            v, k = r["video"], int(float(r["frame"]))
            if v not in fuentes:
                fuentes[v] = Fuente(v)
            ok = True
            for d in (-1, 1, 2):
                dst = sal / f"{v}_{k + d:06d}.png"
                if dst.exists():
                    continue
                im = fuentes[v].compuesto(k + d, 3)
                if im is None:
                    ok = False
                    break
                guardar(dst, im)
            if ok:
                for d in (-1, 1):
                    salida.append({"imagen": f"{v}_{k + d:06d}.png", "video": v, "frame": k + d,
                                   "centro": r["imagen"]})
        with open(Path(a.listas) / f"{f}_vecinos.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["imagen", "video", "frame", "centro"])
            w.writeheader()
            w.writerows(salida)
        print(f"{f}: {len(salida) // 2} imagenes con vecinos -> {f}_vecinos.csv")


if __name__ == "__main__":
    main()
