#!/usr/bin/env python3
"""
Construye ``dataset_denoised`` a partir de ``dataset_sin_hud`` y de las salidas ya
generadas por el modelo ganador de cada regimen (fase 2).

Mismas imagenes, mismos nombres y mismas etiquetas que dataset_sin_hud. Para cada
imagen se toma la salida del ganador de su regimen; las imagenes de regimenes
minoritarios (termico, baja luz, SWIR) se copian tal cual de dataset_sin_hud,
porque ningun modelo se entreno con ese tipo de ruido.

Ganadores (config/denoising/ganadores.json):
    HDEO_DDE_HAZE -> f2f_f2_ycolor   (Frame2Frame, brillo de la red + color de la entrada)
    HDEO_HAZE     -> cvfsid_f2       (CVF-SID)
    HDEO_DDE      -> puca_f2         (PUCA)

Salida:
    datasets/dataset_denoised/
        images/*.png  labels/*.txt  clases.yaml
        correspondencia.csv   <- la de dataset_sin_hud + regimen, familia, metodo,
                                 experimento, checkpoint, paso del checkpoint, origen
        faltantes.csv         <- imagenes sin salida del ganador (deberia estar vacio)
        resumen.json
"""
from __future__ import annotations
import argparse, csv, json, shutil
from collections import Counter
from pathlib import Path

from PIL import Image


def leer_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--sin-hud", default="datasets/dataset_sin_hud")
    ap.add_argument("--listas", default="resultados/denoising/listas_fase2",
                    help="carpeta con <FAMILIA>_evaluacion.csv (imagen -> regimen)")
    ap.add_argument("--ganadores", default="config/denoising/ganadores.json")
    ap.add_argument("--cache", default="cache/denoising")
    ap.add_argument("--salida", default="datasets/dataset_denoised")
    ap.add_argument("--carpeta-salidas", default="salidas",
                    help="subcarpeta de cada experimento con las imagenes limpias (salidas_con_hud = limpiadas con HUD)")
    ap.add_argument("--forzar", action="store_true", help="borra la salida si ya existe")
    a = ap.parse_args()

    sh, sal = Path(a.sin_hud), Path(a.salida)
    gan = json.loads(Path(a.ganadores).read_text(encoding="utf-8"))
    if sal.exists():
        if not a.forzar:
            raise SystemExit(f"{sal} ya existe (use --forzar)")
        shutil.rmtree(sal)
    (sal / "images").mkdir(parents=True)
    shutil.copytree(sh / "labels", sal / "labels")
    if (sh / "clases.yaml").exists():
        shutil.copy2(sh / "clases.yaml", sal / "clases.yaml")

    # imagen -> (familia, regimen)
    asign = {}
    for fam in gan:
        for r in leer_csv(Path(a.listas) / f"{fam}_evaluacion.csv"):
            asign[r["imagen"]] = (fam, r["regimen"])

    # info del checkpoint de cada ganador
    info = {}
    for fam, g in gan.items():
        base = Path(a.cache) / fam / g["entrenamiento"]
        cfg = base / "configuracion.json"
        meta = json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}
        info[fam] = dict(metodo=g["metodo"], experimento=g["experimento"],
                         checkpoint=str(base / "modelo.pt"),
                         paso_checkpoint=meta.get("paso_mejor", meta.get("paso_final", "")),
                         parada=meta.get("parada", ""))

    filas = leer_csv(sh / "correspondencia.csv")
    faltan, conteo, out = [], Counter(), []
    for r in filas:
        im = r["imagen"]
        src_sh = sh / "images" / im
        if im in asign:
            fam, reg = asign[im]
            src = Path(a.cache) / fam / gan[fam]["experimento"] / a.carpeta_salidas / im
            if not src.exists():
                faltan.append(dict(imagen=im, familia=fam, esperado=str(src)))
                continue
            w0, h0 = Image.open(src_sh).size
            w1, h1 = Image.open(src).size
            if (w0, h0) != (w1, h1):
                raise SystemExit(f"tamano distinto en {im}: {w0}x{h0} vs {w1}x{h1}")
            extra = dict(regimen=reg, familia_denoising=fam, origen="denoised", **info[fam])
        else:
            src = src_sh
            extra = dict(regimen=r.get("regimen", "minoritario"), familia_denoising="(ninguna)",
                         origen="copiada de dataset_sin_hud (regimen minoritario)",
                         metodo="", experimento="", checkpoint="", paso_checkpoint="", parada="")
        shutil.copy2(src, sal / "images" / im)
        conteo[extra["familia_denoising"]] += 1
        out.append({**r, **extra})

    campos = list(out[0].keys()) if out else []
    with open(sal / "correspondencia.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader(); w.writerows(out)
    with open(sal / "faltantes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["imagen", "familia", "esperado"]); w.writeheader(); w.writerows(faltan)
    n_lab = len(list((sal / "labels").glob("*.txt")))
    res = dict(imagenes=len(out), etiquetas=n_lab, faltantes=len(faltan),
               por_familia=dict(conteo), ganadores=info)
    (sal / "resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if faltan:
        raise SystemExit(f"!! faltan {len(faltan)} salidas, ver {sal/'faltantes.csv'}")


if __name__ == "__main__":
    main()
