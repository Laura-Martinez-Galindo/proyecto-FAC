#!/usr/bin/env python3
"""
Genera un visor HTML (sin servidor, se abre con doble clic) con tres secciones:

  1. Limpieza por regimen: original / sin HUD / ganador / finalistas de la fase 2, con zoom y
     paneo sincronizados y las anotaciones del dataset que se prenden y apagan.
  2. Deteccion YOLO (test de la particion sin fuga): cajas reales vs predichas en original, sin HUD
     y sin ruido, con umbral de confianza, aciertos (verde), falsas alarmas (rojo) y perdidas
     (punteado).
  3. Captioning: cada ventana con la hoja de contacto que vio el modelo, su respuesta y las
     etiquetas del dataset que caen en esa ventana.

Salida: resultados/visor/ (index.html + img/ + datos.js) y resultados/visor.tar

Ejemplo:  python scripts/generar_visor.py --n-limpieza 12 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
LETRA = {"HDEO_DDE_HAZE": "A", "HDEO_HAZE": "B", "HDEO_DDE": "C"}
NOMBRE_METODO = {"f2f": "Frame2Frame", "pcst": "PCST", "rdrf": "RDRF", "stbn": "STBN", "cvfsid": "CVF-SID",
                 "apbsn": "AP-BSN", "mmbsn": "MM-BSN", "puca": "PUCA"}


def leer_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def leer_labels(p):
    cajas = []
    if p and Path(p).exists():
        for ln in Path(p).read_text().split("\n"):
            x = ln.split()
            if len(x) >= 5:
                cajas.append([int(float(x[0]))] + [round(float(v), 5) for v in x[1:5]])
    return cajas  # clase, xc, yc, w, h normalizados


def guardar_jpg(t):
    src, dst, lado, q = t
    if Path(dst).exists():
        return True
    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if im is None:
        return False
    h, w = im.shape[:2]
    if lado and max(h, w) > lado:
        f = lado / max(h, w)
        im = cv2.resize(im, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, q])


def nombres_clases(a):
    for p in [Path(a.ds_original) / "clases.yaml", Path(a.ds_yolo) / "original" / "data.yaml"]:
        if p.exists():
            try:
                import yaml
                d = yaml.safe_load(p.read_text(encoding="utf-8"))
                n = d.get("names", d)
                if isinstance(n, dict):
                    return [str(n[k]) for k in sorted(n, key=int)]
                if isinstance(n, list):
                    return [str(x) for x in n]
            except Exception as e:
                print("!! no pude leer", p, e)
    return ["building", "road", "vehicle", "maquinaria", "river"]


# ---------------------------------------------------------------------------
def seccion_limpieza(a, tareas):
    corr = leer_csv(Path(a.ds_denoised) / "correspondencia.csv")
    gan = json.loads((RAIZ / "config/denoising/ganadores.json").read_text(encoding="utf-8"))
    fase2 = [l.split("\t") for l in (RAIZ / "config/denoising/fase2.tsv").read_text(encoding="utf-8").split("\n")
             if l and not l.startswith("#")]
    out = []
    for fam, letra in LETRA.items():
        filas = sorted([r for r in corr if r.get("familia_denoising") == fam], key=lambda r: r["imagen"])
        if not filas:
            continue
        idx = np.unique(np.round(np.linspace(0, len(filas) - 1, min(a.n_limpieza, len(filas)))).astype(int))
        versiones = [("Original", Path(a.ds_original) / "images"), ("Sin HUD", Path(a.ds_sin_hud) / "images"),
                     (f"Sin ruido · {gan[fam]['metodo']} (elegido)", Path(a.ds_denoised) / "images")]
        for f in fase2:
            exp, fa, met = f[0], f[1], f[2]
            if fa == fam and exp != gan[fam]["entrenamiento"]:
                versiones.append((f"Finalista · {NOMBRE_METODO.get(met, met)}", Path(a.cache) / fam / exp / "salidas"))
        alt = RAIZ / "config/denoising/ganadores_iqa.json"
        iqa = json.loads(alt.read_text(encoding="utf-8")).get(fam) if alt.exists() else None
        versiones.append((f"Variante · con HUD + {gan[fam]['metodo']}", Path(a.cache) / fam / gan[fam]["experimento"] / "salidas_con_hud"))
        if iqa:
            versiones.append((f"Variante · sin HUD + {iqa['metodo']} (mejor NIQE/BRISQUE)", Path(a.cache) / fam / iqa["experimento"] / "salidas"))
            versiones.append((f"Variante · con HUD + {iqa['metodo']} (mejor NIQE/BRISQUE)", Path(a.cache) / fam / iqa["experimento"] / "salidas_con_hud"))
        imgs = []
        for i in idx:
            r = filas[i]
            stem = Path(r["imagen"]).stem
            vers = []
            for k, (nom, carpeta) in enumerate(versiones):
                src = carpeta / r["imagen"]
                if not src.exists():
                    continue
                rel = f"img/limpieza/{letra}/{stem}_{k}.jpg"
                tareas.append((src, Path(a.salida) / rel, 0, 95))
                vers.append({"nombre": nom, "src": rel})
            imgs.append({"id": stem, "video": r.get("video", ""), "frame": r.get("frame", ""), "versiones": vers,
                         "cajas": leer_labels(Path(a.ds_original) / "labels" / f"{stem}.txt")})
        out.append({"regimen": letra, "familia": fam, "metodo": gan[fam]["metodo"], "n_total": len(filas),
                    "imagenes": imgs})
        print(f"  limpieza {letra}: {len(imgs)} imagenes, {len(versiones)} versiones")
    return out


# ---------------------------------------------------------------------------
def carpeta_test(ds_yolo, version):
    p = Path(ds_yolo) / version / "data.yaml"
    try:
        import yaml
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        base = Path(d.get("path", p.parent))
        if not base.is_absolute():
            base = (p.parent / base) if (p.parent / base).exists() else RAIZ / base
        t = d.get("test")
        t = t[0] if isinstance(t, list) else t
        im = base / t
        return im, Path(str(im).replace(f"{os.sep}images", f"{os.sep}labels"))
    except Exception as e:
        print("!! data.yaml", p, e)
        return Path(ds_yolo) / version / "images" / "test", Path(ds_yolo) / version / "labels" / "test"


def seccion_deteccion(a, tareas):
    nombres = {"original": "Original", "sin_hud": "Sin HUD", "denoised": "Sin HUD + sin ruido (grano)",
               "denoised_iqa": "Sin HUD + sin ruido (NIQE/BRISQUE)", "conhud_grano": "Con HUD + sin ruido (grano)",
               "conhud_iqa": "Con HUD + sin ruido (NIQE/BRISQUE)"}
    versiones = [(v, n) for v, n in nombres.items() if (Path(a.ds_yolo) / v / "data.yaml").exists()]
    test = sorted(r["imagen"] for r in leer_csv(a.split) if r["split"] == "test")
    preds = {}
    for cfg in a.configs:
        for v, _ in versiones:
            p = Path(a.res_yolo) / cfg / v / "test" / "predictions.json"
            if not p.exists():
                print("!! falta", p)
                continue
            for d in json.loads(p.read_text()):
                if d["score"] >= a.conf_min:
                    x, y, w, h = d["bbox"]
                    preds.setdefault((cfg, v, d["image_id"]), []).append(
                        [d["category_id"] - 1, round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(d["score"], 3)])
    con_pred = {k[1] for k in preds}
    versiones = [(v, n) for v, n in versiones if v in con_pred]
    carpetas = {v: carpeta_test(a.ds_yolo, v) for v, _ in versiones}
    imgs = []
    for im in test:
        stem = Path(im).stem
        vers, wh = [], None
        for v, nom in versiones:
            src = carpetas[v][0] / im
            if not src.exists():
                continue
            if wh is None:
                x = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
                wh = [x.shape[1], x.shape[0]] if x is not None else None
            rel = f"img/deteccion/{v}/{stem}.jpg"
            tareas.append((src, Path(a.salida) / rel, a.lado_deteccion, 88))
            vers.append({"clave": v, "nombre": nom, "src": rel,
                         "pred": {c: preds.get((c, v, stem), []) for c in a.configs}})
        if vers and wh:
            imgs.append({"id": stem, "wh": wh, "versiones": vers,
                         "cajas": leer_labels(carpetas["original"][1] / f"{stem}.txt")})
    resumen = leer_csv(Path(a.res_yolo) / "resumen.csv") if (Path(a.res_yolo) / "resumen.csv").exists() else []
    print(f"  deteccion: {len(imgs)} imagenes de test")
    return {"configs": a.configs, "imagenes": imgs, "resumen": resumen}


# ---------------------------------------------------------------------------
def seccion_captioning(a, tareas, clases):
    base = Path(a.res_captioning)
    if not base.exists():
        print("!! no existe", base)
        return []
    fps = {"video1": 30.0, "video2": 29.97, "video3": 30.0}
    etiquetas = []  # (video, segundo, clases)
    corr = Path(a.ds_original) / "correspondencia.csv"
    if corr.exists():
        for r in leer_csv(corr):
            try:
                seg = float(r["segundo"]) if r.get("segundo") else int(r["frame"]) / fps.get(r["video"], 30.0)
            except (KeyError, ValueError):
                continue
            cl = sorted({clases[c[0]] if c[0] < len(clases) else str(c[0])
                         for c in leer_labels(Path(a.ds_original) / "labels" / f"{Path(r['imagen']).stem}.txt")})
            etiquetas.append((r["video"], seg, cl))
    out = []
    for f in sorted(base.glob("*/*/*/ventanas.jsonl")):
        for ln in f.read_text(encoding="utf-8").split("\n"):
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            mos = f.parent / "mosaicos" / f"{d['video']}_{int(round(d['inicio_s'] * 1000)):08d}.jpg"
            rel = None
            if mos.exists():
                rel = f"img/captioning/{f.parent.name}/{mos.name}"
                tareas.append((mos, Path(a.salida) / rel, 0, 85))
            dentro = [e for e in etiquetas if e[0] == d["video"] and d["inicio_s"] <= e[1] < d["fin_s"]]
            j = d.get("json") or {}
            ind = {k: v for k, v in (j.get("indicadores") or {}).items() if v in ("si", "incierto")}
            out.append({"video": d["video"], "version": d["version"], "prompt": d["prompt"],
                        "modelo": d["modelo"].split("/")[-1], "ini": d["inicio_s"], "fin": d["fin_s"],
                        "mosaico": rel, "descripcion": j.get("descripcion", ""), "escena": j.get("escena", ""),
                        "justificacion": j.get("justificacion", ""), "indicadores": ind,
                        "riesgo_inicial": d.get("riesgo_inicial"), "riesgo_final": d.get("riesgo_final"),
                        "n_imagenes_anotadas": len(dentro),
                        "clases_anotadas": sorted({c for e in dentro for c in e[2]})})
    print(f"  captioning: {len(out)} ventanas")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds-original", default="datasets/dataset_original")
    ap.add_argument("--ds-sin-hud", default="datasets/dataset_sin_hud")
    ap.add_argument("--ds-denoised", default="datasets/dataset_denoised")
    ap.add_argument("--cache", default="cache/denoising")
    ap.add_argument("--ds-yolo", default="datasets/yolo/sin_fuga_93grupos")
    ap.add_argument("--res-yolo", default="resultados/yolo_datasets/sin_fuga_93grupos")
    ap.add_argument("--split", default="config/splits/sin_fuga_93grupos.csv")
    ap.add_argument("--configs", nargs="+", default=["heavy_aug", "warm_restart"])
    ap.add_argument("--res-captioning", default="resultados/captioning_mineria")
    ap.add_argument("--n-limpieza", type=int, default=12, help="imagenes por regimen")
    ap.add_argument("--lado-deteccion", type=int, default=1280)
    ap.add_argument("--conf-min", type=float, default=0.05)
    ap.add_argument("--salida", default="resultados/visor")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()
    os.chdir(RAIZ)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    clases = nombres_clases(a)
    tareas, datos = [], {"clases": clases}
    for nombre, fn in (("limpieza", lambda: seccion_limpieza(a, tareas)),
                       ("deteccion", lambda: seccion_deteccion(a, tareas)),
                       ("captioning", lambda: seccion_captioning(a, tareas, clases))):
        try:
            datos[nombre] = fn()
        except Exception as e:
            print(f"!! seccion {nombre} fallo: {e}")
            datos[nombre] = None
    print(f"{len(tareas)} imagenes a convertir")
    with ProcessPoolExecutor(a.workers) as ex:
        ok = sum(ex.map(guardar_jpg, tareas, chunksize=8))
    print(f"  {ok} listas")
    (sal / "datos.js").write_text("window.DATOS = " + json.dumps(datos, ensure_ascii=False) + ";", encoding="utf-8")
    (sal / "index.html").write_text((RAIZ / "scripts" / "visor_plantilla.html").read_text(encoding="utf-8"),
                                    encoding="utf-8")
    print("Listo:", sal / "index.html")


if __name__ == "__main__":
    main()
