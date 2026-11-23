#!/usr/bin/env python3
"""Mismo YOLO, mismo split, tres versiones del dataset: original, sin_hud y denoised.

Split recomendado: config/splits/sin_fuga_93grupos.csv (el de Jorge, 93 grupos de escena sin cruces:
777 train / 172 val / 167 test; es el mismo de Splits/03_Sin_Ruido aplicado a los mismos nombres).

Configuraciones tomadas del articulo IET (Tabla A.1): las dos mejores despues de reentrenar
  heavy_aug     yolo11m  batch 64  lr0 0.01  cos_lr  wd 0.001   dropout 0.2  mosaic 1.0  mixup 0.3  640 px
                (mejor mAP@50 general: 0.708 en la busqueda, 0.739-0.757 reentrenado)
  warm_restart  yolo11m  batch 64  lr0 0.05  cos_lr  wd 0.0005  dropout 0.1  mosaic 0.8  mixup 0.1  640 px
                (mejor recall de maquinaria pesada)
ambas reentrenadas como en el articulo: 150 epocas con parada temprana (paciencia 20).
Sin el preprocesamiento del articulo (mediana + CLAHE): aqui lo que se compara es la limpieza.

El split viene de un CSV (columnas imagen, split con valores train/val/test) con los nombres de
dataset_original. Las tres versiones usan exactamente las mismas imagenes en cada particion. El test
solo se usa una vez al final (la seleccion del checkpoint es con val).

Salida en --salida/<config>/<dataset>/: el run de ultralytics, metricas_test.csv (por clase) y
predicciones de test (para intervalos bootstrap despues). Resumen de todo en --salida/resumen.csv.
"""
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path

CONFIGS = {
    "heavy_aug": dict(model="yolo11m.pt", batch=64, lr0=0.01, optimizer="auto", cos_lr=True, weight_decay=0.001,
                      dropout=0.2, mosaic=1.0, mixup=0.3, imgsz=640),
    "warm_restart": dict(model="yolo11m.pt", batch=64, lr0=0.05, optimizer="auto", cos_lr=True, weight_decay=0.0005,
                         dropout=0.1, mosaic=0.8, mixup=0.1, imgsz=640),
}


def armar(ds: Path, split: dict, dest: Path, nombres: list[str]) -> Path:
    """Estructura YOLO con enlaces simbolicos (no copia imagenes)."""
    for parte in ("train", "val", "test"):
        for sub in ("images", "labels"):
            (dest / sub / parte).mkdir(parents=True, exist_ok=True)
    faltan = 0
    for im, parte in split.items():
        src_i, src_l = ds / "images" / im, ds / "labels" / (Path(im).stem + ".txt")
        if not src_i.exists():
            faltan += 1
            continue
        for src, d in ((src_i, dest / "images" / parte / im), (src_l, dest / "labels" / parte / src_l.name)):
            if src.exists() and not d.exists():
                os.symlink(src.resolve(), d)
    if faltan:
        raise SystemExit(f"!! {faltan} imagenes del split no estan en {ds}")
    y = dest / "data.yaml"
    y.write_text(f"path: {dest.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                 f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(nombres)), encoding="utf-8")
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, help="CSV imagen,split (train/val/test)")
    ap.add_argument("--datasets", nargs="+", default=["original", "sin_hud", "denoised"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--epocas", type=int, default=150)
    ap.add_argument("--paciencia", type=int, default=20)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--device", default="0,1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--salida", default="resultados/yolo_datasets")
    a = ap.parse_args()

    from ultralytics import YOLO
    import yaml

    with open(a.split, encoding="utf-8-sig") as fh:
        split = {r["imagen"]: r["split"].strip().lower() for r in csv.DictReader(fh)}
    assert set(split.values()) <= {"train", "val", "test"}, set(split.values())
    nombre_split = Path(a.split).stem
    sal = Path(a.salida) / nombre_split
    sal.mkdir(parents=True, exist_ok=True)
    resumen = []
    for cfg in a.configs:
        for dsn in a.datasets:
            ds = Path("datasets") / f"dataset_{dsn}"
            nombres = yaml.safe_load((ds / "clases.yaml").read_text(encoding="utf-8"))["names"]
            nombres = [nombres[k] for k in sorted(nombres)] if isinstance(nombres, dict) else list(nombres)
            nombres = ["maquinaria" if n == "SDZI" else n for n in nombres]  # misma clase 4 en las tres versiones
            data = armar(ds, split, Path("datasets/yolo") / nombre_split / dsn, nombres)
            run = sal / cfg / dsn
            if not (run / "train" / "weights" / "best.pt").exists():
                c = dict(CONFIGS[cfg]); modelo = c.pop("model")
                YOLO(modelo).train(data=str(data), epochs=a.epocas, patience=a.paciencia, seed=a.semilla,
                                   deterministic=True, device=a.device, workers=a.workers, project=str(run),
                                   name="train", exist_ok=True, **c)
            best = YOLO(str(run / "train" / "weights" / "best.pt"))
            m = best.val(data=str(data), split="test", imgsz=CONFIGS[cfg]["imgsz"], device=a.device.split(",")[0],
                         project=str(run), name="test", exist_ok=True, save_json=True, plots=True)
            filas = []
            for i, n in enumerate(nombres):
                idx = list(m.box.ap_class_index)
                p, r, ap50, ap = m.box.class_result(idx.index(i)) if i in idx else (None,) * 4
                filas.append(dict(config=cfg, dataset=dsn, clase=n, precision=p, recall=r, map50=ap50, map50_95=ap))
            filas.append(dict(config=cfg, dataset=dsn, clase="todas", precision=m.box.mp, recall=m.box.mr,
                              map50=m.box.map50, map50_95=m.box.map))
            with open(run / "metricas_test.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
            resumen += filas
            print(json.dumps(filas[-1]), flush=True)
    with open(sal / "resumen.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0])); w.writeheader(); w.writerows(resumen)
    print("Resumen:", sal / "resumen.csv")


if __name__ == "__main__":
    main()
