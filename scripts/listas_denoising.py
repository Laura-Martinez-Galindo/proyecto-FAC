#!/usr/bin/env python3
"""
Paso 3 del plan de denoising: listas de frames de cada familia (lo minimo suficiente).

Entradas
  videos/<video>/regimenes_<video>.csv           (detectar_regimen_hud.py carpeta, 1 de cada 15 frames)
  datasets/dataset_original/regimenes_dataset.csv (regimen de cada imagen del dataset)
  datasets/dataset_original/correspondencia.csv   (video y frame de cada imagen del dataset)
  config/denoising/familias.json                   {familia: [regimenes]}

Salidas en --salida (por defecto resultados/denoising/listas):
  <FAMILIA>_entrenamiento.csv  video, frame, regimen       frames de los videos para entrenar
  <FAMILIA>_evaluacion.csv     imagen, video, frame, regimen  imagenes del dataset para comparar
  <FAMILIA>_todas.csv          todas las imagenes del dataset de la familia (para dataset_denoised)
  resumen.csv / resumen.json   cuantos frames hay, cuantos quedan libres, cuantos se eligieron

Reglas
  * Un frame k cuenta para un regimen solo si la lectura anterior y la siguiente del HUD
    coinciden (no se usan las transiciones) y k-2..k+2 tambien (UDVD usa vecinos).
  * Se excluyen +-margen frames alrededor de CADA imagen del dataset: el denoiser nunca ve
    los frames que va a limpiar, sin importar como se haga el split despues.
  * Por familia se toman hasta --max-por-familia frames; cada regimen recibe su parte
    proporcional pero al menos --minimo-por-regimen (si los tiene). Dentro de un regimen se
    toman equiespaciados, asi quedan repartidos en todos sus tramos.
  * Evaluacion: todas las imagenes si la familia tiene <= --max-evaluacion; si no, muestra
    estratificada (al menos --minimo-eval-por-regimen de cada regimen).
  * Si una imagen del dataset tiene un regimen que no esta en ninguna familia, se avisa.

  --exportar-mascaras CARPETA  guarda la mascara del HUD de cada imagen del dataset (mismo nombre)
                               para que evaluar_denoising.py --mascaras excluya esos pixeles.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent


def leer_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def escribir_csv(p, filas, cols):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)


def frames_por_regimen(filas, maximo_frame, vecinos):
    """{regimen: np.array de frames validos} a partir de lecturas cada ~15 frames."""
    lect = sorted((int(r["frame"]), r["regimen"]) for r in filas
                  if r.get("regimen") and r["regimen"] != "DESCONOCIDO")
    if maximo_frame:
        lect = [x for x in lect if x[0] <= maximo_frame]
    if not lect:
        return {}, 0
    fr = np.array([x[0] for x in lect])
    rg = np.array([x[1] for x in lect], dtype=object)
    paso = int(np.median(np.diff(fr))) if len(fr) > 1 else 1
    ultimo = maximo_frame or int(fr[-1] + paso - 1)
    ks = np.arange(1, ultimo + 1)
    ia = np.clip(np.searchsorted(fr, ks, side="right") - 1, 0, len(fr) - 1)  # lectura anterior
    ib = np.clip(np.searchsorted(fr, ks, side="left"), 0, len(fr) - 1)       # lectura siguiente
    ok = rg[ia] == rg[ib]
    lab = np.where(ok, rg[ia], None)
    # vecinos del mismo regimen
    val = lab.copy()
    for d in range(1, vecinos + 1):
        izq = np.r_[[None] * d, lab[:-d]]
        der = np.r_[lab[d:], [None] * d]
        val = np.where((izq == lab) & (der == lab), val, None)
    out = defaultdict(list)
    for k, r in zip(ks, val):
        if r is not None:
            out[r].append(k)
    return {r: np.array(v) for r, v in out.items()}, ultimo


def repartir(disponibles: dict, total: int, minimo: int) -> dict:
    """Cuantos tomar de cada clave: proporcional, con minimo, sin pasarse de lo disponible."""
    n = {k: len(v) for k, v in disponibles.items()}
    if sum(n.values()) <= total:
        return n
    base = {k: min(n[k], minimo) for k in n}
    resto = total - sum(base.values())
    if resto <= 0:
        return base
    extra_disp = {k: n[k] - base[k] for k in n}
    s = sum(extra_disp.values())
    asign = {k: base[k] + int(np.floor(resto * extra_disp[k] / s)) for k in n}
    falta = total - sum(asign.values())
    for k in sorted(n, key=lambda k: -extra_disp[k]):
        if falta <= 0:
            break
        if asign[k] < n[k]:
            asign[k] += 1
            falta -= 1
    return asign


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", default=["video1", "video2", "video3"])
    ap.add_argument("--regimenes", default="videos/{v}/regimenes_{v}.csv", help="patron con {v}")
    ap.add_argument("--maximo-frame", nargs="*", default=["video3=23315"], help="video=N (video3: 1a vuelta)")
    ap.add_argument("--dataset-regimenes", default="datasets/dataset_original/regimenes_dataset.csv")
    ap.add_argument("--correspondencia", default="datasets/dataset_original/correspondencia.csv")
    ap.add_argument("--familias", default="config/denoising/familias.json")
    ap.add_argument("--margen", type=int, default=90, help="frames excluidos a cada lado de una imagen del dataset")
    ap.add_argument("--vecinos", type=int, default=2)
    ap.add_argument("--max-por-familia", type=int, default=1500)
    ap.add_argument("--minimo-por-regimen", type=int, default=150)
    ap.add_argument("--max-evaluacion", type=int, default=120, help="0 = todas")
    ap.add_argument("--minimo-eval-por-regimen", type=int, default=20)
    ap.add_argument("--exportar-mascaras", default="", help="carpeta para las mascaras HUD de las imagenes")
    ap.add_argument("--salida", default="resultados/denoising/listas")
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.semilla)
    familias = json.loads(Path(a.familias).read_text(encoding="utf-8"))
    fam_de = {r: f for f, rs in familias.items() for r in rs}
    maxf = {x.split("=")[0]: int(x.split("=")[1]) for x in a.maximo_frame}
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)

    corr = leer_csv(a.correspondencia)
    excl = defaultdict(set)
    for r in corr:
        k0 = int(r["frame"])
        excl[r["video"]].update(range(k0 - a.margen, k0 + a.margen + 1))

    # ---- frames disponibles en los videos
    disp = defaultdict(dict)     # familia -> (video, regimen) -> frames
    resumen = []
    for v in a.videos:
        ruta = Path(a.regimenes.format(v=v))
        if not ruta.exists():
            print(f"!! falta {ruta}")
            continue
        por_reg, ultimo = frames_por_regimen(leer_csv(ruta), maxf.get(v, 0), a.vecinos)
        for reg, ks in sorted(por_reg.items()):
            libres = np.array([k for k in ks if k not in excl[v]])
            fam = fam_de.get(reg, "")
            resumen.append({"familia": fam or "(sin familia)", "video": v, "regimen": reg,
                            "frames_regimen": len(ks), "frames_libres": len(libres)})
            if fam and len(libres):
                disp[fam][(v, reg)] = libres

    # ---- dataset
    ds = leer_csv(a.dataset_regimenes)
    frame_de = {Path(r["imagen"]).stem: (r["video"], int(r["frame"])) for r in corr}
    sin_familia = Counter()
    ds_fam = defaultdict(list)
    for r in ds:
        st = Path(r["imagen"]).stem
        fam = fam_de.get(r["regimen"])
        if not fam:
            sin_familia[(r["video"], r["regimen"])] += 1
            continue
        v, k = frame_de.get(st, (r["video"], int(r["frame"])))
        ds_fam[fam].append({"imagen": st + ".png", "video": v, "frame": k, "regimen": r["regimen"], "familia": fam})
    if sin_familia:
        print("!! Imagenes del dataset cuyo regimen NO esta en ninguna familia (agregarlo a familias.json):")
        for (v, reg), n in sin_familia.items():
            print(f"     {v} {reg}: {n}")

    # ---- elegir
    elegidos = Counter()
    eval_n = Counter()
    for fam in familias:
        d = disp.get(fam, {})
        cuantos = repartir(d, a.max_por_familia, a.minimo_por_regimen)
        filas = []
        for (v, reg), ks in d.items():
            n = cuantos[(v, reg)]
            if n == 0:
                continue
            sel = ks[np.unique(np.round(np.linspace(0, len(ks) - 1, n)).astype(int))]
            filas += [{"video": v, "frame": int(k), "regimen": reg} for k in sel]
            elegidos[(fam, v, reg)] = len(sel)
        filas.sort(key=lambda r: (r["video"], r["frame"]))
        escribir_csv(sal / f"{fam}_entrenamiento.csv", filas, ["video", "frame", "regimen"])

        todas = sorted(ds_fam.get(fam, []), key=lambda r: r["imagen"])
        escribir_csv(sal / f"{fam}_todas.csv", todas, ["imagen", "video", "frame", "regimen", "familia"])
        por_reg = defaultdict(list)
        for r in todas:
            por_reg[(r["video"], r["regimen"])].append(r)
        tot = a.max_evaluacion or len(todas)
        cuantos_e = repartir(por_reg, tot, a.minimo_eval_por_regimen)
        ev = []
        for kreg, lst in por_reg.items():
            idx = rng.choice(len(lst), cuantos_e[kreg], replace=False)
            ev += [lst[i] for i in sorted(idx)]
            eval_n[(fam, *kreg)] = cuantos_e[kreg]
        ev.sort(key=lambda r: r["imagen"])
        escribir_csv(sal / f"{fam}_evaluacion.csv", ev, ["imagen", "video", "frame", "regimen", "familia"])
        print(f"{fam:11s} entrenamiento={len(filas):5d}  evaluacion={len(ev):4d}  dataset={len(todas):4d}")

    ds_cnt = Counter((r["video"], r["regimen"]) for r in ds)
    for r in resumen:
        k = (r["familia"], r["video"], r["regimen"])
        r["frames_elegidos"] = elegidos.get(k, 0)
        r["imagenes_dataset"] = ds_cnt.get((r["video"], r["regimen"]), 0)
        r["imagenes_evaluacion"] = eval_n.get(k, 0)
    for (v, reg), n in ds_cnt.items():  # regimenes del dataset que no aparecieron en el video
        if not any(r["video"] == v and r["regimen"] == reg for r in resumen):
            resumen.append({"familia": fam_de.get(reg, "(sin familia)"), "video": v, "regimen": reg,
                            "frames_regimen": 0, "frames_libres": 0, "frames_elegidos": 0,
                            "imagenes_dataset": n, "imagenes_evaluacion": eval_n.get((fam_de.get(reg), v, reg), 0)})
    resumen.sort(key=lambda r: (r["familia"], r["video"], -r["frames_regimen"]))
    cols = ["familia", "video", "regimen", "frames_regimen", "frames_libres", "frames_elegidos",
            "imagenes_dataset", "imagenes_evaluacion"]
    escribir_csv(sal / "resumen.csv", resumen, cols)
    (sal / "resumen.json").write_text(json.dumps({"parametros": vars(a), "familias": familias, "tabla": resumen},
                                                 indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"{'familia':12s} {'video':7s} {'regimen':16s} {'frames':>7s} {'libres':>7s} {'elegid':>7s} {'ds':>5s} {'eval':>5s}")
    for r in resumen:
        print(f"{r['familia']:12s} {r['video']:7s} {r['regimen']:16s} {r['frames_regimen']:7d} {r['frames_libres']:7d} "
              f"{r['frames_elegidos']:7d} {r['imagenes_dataset']:5d} {r['imagenes_evaluacion']:5d}")

    if a.exportar_mascaras:
        sys.path.insert(0, str(RAIZ / "scripts"))
        import cv2
        from fac_denoise.datos import Fuente
        dst = Path(a.exportar_mascaras)
        dst.mkdir(parents=True, exist_ok=True)
        fuentes, n = {}, 0
        for r in corr:
            v, k = r["video"], int(r["frame"])
            if v not in fuentes:
                fuentes[v] = Fuente(v)
            f = fuentes[v]
            base = f.idx_orig.get(k)
            if not base:
                continue
            h, w = cv2.imread(base, cv2.IMREAD_GRAYSCALE).shape[:2]
            m = f.mascara(k, (h, w))
            if m is not None:
                cv2.imwrite(str(dst / (Path(r["imagen"]).stem + ".png")), m.astype(np.uint8) * 255)
                n += 1
        print(f"Mascaras HUD: {n}/{len(corr)} -> {dst}")
    print(f"Listas: {sal}")


if __name__ == "__main__":
    main()
