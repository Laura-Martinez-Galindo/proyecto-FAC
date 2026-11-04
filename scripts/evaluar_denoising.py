#!/usr/bin/env python3
"""
Evalua experimentos de denoising SIN imagen limpia de referencia, imagen por imagen.

Para cada imagen de --entrada (las ruidosas, p.ej. dataset_sin_hud/images) y cada experimento
(carpeta con las mismas imagenes limpias, mismo nombre) calcula:

  sigma_ruido          ruido que QUEDA en la salida (DN, 0-255). Metodo de Immerkaer (1996)
                       excluyendo bordes. Menor = mas limpio... pero ver retencion_bordes.
  ruido_quitado        desviacion estandar de (entrada - salida): cuanto se removio.
  retencion_bordes     en los bordes reales de la escena (gradiente de la entrada suavizada),
                       gradiente de la salida / gradiente de la entrada. ~1 conserva detalle;
                       <0.8 esta borrando estructura (sobre-suavizado). Robusto al ruido porque
                       compara versiones suavizadas.
  corr_residuo_bordes  correlacion entre |entrada - salida| y los bordes de la escena. ~0 = lo que
                       se quito no tiene forma (es ruido); alto = se quitaron casas, caminos, etc.
                       Es la prueba de "ruido estructural" ("method noise", Buades et al. 2005).
  autocorr_residuo     correlacion del residuo con su vecino (1 px). Indica si el ruido removido
                       venia en grumos (DDE/H.264) o era independiente por pixel.
  niqe, brisque        calidad perceptual sin referencia (opcional, --iqa, requiere pyiqa).
                       OJO: premian imagenes suaves; nunca usarlas solas.

La linea base "sin_denoising" (la entrada contra si misma) se agrega siempre.

Salidas en --salida:
  metricas_por_imagen.csv   una fila por (experimento, imagen) con regimen
  resumen.csv               por experimento x regimen: n, media, desv, p05, p25, p50, p75, p95
  ranking.csv               por regimen: experimentos que pasan los filtros de calidad,
                            ordenados por sigma_ruido mediano (con el motivo si se descartan)
  graficas/<metrica>.png    distribucion (caja + puntos) por experimento y regimen

Ejemplo:
  python scripts/evaluar_denoising.py --entrada datasets/dataset_sin_hud/images \\
      --experimento udvd_v2=cache/udvd_oficial/video2_ult13min/salidas \\
      --experimento b2u_v2=cache/b2u/video2/salidas \\
      --regimenes datasets/dataset_original/regimenes_dataset.csv \\
      --salida resultados/eval_denoising/ronda1 --iqa niqe brisque
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

METRICAS = ["sigma_ruido", "ruido_quitado", "retencion_bordes", "corr_residuo_bordes", "autocorr_residuo"]
MAYOR_ES_MEJOR = {"retencion_bordes": True}
EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def gris(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)


def mascara_bordes(g_suave: np.ndarray, frac: float) -> np.ndarray:
    mag = cv2.magnitude(cv2.Sobel(g_suave, cv2.CV_32F, 1, 0), cv2.Sobel(g_suave, cv2.CV_32F, 0, 1))
    return mag >= np.quantile(mag, 1 - frac), mag


def sigma_immerkaer(g: np.ndarray, excluir: np.ndarray | None) -> float:
    """Immerkaer, 'Fast noise variance estimation', CVIU 1996. Excluye bordes de la escena."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(g, cv2.CV_32F, k))[1:-1, 1:-1]
    if excluir is not None:
        m = ~excluir[1:-1, 1:-1]
        r = r[m] if m.sum() > 1000 else r.ravel()
    return float(math.sqrt(math.pi / 2) * r.mean() / 6.0)


def metricas_par(entrada: np.ndarray, salida: np.ndarray, frac_bordes: float, sigma_suave: float,
                 mascara_hud: np.ndarray | None = None) -> dict:
    if salida.shape != entrada.shape:
        salida = cv2.resize(salida, (entrada.shape[1], entrada.shape[0]), interpolation=cv2.INTER_AREA)
    gi, go = gris(entrada), gris(salida)
    si = cv2.GaussianBlur(gi, (0, 0), sigma_suave)
    so = cv2.GaussianBlur(go, (0, 0), sigma_suave)
    bordes, mag_i = mascara_bordes(si, frac_bordes)
    validos = np.ones_like(bordes) if mascara_hud is None else ~mascara_hud
    b = bordes & validos
    mag_o = cv2.magnitude(cv2.Sobel(so, cv2.CV_32F, 1, 0), cv2.Sobel(so, cv2.CV_32F, 0, 1))
    res = gi - go
    res_v = res[validos]
    abs_res = cv2.GaussianBlur(np.abs(res), (0, 0), 2.0)
    a, m = abs_res[validos].ravel(), mag_i[validos].ravel()
    corr = float(np.corrcoef(a, m)[0, 1]) if a.std() > 1e-6 else 0.0
    rc = res - res_v.mean()
    den = float((rc[validos] ** 2).mean()) + 1e-9
    ac = 0.5 * (float((rc[:, 1:] * rc[:, :-1]).mean()) + float((rc[1:] * rc[:-1]).mean())) / den
    excl = bordes | (~validos)
    return {
        "sigma_ruido": sigma_immerkaer(go, excl),
        "ruido_quitado": float(res_v.std()),
        "retencion_bordes": float(np.median(mag_o[b]) / (np.median(mag_i[b]) + 1e-6)) if b.any() else float("nan"),
        "corr_residuo_bordes": corr,
        "autocorr_residuo": ac if res_v.std() > 1e-6 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Trabajo en paralelo
# ---------------------------------------------------------------------------

def _tarea(t):
    exp, nombre, p_in, p_out, p_mask, frac, sig = t
    a = cv2.imread(p_in, cv2.IMREAD_COLOR)
    b = cv2.imread(p_out, cv2.IMREAD_COLOR)
    if a is None or b is None:
        return None
    mk = None
    if p_mask:
        mm = cv2.imread(p_mask, cv2.IMREAD_GRAYSCALE)
        if mm is not None:
            mk = cv2.resize(mm, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            mk = cv2.dilate(mk.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    r = metricas_par(a, b, frac, sig, mk)
    r.update(experimento=exp, imagen=nombre)
    return r


def iqa(rutas: list[str], nombres: list[str], metricas: list[str], lote: int) -> dict:
    try:
        import pyiqa
        import torch
    except ImportError:
        print("!! pyiqa/torch no disponibles: se omiten", metricas)
        return {}
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {n: {} for n in nombres}
    for met in metricas:
        f = pyiqa.create_metric(met, device=disp)
        for i in range(0, len(rutas), lote):
            ts = []
            for p in rutas[i:i + lote]:
                im = cv2.imread(p, cv2.IMREAD_COLOR)[..., ::-1]
                ts.append(torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(255))
            # las imagenes de un lote deben tener el mismo tamano
            grupos = {}
            for j, t in enumerate(ts):
                grupos.setdefault(tuple(t.shape), []).append((j, t))
            with torch.inference_mode():
                for _, g in grupos.items():
                    x = torch.stack([t for _, t in g]).to(disp)
                    v = f(x).flatten().cpu().tolist()
                    for (j, _), val in zip(g, v):
                        out[nombres[i + j]][met] = float(val) if math.isfinite(val) else float("nan")
        del f
    return out


# ---------------------------------------------------------------------------
# Resumen, ranking y graficas
# ---------------------------------------------------------------------------

def resumir(filas, metricas):
    grupos = {}
    for r in filas:
        for clave in ((r["experimento"], "TODOS"), (r["experimento"], r["regimen"])):
            grupos.setdefault(clave, []).append(r)
    out = []
    for (exp, reg), g in sorted(grupos.items()):
        fila = {"experimento": exp, "regimen": reg, "n": len(g)}
        for m in metricas:
            v = np.array([x[m] for x in g if x.get(m) is not None and not (isinstance(x[m], float) and math.isnan(x[m]))], float)
            if v.size == 0:
                continue
            fila.update({f"{m}_media": v.mean(), f"{m}_desv": v.std(), **{f"{m}_p{q:02d}": np.percentile(v, q) for q in (5, 25, 50, 75, 95)}})
        out.append(fila)
    return out


def ranking(resumen, min_ret, max_corr):
    out = []
    for reg in sorted({r["regimen"] for r in resumen}):
        filas = [r for r in resumen if r["regimen"] == reg and r["experimento"] != "sin_denoising"]
        for r in filas:
            motivos = []
            if r.get("retencion_bordes_p25", 1) < min_ret:
                motivos.append(f"retencion_bordes p25={r['retencion_bordes_p25']:.2f} < {min_ret}")
            if r.get("corr_residuo_bordes_p50", 0) > max_corr:
                motivos.append(f"corr_residuo_bordes p50={r['corr_residuo_bordes_p50']:.2f} > {max_corr}")
            r["_motivos"] = "; ".join(motivos)
        ok = sorted([r for r in filas if not r["_motivos"]], key=lambda r: r.get("sigma_ruido_p50", 9e9))
        malos = [r for r in filas if r["_motivos"]]
        for pos, r in enumerate(ok, 1):
            out.append({"regimen": reg, "posicion": pos, "experimento": r["experimento"], "n": r["n"],
                        "sigma_ruido_p50": round(r.get("sigma_ruido_p50", float("nan")), 3),
                        "retencion_bordes_p25": round(r.get("retencion_bordes_p25", float("nan")), 3),
                        "corr_residuo_bordes_p50": round(r.get("corr_residuo_bordes_p50", float("nan")), 3),
                        "estado": "OK"})
        for r in malos:
            out.append({"regimen": reg, "posicion": "", "experimento": r["experimento"], "n": r["n"],
                        "sigma_ruido_p50": round(r.get("sigma_ruido_p50", float("nan")), 3),
                        "retencion_bordes_p25": round(r.get("retencion_bordes_p25", float("nan")), 3),
                        "corr_residuo_bordes_p50": round(r.get("corr_residuo_bordes_p50", float("nan")), 3),
                        "estado": "DESCARTADO: " + r["_motivos"]})
    return out


def graficar(filas, metricas, carpeta: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("!! matplotlib no disponible: sin graficas")
        return
    carpeta.mkdir(parents=True, exist_ok=True)
    exps = sorted({r["experimento"] for r in filas}, key=lambda e: (e != "sin_denoising", e))
    regs = sorted({r["regimen"] for r in filas})
    for m in metricas:
        fig, axs = plt.subplots(1, len(regs), figsize=(max(5, 1.2 * len(exps)) * len(regs), 4.5), squeeze=False)
        for ax, reg in zip(axs[0], regs):
            datos = [[r[m] for r in filas if r["experimento"] == e and r["regimen"] == reg
                      and r.get(m) is not None and not math.isnan(r[m])] for e in exps]
            ax.boxplot([d if d else [np.nan] for d in datos], showfliers=False)
            for i, d in enumerate(datos, 1):
                if d:
                    ax.scatter(np.random.normal(i, 0.06, len(d)), d, s=6, alpha=0.35)
            ax.set_xticks(range(1, len(exps) + 1))
            ax.set_xticklabels(exps, rotation=35, ha="right", fontsize=8)
            ax.set_title(f"{reg}  (n={len(datos[0])})", fontsize=9)
            ax.grid(alpha=0.3)
        fig.suptitle(m + ("  (mayor = mejor)" if MAYOR_ES_MEJOR.get(m) else ""), fontsize=11)
        fig.tight_layout()
        fig.savefig(carpeta / f"{m}.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True, help="Carpeta con las imagenes ruidosas (referencia)")
    ap.add_argument("--experimento", action="append", default=[], metavar="NOMBRE=CARPETA")
    ap.add_argument("--regimenes", help="CSV con columnas imagen, regimen (p.ej. regimenes_dataset.csv)")
    ap.add_argument("--mascaras", help="Carpeta de mascaras HUD (mismo nombre) para excluir esos pixeles")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--iqa", nargs="*", default=[], help="Metricas pyiqa extra: niqe brisque piqe ...")
    ap.add_argument("--lote-iqa", type=int, default=8)
    ap.add_argument("--max-imagenes", type=int, default=0, help="Muestra por regimen (0 = todas)")
    ap.add_argument("--frac-bordes", type=float, default=0.10, help="Fraccion de pixeles tomados como bordes")
    ap.add_argument("--sigma-suave", type=float, default=1.5)
    ap.add_argument("--min-retencion", type=float, default=0.80, help="Filtro de ranking: retencion_bordes p25")
    ap.add_argument("--max-corr", type=float, default=0.15, help="Filtro de ranking: corr_residuo_bordes p50")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()

    ent = Path(a.entrada)
    nombres = sorted(p.name for p in ent.iterdir() if p.suffix.lower() in EXT)
    regimen = {}
    if a.regimenes:
        with open(a.regimenes, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                regimen[Path(r["imagen"]).stem] = r.get("regimen", "SIN_REGIMEN")
    if a.max_imagenes:
        rng = np.random.default_rng(a.semilla)
        por_reg = {}
        for n in nombres:
            por_reg.setdefault(regimen.get(Path(n).stem, "SIN_REGIMEN"), []).append(n)
        nombres = sorted(x for lst in por_reg.values()
                         for x in (rng.choice(lst, min(len(lst), a.max_imagenes), replace=False)))
    exps = [("sin_denoising", ent)] + [(e.split("=", 1)[0], Path(e.split("=", 1)[1])) for e in a.experimento]

    tareas = []
    faltan = {}
    for exp, carpeta in exps:
        for n in nombres:
            p_out = carpeta / n
            if not p_out.exists():
                alt = list(carpeta.glob(Path(n).stem + ".*"))
                if not alt:
                    faltan[exp] = faltan.get(exp, 0) + 1
                    continue
                p_out = alt[0]
            p_mask = str(Path(a.mascaras) / n) if a.mascaras else None
            tareas.append((exp, n, str(ent / n), str(p_out), p_mask, a.frac_bordes, a.sigma_suave))
    for e, k in faltan.items():
        print(f"!! {e}: faltan {k} imagenes (se omiten)")
    print(f"{len(nombres)} imagenes x {len(exps)} experimentos = {len(tareas)} pares")

    filas = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(_tarea, tareas, chunksize=4), 1):
            if r:
                r["regimen"] = regimen.get(Path(r["imagen"]).stem, "SIN_REGIMEN")
                filas.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(tareas)}", flush=True)

    metricas = list(METRICAS)
    if a.iqa:
        for exp, carpeta in exps:
            sel = [r for r in filas if r["experimento"] == exp]
            rutas = [t[3] for t in tareas if t[0] == exp]
            nm = [t[1] for t in tareas if t[0] == exp]
            v = iqa(rutas, nm, a.iqa, a.lote_iqa)
            for r in sel:
                r.update(v.get(r["imagen"], {}))
        metricas += a.iqa

    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    cols = ["experimento", "imagen", "regimen"] + metricas
    with open(sal / "metricas_por_imagen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    res = resumir(filas, metricas)
    cols_r = ["experimento", "regimen", "n"] + [f"{m}_{s}" for m in metricas for s in ("media", "desv", "p05", "p25", "p50", "p75", "p95")]
    with open(sal / "resumen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols_r, extrasaction="ignore")
        w.writeheader()
        for r in res:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    rk = ranking(res, a.min_retencion, a.max_corr)
    if rk:
        with open(sal / "ranking.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rk[0].keys()))
            w.writeheader()
            w.writerows(rk)
    graficar(filas, metricas, sal / "graficas")
    print(f"Listo: {sal}")
    for r in rk:
        print(f"  {r['regimen']:18s} {str(r['posicion']):>3s} {r['experimento']:22s} sigma={r['sigma_ruido_p50']}  "
              f"ret_p25={r['retencion_bordes_p25']}  corr={r['corr_residuo_bordes_p50']}  {r['estado']}")


if __name__ == "__main__":
    main()
