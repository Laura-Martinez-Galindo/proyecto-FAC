#!/usr/bin/env python3
"""
Figuras de caracterizacion de los 3 videos para la presentacion (todas con distribucion, no solo
promedios). Por video se toman --por-video frames equiespaciados en todo el video (lecturas del
HUD estables) y en cada uno, FUERA del HUD, se mide:

  intensidades      histograma de niveles de gris (0-255): como se reparte el brillo
  residuo           histograma de (gris - gris suavizado) en zonas planas: la forma del ruido.
                    Campana angosta = poco ruido; campana ancha = mucho; colas largas = impulsos
                    (sal y pimienta)
  hud_pct           % del frame que cubre la mascara del HUD (sobre los frames del video, no del dataset)
  sigma_immerkaer   nivel de ruido por frame (DN 0-255)
  impulsos_pct      % de pixeles que se alejan >30 de su mediana 3x3 (sal y pimienta)
  niqe, brisque     calidad sin referencia (pyiqa), sobre el frame completo (menor = mejor)

Salidas en --salida (por defecto resultados/figuras_caracterizacion/):
  por_frame.csv, resumen.csv (mediana, p25, p75 por video)
  intensidades.png, residuo.png, niqe_brisque.png, sigma_impulsos.png

Ejemplo:  python scripts/figuras_caracterizacion.py --por-video 120 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

from caracterizar_ruido import leer_csv, sigma_immerkaer  # noqa: E402

COLORES = {"video1": "#3f7f7a", "video2": "#e0785a", "video3": "#8e7cc3"}
NOMBRES = {"video1": "Video 1", "video2": "Video 2", "video3": "Video 3"}
BINS_RES = np.arange(-40.5, 41.5, 1.0)


def mascara_valida(img, ruta_mask, margen=12):
    """Devuelve (pixeles validos para medir ruido, % del frame que es HUD segun la mascara)."""
    h, w = img.shape[:2]
    if ruta_mask and os.path.exists(ruta_mask):
        m = cv2.resize(cv2.imread(ruta_mask, cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        hud_pct = float(100 * m.mean())
    else:
        m = np.zeros((h, w), bool)
        m[int(h * 0.60):int(h * 0.97), 0:int(w * 0.16)] = True
        hud_pct = float("nan")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (40, 150, 150), (85, 255, 255)) > 0
    m = m | (cv2.dilate(verde.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0)
    m = cv2.dilate(m.astype(np.uint8), np.ones((2 * margen + 1, 2 * margen + 1), np.uint8)) > 0
    m[:8], m[-8:], m[:, :8], m[:, -8:] = True, True, True, True
    return ~m, hud_pct


def medir(t):
    img = cv2.imread(t["ruta"], cv2.IMREAD_COLOR)
    if img is None:
        return None
    valido, hud_pct = mascara_valida(img, t["mascara"])
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    suave = cv2.GaussianBlur(g, (0, 0), 1.5)
    mag = cv2.magnitude(cv2.Sobel(suave, cv2.CV_32F, 1, 0), cv2.Sobel(suave, cv2.CV_32F, 0, 1))
    plano = valido & (mag <= np.quantile(mag[valido], 0.50))
    r = g - suave
    h_int = np.bincount(g[valido].astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    h_res = np.histogram(r[plano], bins=BINS_RES)[0].astype(np.float64)
    dif = np.abs(g - cv2.cvtColor(cv2.medianBlur(img, 3), cv2.COLOR_BGR2GRAY).astype(np.float32))
    fila = {"video": t["video"], "frame": t["frame"], "regimen": t["regimen"],
            "sigma_immerkaer": sigma_immerkaer(g, (mag > np.quantile(mag[valido], 0.90)) | ~valido),
            "impulsos_pct": float(100 * (dif[plano] > 30).mean()), "hud_pct": hud_pct}
    return fila, h_int / h_int.sum(), h_res / max(h_res.sum(), 1)


def iqa(filas, rutas, lote=8):
    try:
        try:
            import pkg_resources  # noqa: F401
        except ModuleNotFoundError:
            import packaging
            p = types.ModuleType("pkg_resources")
            p.packaging = packaging
            sys.modules["pkg_resources"] = p
        import pyiqa
        import torch
    except ImportError:
        print("!! pyiqa/torch no disponibles: sin NIQE/BRISQUE")
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for met in ("niqe", "brisque"):
        f = pyiqa.create_metric(met, device=dev)
        for i, (fila, ruta) in enumerate(zip(filas, rutas), 1):
            im = cv2.imread(ruta, cv2.IMREAD_COLOR)[..., ::-1]
            x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(255)[None].to(dev)
            with torch.no_grad():
                fila[met] = float(f(x).item())
            if i % 50 == 0:
                print(f"  {met} {i}/{len(filas)}", flush=True)


def estilo(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=13)


def violin(ax, datos, videos, etiqueta, rng):
    pos = list(range(1, len(videos) + 1))
    vals = [np.array([x for x in datos[v] if np.isfinite(x)]) for v in videos]
    partes = ax.violinplot(vals, pos, showextrema=False, widths=0.8)
    for b, v in zip(partes["bodies"], videos):
        b.set_facecolor(COLORES[v]); b.set_alpha(0.35); b.set_edgecolor("none")
    for p, x, v in zip(pos, vals, videos):
        ax.scatter(p + rng.uniform(-0.12, 0.12, len(x)), x, s=9, color=COLORES[v], alpha=0.6, lw=0)
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        ax.vlines(p, q1, q3, color="#1f2d27", lw=5)
        ax.scatter([p], [med], s=45, color="white", edgecolor="#1f2d27", zorder=3)
        ax.text(p + 0.42, med, f"{med:.1f}".replace(".", ","), va="center", fontsize=12, color="#1f2d27")
    ax.set_xticks(pos, [NOMBRES[v] for v in videos])
    ax.set_ylabel(etiqueta, fontsize=14)
    estilo(ax)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", default=["video1", "video2", "video3"])
    ap.add_argument("--regimenes", default="videos/{v}/regimenes_{v}.csv")
    ap.add_argument("--maximo-frame", nargs="*", default=["video3=23315"])
    ap.add_argument("--por-video", type=int, default=120)
    ap.add_argument("--sin-iqa", action="store_true")
    ap.add_argument("--salida", default="resultados/figuras_caracterizacion")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()

    from fac_denoise.datos import Fuente
    maxf = {x.split("=")[0]: int(x.split("=")[1]) for x in a.maximo_frame}
    tareas = []
    for v in a.videos:
        f = Fuente(v)
        print("  " + f.resumen())
        lect = sorted((int(r["frame"]), r) for r in leer_csv(a.regimenes.format(v=v))
                      if r.get("regimen") not in (None, "", "DESCONOCIDO"))
        if v in maxf:
            lect = [x for x in lect if x[0] <= maxf[v]]
        estables = [(k, r) for i, (k, r) in enumerate(lect[1:-1], 1)
                    if lect[i - 1][1]["regimen"] == r["regimen"] == lect[i + 1][1]["regimen"] and k in f.idx_orig]
        idx = np.unique(np.round(np.linspace(0, len(estables) - 1, min(a.por_video, len(estables)))).astype(int))
        for i in idx:
            k, r = estables[i]
            tareas.append({"video": v, "frame": k, "regimen": r["regimen"], "ruta": f.idx_orig[k],
                           "mascara": f.idx_mask.get(k)})
        print(f"    {v}: {len(idx)} frames")
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    filas, rutas, h_int, h_res = [], [], {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        for t, res in zip(tareas, ex.map(medir, tareas, chunksize=2)):
            if res is None:
                continue
            fila, hi, hr = res
            filas.append(fila); rutas.append(t["ruta"])
            h_int.setdefault(fila["video"], []).append(hi)
            h_res.setdefault(fila["video"], []).append(hr)
    if not a.sin_iqa:
        iqa(filas, rutas)
    mets = ["hud_pct", "sigma_immerkaer", "impulsos_pct", "niqe", "brisque"]
    with open(sal / "por_frame.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "frame", "regimen"] + mets, extrasaction="ignore")
        w.writeheader(); w.writerows(filas)
    videos = [v for v in a.videos if v in h_int]
    with open(sal / "resumen.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "metrica", "n", "p25", "mediana", "p75"])
        for v in videos:
            for m in mets:
                x = np.array([r.get(m, np.nan) for r in filas if r["video"] == v], float)
                x = x[np.isfinite(x)]
                if len(x):
                    w.writerow([v, m, len(x), *np.round(np.percentile(x, [25, 50, 75]), 3)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    for v in videos:
        H = np.array(h_int[v]); med = np.median(H, 0)
        ax.fill_between(range(256), np.percentile(H, 25, 0) * 100, np.percentile(H, 75, 0) * 100,
                        color=COLORES[v], alpha=0.2, lw=0)
        ax.plot(range(256), med * 100, color=COLORES[v], lw=2.5, label=NOMBRES[v])
    ax.set_xlabel("nivel de gris (0 = negro, 255 = blanco)", fontsize=14)
    ax.set_ylabel("% de píxeles", fontsize=14)
    ax.set_title("Distribución de intensidades (línea = mediana, banda = p25–p75 entre frames)", fontsize=13)
    ax.legend(frameon=False, fontsize=13); estilo(ax)
    fig.tight_layout(); fig.savefig(sal / "intensidades.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    centros = (BINS_RES[:-1] + BINS_RES[1:]) / 2
    for v in videos:
        H = np.array(h_res[v])
        ax.semilogy(centros, np.median(H, 0) * 100 + 1e-6, color=COLORES[v], lw=2.5, label=NOMBRES[v])
        ax.fill_between(centros, np.percentile(H, 25, 0) * 100 + 1e-6, np.percentile(H, 75, 0) * 100 + 1e-6,
                        color=COLORES[v], alpha=0.2, lw=0)
    ax.set_xlabel("residuo = píxel − su entorno suavizado (niveles de gris)", fontsize=14)
    ax.set_ylabel("% de píxeles (escala log)", fontsize=14)
    ax.set_title("Forma del ruido: más ancha = más ruido; colas largas = sal y pimienta", fontsize=13)
    ax.set_ylim(1e-4, None); ax.legend(frameon=False, fontsize=13); estilo(ax)
    fig.tight_layout(); fig.savefig(sal / "residuo.png", dpi=200); plt.close(fig)

    def datos(m):
        return {v: [r.get(m, np.nan) for r in filas if r["video"] == v] for v in videos}

    fig, ax = plt.subplots(figsize=(6, 5))
    violin(ax, datos("hud_pct"), videos, "% del frame cubierto por el HUD", rng)
    ax.set_title("HUD por frame (máscara)", fontsize=14)
    fig.tight_layout(); fig.savefig(sal / "hud_pct.png", dpi=200); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    violin(axs[0], datos("sigma_immerkaer"), videos, "σ del ruido (niveles de gris)", rng)
    axs[0].set_yscale("log"); axs[0].axhline(0.3, ls=":", color="gray"); axs[0].set_title("Nivel de ruido", fontsize=14)
    violin(axs[1], datos("impulsos_pct"), videos, "% de píxeles", rng)
    axs[1].set_title("Sal y pimienta", fontsize=14)
    fig.tight_layout(); fig.savefig(sal / "sigma_impulsos.png", dpi=200); plt.close(fig)

    if any("niqe" in r for r in filas):
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        violin(axs[0], datos("niqe"), videos, "NIQE (menor = mejor)", rng); axs[0].set_title("NIQE", fontsize=14)
        violin(axs[1], datos("brisque"), videos, "BRISQUE (menor = mejor)", rng); axs[1].set_title("BRISQUE", fontsize=14)
        fig.tight_layout(); fig.savefig(sal / "niqe_brisque.png", dpi=200); plt.close(fig)
    print("Listo:", sal)


if __name__ == "__main__":
    main()
