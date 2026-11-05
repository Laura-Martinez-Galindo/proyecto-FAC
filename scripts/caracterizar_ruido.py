#!/usr/bin/env python3
"""
Caracteriza el RUIDO de cada regimen en cada video, para decidir con datos (no a ojo) que
regimenes se pueden limpiar con el mismo modelo.

Para cada grupo video|regimen se toman --por-grupo frames repartidos en todos sus tramos
(solo frames cuya lectura anterior y siguiente del HUD coinciden) y en cada frame, fuera del
HUD (mascaras_hud dilatadas) y de los bordes de la escena, se mide:

  brillo, contraste           contexto de la escena (media y desviacion del gris)
  sigma_immerkaer             nivel de ruido (DN 0-255), Immerkaer 1996, sin bordes
  sigma_plano                 desviacion del residuo (gris - suavizado) en zonas planas
  ruido_oscuro / medio / claro  sigma_plano en zonas oscuras (<70), medias y claras (>180):
                              si cambia mucho, el ruido depende de la senal (HAZE, baja luz)
  ruido_croma_rel             ruido en color (Cr, Cb) / ruido en luminancia
  autocorr_1, autocorr_2      correlacion del residuo con el vecino a 1 y 2 px (promedio
  autocorr_h1, autocorr_v1    horizontal/vertical). ~0 = grano independiente; alto = grumos (DDE,
                              compresion). Es lo que decide si sirve un blind-spot simple o hace
                              falta romper la correlacion (AP-BSN, PD, etc.)
  rayas_verticales            var(media por columna del residuo) * filas / var(residuo):
  rayas_horizontales          ~1 sin rayas; >>1 con rayas de columna/fila (FPN del termico)
  extremos_pct                % de pixeles <30 o >225 (la regla del 1% de los articulos de la FAC)
  impulsos_pct                % de pixeles que difieren >30 de su mediana 3x3 en zonas planas
                              (sal y pimienta: pixeles defectuosos SWIR, baja luz)
  bloques_8, bloques_16       |gradiente| en las fronteras de bloque / |gradiente| fuera de ellas
                              (~1 = sin bloques de compresion; >1 = bloques H.264 visibles)

Salidas en --salida (por defecto resultados/ruido_regimenes/):
  metricas_por_frame.csv      una fila por frame (con video, regimen, foco, exposicion)
  resumen.csv                 por grupo: n, media, desv, p05, p25, p50, p75, p95 de cada metrica
  resumen_exposicion.csv      lo mismo separando EXP MAN / EXP AUT / ...
  graficas/<metrica>.png      distribucion (caja + puntos) por grupo
  espectros.png               espectro de potencia promedio del residuo por grupo (log):
                              rayas = lineas por el centro; bloques = puntos en rejilla
  recortes.png                recortes 1:1 de cada grupo (original y residuo x4), para verlo a ojo
  distancias.csv / .png       distancia entre grupos (medianas estandarizadas de las metricas de
                              ruido) y agrupamiento jerarquico sugerido (agrupamiento.txt)

Ejemplo:
  python scripts/caracterizar_ruido.py --por-grupo 60 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

METRICAS = ["brillo", "contraste", "sigma_immerkaer", "sigma_plano", "ruido_oscuro", "ruido_medio",
            "ruido_claro", "ruido_croma_rel", "autocorr_1", "autocorr_2", "autocorr_h1", "autocorr_v1",
            "rayas_verticales", "rayas_horizontales", "extremos_pct", "impulsos_pct", "bloques_8", "bloques_16"]
# las que describen el RUIDO (las de contexto no entran a la distancia entre grupos)
METRICAS_RUIDO = ["sigma_immerkaer", "sigma_plano", "ruido_croma_rel", "autocorr_1", "autocorr_2",
                  "rayas_verticales", "rayas_horizontales", "impulsos_pct", "bloques_8"]
LOG = {"sigma_immerkaer", "sigma_plano", "rayas_verticales", "rayas_horizontales", "impulsos_pct"}
LADO_ESPECTRO = 256


# ---------------------------------------------------------------------------
# Medidas sobre un frame
# ---------------------------------------------------------------------------

def sigma_immerkaer(g, excluir):
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(g, cv2.CV_32F, k))[1:-1, 1:-1]
    m = ~excluir[1:-1, 1:-1]
    r = r[m] if m.sum() > 1000 else r.ravel()
    return float(math.sqrt(math.pi / 2) * r.mean() / 6.0)


def autocorr(r, valido, dy, dx):
    a = r[:r.shape[0] - dy, :r.shape[1] - dx]
    b = r[dy:, dx:]
    m = valido[:r.shape[0] - dy, :r.shape[1] - dx] & valido[dy:, dx:]
    if m.sum() < 1000:
        return float("nan")
    a, b = a[m], b[m]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-9))


def rayas(r, valido, eje):
    """var de la media por columna (eje=0) o fila (eje=1), normalizada por lo esperado sin rayas."""
    rr = np.where(valido, r, 0.0)
    n = valido.sum(axis=eje)
    ok = n > 0.5 * valido.shape[eje]
    if ok.sum() < 20:
        return float("nan")
    medias = rr.sum(axis=eje)[ok] / n[ok]
    v = float(r[valido].var()) + 1e-9
    return float(medias.var() * np.mean(n[ok]) / v)


def bloques(g, valido, periodo):
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    vx = valido[:, 1:] & valido[:, :-1]
    vy = valido[1:, :] & valido[:-1, :]
    fx = (np.arange(dx.shape[1]) % periodo) == periodo - 1
    fy = (np.arange(dy.shape[0]) % periodo) == periodo - 1
    en = np.concatenate([dx[:, fx][vx[:, fx]], dy[fy, :][vy[fy, :]]])
    fuera = np.concatenate([dx[:, ~fx][vx[:, ~fx]], dy[~fy, :][vy[~fy, :]]])
    if en.size < 500 or fuera.size < 500:
        return float("nan")
    return float(en.mean() / (fuera.mean() + 1e-9))


def recorte_sin_hud(m, lado):
    """Esquina superior izquierda de un cuadrado lado x lado con el menor HUD (busca en rejilla)."""
    h, w = m.shape
    if h < lado or w < lado:
        return None
    integ = cv2.integral(m.astype(np.uint8))
    mejor = None
    for y in np.linspace(0, h - lado, 5).astype(int):
        for x in np.linspace(0, w - lado, 7).astype(int):
            s = integ[y + lado, x + lado] - integ[y, x + lado] - integ[y + lado, x] + integ[y, x]
            if mejor is None or s < mejor[0]:
                mejor = (s, y, x)
    return mejor[1], mejor[2]


def medir(tarea):
    ruta, ruta_mask, margen, sig_suave = tarea["ruta"], tarea["mascara"], tarea["margen"], tarea["sigma_suave"]
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if ruta_mask and os.path.exists(ruta_mask):
        m = cv2.imread(ruta_mask, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:  # sin mascara: caja fija donde esta el texto del HUD
        m = np.zeros((h, w), bool)
        m[int(h * 0.60):int(h * 0.97), 0:int(w * 0.16)] = True
    # ademas, cualquier texto verde saturado del HUD en todo el frame (coordenadas, retícula...),
    # por si la mascara no lo cubre: verde muy saturado y brillante, que la vegetacion no alcanza
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (40, 150, 150), (85, 255, 255)) > 0
    m = m | (cv2.dilate(verde.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0)
    m = cv2.dilate(m.astype(np.uint8), np.ones((2 * margen + 1, 2 * margen + 1), np.uint8)) > 0
    borde = np.zeros((h, w), bool)
    borde[:8], borde[-8:], borde[:, :8], borde[:, -8:] = True, True, True, True
    valido = ~(m | borde)

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    suave = cv2.GaussianBlur(g, (0, 0), sig_suave)
    mag = cv2.magnitude(cv2.Sobel(suave, cv2.CV_32F, 1, 0), cv2.Sobel(suave, cv2.CV_32F, 0, 1))
    umbral_borde = np.quantile(mag[valido], 0.90)
    plano = valido & (mag <= np.quantile(mag[valido], 0.50))
    r = g - suave

    out = {"brillo": float(g[valido].mean()), "contraste": float(g[valido].std()),
           "sigma_immerkaer": sigma_immerkaer(g, (mag > umbral_borde) | ~valido),
           "sigma_plano": float(r[plano].std())}
    for nom, lo, hi in (("ruido_oscuro", 0, 70), ("ruido_medio", 70, 180), ("ruido_claro", 180, 256)):
        sel = plano & (suave >= lo) & (suave < hi)
        out[nom] = float(r[sel].std()) if sel.sum() > 2000 else float("nan")
    rc = [ycc[..., c] - cv2.GaussianBlur(ycc[..., c], (0, 0), sig_suave) for c in (1, 2)]
    out["ruido_croma_rel"] = float(np.mean([x[plano].std() for x in rc]) / (out["sigma_plano"] + 1e-9))
    out["autocorr_h1"] = autocorr(r, plano, 0, 1)
    out["autocorr_v1"] = autocorr(r, plano, 1, 0)
    out["autocorr_1"] = float(np.nanmean([out["autocorr_h1"], out["autocorr_v1"]]))
    out["autocorr_2"] = float(np.nanmean([autocorr(r, plano, 0, 2), autocorr(r, plano, 2, 0)]))
    out["rayas_verticales"] = rayas(r, plano, 0)
    out["rayas_horizontales"] = rayas(r, plano, 1)
    out["extremos_pct"] = float(100 * ((g[valido] < 30) | (g[valido] > 225)).mean())
    med = cv2.medianBlur(img, 3)
    dif = np.abs(g - cv2.cvtColor(med, cv2.COLOR_BGR2GRAY).astype(np.float32))
    out["impulsos_pct"] = float(100 * (dif[plano] > 30).mean())
    out["bloques_8"] = bloques(g, valido, 8)
    out["bloques_16"] = bloques(g, valido, 16)

    # espectro del residuo en un cuadrado sin HUD
    esp = None
    pos = recorte_sin_hud(m, LADO_ESPECTRO)
    if pos:
        y, x = pos
        p = r[y:y + LADO_ESPECTRO, x:x + LADO_ESPECTRO]
        p = (p - p.mean()) * np.outer(np.hanning(LADO_ESPECTRO), np.hanning(LADO_ESPECTRO))
        esp = (np.abs(np.fft.fftshift(np.fft.fft2(p))) ** 2).astype(np.float32)
    # recorte para mirar (el del centro de la zona sin HUD)
    rec = None
    pos = recorte_sin_hud(m, 160)
    if pos:
        y, x = pos
        rec = img[y:y + 160, x:x + 160].copy()
    return {**tarea["fila"], **{k: (round(v, 5) if isinstance(v, float) and math.isfinite(v) else v)
                                for k, v in out.items()}}, esp, rec


# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------

def leer_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def elegir_frames(filas, por_grupo, maximo_frame, rng, disponibles=None):
    """Lecturas del HUD (cada ~15 frames) cuyo vecino anterior y siguiente tienen el mismo regimen,
    equiespaciadas dentro de cada regimen (asi quedan todos los tramos)."""
    lect = sorted((int(r["frame"]), r) for r in filas if r.get("regimen") not in (None, "", "DESCONOCIDO"))
    if maximo_frame:
        lect = [x for x in lect if x[0] <= maximo_frame]
    por_reg = defaultdict(list)
    for i in range(1, len(lect) - 1):
        k, r = lect[i]
        if lect[i - 1][1]["regimen"] == r["regimen"] == lect[i + 1][1]["regimen"]:
            por_reg[r["regimen"]].append(r)
    out = {}
    for reg, lst in por_reg.items():
        if disponibles is not None:  # solo frames que existen en disco
            lst = [r for r in lst if int(r["frame"]) in disponibles]
        if not lst:
            continue
        n = min(por_grupo, len(lst))
        idx = np.unique(np.round(np.linspace(0, len(lst) - 1, n)).astype(int))
        out[reg] = [lst[i] for i in idx]
    return out


# ---------------------------------------------------------------------------
# Resumen, graficas y distancias
# ---------------------------------------------------------------------------

def percentiles(v):
    v = np.array([x for x in v if isinstance(x, (int, float)) and math.isfinite(x)], float)
    if v.size == 0:
        return {}
    return {"media": v.mean(), "desv": v.std(), **{f"p{q:02d}": np.percentile(v, q) for q in (5, 25, 50, 75, 95)}}


def escribir_resumen(filas, clave, ruta):
    grupos = defaultdict(list)
    for r in filas:
        grupos[clave(r)].append(r)
    cols = ["grupo", "n"] + [f"{m}_{s}" for m in METRICAS for s in ("media", "desv", "p05", "p25", "p50", "p75", "p95")]
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for g in sorted(grupos):
            fila = {"grupo": g, "n": len(grupos[g])}
            for m in METRICAS:
                for s, v in percentiles([r[m] for r in grupos[g]]).items():
                    fila[f"{m}_{s}"] = round(float(v), 4)
            w.writerow(fila)
    return grupos


def graficar(grupos, carpeta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    carpeta.mkdir(parents=True, exist_ok=True)
    nombres = sorted(grupos)
    for m in METRICAS:
        datos = [[r[m] for r in grupos[g] if isinstance(r[m], float) and math.isfinite(r[m])] for g in nombres]
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(nombres)), 4.8))
        ax.boxplot([d if d else [np.nan] for d in datos], showfliers=False)
        for i, d in enumerate(datos, 1):
            if d:
                ax.scatter(np.random.normal(i, 0.07, len(d)), d, s=7, alpha=0.4)
        ax.set_xticks(range(1, len(nombres) + 1))
        ax.set_xticklabels([f"{g}\n(n={len(grupos[g])})" for g in nombres], rotation=40, ha="right", fontsize=8)
        if m in LOG:
            ax.set_yscale("log")
        ax.set_title(m)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(carpeta / f"{m}.png", dpi=120)
        plt.close(fig)


def hoja_espectros(espectros, ruta):
    tiles = []
    for g in sorted(espectros):
        e = np.log10(np.mean(espectros[g], axis=0) + 1e-6)
        e = (e - np.percentile(e, 1)) / (np.percentile(e, 99.5) - np.percentile(e, 1) + 1e-9)
        im = cv2.applyColorMap(np.clip(e * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        bar = np.zeros((22, LADO_ESPECTRO, 3), np.uint8)
        cv2.putText(bar, f"{g} (n={len(espectros[g])})"[:40], (3, 15), 0, 0.42, (255, 255, 255), 1)
        tiles.append(np.vstack([bar, im]))
    guardar_rejilla(tiles, ruta, 5)


def hoja_recortes(recortes, ruta, por_grupo=4):
    filas_img = []
    for g in sorted(recortes):
        lst = recortes[g][:: max(1, len(recortes[g]) // por_grupo)][:por_grupo]
        celdas = []
        for rc in lst:
            gg = cv2.cvtColor(rc, cv2.COLOR_BGR2GRAY).astype(np.float32)
            res = np.clip((gg - cv2.GaussianBlur(gg, (0, 0), 1.5)) * 4 + 128, 0, 255).astype(np.uint8)
            celdas.append(np.hstack([rc, cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)]))
        while len(celdas) < por_grupo:
            celdas.append(np.zeros((160, 320, 3), np.uint8))
        fila = np.hstack(celdas)
        bar = np.zeros((24, fila.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{g}   (cada par: original 1:1 | residuo x4)", (4, 17), 0, 0.5, (255, 255, 255), 1)
        filas_img.append(np.vstack([bar, fila]))
    if filas_img:
        cv2.imwrite(str(ruta), np.vstack(filas_img), [cv2.IMWRITE_JPEG_QUALITY, 92])


def guardar_rejilla(tiles, ruta, cols):
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(ruta), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def distancias(grupos, sal, min_n=10):
    nombres = [g for g in sorted(grupos) if len(grupos[g]) >= min_n]
    if len(nombres) < 2:
        return
    med = np.array([[np.nanmedian([r[m] if isinstance(r[m], float) else np.nan for r in grupos[g]])
                     for m in METRICAS_RUIDO] for g in nombres], float)
    for j, m in enumerate(METRICAS_RUIDO):
        if m in LOG:
            med[:, j] = np.log10(np.maximum(med[:, j], 1e-6))
    # escala robusta: dispersion tipica DENTRO de los grupos (IQR mediano), para que la distancia
    # se lea como "cuantas veces la variacion normal entre frames de un mismo grupo"
    escala = []
    for j, m in enumerate(METRICAS_RUIDO):
        iqr = []
        for g in nombres:
            v = np.array([r[m] for r in grupos[g] if isinstance(r[m], float) and math.isfinite(r[m])], float)
            if m in LOG:
                v = np.log10(np.maximum(v, 1e-6))
            if v.size > 3:
                iqr.append(np.subtract(*np.percentile(v, [75, 25])))
        escala.append(max(np.median(iqr) if iqr else 1.0, 1e-3))
    z = med / np.array(escala)
    n = len(nombres)
    d = np.sqrt(np.nansum((z[:, None, :] - z[None, :, :]) ** 2, axis=2) / len(METRICAS_RUIDO))
    with open(sal / "distancias.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grupo"] + nombres)
        for i, g in enumerate(nombres):
            w.writerow([g] + [round(float(x), 3) for x in d[i]])
    # agrupamiento jerarquico (enlace promedio), sin scipy
    clusters = {i: [i] for i in range(n)}
    pasos = []
    while len(clusters) > 1:
        claves = list(clusters)
        mejor = None
        for a in range(len(claves)):
            for b in range(a + 1, len(claves)):
                ca, cb = clusters[claves[a]], clusters[claves[b]]
                dd = float(np.mean([d[i, j] for i in ca for j in cb]))
                if mejor is None or dd < mejor[0]:
                    mejor = (dd, claves[a], claves[b])
        dd, ka, kb = mejor
        pasos.append((dd, [nombres[i] for i in clusters[ka]], [nombres[i] for i in clusters[kb]]))
        clusters[ka] = clusters[ka] + clusters.pop(kb)
    orden = clusters[list(clusters)[0]]
    with open(sal / "agrupamiento.txt", "w", encoding="utf-8") as f:
        f.write("Union de grupos de menor a mayor distancia (enlace promedio).\n"
                "Distancia ~1 = se diferencian tanto como dos frames normales del mismo grupo;\n"
                ">3 = ruido claramente distinto.\n\n")
        for dd, a, b in pasos:
            f.write(f"{dd:6.2f}   {' + '.join(a)}   <->   {' + '.join(b)}\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dd = d[np.ix_(orden, orden)]
        fig, ax = plt.subplots(figsize=(1 + 0.75 * n, 0.8 + 0.65 * n))
        im = ax.imshow(dd, cmap="viridis")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([nombres[i] for i in orden], rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels([nombres[i] for i in orden], fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{dd[i, j]:.1f}", ha="center", va="center", fontsize=7, color="w")
        fig.colorbar(im, ax=ax, fraction=0.04)
        ax.set_title("Distancia entre grupos (metricas de ruido, estandarizadas)")
        fig.tight_layout()
        fig.savefig(sal / "distancias.png", dpi=120)
        plt.close(fig)
    except ImportError:
        pass


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", default=["video1", "video2", "video3"])
    ap.add_argument("--regimenes", default="videos/{v}/regimenes_{v}.csv", help="patron con {v}")
    ap.add_argument("--maximo-frame", nargs="*", default=["video3=23315"])
    ap.add_argument("--por-grupo", type=int, default=60, help="frames por video|regimen")
    ap.add_argument("--margen-hud", type=int, default=12, help="px alrededor del HUD que no se miden")
    ap.add_argument("--sigma-suave", type=float, default=1.5, help="suavizado para separar residuo/escena")
    ap.add_argument("--min-n-distancia", type=int, default=10, help="grupos con menos frames no entran a la distancia")
    ap.add_argument("--salida", default="resultados/ruido_regimenes")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()

    from fac_denoise.datos import Fuente  # noqa: E402  (localiza frames_originales y mascaras_hud)
    rng = np.random.default_rng(a.semilla)
    maxf = {x.split("=")[0]: int(x.split("=")[1]) for x in a.maximo_frame}
    tareas = []
    for v in a.videos:
        ruta = Path(a.regimenes.format(v=v))
        if not ruta.exists():
            print(f"!! falta {ruta}")
            continue
        f = Fuente(v)
        print("  " + f.resumen())
        sel = elegir_frames(leer_csv(ruta), a.por_grupo, maxf.get(v, 0), rng, set(f.idx_orig))
        for reg, filas in sorted(sel.items()):
            n_ok = 0
            for r in filas:
                k = int(r["frame"])
                if k not in f.idx_orig:
                    continue
                n_ok += 1
                tareas.append({"ruta": f.idx_orig[k], "mascara": f.idx_mask.get(k), "margen": a.margen_hud,
                               "sigma_suave": a.sigma_suave,
                               "fila": {"grupo": f"{v}|{reg}", "video": v, "regimen": reg, "frame": k,
                                        "foco": r.get("foco", ""), "exposicion": r.get("exposicion", "")}})
            print(f"    {v} {reg:18s} {n_ok} frames")
    print(f"{len(tareas)} frames a medir")
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    filas, espectros, recortes = [], defaultdict(list), defaultdict(list)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(medir, tareas, chunksize=2), 1):
            if res is None:
                continue
            fila, esp, rec = res
            filas.append(fila)
            if esp is not None:
                espectros[fila["grupo"]].append(esp)
            if rec is not None:
                recortes[fila["grupo"]].append(rec)
            if i % 50 == 0:
                print(f"  {i}/{len(tareas)}", flush=True)
    cols = ["grupo", "video", "regimen", "frame", "foco", "exposicion"] + METRICAS
    with open(sal / "metricas_por_frame.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    grupos = escribir_resumen(filas, lambda r: r["grupo"], sal / "resumen.csv")
    escribir_resumen(filas, lambda r: f"{r['grupo']}|{r['exposicion']}", sal / "resumen_exposicion.csv")
    try:
        graficar(grupos, sal / "graficas")
    except ImportError:
        print("!! sin matplotlib: no hay graficas")
    hoja_espectros(espectros, sal / "espectros.png")
    hoja_recortes(recortes, sal / "recortes.jpg")
    distancias(grupos, sal, a.min_n_distancia)
    (sal / "parametros.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")
    print()
    print(f"{'grupo':30s} {'n':>4s} " + " ".join(f"{m[:12]:>12s}" for m in METRICAS_RUIDO))
    for g in sorted(grupos):
        med = [np.nanmedian([r[m] if isinstance(r[m], float) else np.nan for r in grupos[g]]) for m in METRICAS_RUIDO]
        print(f"{g:30s} {len(grupos[g]):4d} " + " ".join(f"{x:12.3f}" for x in med))
    print(f"\nListo: {sal}")


if __name__ == "__main__":
    main()
