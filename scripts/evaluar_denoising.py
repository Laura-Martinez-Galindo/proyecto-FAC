#!/usr/bin/env python3
"""
Evalua experimentos de denoising SIN imagen limpia de referencia, imagen por imagen.

Para cada imagen de --entrada (las ruidosas, p.ej. dataset_sin_hud/images) y cada experimento
(carpeta con las mismas imagenes limpias, mismo nombre) calcula:

  sigma_ruido          ruido que QUEDA en la salida (DN, 0-255). Metodo de Immerkaer (1996)
                       excluyendo bordes. Menor = mas limpio... pero ver retencion_bordes.
  grano                desviacion del detalle fino (gris - gaussiano sigma 2) fuera de los bordes:
                       el grano que SE VE (grumos de 2-4 px). sigma_ruido (Immerkaer) solo ve ruido
                       de 1 px y no ve los grumos que dejan B2U/NB2NB; por eso el ranking ordena por
                       grano. Incluye algo de textura real: sirve para comparar metodos.
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
  croma_ratio          color de la salida / color de la entrada (croma medio en Lab, sin HUD).
                       1 = conserva el color; <0.85 la red lo esta lavando (ronda 1: UDVD 0,24-0,57).
  delta_e_suave        diferencia de color/brillo (CIE76) entre entrada y salida SUAVIZADAS
                       (gaussiano sigma 4): mide si cambia el tono de la escena, no el ruido. <2-3 ok.
  bloques_8px          artefacto de bloques del H.264 en la salida: salto de gris en las fronteras
                       de 8 px / salto en el resto. 1 = sin rejilla; >1 = se ven bloques.
  corr_residuo_temporal  ruido vs textura usando el frame vecino (k = --vecino-temporal):
                       se alinea el frame k+d (flujo optico DIS) y se correlaciona lo que el metodo
                       quito (entrada - salida) con lo que el vecino tiene y la salida no
                       (vecino - salida). El ruido cambia de un frame a otro y la textura real no:
                       ~0 = quito ruido; alto = quito textura que tambien esta en el vecino.
                       OJO: el H.264 copia parte del ruido entre frames, asi que nunca es 0 exacto;
                       sirve para COMPARAR metodos, no como umbral absoluto.
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

METRICAS = ["sigma_ruido", "grano", "ruido_quitado", "retencion_bordes", "corr_residuo_bordes", "autocorr_residuo",
            "croma_ratio", "delta_e_suave", "bloques_8px", "corr_residuo_temporal"]
MAYOR_ES_MEJOR = {"retencion_bordes": True, "musiq": True, "clipiqa": True, "upsnr": True}
# metricas sin referencia (pyiqa): menor = mejor salvo las de MAYOR_ES_MEJOR
IQA_MENOR = ("niqe", "brisque")
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


def bloques(g: np.ndarray, b: int = 8) -> float:
    """Salto medio en las fronteras de bloque (x % 8 == 7 -> x+1) / salto medio en el resto."""
    dx, dy = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    fx = (np.arange(dx.shape[1]) % b) == b - 1
    fy = (np.arange(dy.shape[0]) % b) == b - 1
    en = dx[:, fx].mean() + dy[fy].mean()
    fuera = dx[:, ~fx].mean() + dy[~fy].mean()
    return float(en / (fuera + 1e-6))


_FUENTES: dict = {}
_DIS = None


def vecino_alineado(video: str, frame: int, d: int, ref_bgr: np.ndarray):
    """Frame (frame + d) del video original, alineado al frame de referencia con flujo DIS.
    Devuelve (vecino_gris_alineado, validos) o None si no hay vecino."""
    global _DIS
    from fac_denoise.datos import Fuente  # import aqui: solo hace falta para esta metrica
    if video not in _FUENTES:
        _FUENTES[video] = Fuente(video)
    f = _FUENTES[video]
    p = f.idx_orig.get(frame + d)
    if p is None:
        return None
    nb = cv2.imread(p, cv2.IMREAD_COLOR)
    if nb is None:
        return None
    h, w = ref_bgr.shape[:2]
    if nb.shape[:2] != (h, w):
        nb = cv2.resize(nb, (w, h), interpolation=cv2.INTER_AREA)
    g0 = cv2.GaussianBlur(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    g1 = cv2.GaussianBlur(cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flujo = _DIS.calc(g0, g1, None)
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    nb_al = cv2.remap(gris(nb), mx + flujo[..., 0], my + flujo[..., 1], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    validos = nb_al >= 0
    # HUD del frame de referencia y del vecino (el vecino tiene HUD: se excluye)
    for k in (frame, frame + d):
        m = f.mascara(k, (h, w), margen=6)
        if m is not None:
            validos &= ~m
    # oclusiones / fallas del flujo: donde ni suavizado coinciden
    dif = np.abs(cv2.GaussianBlur(nb_al, (0, 0), 2) - cv2.GaussianBlur(gris(ref_bgr), (0, 0), 2))
    validos &= dif < 12
    validos[:16], validos[-16:], validos[:, :16], validos[:, -16:] = False, False, False, False
    return nb_al, validos


def corr_temporal(entrada, salida, vecino, excluir) -> float:
    nb_al, val = vecino
    gi, go = gris(entrada), gris(salida)
    m = val & ~excluir
    if m.sum() < 5000:
        return float("nan")
    r = (gi - go)[m]
    v = (nb_al - go)[m]
    if r.std() < 1e-6 or v.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(r, v)[0, 1])


def metricas_par(entrada: np.ndarray, salida: np.ndarray, frac_bordes: float, sigma_suave: float,
                 mascara_hud: np.ndarray | None = None) -> dict:
    if salida.shape != entrada.shape:
        salida = cv2.resize(salida, (entrada.shape[1], entrada.shape[0]), interpolation=cv2.INTER_AREA)
    gi, go = gris(entrada), gris(salida)
    # bordes de la escena sobre versiones SIN ruido: mediana 5x5 (quita impulsos y grano) + gaussiano.
    # Asi el grano del video2 no cuenta como "borde" y quitarlo no se castiga.
    si = cv2.GaussianBlur(cv2.medianBlur(np.clip(gi, 0, 255).astype(np.uint8), 5).astype(np.float32), (0, 0), sigma_suave)
    so = cv2.GaussianBlur(cv2.medianBlur(np.clip(go, 0, 255).astype(np.uint8), 5).astype(np.float32), (0, 0), sigma_suave)
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
    # color
    li = cv2.cvtColor(entrada.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    lo = cv2.cvtColor(salida.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    ci = np.hypot(li[..., 1], li[..., 2])[validos].mean()
    co = np.hypot(lo[..., 1], lo[..., 2])[validos].mean()
    lis, los_ = cv2.GaussianBlur(li, (0, 0), 4), cv2.GaussianBlur(lo, (0, 0), 4)
    de = np.sqrt(((lis - los_) ** 2).sum(axis=2))[validos].mean()
    return {
        "grano": float((go - cv2.GaussianBlur(go, (0, 0), 2.0))[~excl].std()),
        "croma_ratio": float(co / (ci + 1e-6)),
        "delta_e_suave": float(de),
        "bloques_8px": bloques(go),
        "sigma_ruido": sigma_immerkaer(go, excl),
        "ruido_quitado": float(res_v.std()),
        "retencion_bordes": float(np.median(mag_o[b]) / (np.median(mag_i[b]) + 1e-6)) if b.any() else float("nan"),
        "corr_residuo_bordes": corr,
        "autocorr_residuo": ac if res_v.std() > 1e-6 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Metricas temporales (con frames vecinos, ver scripts/preparar_vecinos.py)
def _alinear_dis(ref, nb):
    """Flujo DIS de ref (BGR) a nb sobre versiones suavizadas; devuelve funcion de alineado y validos."""
    global _DIS
    g0 = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    g1 = cv2.GaussianBlur(cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fl = _DIS.calc(g0, g1, None)
    h, w = g0.shape
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = mx + fl[..., 0], my + fl[..., 1]
    validos = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    al = lambda im: cv2.remap(gris(im), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dif = np.abs(cv2.GaussianBlur(al(nb), (0, 0), 2) - cv2.GaussianBlur(gris(ref), (0, 0), 2))
    validos &= dif < 12  # oclusiones / fallas del flujo
    validos[:16], validos[-16:], validos[:, :16], validos[:, -16:] = False, False, False, False
    return al, validos


def _tarea_temporal(t):
    exp, nombre, p_in, p_out, dir_vec_in, dir_vec_out, video, frame = t
    a, b = cv2.imread(p_in, cv2.IMREAD_COLOR), cv2.imread(p_out, cv2.IMREAD_COLOR)
    leer = lambda d, k: cv2.imread(str(Path(d) / f"{video}_{k:06d}.png"), cv2.IMREAD_COLOR)
    ym1, yp1, yp2 = leer(dir_vec_in, frame - 1), leer(dir_vec_in, frame + 1), leer(dir_vec_in, frame + 2)
    om1, op1 = leer(dir_vec_out, frame - 1), leer(dir_vec_out, frame + 1)
    r = {"experimento": exp, "imagen": nombre, "warping_error": float("nan"), "upsnr": float("nan")}
    if a is None or b is None or ym1 is None or yp1 is None:
        return r
    go = gris(b)
    al_m1, v_m1 = _alinear_dis(a, ym1)
    al_p1, v_p1 = _alinear_dis(a, yp1)
    if om1 is not None and op1 is not None:
        e1 = np.abs(go - al_m1(om1))[v_m1]
        e2 = np.abs(go - al_p1(op1))[v_p1]
        if e1.size > 1000 and e2.size > 1000:
            r["warping_error"] = float(0.5 * (e1.mean() + e2.mean()))
    if yp2 is not None:
        al_p2, v_p2 = _alinear_dis(a, yp2)
        v = v_m1 & v_p1 & v_p2
        if v.sum() > 1000:
            umse = float(np.mean((go - al_p1(yp1))[v] ** 2) - 0.5 * np.mean((al_m1(ym1) - al_p2(yp2))[v] ** 2))
            r["upsnr"] = float(10 * math.log10(255.0 ** 2 / umse)) if umse > 0 else float("nan")
    return r


# ---------------------------------------------------------------------------
# Trabajo en paralelo
# ---------------------------------------------------------------------------

def _tarea(t):
    exp, nombre, p_in, p_out, p_mask, frac, sig, video, frame, d = t
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
    r["corr_residuo_temporal"] = float("nan")
    if d and video and frame is not None and exp != "sin_denoising":
        try:
            vec = vecino_alineado(video, frame, d, a)
            if vec is not None:
                g = cv2.GaussianBlur(cv2.medianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), 5).astype(np.float32), (0, 0), sig)
                bordes, _ = mascara_bordes(g, frac)
                r["corr_residuo_temporal"] = corr_temporal(a, b, vec, bordes | (mk if mk is not None else False))
        except Exception as e:  # la metrica temporal nunca tumba la evaluacion
            print(f"!! temporal {nombre}: {e}", flush=True)
    r.update(experimento=exp, imagen=nombre)
    return r


def iqa(rutas: list[str], nombres: list[str], metricas: list[str], lote: int) -> dict:
    try:
        import sys
        try:
            import pkg_resources
        except ModuleNotFoundError:
            try:
                import packaging, types
                _p = types.ModuleType("pkg_resources")
                _p.packaging = packaging
                sys.modules["pkg_resources"] = _p
            except Exception:
                pass
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


def ranking(resumen, min_ret, max_corr, min_croma=0.85, max_croma=1.15, max_de=3.0):
    """Descarta solo por COLOR (croma_ratio, delta_e_suave): eso si se puede medir sin referencia.
    retencion_bordes y corr_residuo_bordes quedan como AVISOS: en la ronda 1 confundian grano y
    bloques con bordes y descartaban hasta lo que se veia bien.
    Orden: si hay NIQE y BRISQUE (objetivo de la propuesta), por el % de reduccion promedio de
    las dos frente a sin_denoising (mayor = mejor); si no, por grano p50."""
    out = []
    iqa_pres = [m for m in ("niqe", "brisque", "musiq", "clipiqa") if any(f"{m}_media" in r for r in resumen)]
    cols = [f"{m}_cambio_pct" for m in iqa_pres] + [f"{m}_p50" for m in iqa_pres] + ["grano_p50", "sigma_ruido_p50", "ruido_quitado_p50", "retencion_bordes_p25", "corr_residuo_bordes_p50",
            "croma_ratio_p50", "delta_e_suave_p50", "bloques_8px_p50", "corr_residuo_temporal_p50",
            "warping_error_p50", "upsnr_p50"]
    for reg in sorted({r["regimen"] for r in resumen}):
        filas = [r for r in resumen if r["regimen"] == reg and r["experimento"] != "sin_denoising"]
        base = next((r for r in resumen if r["regimen"] == reg and r["experimento"] == "sin_denoising"), {})
        for r in filas:  # % de cambio de la media frente al original (positivo = mejora)
            for m in iqa_pres:
                b0, v = base.get(f"{m}_media"), r.get(f"{m}_media")
                if b0 and v is not None:
                    d = 100.0 * (b0 - v) / abs(b0)
                    r[f"{m}_cambio_pct"] = d if m in IQA_MENOR else -d
        for r in filas:
            desc, avisos = [], []
            c = r.get("croma_ratio_p50", 1)
            if not (min_croma <= c <= max_croma):
                desc.append(f"croma_ratio p50={c:.2f} fuera de [{min_croma}, {max_croma}]")
            if r.get("delta_e_suave_p50", 0) > max_de:
                desc.append(f"delta_e_suave p50={r['delta_e_suave_p50']:.2f} > {max_de}")
            if r.get("retencion_bordes_p25", 1) < min_ret:
                avisos.append(f"retencion_bordes p25={r['retencion_bordes_p25']:.2f} < {min_ret}")
            if r.get("corr_residuo_bordes_p50", 0) > max_corr:
                avisos.append(f"corr_residuo_bordes p50={r['corr_residuo_bordes_p50']:.2f} > {max_corr}")
            r["_desc"], r["_avisos"] = "; ".join(desc), "; ".join(avisos)
        if "niqe" in iqa_pres and "brisque" in iqa_pres:
            clave = lambda r: -0.5 * (r.get("niqe_cambio_pct", -1e9) + r.get("brisque_cambio_pct", -1e9))
        else:
            clave = lambda r: r.get("grano_p50", 9e9)
        ok = sorted([r for r in filas if not r["_desc"]], key=clave)
        for pos, r in enumerate(ok, 1):
            meta = ""
            if "niqe_cambio_pct" in r and "brisque_cambio_pct" in r:
                meta = " | meta 8%: " + ("SI" if min(r["niqe_cambio_pct"], r["brisque_cambio_pct"]) >= 8 else "no")
            out.append({"regimen": reg, "posicion": pos, "experimento": r["experimento"], "n": r["n"],
                        **{k: round(r.get(k, float("nan")), 3) for k in cols},
                        "estado": "OK" + meta + (" (aviso: " + r["_avisos"] + ")" if r["_avisos"] else "")})
        for r in [r for r in filas if r["_desc"]]:
            out.append({"regimen": reg, "posicion": "", "experimento": r["experimento"], "n": r["n"],
                        **{k: round(r.get(k, float("nan")), 3) for k in cols},
                        "estado": "DESCARTADO: " + r["_desc"]})
    return out


def ranking_viejo(resumen, min_ret, max_corr):
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
    ap.add_argument("--lista", help="CSV con columna imagen: evaluar solo esas (p.ej. <FAMILIA>_evaluacion.csv)")
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
    ap.add_argument("--vecino-temporal", type=int, default=2,
                    help="d del frame vecino (k+d) para corr_residuo_temporal; 0 = no calcularla")
    ap.add_argument("--vecinos-entrada", default="", help="carpeta con los frames vecinos ruidosos (preparar_vecinos.py); "
                    "la salida del metodo en los vecinos se busca en <carpeta del experimento>/../vecinos")
    ap.add_argument("--min-croma", type=float, default=0.85, help="Filtro: croma_ratio p50 minimo")
    ap.add_argument("--max-croma", type=float, default=1.15, help="Filtro: croma_ratio p50 maximo")
    ap.add_argument("--max-delta-e", type=float, default=3.0, help="Filtro: delta_e_suave p50 maximo")
    a = ap.parse_args()

    ent = Path(a.entrada)
    nombres = sorted(p.name for p in ent.iterdir() if p.suffix.lower() in EXT)
    regimen = {}
    vf = {}  # stem -> (video, frame), de --lista, para la metrica temporal
    if a.regimenes:
        with open(a.regimenes, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                regimen[Path(r["imagen"]).stem] = r.get("regimen", "SIN_REGIMEN")
    if a.lista:
        with open(a.lista, encoding="utf-8-sig", newline="") as f:
            filas_l = list(csv.DictReader(f))
        quiero = {Path(r["imagen"]).stem for r in filas_l}
        for r in filas_l:
            if r.get("video") and r.get("frame"):
                vf[Path(r["imagen"]).stem] = (r["video"], int(float(r["frame"])))
        for r in filas_l:
            if r.get("regimen"):
                regimen.setdefault(Path(r["imagen"]).stem, r["regimen"])
        nombres = [n for n in nombres if Path(n).stem in quiero]
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
            v, k = vf.get(Path(n).stem, (None, None))
            tareas.append((exp, n, str(ent / n), str(p_out), p_mask, a.frac_bordes, a.sigma_suave, v, k, a.vecino_temporal))
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
    if a.vecinos_entrada and vf:
        cand = [n for n in nombres if Path(n).stem in vf
                and (Path(a.vecinos_entrada) / f"{vf[Path(n).stem][0]}_{vf[Path(n).stem][1] - 1:06d}.png").exists()]
        tt = []
        for exp, carpeta in exps:
            d_out = Path(a.vecinos_entrada) if exp == "sin_denoising" else Path(carpeta).parent / "vecinos"
            if not d_out.is_dir():
                continue
            for n in cand:
                p_out = Path(carpeta) / n
                if p_out.exists():
                    v, k = vf[Path(n).stem]
                    tt.append((exp, n, str(ent / n), str(p_out), a.vecinos_entrada, str(d_out), v, k))
        if tt:
            print(f"metricas temporales: {len(tt)} pares (imagen, experimento) con vecinos")
            idx = {(r["experimento"], r["imagen"]): r for r in filas}
            with ProcessPoolExecutor(a.workers) as ex:
                for rt in ex.map(_tarea_temporal, tt, chunksize=2):
                    if (rt["experimento"], rt["imagen"]) in idx:
                        idx[(rt["experimento"], rt["imagen"])].update(warping_error=rt["warping_error"], upsnr=rt["upsnr"])
            metricas += ["warping_error", "upsnr"]
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
    rk = ranking(res, a.min_retencion, a.max_corr, a.min_croma, a.max_croma, a.max_delta_e)
    if rk:
        with open(sal / "ranking.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rk[0].keys()))
            w.writeheader()
            w.writerows(rk)
    graficar(filas, metricas, sal / "graficas")
    print(f"Listo: {sal}")
    for r in rk:
        iq = "".join(f"{k.split('_')[0]}={r[k]:+.1f}%  " for k in r if k.endswith("_cambio_pct") and r[k] == r[k])
        print(f"  {r['regimen']:15s} {str(r['posicion']):>3s} {r['experimento']:24s} {iq}grano={r['grano_p50']}  "
              f"croma={r['croma_ratio_p50']}  dE={r['delta_e_suave_p50']}  temporal={r['corr_residuo_temporal_p50']}  "
              f"ret_p25={r['retencion_bordes_p25']}  {r['estado']}")


if __name__ == "__main__":
    main()
