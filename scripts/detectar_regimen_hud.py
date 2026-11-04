#!/usr/bin/env python3
"""
Detecta el REGIMEN de cada frame leyendo las banderas del HUD (esquina inferior izquierda):

    linea 1  sensor     HDEO (optico) | HDIR (termico) | HDLL (baja luz)
    linea 2  filtros    DDE | HAZE | DDE HAZE | C WH | C BH | (vacia)
    linea 3  foco       FOC MAN | FOC AUT
    linea 4  exposicion EXP MAN | EXP AUT

No usa OCR ni heuristicas de color de la escena: aisla el texto verde del HUD, lleva la zona a
escala 720p, busca la posicion vertical de las 4 lineas (separadas ~19 px) y compara cada linea
con plantillas sacadas de frames de ejemplo. Si una linea no se parece a ninguna plantilla
(puntaje < --umbral) se marca DESCONOCIDO y se guarda el recorte: asi aparecen regimenes nuevos.

Como el operador cambia de modo pocas veces, en modo carpeta (frames consecutivos) los frames
ilegibles (texto verde sobre vegetacion, menus encima, etc.) toman el valor del frame legible mas
cercano y se reportan como 'relleno'.

Subcomandos:
  plantillas  crea config/plantillas_hud.npz a partir de un CSV de ejemplos
              (columnas: ruta, sensor, filtros, foco, exposicion)
  carpeta     recorre una carpeta de frames (p.ej. videos/video2/frames_originales) cada --paso
              frames y escribe <salida>.csv con un regimen por frame + intervalos y resumen
  dataset     etiqueta las imagenes de un dataset (usa correspondencia.csv) y agrega la columna
              'regimen'

Regimen = sensor + filtros (p.ej. HDEO_DDE_HAZE, HDIR_C_WH, HDLL_DDE). Foco y exposicion se guardan
aparte (EXP AUT cambia la ganancia y por tanto el nivel de ruido, pero no el tipo).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
PLANTILLAS = RAIZ / "config" / "plantillas_hud.npz"
LINEAS = ["sensor", "filtros", "foco", "exposicion"]
ALTO_LINEA, PASO_LINEA = 13, 19.3        # en escala 720p
ANCHO_TEXTO = 140                        # px (720p) desde el borde izquierdo: solo el texto
PATRON = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")


# ---------------------------------------------------------------------------
# Extraccion
# ---------------------------------------------------------------------------

def mascara_texto(img_bgr: np.ndarray) -> np.ndarray:
    """Texto verde del HUD en la zona inferior izquierda, reescalado a 720p. Devuelve float 0/1."""
    h, w = img_bgr.shape[:2]
    roi = img_bgr[int(h * 0.62):int(h * 0.95), 0:int(w * 0.13)]
    esc = 720.0 / h
    roi = cv2.resize(roi, (max(1, int(roi.shape[1] * esc)), max(1, int(roi.shape[0] * esc))),
                     interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (38, 110, 110), (88, 255, 255))
    return (m[:, :ANCHO_TEXTO] > 0).astype(np.float32)


def recorte_linea(m: np.ndarray, y0: float, i: int) -> np.ndarray:
    y = int(round(y0 + i * PASO_LINEA))
    out = np.zeros((ALTO_LINEA, ANCHO_TEXTO), np.float32)
    seg = m[max(0, y):y + ALTO_LINEA]
    out[:seg.shape[0], :seg.shape[1]] = seg
    return out


def similitud(a: np.ndarray, b: np.ndarray, peso: np.ndarray | None = None, dx: int = 4) -> float:
    """Correlacion normalizada (ponderada) maxima con desplazamientos horizontales pequenos.
    El peso da mas importancia a las columnas donde las plantillas de esa linea se diferencian
    (p.ej. el final de HDEO vs HDLL), para no decidir por el prefijo comun."""
    best = -1.0
    w = np.ones_like(a) if peso is None else peso
    for s in range(-dx, dx + 1):
        bb = np.roll(b, s, axis=1)
        ww = np.roll(w, s, axis=1)
        num = float((ww * a * bb).sum())
        den = float(np.sqrt((ww * a * a).sum() * (ww * bb * bb).sum())) + 1e-6
        best = max(best, num / den)
    return best


def pesos_discriminativos(plantillas) -> np.ndarray:
    """1 + (varianza entre plantillas por pixel, suavizada): realza lo que distingue a las palabras."""
    pilas = np.stack([t for _, t in plantillas])
    v = cv2.GaussianBlur(pilas.std(0).astype(np.float32), (0, 0), 1.0)
    return 1.0 + 4.0 * v / (v.max() + 1e-6)


Y0_MIN, Y0_MAX = 120.0, 142.0  # posicion de la 1a linea en la zona (escala 720p), medida en los 3 videos


def ubicar_lineas(m: np.ndarray) -> float:
    """y0 que maximiza texto dentro de las 4 bandas y minimiza texto entre ellas."""
    mejor = (-1e9, (Y0_MIN + Y0_MAX) / 2)
    for y0 in np.arange(Y0_MIN, Y0_MAX + 0.01, 0.5):
        dentro = sum(recorte_linea(m, y0, i).sum() for i in range(4))
        fuera = sum(m[int(y0 + i * PASO_LINEA + ALTO_LINEA):int(y0 + (i + 1) * PASO_LINEA)].sum() for i in range(3))
        s = dentro - 3 * fuera
        if s > mejor[0]:
            mejor = (s, y0)
    return mejor[1]


MARGEN = 0.04  # el mejor candidato debe superar al segundo por al menos esto


def leer(img_bgr: np.ndarray, plant: dict, umbral: float):
    m = mascara_texto(img_bgr)
    yc = ubicar_lineas(m)
    pesos = plant.get("_pesos") or {n: pesos_discriminativos(plant[n]) for n in LINEAS}
    plant["_pesos"] = pesos
    mejor = None
    for y0 in (yc - 1, yc - 0.5, yc, yc + 0.5, yc + 1):
        lect, puntos, margenes = [], [], []
        for i, nombre in enumerate(LINEAS):
            rec = recorte_linea(m, y0, i)
            if rec.sum() < 5:
                lect.append("")
                puntos.append(0.0)
                margenes.append(0.0)
                continue
            cands = sorted(((similitud(rec, t, pesos[nombre]), et) for et, t in plant[nombre]), reverse=True)
            s, et = cands[0]
            lect.append(et)
            puntos.append(s)
            margenes.append(s - cands[1][0] if len(cands) > 1 else 1.0)
        tot = float(np.mean(puntos))
        if mejor is None or tot > mejor[0]:
            mejor = (tot, y0, lect, puntos, margenes)
    tot, y0, lect, puntos, margenes = mejor
    lect = [et if (p >= umbral and mg >= MARGEN) else "DESCONOCIDO"
            for et, p, mg in zip(lect, puntos, margenes)]
    return lect, puntos, y0, m


def regimen_de(lect) -> str:
    if "DESCONOCIDO" in lect[:2]:
        return "DESCONOCIDO"
    partes = [lect[0]] + ([lect[1].replace(" ", "_")] if lect[1] else [])
    return "_".join(partes)


# ---------------------------------------------------------------------------
# Plantillas
# ---------------------------------------------------------------------------

def cargar_plantillas(ruta: Path) -> dict:
    if not ruta.exists():
        sys.exit(f"No existe {ruta}. Crea las plantillas con el subcomando 'plantillas'.")
    z = np.load(ruta, allow_pickle=True)
    pl = {k: list(z[k]) for k in LINEAS}
    pl["_pesos"] = {k: pesos_discriminativos(pl[k]) for k in LINEAS}
    return pl


def cmd_plantillas(a):
    plant = {k: [] for k in LINEAS}
    with open(a.ejemplos, encoding="utf-8-sig", newline="") as f:
        ejemplos = list(csv.DictReader(f))
    for e in ejemplos:
        img = cv2.imread(e["ruta"])
        m = mascara_texto(img)
        y0 = ubicar_lineas(m)
        for i, nombre in enumerate(LINEAS):
            et = e[nombre].strip()
            rec = recorte_linea(m, y0, i)
            plant[nombre].append((et, rec))
        print(f"  {Path(e['ruta']).name}: y0={y0:.0f}  " + " | ".join(e[k] for k in LINEAS))
    # una sola plantilla por etiqueta: promedio de los ejemplos
    final = {}
    for nombre, lst in plant.items():
        agr = {}
        for et, rec in lst:
            agr.setdefault(et, []).append(rec)
        final[nombre] = np.array([(et, np.mean(r, 0)) for et, r in agr.items()], dtype=object)
        print(f"{nombre}: {sorted(agr)}")
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.salida, **final)
    print(f"Plantillas: {a.salida}")


# ---------------------------------------------------------------------------
# Lectura de muchos frames
# ---------------------------------------------------------------------------

_PL = None
_UM = 0.6


def _init(ruta_pl, umbral):
    global _PL, _UM
    _PL, _UM = cargar_plantillas(Path(ruta_pl)), umbral


def _leer_uno(args):
    ruta, clave = args
    img = cv2.imread(ruta)
    if img is None:
        return clave, None
    lect, puntos, y0, _ = leer(img, _PL, _UM)
    return clave, (lect, [round(p, 3) for p in puntos])


def guardar_dudosos(filas, carpeta: Path, maximo=60):
    carpeta.mkdir(parents=True, exist_ok=True)
    for r in filas[:maximo]:
        img = cv2.imread(r["ruta"])
        h, w = img.shape[:2]
        roi = img[int(h * 0.62):int(h * 0.95), 0:int(w * 0.13)]
        cv2.imwrite(str(carpeta / f"{Path(r['ruta']).stem}.png"), roi)


def intervalos(filas):
    out, ini, prev = [], None, None
    for r in filas:
        if prev is None or r["regimen"] != prev["regimen"]:
            if prev is not None:
                out.append((prev["regimen"], ini, prev["frame"]))
            ini = r["frame"]
        prev = r
    if prev is not None:
        out.append((prev["regimen"], ini, prev["frame"]))
    return out


def cmd_carpeta(a):
    carpeta = Path(a.carpeta)
    idx = {}
    for e in os.scandir(carpeta):
        m = PATRON.search(e.name)
        if e.is_file() and m:
            idx[int(m.group(1))] = e.path
    nums = sorted(idx)
    if a.maximo_frame:
        nums = [k for k in nums if k <= a.maximo_frame]
    nums = nums[::a.paso]
    print(f"{carpeta}: leyendo {len(nums)} frames (cada {a.paso})")
    res = {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.plantillas, a.umbral)) as ex:
        for k, r in ex.map(_leer_uno, [(idx[k], k) for k in nums], chunksize=16):
            res[k] = r
    filas = []
    for k in nums:
        r = res.get(k)
        lect, pts = (r if r else (["DESCONOCIDO"] * 4, [0, 0, 0, 0]))
        filas.append(dict(video=a.video, frame=k, ruta=idx[k], sensor=lect[0], filtros=lect[1], foco=lect[2],
                          exposicion=lect[3], p_sensor=pts[0], p_filtros=pts[1], p_foco=pts[2],
                          p_exposicion=pts[3], regimen=regimen_de(lect), origen="leido"))
    # relleno temporal de ilegibles con el legible mas cercano (el modo cambia poco)
    legibles = [i for i, f in enumerate(filas) if f["regimen"] != "DESCONOCIDO"]
    dudosos = [dict(f) for f in filas if f["regimen"] == "DESCONOCIDO"]
    if legibles:
        li = np.array(legibles)
        for i, f in enumerate(filas):
            if f["regimen"] == "DESCONOCIDO":
                j = li[np.argmin(np.abs(li - i))]
                if abs(filas[j]["frame"] - f["frame"]) <= a.max_relleno:
                    for c in ("sensor", "filtros", "foco", "exposicion", "regimen"):
                        f[c] = filas[j][c]
                    f["origen"] = f"relleno_desde_{filas[j]['frame']}"
    sal = Path(a.salida)
    sal.parent.mkdir(parents=True, exist_ok=True)
    cols = ["video", "frame", "regimen", "sensor", "filtros", "foco", "exposicion",
            "p_sensor", "p_filtros", "p_foco", "p_exposicion", "origen"]
    with open(sal, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    inter = intervalos(filas)
    with open(sal.with_name(sal.stem + "_intervalos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["regimen", "frame_inicio", "frame_fin", "frames_aprox"])
        for reg, i0, i1 in inter:
            w.writerow([reg, i0, i1, i1 - i0 + a.paso])
    cnt = Counter(f["regimen"] for f in filas)
    combos = Counter((f["sensor"], f["filtros"], f["foco"], f["exposicion"]) for f in filas)
    resumen = {"frames_leidos": len(filas), "paso": a.paso,
               "regimenes_%": {k: round(100 * v / len(filas), 2) for k, v in cnt.most_common()},
               "combinaciones_completas": {" | ".join(k): v for k, v in combos.most_common()},
               "ilegibles_antes_de_relleno": len(dudosos),
               "sin_resolver": cnt.get("DESCONOCIDO", 0)}
    sal.with_name(sal.stem + "_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    guardar_dudosos(dudosos, sal.with_name(sal.stem + "_dudosos"))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


def cmd_dataset(a):
    ds = Path(a.dataset)
    with open(ds / "correspondencia.csv", encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    tareas = [(str(ds / "images" / r["imagen"]), i) for i, r in enumerate(filas)]
    res = {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.plantillas, a.umbral)) as ex:
        for i, r in ex.map(_leer_uno, tareas, chunksize=16):
            res[i] = r
    tabla = None
    if a.tabla_videos:  # regimen ya calculado sobre los videos completos: se usa para los ilegibles
        tabla = {}
        for t in a.tabla_videos:
            with open(t, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    tabla.setdefault(r["video"], []).append((int(r["frame"]), r))
        for v in tabla:
            tabla[v].sort(key=lambda x: x[0])
    dudosos = []
    for i, r in enumerate(filas):
        lect, pts = res[i] if res[i] else (["DESCONOCIDO"] * 4, [0] * 4)
        reg = regimen_de(lect)
        origen = "leido"
        if reg == "DESCONOCIDO":
            dudosos.append({"ruta": tareas[i][0]})
            if tabla and r["video"] in tabla:
                fr = int(r["frame"])
                lst = tabla[r["video"]]
                j = int(np.argmin([abs(k - fr) for k, _ in lst]))
                ref = lst[j][1]
                lect = [ref["sensor"], ref["filtros"], ref["foco"], ref["exposicion"]]
                reg, origen = ref["regimen"], f"tabla_video_frame_{lst[j][0]}"
        r.update(regimen=reg, sensor=lect[0], filtros=lect[1], foco=lect[2], exposicion=lect[3],
                 p_lectura=round(float(np.mean(pts)), 3), origen_regimen=origen)
    salida = Path(a.salida) if a.salida else ds / "regimenes_dataset.csv"
    cols = ["imagen", "video", "frame", "regimen", "sensor", "filtros", "foco", "exposicion", "p_lectura", "origen_regimen"]
    with open(salida, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    guardar_dudosos(dudosos, salida.with_name(salida.stem + "_dudosos"))
    cnt = Counter((r["video"], r["regimen"]) for r in filas)
    for k, v in sorted(cnt.items()):
        print(f"{k[0]:8s} {k[1]:22s} {v}")
    print(f"ilegibles: {len(dudosos)}  -> {salida}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plantillas")
    p.add_argument("--ejemplos", required=True)
    p.add_argument("--salida", default=str(PLANTILLAS))
    for nombre in ("carpeta", "dataset"):
        q = sub.add_parser(nombre)
        q.add_argument("--plantillas", default=str(PLANTILLAS))
        q.add_argument("--umbral", type=float, default=0.6, help="Similitud minima para aceptar una linea")
        q.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
        q.add_argument("--salida", default=None)
    c = sub.choices["carpeta"]
    c.add_argument("--carpeta", required=True)
    c.add_argument("--video", required=True)
    c.add_argument("--paso", type=int, default=15, help="Leer 1 de cada N frames")
    c.add_argument("--maximo-frame", type=int, default=0, help="Ignorar frames mayores (video3: 23315)")
    c.add_argument("--max-relleno", type=int, default=300, help="Max. distancia en frames para rellenar")
    d = sub.choices["dataset"]
    d.add_argument("--dataset", default="datasets/dataset_original")
    d.add_argument("--tabla-videos", nargs="*", help="CSVs de 'carpeta' para resolver ilegibles")
    a = ap.parse_args()
    {"plantillas": cmd_plantillas, "carpeta": cmd_carpeta, "dataset": cmd_dataset}[a.cmd](a)


if __name__ == "__main__":
    main()
