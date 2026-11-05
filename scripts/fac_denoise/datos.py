"""
Lectura de frames para entrenar e inferir.

Fuente(video): sabe donde estan, para un video, sus carpetas
    frames_originales  (resolucion completa, con HUD)
    mascaras_hud       (pixeles del HUD, normalmente a 960x540)
    frames_sin_hud     (ProPainter, 960x540)
y entrega:
    original(k)   frame k a resolucion completa
    mascara(k)    HUD del frame k a resolucion completa (bool, dilatada)
    compuesto(k)  frame k sin HUD, construido IGUAL que dataset_sin_hud (modo compuesto):
                  original + solo los pixeles del HUD reemplazados por ProPainter reescalado

Entrenamiento: se usan los frames originales, pero los parches se sortean FUERA del HUD
(no hace falta reconstruir el compuesto y la red nunca aprende ni el texto ni el relleno).
Inferencia con vecinos (UDVD): los vecinos son compuestos; si a un vecino le falta algo,
se usa el frame central repetido (nunca un vecino con HUD, porque la red copiaria el texto).
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:  # torch solo hace falta para entrenar (Parches); medir/inferir clasicos funciona sin torch
    import torch
    _Base = torch.utils.data.Dataset
except ImportError:  # pragma: no cover
    torch = None
    _Base = object

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts"))
from construir_dataset_sin_hud import candidatos_carpeta  # noqa: E402

PATRON = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")
EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def indexar(carpeta: Path) -> dict[int, str]:
    idx: dict[int, str] = {}
    if not carpeta or not Path(carpeta).is_dir():
        return idx
    with os.scandir(carpeta) as it:
        for e in it:
            if e.is_file() and os.path.splitext(e.name)[1].lower() in EXT:
                m = PATRON.search(e.name)
                if m:
                    idx[int(m.group(1))] = e.path
    return idx


def leer(ruta: str, canales: int) -> np.ndarray:
    """HxWxC uint8 (RGB si C=3)."""
    if canales == 1:
        im = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        return None if im is None else im[..., None]
    im = cv2.imread(ruta, cv2.IMREAD_COLOR)
    return None if im is None else np.ascontiguousarray(im[..., ::-1])


def a_canales(im_bgr: np.ndarray, canales: int) -> np.ndarray:
    if canales == 1:
        return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)[..., None]
    return np.ascontiguousarray(im_bgr[..., ::-1])


def guardar(ruta: Path, im: np.ndarray):
    """im HxWxC RGB/gris uint8 -> PNG BGR de 3 canales (mismo formato que el dataset)."""
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    cv2.imwrite(str(ruta), np.ascontiguousarray(im[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 3])


def cargar_config() -> dict:
    p = RAIZ / "config" / "videos.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


class Fuente:
    def __init__(self, video: str, dilatar: int = 2, suavizar: float = 1.5, margen_hud: int = 8):
        cfg = cargar_config()
        self.video = video
        self.dilatar, self.suavizar, self.margen_hud = dilatar, suavizar, margen_hud
        self.idx_orig = self._primera(candidatos_carpeta(RAIZ, cfg, video, "frames_originales"))
        self.idx_mask = self._primera(candidatos_carpeta(RAIZ, cfg, video, "mascaras_hud"))
        self.idx_sin = self._primera(candidatos_carpeta(RAIZ, cfg, video, "frames_sin_hud"))

    @staticmethod
    def _primera(cands) -> dict[int, str]:
        mejor = {}
        for c in cands:
            idx = indexar(c)
            if len(idx) > len(mejor):
                mejor = idx
        return mejor

    def resumen(self) -> str:
        return (f"{self.video}: originales={len(self.idx_orig)} mascaras={len(self.idx_mask)} "
                f"sin_hud={len(self.idx_sin)}")

    def original(self, k: int, canales: int):
        p = self.idx_orig.get(k)
        return None if p is None else leer(p, canales)

    def mascara(self, k: int, hw, margen: int | None = None):
        """bool HxW. margen = dilatacion extra (para sortear parches lejos del HUD)."""
        p = self.idx_mask.get(k)
        if p is None:
            return None
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        h, w = hw
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
        r = self.dilatar + (margen or 0)
        if r > 0:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        return m.astype(bool)

    def compuesto(self, k: int, canales: int):
        """Igual que construir_dataset_sin_hud.py --modo compuesto."""
        po, pm, ps = self.idx_orig.get(k), self.idx_mask.get(k), self.idx_sin.get(k)
        if not (po and pm and ps):
            return None
        base = cv2.imread(po, cv2.IMREAD_COLOR)
        sin = cv2.imread(ps, cv2.IMREAD_COLOR)
        if base is None or sin is None:
            return None
        h, w = base.shape[:2]
        if sin.shape[:2] != (h, w):
            sin = cv2.resize(sin, (w, h), interpolation=cv2.INTER_CUBIC)
        a = self.mascara(k, (h, w)).astype(np.float32)
        if self.suavizar > 0:
            a = np.maximum(a, cv2.GaussianBlur(a, (0, 0), self.suavizar))
        a = a[..., None]
        out = np.clip(np.rint(base.astype(np.float32) * (1 - a) + sin.astype(np.float32) * a), 0, 255).astype(np.uint8)
        return a_canales(out, canales)


# ---------------------------------------------------------------------------
# Listas
# ---------------------------------------------------------------------------

def leer_lista(ruta: str, video: str | None = None) -> list[dict]:
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    return [r for r in filas if not video or r["video"] == video]


# ---------------------------------------------------------------------------
# Parches para entrenar
# ---------------------------------------------------------------------------

class Parches(_Base):
    """
    Cada item = una carga de frames (1 o 5) de la que se sacan `por_carga` parches.
    Devuelve float32 [por_carga, T*C, t, t] en [0,1]. Los parches no tocan el HUD.
    """

    def __init__(self, muestras, fuentes, canales, tam, n_items, temporal=1, por_carga=4,
                 multiplo=1, intentos=30):
        self.muestras, self.fuentes = muestras, fuentes
        self.canales, self.tam, self.n = canales, tam, n_items
        self.temporal, self.por_carga, self.multiplo, self.intentos = temporal, por_carga, multiplo, intentos

    def __len__(self):
        return self.n

    def _cargar(self, video, k):
        f = self.fuentes[video]
        r = self.temporal // 2
        ims = []
        for d in range(-r, r + 1):
            im = f.original(k + d, self.canales)
            if im is None:
                im = f.original(k, self.canales)
            ims.append(im)
        return ims, f.mascara(k, ims[0].shape[:2], margen=f.margen_hud)

    def __getitem__(self, i):
        for _ in range(10):
            video, k = random.choice(self.muestras)
            ims, m = self._cargar(video, k)
            if all(x is not None for x in ims):
                break
        h, w = ims[0].shape[:2]
        t = min(self.tam, h, w)
        t -= t % self.multiplo
        integ = None
        if m is not None:
            integ = cv2.integral(m.astype(np.uint8))
        st = np.concatenate(ims, axis=2)
        salida = []
        for _ in range(self.por_carga):
            mejor = None
            for _ in range(self.intentos):
                y, x = random.randint(0, h - t), random.randint(0, w - t)
                hud = 0 if integ is None else int(integ[y + t, x + t] - integ[y, x + t] - integ[y + t, x] + integ[y, x])
                if mejor is None or hud < mejor[0]:
                    mejor = (hud, y, x)
                if hud == 0:
                    break
            _, y, x = mejor
            p = st[y:y + t, x:x + t]
            if random.random() < 0.5:
                p = p[:, ::-1]
            if random.random() < 0.5:
                p = p[::-1]
            p = np.rot90(p, random.randint(0, 3))
            if self.temporal > 1 and random.random() < 0.5:  # invertir el tiempo tambien es valido
                c = self.canales
                p = np.concatenate([p[..., j * c:(j + 1) * c] for j in reversed(range(self.temporal))], axis=2)
            salida.append(np.ascontiguousarray(p).transpose(2, 0, 1))
        return torch.from_numpy(np.stack(salida).astype(np.float32) / 255.0)
