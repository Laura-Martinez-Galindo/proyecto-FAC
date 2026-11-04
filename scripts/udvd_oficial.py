#!/usr/bin/env python3
"""
UDVD oficial (Sheth et al., ICCV 2021) para los videos FLIR del proyecto.

Usa la red original `blind-video-net-4` (scripts/udvd_oficial/modelo.py, licencia MIT)
con la configuracion que los autores usan para ruido REAL (RAW y microscopia):
perdida MSE contra el frame central ruidoso, salida de C canales, Adam lr=1e-4,
5 frames de entrada, parches aleatorios con flips/rotaciones.

Por que no necesita imagenes limpias: la red es "ciega" al pixel que predice (su campo
receptivo excluye ese pixel del frame central). Lo unico que puede hacer es predecirlo a
partir de los vecinos (espaciales y de los otros 4 frames). La senal real es predecible
desde los vecinos; el ruido independiente no. Al minimizar el error contra el frame
ruidoso, la mejor prediccion posible es la senal limpia.
OJO: esto solo vale si el ruido es independiente pixel a pixel. Si el ruido viene
correlacionado (DDE, H.264), la red puede "predecir" parte del ruido desde los vecinos y no
lo quita: hay que medirlo (ver `--residuo` y el mapa de autocorrelacion).

Subcomandos:

  entrenar   entrena sobre frames de una o varias carpetas (p.ej. frames_sin_hud) y guarda
             un checkpoint. Se puede limitar a un regimen con --regimenes-csv/--regimen.
  inferir    aplica un checkpoint a una lista de frames (usa los vecinos +-2 de la carpeta).

Ejemplos:
  python scripts/udvd_oficial.py entrenar \\
      --carpeta videos/video2/frames_sin_hud --rango 59000:65000 \\
      --salida cache/udvd_oficial/video2_prueba --pasos 6000 --batch 16

  python scripts/udvd_oficial.py inferir \\
      --checkpoint cache/udvd_oficial/video2_prueba/modelo.pt \\
      --carpeta videos/video2/frames_sin_hud --frames 59343 66713 \\
      --salida cache/udvd_oficial/video2_prueba/salidas --residuo
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from udvd_oficial.modelo import BlindVideoNet  # noqa: E402

PATRON = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")
EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
N_FRAMES = 5  # la arquitectura oficial usa exactamente 5 (tres tripletes)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def indexar(carpeta: Path) -> dict[int, str]:
    idx = {}
    with os.scandir(carpeta) as it:
        for e in it:
            if e.is_file() and os.path.splitext(e.name)[1].lower() in EXT:
                m = PATRON.search(e.name)
                if m:
                    idx[int(m.group(1))] = e.path
    return idx


def leer(ruta: str, canales: int) -> np.ndarray:
    if canales == 1:
        im = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)[..., None]
    else:
        im = cv2.imread(ruta, cv2.IMREAD_COLOR)[..., ::-1]  # BGR -> RGB
    return np.ascontiguousarray(im)


def ventana(idx: dict[int, str], k: int) -> list[str] | None:
    """Rutas de los frames k-2..k+2. En los bordes del video repite el extremo."""
    if k not in idx:
        return None
    rutas = []
    for d in range(-(N_FRAMES // 2), N_FRAMES // 2 + 1):
        j = k + d
        while j not in idx and j != k:
            j += 1 if d < 0 else -1
        rutas.append(idx[j])
    return rutas


def frames_de_regimen(csv_path: str, video: str, regimenes: list[str]) -> set[int]:
    """CSV con columnas video, frame, regimen (frame = numero de archivo, desde 1)."""
    sel = set()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (not video or r.get("video") == video) and r.get("regimen") in regimenes:
                sel.add(int(r["frame"]))
    return sel


class Parches(torch.utils.data.Dataset):
    def __init__(self, muestras, canales, tam, n_por_epoca):
        self.muestras, self.canales, self.tam, self.n = muestras, canales, tam, n_por_epoca

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rutas = random.choice(self.muestras)
        ims = [leer(p, self.canales) for p in rutas]
        h, w = ims[0].shape[:2]
        t = min(self.tam, h, w)
        y, x = random.randint(0, h - t), random.randint(0, w - t)
        st = np.concatenate([im[y:y + t, x:x + t] for im in ims], axis=2)  # t x t x 5C
        if random.random() < 0.5:
            st = st[:, ::-1]
        if random.random() < 0.5:
            st = st[::-1]
        st = np.rot90(st, random.randint(0, 3))
        if random.random() < 0.5:  # invertir el orden temporal tambien es valido
            c = self.canales
            st = np.concatenate([st[..., j * c:(j + 1) * c] for j in reversed(range(N_FRAMES))], axis=2)
        return torch.from_numpy(np.ascontiguousarray(st).transpose(2, 0, 1).astype(np.float32) / 255.0)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def crear_modelo(canales: int) -> BlindVideoNet:
    # bias=False, blind=True, sigma_known=True = configuracion del repo oficial con --loss mse
    return BlindVideoNet(channels_per_frame=canales, out_channels=canales, bias=False, blind=True, sigma_known=True)


@torch.no_grad()
def denoise_ventana(modelo, stack: torch.Tensor, tile: int, overlap: int, amp: bool) -> torch.Tensor:
    """stack: 1 x 5C x H x W en [0,1]. Inferencia por mosaicos con mezcla suave en los bordes."""
    _, _, H, W = stack.shape
    c = modelo.c
    if max(H, W) <= tile:
        with torch.autocast(stack.device.type, enabled=amp and stack.device.type == "cuda"):
            out, _ = modelo(stack)
        return out.float()[:, :c]
    out = torch.zeros(1, c, H, W, device=stack.device)
    peso = torch.zeros(1, 1, H, W, device=stack.device)
    paso = tile - overlap
    rampa = torch.ones(tile, device=stack.device)
    if overlap > 0:
        r = torch.linspace(0, 1, overlap + 2, device=stack.device)[1:-1]
        rampa[:overlap], rampa[-overlap:] = r, r.flip(0)
    w2d = (rampa[:, None] * rampa[None, :])[None, None]
    ys = list(range(0, max(H - tile, 0) + 1, paso)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, paso)) or [0]
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    for y in ys:
        for x in xs:
            t = stack[:, :, y:y + tile, x:x + tile]
            with torch.autocast(stack.device.type, enabled=amp and stack.device.type == "cuda"):
                o, _ = modelo(t)
            th, tw = t.shape[2:]
            ww = w2d[:, :, :th, :tw]
            out[:, :, y:y + th, x:x + tw] += o.float()[:, :c] * ww
            peso[:, :, y:y + th, x:x + tw] += ww
    return out / peso.clamp_min(1e-6)


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------

def cmd_entrenar(a):
    disp = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    random.seed(a.semilla)
    np.random.seed(a.semilla)
    torch.manual_seed(a.semilla)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)

    muestras = []
    for carpeta in a.carpeta:
        idx = indexar(Path(carpeta))
        nums = sorted(idx)
        if a.rango:
            ini, fin = (int(x) for x in a.rango.split(":"))
            nums = [k for k in nums if ini <= k <= fin]
        if a.regimenes_csv:
            sel = frames_de_regimen(a.regimenes_csv, a.video, a.regimen)
            nums = [k for k in nums if k in sel]
        if a.excluir_csv:  # p.ej. frames del test de YOLO (+- margen) que no deben verse
            excl = set()
            with open(a.excluir_csv, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    if not a.video or r.get("video") == a.video:
                        k0 = int(r["frame"])
                        excl.update(range(k0 - a.margen_exclusion, k0 + a.margen_exclusion + 1))
            nums = [k for k in nums if k not in excl]
        if a.max_frames and len(nums) > a.max_frames:
            nums = sorted(random.sample(nums, a.max_frames))
        for k in nums:
            v = ventana(idx, k)
            if v:
                muestras.append(v)
        print(f"{carpeta}: {len(nums)} frames centrales disponibles")
    if not muestras:
        sys.exit("No hay frames para entrenar con esos filtros.")

    ds = Parches(muestras, a.canales, a.parche, a.pasos * a.batch)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers,
                                     pin_memory=disp.type == "cuda", drop_last=True)
    modelo = crear_modelo(a.canales).to(disp)
    opt = torch.optim.Adam(modelo.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.pasos, eta_min=a.lr * 0.05)
    escalador = torch.amp.GradScaler("cuda", enabled=a.amp and disp.type == "cuda")
    mid = N_FRAMES // 2
    c = a.canales
    log = open(sal / "entrenamiento.csv", "w")
    log.write("paso,loss,lr,segundos\n")
    t0 = time.time()
    media = None
    modelo.train()
    for paso, x in enumerate(dl, 1):
        x = x.to(disp, non_blocking=True)
        objetivo = x[:, mid * c:(mid + 1) * c]  # frame central RUIDOSO (no hay imagen limpia)
        with torch.autocast(disp.type, enabled=a.amp and disp.type == "cuda"):
            out, _ = modelo(x)
            loss = F.mse_loss(out[:, :c].float(), objetivo)
        opt.zero_grad(set_to_none=True)
        escalador.scale(loss).backward()
        escalador.step(opt)
        escalador.update()
        sched.step()
        media = loss.item() if media is None else 0.98 * media + 0.02 * loss.item()
        if paso % a.log_cada == 0 or paso == a.pasos:
            log.write(f"{paso},{media:.6f},{sched.get_last_lr()[0]:.2e},{time.time() - t0:.0f}\n")
            log.flush()
            print(f"paso {paso}/{a.pasos}  loss {media:.6f}  ({time.time() - t0:.0f} s)", flush=True)
        if paso % a.guardar_cada == 0 or paso == a.pasos:
            torch.save({"modelo": modelo.state_dict(), "canales": c, "args": vars(a), "paso": paso},
                       sal / "modelo.pt")
    meta = {"frames_entrenamiento": len(muestras), "segundos": round(time.time() - t0), **vars(a)}
    (sal / "configuracion.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Checkpoint: {sal / 'modelo.pt'}")


def cmd_inferir(a):
    disp = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    ck = torch.load(a.checkpoint, map_location=disp)
    c = int(ck["canales"])
    modelo = crear_modelo(c).to(disp)
    modelo.load_state_dict(ck["modelo"], strict=True)  # strict: si algo no carga, falla (no en silencio)
    modelo.eval()
    idx = indexar(Path(a.carpeta))
    frames = a.frames or sorted(idx)
    if a.lista_csv:
        with open(a.lista_csv, encoding="utf-8-sig", newline="") as f:
            frames = [int(r["frame"]) for r in csv.DictReader(f) if not a.video or r.get("video") == a.video]
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    if a.residuo:
        (sal / "residuo").mkdir(exist_ok=True)
    t0 = time.time()
    for i, k in enumerate(frames, 1):
        rutas = ventana(idx, k)
        if rutas is None:
            print(f"!! frame {k} no esta en {a.carpeta}")
            continue
        ims = [leer(p, c) for p in rutas]
        st = torch.from_numpy(np.concatenate(ims, axis=2).transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(disp)
        out = denoise_ventana(modelo, st, a.tile, a.overlap, a.amp)
        res = (out[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0)
        img = np.rint(res).astype(np.uint8)
        nombre = Path(rutas[N_FRAMES // 2]).stem + ".png"
        cv2.imwrite(str(sal / nombre), img[..., ::-1] if c == 3 else img[..., 0])
        if a.residuo:  # ruido estimado = ruidoso - limpio, centrado en 128 y amplificado x4
            orig = ims[N_FRAMES // 2].astype(np.float32)
            r = np.clip((orig - res) * 4 + 128, 0, 255).astype(np.uint8)
            cv2.imwrite(str(sal / "residuo" / nombre), r[..., ::-1] if c == 3 else r[..., 0])
        if i % 20 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)}  ({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("entrenar")
    e.add_argument("--carpeta", nargs="+", required=True, help="Carpeta(s) de frames (p.ej. frames_sin_hud)")
    e.add_argument("--salida", required=True)
    e.add_argument("--video", default="", help="Nombre del video (para filtrar los CSV)")
    e.add_argument("--rango", help="ini:fin de numeros de frame")
    e.add_argument("--regimenes-csv", help="CSV con video,frame,regimen")
    e.add_argument("--regimen", nargs="*", default=[], help="Regimen(es) a usar del CSV")
    e.add_argument("--excluir-csv", help="CSV con video,frame a excluir (p.ej. test de YOLO)")
    e.add_argument("--margen-exclusion", type=int, default=90, help="Frames a cada lado a excluir")
    e.add_argument("--max-frames", type=int, default=0)
    e.add_argument("--canales", type=int, default=3, choices=[1, 3])
    e.add_argument("--parche", type=int, default=128)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--pasos", type=int, default=6000)
    e.add_argument("--lr", type=float, default=1e-4)
    e.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    e.add_argument("--log-cada", type=int, default=50)
    e.add_argument("--guardar-cada", type=int, default=1000)
    e.add_argument("--semilla", type=int, default=0)
    e.add_argument("--amp", action="store_true", help="Precision mixta en GPU")
    e.add_argument("--cpu", action="store_true")

    i = sub.add_parser("inferir")
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--carpeta", required=True)
    i.add_argument("--salida", required=True)
    i.add_argument("--frames", type=int, nargs="*", help="Numeros de frame (por defecto todos)")
    i.add_argument("--lista-csv", help="CSV con columna frame (y video)")
    i.add_argument("--video", default="")
    i.add_argument("--tile", type=int, default=512)
    i.add_argument("--overlap", type=int, default=48)
    i.add_argument("--residuo", action="store_true", help="Guardar tambien ruidoso - limpio (x4)")
    i.add_argument("--amp", action="store_true")
    i.add_argument("--cpu", action="store_true")

    a = ap.parse_args()
    cmd_entrenar(a) if a.cmd == "entrenar" else cmd_inferir(a)


if __name__ == "__main__":
    main()
