#!/usr/bin/env python3
"""
Pipeline Noise2Noise (Lehtinen et al., ICML 2018) nativo en PyTorch para video FLIR.
Implementacion optimizada de bajo consumo de memoria (0 archivos temporales TIFF en disco, <300MB VRAM).
"""

import argparse
import fcntl
import gc
import json
import math
import os
import random
import re
import shutil
import sys
import time

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
RUTA_JSON = RAIZ / "config" / "videos.json"


# 1. Arquitectura UNet 2D PyTorch nativa
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_channels_init=48, depth=4):
        super().__init__()
        self.depth = depth
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = num_channels_init
        self.downs.append(DoubleConv(in_channels, ch))
        for _ in range(depth - 1):
            self.downs.append(DoubleConv(ch, ch * 2))
            ch *= 2

        self.bottleneck = DoubleConv(ch, ch * 2)
        ch *= 2

        for _ in range(depth):
            self.ups.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
            self.ups.append(DoubleConv(ch, ch // 2))
            ch //= 2

        self.out_conv = nn.Conv2d(num_channels_init, out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.out_conv(x)


# 2. Dataset Noise2Noise con filtrado de movimiento
def estimar_movimiento(g1, g2):
    escala = 0.25
    p1 = cv2.resize(g1, (0, 0), fx=escala, fy=escala)
    p2 = cv2.resize(g2, (0, 0), fx=escala, fy=escala)
    flow = cv2.calcOpticalFlowFarneback(p1, p2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))) / escala
    diff = float(np.mean(np.abs(g1.astype(np.float32) - g2.astype(np.float32))))
    return diff, mag


class N2NPairsDataset(Dataset):
    def __init__(self, pares_rutas, patch_size=128, es_entrenamiento=True):
        self.pares = pares_rutas
        self.patch_size = patch_size
        self.es_entrenamiento = es_entrenamiento

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        ruta1, ruta2 = self.pares[idx]
        bgr1 = cv2.imread(str(ruta1), cv2.IMREAD_UNCHANGED)
        bgr2 = cv2.imread(str(ruta2), cv2.IMREAD_UNCHANGED)

        if bgr1 is None or bgr2 is None:
            raise RuntimeError(f"Error al leer par: {ruta1}, {ruta2}")

        def a_rgb(bgr):
            if bgr.ndim == 2:
                return cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
            elif bgr.shape[2] == 4:
                return cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        rgb1 = a_rgb(bgr1)
        rgb2 = a_rgb(bgr2)

        h, w, _ = rgb1.shape
        if self.es_entrenamiento and h >= self.patch_size and w >= self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            rgb1 = rgb1[top : top + self.patch_size, left : left + self.patch_size]
            rgb2 = rgb2[top : top + self.patch_size, left : left + self.patch_size]

            # Aumentos coherentes en ambas imagenes
            if random.random() > 0.5:
                rgb1 = np.fliplr(rgb1)
                rgb2 = np.fliplr(rgb2)
            if random.random() > 0.5:
                rgb1 = np.flipud(rgb1)
                rgb2 = np.flipud(rgb2)
            rot = random.choice([0, 1, 2, 3])
            if rot > 0:
                rgb1 = np.rot90(rgb1, rot)
                rgb2 = np.rot90(rgb2, rot)

        t1 = torch.from_numpy(np.ascontiguousarray(rgb1)).permute(2, 0, 1).float().div(255.0)
        t2 = torch.from_numpy(np.ascontiguousarray(rgb2)).permute(2, 0, 1).float().div(255.0)
        return t1, t2


# 3. Argumentos
def argumentos():
    p = argparse.ArgumentParser(description="Pipeline Noise2Noise (N2N) optimizado.")
    p.add_argument("--video", required=True)
    p.add_argument("--modo", choices=("original", "sin_hud"), required=True)
    p.add_argument("--etapa", choices=("entrenar", "inferir", "todo"), default="todo")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--pasos-por-epoca", type=int, default=300)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-channels-init", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--filtrar-movimiento", action="store_true", default=True)
    p.add_argument("--sin-filtro-movimiento", dest="filtrar_movimiento", action="store_false")
    p.add_argument("--umbral-movimiento", type=float, default=3.5)
    p.add_argument("--umbral-diferencia", type=float, default=60.0)
    p.add_argument("--id-experimento", help="Nombre del experimento.")
    p.add_argument("--reiniciar", action="store_true")
    return p.parse_args()


def cargar_json():
    with RUTA_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def natural(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def listar(carpeta, limite=None):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    rutas = sorted((p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in exts), key=natural)
    return rutas[:limite] if limite else rutas


def rutas_base(a, cfg):
    v = cfg[a.video]
    fuente_cfg = v.get("extraccion", {}) if a.modo == "original" else v.get("hud", {}).get("limpieza", {})
    if fuente_cfg.get("estado") != "completada":
        raise RuntimeError(f"La fuente {a.modo} no esta completada")
    fuente = Path(fuente_cfg["carpeta_salida"]).resolve()
    base_video = Path(v["ruta"]).resolve().parent.parent

    nombre = a.id_experimento or ("n2n" if a.modo == "original" else "n2n_sin_hud")
    cache = RAIZ / "cache" / "denoising" / a.video / nombre
    return {
        "fuente": fuente,
        "salida": base_video / nombre,
        "cache": cache,
        "ckpt": cache / "modelo.pth",
        "nombre": nombre,
    }


def construir_parejas(rutas_frames, filtrar, umbral_mov, umbral_diff):
    pares = []
    excluidas = 0

    for i in range(len(rutas_frames) - 1):
        f1, f2 = rutas_frames[i], rutas_frames[i + 1]
        if filtrar:
            im1 = cv2.imread(str(f1), cv2.IMREAD_GRAYSCALE)
            im2 = cv2.imread(str(f2), cv2.IMREAD_GRAYSCALE)
            if im1 is None or im2 is None:
                continue
            diff, mag = estimar_movimiento(im1, im2)
            if mag > umbral_mov or diff > umbral_diff:
                excluidas += 1
                continue
        pares.append((f1, f2))

    print(f"Parejas N2N construidas: {len(pares)} validas, {excluidas} excluidas por movimiento.")
    if len(pares) < 5:
        pares = [(rutas_frames[i], rutas_frames[i + 1]) for i in range(len(rutas_frames) - 1)]
    return pares


def actualizar_registro(a, r, etiqueta, inicio, cantidad):
    ruta_lock = RUTA_JSON.with_suffix(".lock")
    registro = {
        "nombre": etiqueta,
        "estado": "completada",
        "modo": a.modo,
        "carpeta_salida": str(r["salida"].relative_to(RAIZ)),
        "checkpoint": str(r["ckpt"].relative_to(RAIZ)),
        "frames_procesados": cantidad,
        "depth": a.depth,
        "lr": a.lr,
        "filtrado_movimiento": a.filtrar_movimiento,
        "fecha_ejecucion": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tiempo_total_minutos": (time.monotonic() - inicio) / 60.0,
    }

    with ruta_lock.open("w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        configuracion = cargar_json()
        configuracion[a.video].setdefault("modelos", {})
        configuracion[a.video]["modelos"][r["nombre"]] = registro
        temporal = RUTA_JSON.with_suffix(".json.tmp")
        temporal.write_text(json.dumps(configuracion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporal.replace(RUTA_JSON)
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def entrenar(a, r, rutas_frames, dispositivo):
    r["cache"].mkdir(parents=True, exist_ok=True)
    pares = construir_parejas(rutas_frames, a.filtrar_movimiento, a.umbral_movimiento, a.umbral_diferencia)

    dataset = N2NPairsDataset(pares, patch_size=a.patch_size, es_entrenamiento=True)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    modelo = UNet(in_channels=3, out_channels=3, num_channels_init=a.num_channels_init, depth=a.depth).to(dispositivo)
    optimizador = optim.Adam(modelo.parameters(), lr=a.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=a.epocas, eta_min=1e-6)
    criterio = nn.L1Loss()

    print(f"Iniciando entrenamiento Noise2Noise ({a.epocas} epocas, lr={a.lr}, depth={a.depth})...")
    mejor_loss = float("inf")

    for epoca in range(1, a.epocas + 1):
        modelo.train()
        loss_total = 0.0
        pasos = 0

        pbar = tqdm(loader, desc=f"Epoca {epoca}/{a.epocas}", dynamic_ncols=True)
        for x, y in pbar:
            x = x.to(dispositivo, non_blocking=True)
            y = y.to(dispositivo, non_blocking=True)

            optimizador.zero_grad()
            out = modelo(x)
            loss = criterio(out, y)
            loss.backward()
            optimizador.step()

            loss_total += loss.item()
            pasos += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if a.pasos_por_epoca and pasos >= a.pasos_por_epoca:
                break

        scheduler.step()
        promedio = loss_total / max(1, pasos)
        print(f"Epoca {epoca} completada - Loss promedio: {promedio:.5f}")

        if promedio < mejor_loss:
            mejor_loss = promedio
            torch.save({"estado": modelo.state_dict(), "depth": a.depth, "num_channels_init": a.num_channels_init}, r["ckpt"])

    print(f"Entrenamiento completado. Checkpoint guardado en {r['ckpt']}.")
    del modelo, optimizador, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inferir(a, r, rutas_frames, dispositivo, inicio):
    if not r["ckpt"].is_file():
        raise FileNotFoundError(f"No existe checkpoint en {r['ckpt']}")

    modelo = UNet(in_channels=3, out_channels=3, num_channels_init=a.num_channels_init, depth=a.depth).to(dispositivo)
    checkpoint = torch.load(r["ckpt"], map_location=dispositivo)
    modelo.load_state_dict(checkpoint["estado"])
    modelo.eval()

    r["salida"].mkdir(parents=True, exist_ok=True)
    print(f"Ejecutando inferencia Noise2Noise en {len(rutas_frames)} frames...")

    with torch.inference_mode():
        for ruta in tqdm(rutas_frames, desc="Inferiendo frames", dynamic_ncols=True):
            bgr = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                continue
            if bgr.ndim == 2:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
            elif bgr.shape[2] == 4:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            h, w, _ = rgb.shape
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                rgb_padded = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            else:
                rgb_padded = rgb

            tensor = torch.from_numpy(np.ascontiguousarray(rgb_padded)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dispositivo)
            denoised = modelo(tensor)

            denoised = torch.clamp(denoised, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
            if pad_h > 0 or pad_w > 0:
                denoised = denoised[:h, :w, :]

            out_uint8 = (denoised * 255.0).round().astype(np.uint8)
            salida_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)

            destino = r["salida"] / f"{ruta.stem}.png"
            cv2.imwrite(str(destino), salida_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    etiqueta = a.id_experimento or ("Noise2Noise" if a.modo == "original" else "Noise2Noise sin HUD")
    actualizar_registro(a, r, etiqueta, inicio, len(rutas_frames))
    print(f"Inferencia N2N completada exitosamente en {len(rutas_frames)} frames.")

    del modelo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    a = argumentos()
    inicio = time.monotonic()
    cfg = cargar_json()
    r = rutas_base(a, cfg)
    rutas_frames = listar(r["fuente"], a.max_frames)
    dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if a.reiniciar:
        shutil.rmtree(r["cache"], ignore_errors=True)
        shutil.rmtree(r["salida"], ignore_errors=True)

    if a.etapa in ("entrenar", "todo"):
        entrenar(a, r, rutas_frames, dispositivo)
    if a.etapa in ("inferir", "todo"):
        inferir(a, r, rutas_frames, dispositivo, inicio)


if __name__ == "__main__":
    main()
