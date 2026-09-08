#!/usr/bin/env python3
"""Pipeline Blind2Unblind (Wang et al., CVPR 2022) para denoising autosupervisado de video FLIR."""

import argparse
import fcntl
import gc
import json
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


# 1. Red UNet para Blind2Unblind
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


class UNetB2U(nn.Module):
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


# 2. Generador de mascaras Blind2Unblind (B2U Masking)
def generar_mascara_b2u(shape, device, k=2):
    """
    Genera una mascara binaria de muestreo estructurado B2U con celdas de tamano kxk.
    """
    b, c, h, w = shape
    mask = torch.zeros((b, 1, h, w), device=device)

    # Selecciona una posicion (r, c) aleatoria en cada celda kxk
    r_offset = random.randint(0, k - 1)
    c_offset = random.randint(0, k - 1)

    mask[:, :, r_offset::k, c_offset::k] = 1.0
    return mask


# 3. Dataset
class VideoFramesDataset(Dataset):
    def __init__(self, rutas_frames, patch_size=128, es_entrenamiento=True):
        self.rutas = rutas_frames
        self.patch_size = patch_size
        self.es_entrenamiento = es_entrenamiento

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, idx):
        ruta = self.rutas[idx]
        bgr = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise RuntimeError(f"No se pudo leer {ruta}")
        if bgr.ndim == 2:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
        elif bgr.shape[2] == 4:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        h, w, _ = rgb.shape
        if self.es_entrenamiento and h >= self.patch_size and w >= self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            rgb = rgb[top : top + self.patch_size, left : left + self.patch_size]

            if random.random() > 0.5:
                rgb = np.fliplr(rgb)
            if random.random() > 0.5:
                rgb = np.flipud(rgb)
            rot = random.choice([0, 1, 2, 3])
            if rot > 0:
                rgb = np.rot90(rgb, rot)

        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
        return tensor


# 4. Argumentos
def argumentos():
    p = argparse.ArgumentParser(description="Pipeline Blind2Unblind (CVPR 2022) para video FLIR.")
    p.add_argument("--video", required=True)
    p.add_argument("--modo", choices=("original", "sin_hud"), required=True)
    p.add_argument("--etapa", choices=("entrenar", "inferir", "todo"), default="todo")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--pasos-por-epoca", type=int, default=300)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-channels-init", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--id-experimento", help="Identificador personalizado del experimento.")
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

    nombre = a.id_experimento or (f"blind2unblind_{a.modo}")
    cache = RAIZ / "cache" / "denoising" / a.video / nombre
    return {
        "fuente": fuente,
        "salida": base_video / nombre,
        "cache": cache,
        "ckpt": cache / "modelo.pth",
        "nombre": nombre,
    }


def entrenar(a, r, rutas_frames, dispositivo):
    r["cache"].mkdir(parents=True, exist_ok=True)
    modelo = UNetB2U(in_channels=3, out_channels=3, num_channels_init=a.num_channels_init, depth=a.depth).to(dispositivo)
    optimizador = optim.Adam(modelo.parameters(), lr=a.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=a.epocas, eta_min=1e-6)
    criterio_l1 = nn.L1Loss(reduction="none")

    dataset = VideoFramesDataset(rutas_frames, patch_size=a.patch_size, es_entrenamiento=True)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    print(f"Iniciando entrenamiento Blind2Unblind ({a.epocas} epocas, lr={a.lr}, depth={a.depth})...")
    mejor_loss = float("inf")

    for epoca in range(1, a.epocas + 1):
        modelo.train()
        loss_total = 0.0
        pasos = 0

        pbar = tqdm(loader, desc=f"Epoca {epoca}/{a.epocas}", dynamic_ncols=True)
        for batch in pbar:
            batch = batch.to(dispositivo, non_blocking=True)
            # Generar mascara B2U de puntos ciegos estructurados
            mask = generar_mascara_b2u(batch.shape, dispositivo, k=2)

            # Entrada con puntos ciegos reemplazados por media local
            kernel = torch.ones((1, 1, 3, 3), device=dispositivo) / 8.0
            kernel[:, :, 1, 1] = 0.0
            batch_smooth = torch.cat([torch.nn.functional.conv2d(batch[:, i:i+1], kernel, padding=1) for i in range(3)], dim=1)
            entrada_b2u = batch * (1.0 - mask) + batch_smooth * mask

            optimizador.zero_grad()
            salida = modelo(entrada_b2u)

            # Perdida calculada exclusivamente sobre los pixeles ciegos (mascaras) y regularizada en visibles
            loss_mask = (criterio_l1(salida, batch) * mask).sum() / (mask.sum() * 3 + 1e-6)
            loss_vis = (criterio_l1(salida, batch) * (1.0 - mask)).sum() / ((1.0 - mask).sum() * 3 + 1e-6)
            loss = loss_mask + 0.5 * loss_vis

            loss.backward()
            optimizador.step()

            loss_total += loss.item()
            pasos += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mask": f"{loss_mask.item():.4f}"})

            if a.pasos_por_epoca and pasos >= a.pasos_por_epoca:
                break

        scheduler.step()
        promedio = loss_total / max(1, pasos)
        print(f"Epoca {epoca} completada - Loss promedio: {promedio:.5f}")

        if promedio < mejor_loss:
            mejor_loss = promedio
            torch.save({"estado": modelo.state_dict(), "depth": a.depth, "num_channels_init": a.num_channels_init}, r["ckpt"])

    print(f"Entrenamiento completado. Checkpoint guardado en {r['ckpt']}.")


def inferir(a, r, rutas_frames, dispositivo):
    if not r["ckpt"].is_file():
        raise FileNotFoundError(f"No existe checkpoint en {r['ckpt']}")

    modelo = UNetB2U(in_channels=3, out_channels=3, num_channels_init=a.num_channels_init, depth=a.depth).to(dispositivo)
    checkpoint = torch.load(r["ckpt"], map_location=dispositivo)
    modelo.load_state_dict(checkpoint["estado"])
    modelo.eval()

    r["salida"].mkdir(parents=True, exist_ok=True)
    print(f"Ejecutando inferencia Blind2Unblind en {len(rutas_frames)} frames...")

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


def main():
    a = argumentos()
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
        inferir(a, r, rutas_frames, dispositivo)


if __name__ == "__main__":
    main()
