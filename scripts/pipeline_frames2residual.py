#!/usr/bin/env python3
"""
Pipeline Frames2Residual (Reyes-Gomez et al.) para Denoising Espacio-Temporal en video FLIR.
Utiliza una ventana temporal de T=5 frames consecutivos (x_{t-2}, ..., x_{t+2}) para estimar
el residuo de ruido estocastico y sustraerlo del frame central.
Consumo optimizado de VRAM (<400 MB) y procesamiento por mini-batches.
"""

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


# 1. Arquitectura Spatio-Temporal UNet 2.5D (Frames2Residual)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )

    def forward(self, x):
        return x + self.conv(x)


class SpatioTemporalUNet(nn.Module):
    def __init__(self, num_frames=5, in_channels=3, num_channels_init=48, depth=4):
        super().__init__()
        self.num_frames = num_frames
        self.depth = depth
        in_total = num_frames * in_channels  # 5 * 3 = 15 canales de entrada

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_total, num_channels_init, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = num_channels_init
        for _ in range(depth - 1):
            self.downs.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 3, padding=1, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                ResidualBlock(ch * 2),
            ))
            ch *= 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(ch * 2),
        )
        ch *= 2

        for _ in range(depth - 1):
            self.ups.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 3, padding=1, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
            ))
            ch //= 2

        self.final_up = nn.ConvTranspose2d(ch, ch // 2, 2, stride=2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch, num_channels_init, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_channels_init, in_channels, 1),  # Estima el residuo (3 canales)
        )

    def forward(self, x_stack):
        # x_stack: (B, T*C, H, W)
        x0 = self.in_conv(x_stack)
        skips = [x0]

        curr = x0
        for down in self.downs:
            curr = self.pool(curr)
            curr = down(curr)
            skips.append(curr)

        curr = self.pool(curr)
        curr = self.bottleneck(curr)

        for i in range(0, len(self.ups), 2):
            curr = self.ups[i](curr)
            skip = skips.pop()
            if curr.shape != skip.shape:
                curr = torch.nn.functional.interpolate(curr, size=skip.shape[2:])
            curr = torch.cat((skip, curr), dim=1)
            curr = self.ups[i + 1](curr)

        curr = self.final_up(curr)
        skip0 = skips.pop()
        if curr.shape != skip0.shape:
            curr = torch.nn.functional.interpolate(curr, size=skip0.shape[2:])
        curr = torch.cat((skip0, curr), dim=1)
        residual = self.out_conv(curr)
        return residual


# 2. Dataset Temporal de Ventanas Deslizantes
class TemporalFramesDataset(Dataset):
    def __init__(self, rutas_frames, num_frames=5, patch_size=128, es_entrenamiento=True):
        self.rutas = rutas_frames
        self.num_frames = num_frames
        self.radio = num_frames // 2
        self.patch_size = patch_size
        self.es_entrenamiento = es_entrenamiento

    def __len__(self):
        return max(0, len(self.rutas) - self.num_frames + 1)

    def __getitem__(self, idx):
        # Extraer ventana de frames consecutivos [idx : idx + num_frames]
        ventana = []
        for i in range(idx, idx + self.num_frames):
            bgr = cv2.imread(str(self.rutas[i]), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise RuntimeError(f"No se pudo leer frame: {self.rutas[i]}")
            if bgr.ndim == 2:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
            elif bgr.shape[2] == 4:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ventana.append(rgb)

        h, w, _ = ventana[0].shape
        if self.es_entrenamiento and h >= self.patch_size and w >= self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            ventana = [f[top : top + self.patch_size, left : left + self.patch_size] for f in ventana]

            # Aumentos sincronizados en toda la ventana
            if random.random() > 0.5:
                ventana = [np.fliplr(f) for f in ventana]
            if random.random() > 0.5:
                ventana = [np.flipud(f) for f in ventana]
            rot = random.choice([0, 1, 2, 3])
            if rot > 0:
                ventana = [np.rot90(f, rot) for f in ventana]

        tensores = [torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float().div(255.0) for f in ventana]
        stack = torch.cat(tensores, dim=0)  # (T*C, H, W)
        center_frame = tensores[self.radio]  # Frame central x_t (C, H, W)
        return stack, center_frame


# 3. Argumentos y Rutas
def argumentos():
    p = argparse.ArgumentParser(description="Pipeline Frames2Residual (Spatio-Temporal Denoising).")
    p.add_argument("--video", required=True)
    p.add_argument("--modo", choices=("original", "sin_hud"), required=True)
    p.add_argument("--etapa", choices=("entrenar", "inferir", "todo"), default="todo")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--pasos-por-epoca", type=int, default=300)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-frames", type=int, default=5, help="Tamano de ventana temporal (default: 5).")
    p.add_argument("--num-channels-init", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=128)
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

    nombre = a.id_experimento or (f"frames2residual_{a.modo}")
    cache = RAIZ / "cache" / "denoising" / a.video / nombre
    return {
        "fuente": fuente,
        "salida": base_video / nombre,
        "cache": cache,
        "ckpt": cache / "modelo.pth",
        "nombre": nombre,
    }


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
        "num_frames_temporal": a.num_frames,
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
    dataset = TemporalFramesDataset(rutas_frames, num_frames=a.num_frames, patch_size=a.patch_size, es_entrenamiento=True)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    modelo = SpatioTemporalUNet(num_frames=a.num_frames, in_channels=3, num_channels_init=a.num_channels_init, depth=a.depth).to(dispositivo)
    optimizador = optim.Adam(modelo.parameters(), lr=a.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=a.epocas, eta_min=1e-6)

    # Perdida autosupervisada espacio-temporal:
    # La red predice el residuo r_t tal que (x_t - r_t) debe ser consistente con la media temporal de los vecinos
    criterio_l1 = nn.L1Loss()
    criterio_mse = nn.MSELoss()

    print(f"Iniciando entrenamiento Frames2Residual (T={a.num_frames}, {a.epocas} epocas, lr={a.lr})...")
    mejor_loss = float("inf")

    for epoca in range(1, a.epocas + 1):
        modelo.train()
        loss_total = 0.0
        pasos = 0

        pbar = tqdm(loader, desc=f"Epoca {epoca}/{a.epocas}", dynamic_ncols=True)
        for stack, center in pbar:
            stack = stack.to(dispositivo, non_blocking=True)
            center = center.to(dispositivo, non_blocking=True)

            optimizador.zero_grad()
            residual = modelo(stack)
            denoised_center = center - residual

            # Objetivo: Consistencia con media temporal ponderada de vecinos
            # stack: (B, 15, H, W) -> reshape a (B, 5, 3, H, W)
            b, _, ph, pw = stack.shape
            frames_t = stack.view(b, a.num_frames, 3, ph, pw)
            media_temporal = frames_t.mean(dim=1)  # Media temporal de la ventana

            # Perdida L1 frente a la media temporal + regularizador de norma del residuo
            loss_temp = criterio_l1(denoised_center, media_temporal)
            loss_reg = criterio_mse(residual, torch.zeros_like(residual)) * 0.1
            loss = loss_temp + loss_reg

            loss.backward()
            optimizador.step()

            loss_total += loss.item()
            pasos += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "temp": f"{loss_temp.item():.4f}"})

            if a.pasos_por_epoca and pasos >= a.pasos_por_epoca:
                break

        scheduler.step()
        promedio = loss_total / max(1, pasos)
        print(f"Epoca {epoca} completada - Loss promedio: {promedio:.5f}")

        if promedio < mejor_loss:
            mejor_loss = promedio
            torch.save({"estado": modelo.state_dict(), "depth": a.depth, "num_frames": a.num_frames, "num_channels_init": a.num_channels_init}, r["ckpt"])

    print(f"Entrenamiento completado. Checkpoint guardado en {r['ckpt']}.")
    del modelo, optimizador, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inferir(a, r, rutas_frames, dispositivo, inicio):
    if not r["ckpt"].is_file():
        raise FileNotFoundError(f"No existe checkpoint en {r['ckpt']}")

    modelo = SpatioTemporalUNet(num_frames=a.num_frames, in_channels=3, num_channels_init=a.num_channels_init, depth=a.depth).to(dispositivo)
    checkpoint = torch.load(r["ckpt"], map_location=dispositivo)
    modelo.load_state_dict(checkpoint["estado"])
    modelo.eval()

    r["salida"].mkdir(parents=True, exist_ok=True)
    print(f"Ejecutando inferencia Frames2Residual en {len(rutas_frames)} frames...")

    radio = a.num_frames // 2
    n_frames = len(rutas_frames)

    with torch.inference_mode():
        for i in tqdm(range(n_frames), desc="Inferiendo frames temporales", dynamic_ncols=True):
            # Obtener indices con reflejo en los bordes para ventana completa de T frames
            indices = [min(max(i + k, 0), n_frames - 1) for k in range(-radio, radio + 1)]
            ventana = []
            for idx in indices:
                bgr = cv2.imread(str(rutas_frames[idx]), cv2.IMREAD_UNCHANGED)
                if bgr.ndim == 2:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
                elif bgr.shape[2] == 4:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ventana.append(rgb)

            h, w, _ = ventana[0].shape
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                ventana = [cv2.copyMakeBorder(f, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT) for f in ventana]

            tensores = [torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float().div(255.0) for f in ventana]
            stack = torch.cat(tensores, dim=0).unsqueeze(0).to(dispositivo)  # (1, T*C, H_pad, W_pad)
            center_tensor = tensores[radio].unsqueeze(0).to(dispositivo)

            residual = modelo(stack)
            denoised = center_tensor - residual
            denoised = torch.clamp(denoised, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()

            if pad_h > 0 or pad_w > 0:
                denoised = denoised[:h, :w, :]

            out_uint8 = (denoised * 255.0).round().astype(np.uint8)
            salida_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)

            destino = r["salida"] / f"{rutas_frames[i].stem}.png"
            cv2.imwrite(str(destino), salida_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    etiqueta = a.id_experimento or (f"Frames2Residual_{a.modo}")
    actualizar_registro(a, r, etiqueta, inicio, n_frames)
    print(f"Inferencia Frames2Residual completada exitosamente en {n_frames} frames.")

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
