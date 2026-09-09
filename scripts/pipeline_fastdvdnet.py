#!/usr/bin/env python3
"""
Pipeline FastDVDnet (Tassano et al., CVPR 2020) para Denoising Espacio-Temporal en video FLIR.
Arquitectura de doble etapa en cascada (Dual-Stage UNet) sin estimacion de flujo optico explocito.
Consumo de VRAM optimizado (<400MB) y procesamiento por mini-batches.
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


# 1. Bloques FastDVDnet (Dual-Stage UNet)
class ConvBlock(nn.Module):
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


class StageUNet(nn.Module):
    def __init__(self, in_channels, out_channels=3, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool2d(2, 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        if d2.shape != e2.shape:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        return self.out_conv(d1)


class FastDVDnet(nn.Module):
    """Arquitectura FastDVDnet en 2 etapas: Etapa 1 procesa tripletes, Etapa 2 fusiona."""
    def __init__(self, in_channels_per_frame=3, base_ch=32):
        super().__init__()
        # Etapa 1: procesa tripletes de 3 frames (3 * 3 = 9 canales)
        self.stage1 = StageUNet(in_channels=in_channels_per_frame * 3, out_channels=in_channels_per_frame, base_ch=base_ch)
        # Etapa 2: procesa 3 salidas de etapa 1 (3 * 3 = 9 canales)
        self.stage2 = StageUNet(in_channels=in_channels_per_frame * 3, out_channels=in_channels_per_frame, base_ch=base_ch)

    def forward(self, x_stack):
        # x_stack: (B, 15, H, W) con 5 frames (f0, f1, f2, f3, f4)
        f0, f1, f2, f3, f4 = torch.chunk(x_stack, 5, dim=1)

        # 3 tripletes: (f0, f1, f2), (f1, f2, f3), (f2, f3, f4)
        t1 = torch.cat([f0, f1, f2], dim=1)
        t2 = torch.cat([f1, f2, f3], dim=1)
        t3 = torch.cat([f2, f3, f4], dim=1)

        out1_1 = self.stage1(t1)
        out1_2 = self.stage1(t2)
        out1_3 = self.stage1(t3)

        # Fusion en Etapa 2
        t_stage2 = torch.cat([out1_1, out1_2, out1_3], dim=1)
        out_final = self.stage2(t_stage2)
        return out_final


# 2. Dataset Temporal
class FastDVDDataset(Dataset):
    def __init__(self, rutas_frames, patch_size=128, es_entrenamiento=True):
        self.rutas = rutas_frames
        self.patch_size = patch_size
        self.es_entrenamiento = es_entrenamiento

    def __len__(self):
        return max(0, len(self.rutas) - 4)

    def __getitem__(self, idx):
        ventana = []
        for i in range(idx, idx + 5):
            bgr = cv2.imread(str(self.rutas[i]), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise RuntimeError(f"Error al leer {self.rutas[i]}")
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

            if random.random() > 0.5:
                ventana = [np.fliplr(f) for f in ventana]
            if random.random() > 0.5:
                ventana = [np.flipud(f) for f in ventana]
            rot = random.choice([0, 1, 2, 3])
            if rot > 0:
                ventana = [np.rot90(f, rot) for f in ventana]

        tensores = [torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float().div(255.0) for f in ventana]
        stack = torch.cat(tensores, dim=0)  # (15, H, W)
        center = tensores[2]  # Frame central
        return stack, center


# 3. Argumentos y Rutas
def argumentos():
    p = argparse.ArgumentParser(description="Pipeline FastDVDnet Espacio-Temporal.")
    p.add_argument("--video", required=True)
    p.add_argument("--modo", choices=("original", "sin_hud"), required=True)
    p.add_argument("--etapa", choices=("entrenar", "inferir", "todo"), default="todo")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--pasos-por-epoca", type=int, default=300)
    p.add_argument("--base-channels", type=int, default=32)
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

    nombre = a.id_experimento or (f"fastdvdnet_{a.modo}")
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
        "base_channels": a.base_channels,
        "lr": a.lr,
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
    dataset = FastDVDDataset(rutas_frames, patch_size=a.patch_size, es_entrenamiento=True)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    modelo = FastDVDnet(in_channels_per_frame=3, base_ch=a.base_channels).to(dispositivo)
    optimizador = optim.Adam(modelo.parameters(), lr=a.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=a.epocas, eta_min=1e-6)
    criterio_l1 = nn.L1Loss()

    print(f"Iniciando entrenamiento FastDVDnet ({a.epocas} epocas, lr={a.lr}, base_ch={a.base_channels})...")
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
            denoised = modelo(stack)

            # Consistencia temporal multi-triplete
            b, _, ph, pw = stack.shape
            media_temp = stack.view(b, 5, 3, ph, pw).mean(dim=1)
            loss = criterio_l1(denoised, media_temp)

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
            torch.save({"estado": modelo.state_dict(), "base_channels": a.base_channels}, r["ckpt"])

    print(f"Entrenamiento completado. Checkpoint guardado en {r['ckpt']}.")
    del modelo, optimizador, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inferir(a, r, rutas_frames, dispositivo, inicio):
    if not r["ckpt"].is_file():
        raise FileNotFoundError(f"No existe checkpoint en {r['ckpt']}")

    modelo = FastDVDnet(in_channels_per_frame=3, base_ch=a.base_channels).to(dispositivo)
    checkpoint = torch.load(r["ckpt"], map_location=dispositivo)
    modelo.load_state_dict(checkpoint["estado"])
    modelo.eval()

    r["salida"].mkdir(parents=True, exist_ok=True)
    print(f"Ejecutando inferencia FastDVDnet en {len(rutas_frames)} frames...")

    n_frames = len(rutas_frames)

    with torch.inference_mode():
        for i in tqdm(range(n_frames), desc="Inferiendo FastDVDnet", dynamic_ncols=True):
            indices = [min(max(i + k, 0), n_frames - 1) for k in range(-2, 3)]
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
            stack = torch.cat(tensores, dim=0).unsqueeze(0).to(dispositivo)

            denoised = modelo(stack)
            denoised = torch.clamp(denoised, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()

            if pad_h > 0 or pad_w > 0:
                denoised = denoised[:h, :w, :]

            out_uint8 = (denoised * 255.0).round().astype(np.uint8)
            salida_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)

            destino = r["salida"] / f"{rutas_frames[i].stem}.png"
            cv2.imwrite(str(destino), salida_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    etiqueta = a.id_experimento or (f"FastDVDnet_{a.modo}")
    actualizar_registro(a, r, etiqueta, inicio, n_frames)
    print(f"Inferencia FastDVDnet completada exitosamente en {n_frames} frames.")

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
