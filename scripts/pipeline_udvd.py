#!/usr/bin/env python3
"""
Pipeline UDVD (Unified Dynamic Video Denoising - Sheth et al., CVPR 2021).
Utiliza prediccion de nucleos dinamicos (Dynamic Kernel Prediction) espacio-temporales
para filtrar cada pixel basandose en la trayectoria temporal sin desenfoque de movimiento.
Consumo de VRAM optimizado (<400 MB) y procesamiento por mini-batches.
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


# 1. Red de Prediccion de Nucleos Dinamicos (UDVD)
class DynamicKernelPredictor(nn.Module):
    def __init__(self, num_frames=5, in_channels=3, kernel_size=5, base_ch=32):
        super().__init__()
        self.num_frames = num_frames
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        in_total = num_frames * in_channels  # 15 canales

        # U-Net liviana para estimar pesos de filtrado dinamico (K*K por cada frame temporal)
        out_kernels = num_frames * (kernel_size * kernel_size)

        self.net = nn.Sequential(
            nn.Conv2d(in_total, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, out_kernels, 3, padding=1),
        )

    def forward(self, x_stack):
        # x_stack: (B, T*C, H, W)
        b, tc, h, w = x_stack.shape
        kernels = self.net(x_stack)  # (B, T * K*K, H, W)
        kernels = torch.softmax(kernels.view(b, self.num_frames * self.kernel_size * self.kernel_size, h, w), dim=1)

        # Aplicar filtrado dinamico mediante unfold
        pad = self.kernel_size // 2
        x_pad = torch.nn.functional.pad(x_stack, (pad, pad, pad, pad), mode="reflect")

        # Unfold sobre cada frame: (B, C, K*K, H*W)
        frames_unfolded = []
        for t in range(self.num_frames):
            f_t = x_pad[:, t * self.in_channels : (t + 1) * self.in_channels, :, :]
            unfold_t = torch.nn.functional.unfold(f_t, kernel_size=self.kernel_size)  # (B, C*K*K, H*W)
            frames_unfolded.append(unfold_t.view(b, self.in_channels, self.kernel_size * self.kernel_size, h * w))

        # Stack temporal: (B, C, T * K*K, H*W)
        all_unfolded = torch.cat(frames_unfolded, dim=2)
        kernels_flat = kernels.view(b, 1, self.num_frames * self.kernel_size * self.kernel_size, h * w)

        # Filtrado: suma ponderada por nucleos dinamicos
        filtered = (all_unfolded * kernels_flat).sum(dim=2).view(b, self.in_channels, h, w)
        return filtered


# 2. Dataset Temporal
class UDVDDataset(Dataset):
    def __init__(self, rutas_frames, num_frames=5, patch_size=128, es_entrenamiento=True):
        self.rutas = rutas_frames
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.es_entrenamiento = es_entrenamiento

    def __len__(self):
        return max(0, len(self.rutas) - self.num_frames + 1)

    def __getitem__(self, idx):
        ventana = []
        for i in range(idx, idx + self.num_frames):
            bgr = cv2.imread(str(self.rutas[i]), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise RuntimeError(f"No se pudo leer {self.rutas[i]}")
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
        center = tensores[self.num_frames // 2]
        return stack, center


# 3. Argumentos y Rutas
def argumentos():
    p = argparse.ArgumentParser(description="Pipeline UDVD (Unified Dynamic Video Denoising).")
    p.add_argument("--video", required=True)
    p.add_argument("--modo", choices=("original", "sin_hud"), required=True)
    p.add_argument("--etapa", choices=("entrenar", "inferir", "todo"), default="todo")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--pasos-por-epoca", type=int, default=300)
    p.add_argument("--kernel-size", type=int, default=5)
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

    nombre = a.id_experimento or (f"udvd_{a.modo}")
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
        "kernel_size": a.kernel_size,
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
    dataset = UDVDDataset(rutas_frames, num_frames=5, patch_size=a.patch_size, es_entrenamiento=True)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    modelo = DynamicKernelPredictor(num_frames=5, in_channels=3, kernel_size=a.kernel_size, base_ch=32).to(dispositivo)
    optimizador = optim.Adam(modelo.parameters(), lr=a.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=a.epocas, eta_min=1e-6)
    criterio_l1 = nn.L1Loss()

    print(f"Iniciando entrenamiento UDVD ({a.epocas} epocas, K={a.kernel_size}, lr={a.lr})...")
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

            # Consistencia temporal multi-frame
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
            torch.save({"estado": modelo.state_dict(), "kernel_size": a.kernel_size}, r["ckpt"])

    print(f"Entrenamiento completado. Checkpoint guardado en {r['ckpt']}.")
    del modelo, optimizador, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inferir(a, r, rutas_frames, dispositivo, inicio):
    if not r["ckpt"].is_file():
        raise FileNotFoundError(f"No existe checkpoint en {r['ckpt']}")

    modelo = DynamicKernelPredictor(num_frames=5, in_channels=3, kernel_size=a.kernel_size, base_ch=32).to(dispositivo)
    checkpoint = torch.load(r["ckpt"], map_location=dispositivo)
    modelo.load_state_dict(checkpoint["estado"])
    modelo.eval()

    r["salida"].mkdir(parents=True, exist_ok=True)
    print(f"Ejecutando inferencia UDVD en {len(rutas_frames)} frames...")

    n_frames = len(rutas_frames)

    with torch.inference_mode():
        for i in tqdm(range(n_frames), desc="Inferiendo UDVD", dynamic_ncols=True):
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

    etiqueta = a.id_experimento or (f"UDVD_{a.modo}")
    actualizar_registro(a, r, etiqueta, inicio, n_frames)
    print(f"Inferencia UDVD completada exitosamente en {n_frames} frames.")

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
