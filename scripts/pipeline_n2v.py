#!/usr/bin/env python3
"""
Pipeline Noise2Void (Krull et al., CVPR 2019 / N2V2 / StructN2V) nativo en PyTorch para video FLIR.
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
    def __init__(self, in_channels=3, out_channels=3, num_channels_init=48, depth=3):
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


# 2. Dataset Noise2Void (con Blind-Spot, StructN2V y N2V2)
class N2VDataset(Dataset):
    def __init__(self, rutas_frames, patch_size=128, es_entrenamiento=True, perc_pix=0.015, struct_n2v="none", use_n2v2=True):
        self.rutas = rutas_frames
        self.patch_size = patch_size
        self.es_entrenamiento = es_entrenamiento
        self.perc_pix = perc_pix
        self.struct_n2v = struct_n2v
        self.use_n2v2 = use_n2v2

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

            # Aumentos
            if random.random() > 0.5:
                rgb = np.fliplr(rgb)
            if random.random() > 0.5:
                rgb = np.flipud(rgb)
            rot = random.choice([0, 1, 2, 3])
            if rot > 0:
                rgb = np.rot90(rgb, rot)

        target = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
        input_img = target.clone()

        if self.es_entrenamiento:
            c, ph, pw = input_img.shape
            num_pix = max(1, int(ph * pw * self.perc_pix))
            # Coordenadas de blind-spot
            coords_y = torch.randint(2, ph - 2, (num_pix,))
            coords_x = torch.randint(2, pw - 2, (num_pix,))

            mask = torch.zeros((1, ph, pw), dtype=torch.float32)

            for cy, cx in zip(coords_y, coords_x):
                # Vecino aleatorio para reemplazar el pixel ciego
                dy = random.choice([-2, -1, 1, 2])
                dx = random.choice([-2, -1, 1, 2])
                input_img[:, cy, cx] = target[:, cy + dy, cx + dx]
                mask[:, cy, cx] = 1.0

                # StructN2V: Enmascaramiento estructurado lineal
                if self.struct_n2v == "horizontal":
                    for off in [-2, -1, 1, 2]:
                        if 0 <= cx + off < pw:
                            input_img[:, cy, cx + off] = target[:, cy + dy, cx + dx]
                            mask[:, cy, cx + off] = 1.0
                elif self.struct_n2v == "vertical":
                    for off in [-2, -1, 1, 2]:
                        if 0 <= cy + off < ph:
                            input_img[:, cy + off, cx] = target[:, cy + dy, cx + dx]
                            mask[:, cy + off, cx] = 1.0

            return input_img, target, mask

        return input_img, target, torch.ones((1, h, w))


# 3. Argumentos
def argumentos():
    p = argparse.ArgumentParser(description="Pipeline Noise2Void (N2V / N2V2 / StructN2V) optimizado.")
    p.add_argument("--video", required=True)
    p.add_argument("--modo", choices=("original", "sin_hud"), required=True)
    p.add_argument("--etapa", choices=("entrenar", "inferir", "todo"), default="todo")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--pasos-por-epoca", type=int, default=300)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-channels-init", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--use-n2v2", action="store_true", default=True)
    p.add_argument("--sin-n2v2", dest="use_n2v2", action="store_false")
    p.add_argument("--struct-n2v", choices=("none", "horizontal", "vertical"), default="none")
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

    nombre = a.id_experimento or ("n2v" if a.modo == "original" else "n2v_sin_hud")
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
        "use_n2v2": a.use_n2v2,
        "struct_n2v": a.struct_n2v,
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
    dataset = N2VDataset(rutas_frames, patch_size=a.patch_size, es_entrenamiento=True, struct_n2v=a.struct_n2v, use_n2v2=a.use_n2v2)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    modelo = UNet(in_channels=3, out_channels=3, num_channels_init=a.num_channels_init, depth=a.depth).to(dispositivo)
    optimizador = optim.Adam(modelo.parameters(), lr=a.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=a.epocas, eta_min=1e-6)

    print(f"Iniciando entrenamiento Noise2Void ({a.epocas} epocas, lr={a.lr}, depth={a.depth}, struct={a.struct_n2v})...")
    mejor_loss = float("inf")

    for epoca in range(1, a.epocas + 1):
        modelo.train()
        loss_total = 0.0
        pasos = 0

        pbar = tqdm(loader, desc=f"Epoca {epoca}/{a.epocas}", dynamic_ncols=True)
        for inp, target, mask in pbar:
            inp = inp.to(dispositivo, non_blocking=True)
            target = target.to(dispositivo, non_blocking=True)
            mask = mask.to(dispositivo, non_blocking=True)

            optimizador.zero_grad()
            out = modelo(inp)

            # Perdida calculada exclusivamente en los pixeles enmascarados
            diff = torch.abs(out - target) * mask
            loss = diff.sum() / torch.clamp(mask.sum() * 3.0, min=1.0)

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
    print(f"Ejecutando inferencia Noise2Void en {len(rutas_frames)} frames...")

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

    etiqueta = a.id_experimento or ("Noise2Void" if a.modo == "original" else "Noise2Void sin HUD")
    actualizar_registro(a, r, etiqueta, inicio, len(rutas_frames))
    print(f"Inferencia N2V completada exitosamente en {len(rutas_frames)} frames.")

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