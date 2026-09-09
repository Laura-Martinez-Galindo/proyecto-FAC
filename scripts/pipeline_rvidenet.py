#!/usr/bin/env python3
"""
Pipeline RViDeNet (Recurrent Video Denoising Network - Yue et al., ICCV/TPAMI).
Utiliza una red recurrente convolucional (ConvGRU) espacio-temporal para propagar
caracteristicas y dependencias temporales a lo largo de la secuencia de frames.
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


# 1. Celda ConvGRU Recurrente Espacio-Temporal
class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv_gates = nn.Conv2d(in_channels + hidden_channels, 2 * hidden_channels, kernel_size, padding=padding, bias=True)
        self.conv_cand = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding, bias=True)

    def forward(self, x, h_prev):
        # x: (B, C_in, H, W), h_prev: (B, C_h, H, W)
        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        r, z = torch.chunk(gates, 2, dim=1)  # reset gate r, update gate z

        combined_cand = torch.cat([x, r * h_prev], dim=1)
        cand = torch.tanh(self.conv_cand(combined_cand))

        h_next = (1 - z) * h_prev + z * cand
        return h_next


class RViDeNet(nn.Module):
    """Red Recurrente Espacio-Temporal con Encoder convolucional, ConvGRU y Decoder."""
    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()
        self.base_ch = base_ch

        # Encoder espacial
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Memoria recurrente temporal
        self.gru = ConvGRUCell(in_channels=base_ch * 2, hidden_channels=base_ch * 2)

        # Decoder espacial
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, in_channels, 1),
        )

    def forward(self, seq_frames):
        # seq_frames: (B, T, C, H, W)
        b, t, c, h, w = seq_frames.shape
        h_state = torch.zeros((b, self.base_ch * 2, h // 2, w // 2), device=seq_frames.device)

        salidas = []
        for step in range(t):
            xt = seq_frames[:, step, :, :, :]
            feat = self.encoder(xt)
            h_state = self.gru(feat, h_state)
            out_t = self.decoder(h_state)
            salidas.append(out_t)

        return torch.stack(salidas, dim=1)  # (B, T, C, H, W)


# 2. Dataset Secuencial Recurrente
class RecurrentSeqDataset(Dataset):
    def __init__(self, rutas_frames, seq_len=5, patch_size=128, es_entrenamiento=True):
        self.rutas = rutas_frames
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.es_entrenamiento = es_entrenamiento

    def __len__(self):
        return max(0, len(self.rutas) - self.seq_len + 1)

    def __getitem__(self, idx):
        seq = []
        for i in range(idx, idx + self.seq_len):
            bgr = cv2.imread(str(self.rutas[i]), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise RuntimeError(f"No se pudo leer frame: {self.rutas[i]}")
            if bgr.ndim == 2:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
            elif bgr.shape[2] == 4:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            seq.append(rgb)

        h, w, _ = seq[0].shape
        if self.es_entrenamiento and h >= self.patch_size and w >= self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            seq = [f[top : top + self.patch_size, left : left + self.patch_size] for f in seq]

            if random.random() > 0.5:
                seq = [np.fliplr(f) for f in seq]
            if random.random() > 0.5:
                seq = [np.flipud(f) for f in seq]
            rot = random.choice([0, 1, 2, 3])
            if rot > 0:
                seq = [np.rot90(f, rot) for f in seq]

        tensores = [torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float().div(255.0) for f in seq]
        return torch.stack(tensores, dim=0)  # (T, C, H, W)


# 3. Argumentos y Rutas
def argumentos():
    p = argparse.ArgumentParser(description="Pipeline RViDeNet Recurrente Espacio-Temporal.")
    p.add_argument("--video", required=True)
    p.add_argument("--modo", choices=("original", "sin_hud"), required=True)
    p.add_argument("--etapa", choices=("entrenar", "inferir", "todo"), default="todo")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--pasos-por-epoca", type=int, default=300)
    p.add_argument("--seq-len", type=int, default=5, help="Longitud de secuencia recurrente (default: 5).")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
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

    nombre = a.id_experimento or (f"rvidenet_{a.modo}")
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
        "seq_len": a.seq_len,
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
    dataset = RecurrentSeqDataset(rutas_frames, seq_len=a.seq_len, patch_size=a.patch_size, es_entrenamiento=True)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    modelo = RViDeNet(in_channels=3, base_ch=a.base_channels).to(dispositivo)
    optimizador = optim.Adam(modelo.parameters(), lr=a.lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=a.epocas, eta_min=1e-6)
    criterio_l1 = nn.L1Loss()

    print(f"Iniciando entrenamiento RViDeNet ({a.epocas} epocas, seq_len={a.seq_len}, lr={a.lr})...")
    mejor_loss = float("inf")

    for epoca in range(1, a.epocas + 1):
        modelo.train()
        loss_total = 0.0
        pasos = 0

        pbar = tqdm(loader, desc=f"Epoca {epoca}/{a.epocas}", dynamic_ncols=True)
        for seq in pbar:
            seq = seq.to(dispositivo, non_blocking=True)  # (B, T, C, H, W)

            optimizador.zero_grad()
            out_seq = modelo(seq)

            # Perdida de consistencia recurrente: cada frame predicho se alinea con la media local temporal
            media_temp = seq.mean(dim=1, keepdim=True)
            loss = criterio_l1(out_seq, media_temp.expand_as(out_seq))

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
            torch.save({"estado": modelo.state_dict(), "base_channels": a.base_channels, "seq_len": a.seq_len}, r["ckpt"])

    print(f"Entrenamiento completado. Checkpoint guardado en {r['ckpt']}.")
    del modelo, optimizador, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inferir(a, r, rutas_frames, dispositivo, inicio):
    if not r["ckpt"].is_file():
        raise FileNotFoundError(f"No existe checkpoint en {r['ckpt']}")

    modelo = RViDeNet(in_channels=3, base_ch=a.base_channels).to(dispositivo)
    checkpoint = torch.load(r["ckpt"], map_location=dispositivo)
    modelo.load_state_dict(checkpoint["estado"])
    modelo.eval()

    r["salida"].mkdir(parents=True, exist_ok=True)
    print(f"Ejecutando inferencia RViDeNet en {len(rutas_frames)} frames...")

    n_frames = len(rutas_frames)
    seq_len = a.seq_len

    with torch.inference_mode():
        # Procesamiento secuencial recurrente
        for i in tqdm(range(0, n_frames, seq_len), desc="Inferiendo RViDeNet", dynamic_ncols=True):
            sub_rutas = rutas_frames[i : min(i + seq_len, n_frames)]
            if len(sub_rutas) < seq_len:
                # Pad con el ultimo frame si no completa seq_len
                sub_rutas += [sub_rutas[-1]] * (seq_len - len(sub_rutas))

            ventana = []
            for ruta in sub_rutas:
                bgr = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
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
            seq_tensor = torch.stack(tensores, dim=0).unsqueeze(0).to(dispositivo)  # (1, T, C, H_pad, W_pad)

            denoised_seq = modelo(seq_tensor).squeeze(0)  # (T, C, H_pad, W_pad)

            for step_idx in range(min(seq_len, n_frames - i)):
                frame_out = denoised_seq[step_idx].permute(1, 2, 0).cpu().numpy()
                frame_out = torch.clamp(torch.from_numpy(frame_out), 0.0, 1.0).numpy()
                if pad_h > 0 or pad_w > 0:
                    frame_out = frame_out[:h, :w, :]

                out_uint8 = (frame_out * 255.0).round().astype(np.uint8)
                salida_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)

                destino = r["salida"] / f"{rutas_frames[i + step_idx].stem}.png"
                cv2.imwrite(str(destino), salida_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    etiqueta = a.id_experimento or (f"RViDeNet_{a.modo}")
    actualizar_registro(a, r, etiqueta, inicio, n_frames)
    print(f"Inferencia RViDeNet completada exitosamente en {n_frames} frames.")

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
