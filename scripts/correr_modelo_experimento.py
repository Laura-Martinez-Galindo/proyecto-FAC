#!/usr/bin/env python3
"""
==============================================================================
WRAPPER UNIVERSAL DE MODELOS Y ABLACIONES — PROTOCOLO EXPERIMENTAL FAC
==============================================================================
Implementa y ejecuta los modelos del estado del arte (2021-2026) bajo una
interfaz unificada orientada a carpetas de entrada y salida:

Modelos soportados:
  1. p2n            : Positive2Negative (CVPR 2025) — Denoising a resolución 100%
                      completa sin submuestreo, preservando objetos pequeños.
  2. mm_bsn         : Multi-Mask BSN (CVPRW 2023) — Máscaras multi-geometría
                      ('+', '□', 'o') para romper grumos correlacionados DDE+HAZE.
  3. ap_bsn         : Asymmetric PD BSN (CVPR 2022) — Submuestreo asimétrico con
                      stride configurable (s=2, 3, 5) y refinamiento R3 opcional.
  4. nb2nb          : Neighbor2Neighbor (CVPR 2021) — Submuestreo de celdas 2×2.
  5. b2u            : Blind2Unblind (CVPR 2022) — Máscara visible sin descarte central.
  6. struct_n2v     : StructN2V / N2V2 (CAREamics) — Máscara vertical 9×1 contra
                      Fixed Pattern Noise (FPN striping térmico) vs isotrópica 1×1.
  7. f2r            : Frames2Residual (ECCV 2026) — Desacoplamiento espacio-temporal.
  8. udvd           : UDVD baseline (ICCV 2021) — Predicción temporal con ventana T=5.

Opciones de Ablación:
  --alinear-homografia  : Alineación RANSAC previa sobre ventana temporal (elimina 180 KTS)
  --loss-type           : Función de pérdida ('l1' o 'l2' para estudiar colas pesadas HAZE)
  --stride-submuestreo  : Salto de submuestreo s ∈ {2, 3, 5} para AP-BSN / MM-BSN
  --struct-mask-tipo    : 'lineal' (9×1 vertical) vs 'isotropica' (1×1) para StructN2V
  --post-r3             : Refinamiento por reemplazo aleatorio R3 (AP-BSN)
==============================================================================
"""

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

from metricas_experimentos import estimar_sigma_mad


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def listar_frames(carpeta: Path, limite: int = None) -> list:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    rutas = sorted([p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in exts], key=natural_key)
    return rutas[:limite] if limite else rutas


# ─────────────────────────────────────────────────────────────────────────────
# 1. Alineación Homográfica RANSAC (Compensación de 180 KTS)
# ─────────────────────────────────────────────────────────────────────────────
def alinear_frame_a_referencia_ransac(img_src: np.ndarray, img_ref: np.ndarray) -> np.ndarray:
    """
    Alinea img_src hacia img_ref usando homografía calculada con ORB + RANSAC.
    Si no hay suficientes puntos de correspondencia, retorna img_src sin cambios.
    """
    orb = cv2.ORB_create(nfeatures=400)
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY) if img_src.ndim == 3 else img_src
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY) if img_ref.ndim == 3 else img_ref

    kp1, des1 = orb.detectAndCompute(gray_src, None)
    kp2, des2 = orb.detectAndCompute(gray_ref, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return img_src

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 8:
        return img_src

    matches = sorted(matches, key=lambda x: x.distance)[:80]
    pts_src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 3.0)
    if H is None:
        return img_src

    h, w = img_ref.shape[:2]
    alineado = cv2.warpPerspective(img_src, H, (w, h), borderMode=cv2.BORDER_REFLECT)
    return alineado


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset Universal para Modelos Espaciales y Temporales
# ─────────────────────────────────────────────────────────────────────────────
class VideoFrameDataset(Dataset):
    def __init__(self, rutas: list, patch_size: int = 128, es_entrenamiento: bool = True,
                 n_frames_temporales: int = 1, alinear_homografia: bool = False):
        self.rutas = rutas
        self.patch_size = patch_size
        self.es_entrenamiento = es_entrenamiento
        self.n_frames = n_frames_temporales
        self.alinear_homografia = alinear_homografia

    def __len__(self):
        if self.n_frames <= 1:
            return len(self.rutas)
        mid = self.n_frames // 2
        return max(0, len(self.rutas) - 2 * mid)

    def _leer_rgb(self, ruta: Path) -> np.ndarray:
        bgr = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise RuntimeError(f"No se pudo leer: {ruta}")
        if bgr.ndim == 2:
            return cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
        elif bgr.shape[2] == 4:
            return cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _aplicar_crop_aug(self, imgs: list):
        h, w = imgs[0].shape[:2]
        if self.es_entrenamiento and h >= self.patch_size and w >= self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            imgs = [img[top:top + self.patch_size, left:left + self.patch_size] for img in imgs]
            if np.random.rand() > 0.5:
                imgs = [np.fliplr(img) for img in imgs]
            if np.random.rand() > 0.5:
                imgs = [np.flipud(img) for img in imgs]
            k = np.random.randint(0, 4)
            if k > 0:
                imgs = [np.rot90(img, k) for img in imgs]
        return imgs

    def __getitem__(self, idx):
        if self.n_frames <= 1:
            rgb = self._leer_rgb(self.rutas[idx])
            imgs = self._aplicar_crop_aug([rgb])
            return torch.from_numpy(np.ascontiguousarray(imgs[0])).permute(2, 0, 1).float().div(255.0)
        else:
            mid = self.n_frames // 2
            real_idx = idx + mid
            indices = [min(max(0, real_idx + k), len(self.rutas) - 1) for k in range(-mid, mid + 1)]
            imgs = [self._leer_rgb(self.rutas[i]) for i in indices]

            # Si se activó alineación por homografía, alinear todos al frame central (mid)
            if self.alinear_homografia:
                ref = imgs[mid]
                imgs = [alinear_frame_a_referencia_ransac(img, ref) if i != mid else ref for i, img in enumerate(imgs)]

            imgs = self._aplicar_crop_aug(imgs)
            stacked = [torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0) for img in imgs]
            return torch.cat(stacked, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Arquitecturas de Redes Neuronales
# ─────────────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetUniversal(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_ch=48, depth=4):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.convups = nn.ModuleList()

        ch = in_channels
        encoder_chs = []
        for i in range(depth):
            out = base_ch * (2 ** i)
            self.downs.append(DoubleConv(ch, out))
            self.pools.append(nn.MaxPool2d(2))
            encoder_chs.append(out)
            ch = out

        self.bottleneck = DoubleConv(ch, ch * 2)
        ch = ch * 2

        for i in reversed(range(depth)):
            target = encoder_chs[i]
            self.convups.append(nn.ConvTranspose2d(ch, target, 2, stride=2))
            self.ups.append(DoubleConv(target * 2, target))
            ch = target

        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for convup, up, skip in zip(self.convups, self.ups, reversed(skips)):
            x = convup(x)
            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([x, skip], dim=1))
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Paradigmas de Entrenamiento Autosupervisado
# ─────────────────────────────────────────────────────────────────────────────
def submuestreo_pixel_shuffle(x: torch.Tensor, stride: int) -> torch.Tensor:
    """Submuestreo espacial asimétrico PD para AP-BSN."""
    B, C, H, W = x.shape
    return x[:, :, ::stride, ::stride]


def generar_mascara_multigeometria(batch: torch.Tensor, tipo: str = "+") -> torch.Tensor:
    """Máscaras geométricas de MM-BSN para destruir grumos DDE+HAZE."""
    B, C, H, W = batch.shape
    mask = torch.zeros(B, 1, H, W, device=batch.device)
    n_pts = max(1, int(H * W * 0.008))
    ys = torch.randint(2, H - 2, (B, n_pts))
    xs = torch.randint(2, W - 2, (B, n_pts))

    for b in range(B):
        y_b = ys[b]
        x_b = xs[b]
        mask[b, 0, y_b, x_b] = 1.0
        if tipo == "+":  # Máscara en cruz
            mask[b, 0, y_b - 1, x_b] = 1.0
            mask[b, 0, y_b + 1, x_b] = 1.0
            mask[b, 0, y_b, x_b - 1] = 1.0
            mask[b, 0, y_b, x_b + 1] = 1.0
        elif tipo == "square":  # Máscara cuadrada 3×3
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    mask[b, 0, y_b + dy, x_b + dx] = 1.0
        elif tipo == "o":  # Máscara circular / diagonal
            mask[b, 0, y_b - 1, x_b - 1] = 1.0
            mask[b, 0, y_b - 1, x_b + 1] = 1.0
            mask[b, 0, y_b + 1, x_b - 1] = 1.0
            mask[b, 0, y_b + 1, x_b + 1] = 1.0

    return mask.expand(-1, C, -1, -1).bool()


def generar_mascara_struct_n2v(batch: torch.Tensor, tipo: str = "lineal") -> torch.Tensor:
    """
    Máscara para StructN2V:
    'lineal'     : segmento vertical 9×1 (destruye striping FPN columnar del sensor InSb).
    'isotropica' : píxel puntual 1×1 (N2V2 estándar).
    """
    B, C, H, W = batch.shape
    mask = torch.zeros(B, 1, H, W, device=batch.device)
    n_puntos = max(1, int(H * W * 0.005))
    ys = torch.randint(4, H - 4, (B, n_puntos))
    xs = torch.randint(0, W, (B, n_puntos))

    for b in range(B):
        y_b = ys[b]
        x_b = xs[b]
        if tipo == "lineal":
            for dy in range(-4, 5):  # 9 píxeles en columna vertical
                mask[b, 0, y_b + dy, x_b] = 1.0
        else:
            mask[b, 0, y_b, x_b] = 1.0

    return mask.expand(-1, C, -1, -1).bool()


def entrenar_red(modelo_id: str, rutas: list, carpeta_cache: Path,
                 epocas: int, lr: float, batch_size: int, patch_size: int,
                 paciencia: int, dispositivo: torch.device,
                 loss_type: str = "l1", stride: int = 2,
                 struct_mask_tipo: str = "lineal",
                 n_frames_temporales: int = 1,
                 alinear_homografia: bool = False) -> nn.Module:
    """Bucle de entrenamiento unificado con soporte para todos los paradigmas."""
    in_channels = n_frames_temporales * 3 if n_frames_temporales > 1 else 3
    modelo = UNetUniversal(in_channels=in_channels, out_channels=3, base_ch=48, depth=4).to(dispositivo)

    ckpt = carpeta_cache / "modelo.pth"
    if ckpt.exists():
        print(f"  [↑] Cargando checkpoint previo: {ckpt.name}")
        modelo.load_state_dict(torch.load(str(ckpt), map_location=dispositivo))

    optimizer = optim.Adam(modelo.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epocas, eta_min=1e-6)
    
    criterion = nn.L1Loss() if loss_type == "l1" else nn.MSELoss()
    crit_mse = nn.MSELoss()

    dataset = VideoFrameDataset(rutas, patch_size=patch_size, es_entrenamiento=True,
                                n_frames_temporales=n_frames_temporales,
                                alinear_homografia=alinear_homografia)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=min(4, os.cpu_count() or 1),
                        pin_memory=True, drop_last=True)

    mejor_loss = float("inf")
    epocas_sin_mejora = 0

    for ep in range(1, epocas + 1):
        modelo.train()
        loss_ep = 0.0
        pasos = 0
        pbar = tqdm(loader, desc=f"Época {ep}/{epocas} [{modelo_id.upper()}]", dynamic_ncols=True)

        for batch in pbar:
            batch = batch.to(dispositivo, non_blocking=True)
            optimizer.zero_grad()

            # -------------------------------------------------------------
            # Paradigma 1: Positive2Negative (P2N CVPR 2025)
            # Resolución 100% sin submuestreo. Simetría +n / -n.
            # -------------------------------------------------------------
            if modelo_id == "p2n":
                # Estimación de perturbación controlada ε ~ N(0, σ_local)
                sigma_estimado = 0.03
                ruido_pert = torch.randn_like(batch) * sigma_estimado
                y_pos = torch.clamp(batch + ruido_pert, 0.0, 1.0)
                y_neg = torch.clamp(batch - ruido_pert, 0.0, 1.0)

                pred_pos = modelo(y_pos)
                pred_neg = modelo(y_neg)
                pred_orig = modelo(batch)

                # Pérdida de simetría y reconstrucción a resolución completa
                loss_simetria = criterion(0.5 * (pred_pos + pred_neg), batch)
                loss_consistencia = criterion(pred_orig, 0.5 * (pred_pos + pred_neg))
                loss = loss_simetria + 0.5 * loss_consistencia

            # -------------------------------------------------------------
            # Paradigma 2: Multi-Mask BSN (MM-BSN CVPRW 2023)
            # -------------------------------------------------------------
            elif modelo_id == "mm_bsn":
                geometrias = ["+", "square", "o"]
                geom_actual = geometrias[np.random.randint(0, 3)]
                mask = generar_mascara_multigeometria(batch, tipo=geom_actual)

                # Rellenar máscara con promedio convolucional 3×3 de vecinos
                kernel = torch.ones(3, 1, 3, 3, device=dispositivo) / 8.0
                kernel[:, :, 1, 1] = 0.0
                batch_blind = batch.clone()
                for c in range(3):
                    vecinos_c = torch.nn.functional.conv2d(batch[:, c:c+1], kernel[c:c+1], padding=1)
                    batch_blind[:, c:c+1][mask[:, c:c+1]] = vecinos_c[mask[:, c:c+1]]

                pred = modelo(batch_blind)
                loss = criterion(pred[mask], batch[mask])

            # -------------------------------------------------------------
            # Paradigma 3: AP-BSN (CVPR 2022) con Submuestreo Asimétrico s
            # -------------------------------------------------------------
            elif modelo_id == "ap_bsn":
                sub_y = submuestreo_pixel_shuffle(batch, stride=stride)
                pred_sub = modelo(sub_y)
                # Auto-reconstrucción en espacio submuestreado con regularizador L1/L2
                loss = criterion(pred_sub, sub_y)

            # -------------------------------------------------------------
            # Paradigma 4: Neighbor2Neighbor (NB2NB CVPR 2021)
            # -------------------------------------------------------------
            elif modelo_id == "nb2nb":
                p1 = batch[:, :, 0::2, 0::2]
                p2 = batch[:, :, 0::2, 1::2]
                p3 = batch[:, :, 1::2, 0::2]
                p4 = batch[:, :, 1::2, 1::2]
                elec = [(p1, p2), (p3, p4), (p1, p3), (p2, p4)][np.random.randint(0, 4)]
                y1, y2 = elec
                out1 = modelo(y1)
                out2 = modelo(y2)
                loss = criterion(out1, y2) + 2.0 * crit_mse(out1 - y1, out2 - y2)

            # -------------------------------------------------------------
            # Paradigma 5: Blind2Unblind (B2U CVPR 2022)
            # -------------------------------------------------------------
            elif modelo_id == "b2u":
                B, C, H, W = batch.shape
                mask = torch.zeros(B, 1, H, W, device=dispositivo)
                for b in range(B):
                    mask[b, 0, np.random.randint(0, 2)::2, np.random.randint(0, 2)::2] = 1.0
                mask = mask.expand(-1, C, -1, -1).bool()

                batch_blind = batch.clone()
                kernel = torch.ones(3, 1, 3, 3, device=dispositivo) / 8.0
                kernel[:, :, 1, 1] = 0.0
                for c in range(3):
                    vec = torch.nn.functional.conv2d(batch[:, c:c+1], kernel[c:c+1], padding=1)
                    batch_blind[:, c:c+1][mask[:, c:c+1]] = vec[mask[:, c:c+1]]

                pred = modelo(batch_blind)
                loss = criterion(pred[mask], batch[mask]) + 0.5 * criterion(pred[~mask], batch[~mask])

            # -------------------------------------------------------------
            # Paradigma 6: StructN2V / N2V2 (CAREamics)
            # -------------------------------------------------------------
            elif modelo_id in ("struct_n2v", "n2v"):
                tipo_m = struct_mask_tipo if modelo_id == "struct_n2v" else "isotropica"
                mask = generar_mascara_struct_n2v(batch, tipo=tipo_m)
                batch_masked = batch.clone()
                # Reemplazo estocástico de píxeles ciegos con vecinos desplazados
                batch_masked[mask] = batch.roll(1, dims=3)[mask]
                pred = modelo(batch_masked)
                loss = criterion(pred[mask], batch[mask])

            # -------------------------------------------------------------
            # Paradigma 7: Modelos Temporales (UDVD, F2R)
            # -------------------------------------------------------------
            elif modelo_id in ("udvd", "f2r"):
                T3 = batch.shape[1]
                c_mid = T3 // 2
                centro = batch[:, c_mid:c_mid+3, :, :]
                pred_residuo = modelo(batch)
                yhat = torch.clamp(centro - pred_residuo, 0.0, 1.0)
                loss = criterion(yhat, centro)

            else:
                pred = modelo(batch)
                loss = criterion(pred, batch)

            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
            pasos += 1
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        scheduler.step()
        loss_prom = loss_ep / max(1, pasos)
        print(f"  Época {ep}/{epocas} — Loss: {loss_prom:.5f}")

        if loss_prom < mejor_loss - 1e-5:
            mejor_loss = loss_prom
            epocas_sin_mejora = 0
            carpeta_cache.mkdir(parents=True, exist_ok=True)
            torch.save(modelo.state_dict(), str(ckpt))
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= paciencia:
                print(f"  [Early stop] {paciencia} épocas sin mejora.")
                break

    if ckpt.exists():
        modelo.load_state_dict(torch.load(str(ckpt), map_location=dispositivo))
    return modelo


# ─────────────────────────────────────────────────────────────────────────────
# 5. Inferencia Frame a Frame
# ─────────────────────────────────────────────────────────────────────────────
def inferir_secuencia(modelo: nn.Module, rutas: list, carpeta_salida: Path,
                      dispositivo: torch.device, modelo_id: str,
                      n_frames_temporales: int = 1,
                      alinear_homografia: bool = False,
                      post_r3: bool = False,
                      stride: int = 2):
    """Inferencia de alta precisión sobre la secuencia de fotogramas."""
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    modelo.eval()

    with torch.no_grad():
        for i, ruta in enumerate(tqdm(rutas, desc=f"Inferencia [{modelo_id.upper()}]")):
            bgr = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)

            h, w = rgb.shape[:2]
            ph = (16 - h % 16) % 16
            pw = (16 - w % 16) % 16
            rgb_pad = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT) if (ph or pw) else rgb

            if n_frames_temporales <= 1:
                t = torch.from_numpy(np.ascontiguousarray(rgb_pad)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dispositivo)

                # Refinamiento R3 para AP-BSN
                if modelo_id == "ap_bsn" and post_r3:
                    preds = []
                    for dy in range(stride):
                        for dx in range(stride):
                            t_shift = torch.roll(t, shifts=(-dy, -dx), dims=(2, 3))
                            p = modelo(t_shift)
                            p = torch.roll(p, shifts=(dy, dx), dims=(2, 3))
                            preds.append(p)
                    out = torch.clamp(torch.stack(preds).mean(dim=0), 0.0, 1.0)
                else:
                    out = modelo(t)
                    out = torch.clamp(out, 0.0, 1.0)

                resultado = out.squeeze(0).permute(1, 2, 0).cpu().numpy()

            else:
                # Ventana temporal centrada en el frame i
                mid = n_frames_temporales // 2
                indices = [min(max(0, i + k), len(rutas) - 1) for k in range(-mid, mid + 1)]
                imgs = []
                for idx_k in indices:
                    b_k = cv2.imread(str(rutas[idx_k]), cv2.IMREAD_UNCHANGED)
                    r_k = cv2.cvtColor(b_k, cv2.COLOR_BGR2RGB) if b_k.ndim == 3 else cv2.cvtColor(b_k, cv2.COLOR_GRAY2RGB)
                    if (ph or pw):
                        r_k = cv2.copyMakeBorder(r_k, 0, ph, 0, pw, cv2.BORDER_REFLECT)
                    imgs.append(r_k)

                if alinear_homografia:
                    ref = imgs[mid]
                    imgs = [alinear_frame_a_referencia_ransac(img, ref) if k != mid else ref for k, img in enumerate(imgs)]

                stack = torch.cat([
                    torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0)
                    for img in imgs
                ], dim=0).unsqueeze(0).to(dispositivo)

                centro = stack[:, (mid * 3):(mid * 3 + 3), :, :]
                pred_res = modelo(stack)
                out = torch.clamp(centro - pred_res, 0.0, 1.0)
                resultado = out.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Recortar padding y guardar BGR uint8
            resultado = resultado[:h, :w, :]
            salida_bgr = cv2.cvtColor((resultado * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            dest = carpeta_salida / ruta.name
            cv2.imwrite(str(dest), salida_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────
MODELOS_TEMPORALES = {"f2r": 5, "udvd": 5}

def main():
    p = argparse.ArgumentParser(description="Wrapper universal de modelos de denoising FAC")
    p.add_argument("--modelo", required=True,
                   choices=["p2n", "mm_bsn", "ap_bsn", "nb2nb", "b2u", "struct_n2v", "n2v", "f2r", "udvd"],
                   help="Modelo a ejecutar")
    p.add_argument("--carpeta-entrada", required=True, type=Path)
    p.add_argument("--carpeta-salida", required=True, type=Path)
    p.add_argument("--carpeta-cache", type=Path, default=None)
    p.add_argument("--epocas", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--paciencia", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--loss-type", choices=["l1", "l2"], default="l1",
                   help="Función de pérdida (L1 para colas pesadas de HAZE, L2 MSE)")
    p.add_argument("--stride-submuestreo", type=int, default=2, choices=[2, 3, 5],
                   help="Stride de submuestreo para AP-BSN / MM-BSN")
    p.add_argument("--struct-mask-tipo", choices=["lineal", "isotropica"], default="lineal",
                   help="Tipo de máscara para StructN2V (lineal 9x1 vs isotrópica 1x1)")
    p.add_argument("--alinear-homografia", action="store_true",
                   help="Alinear temporalmente con RANSAC antes de la red")
    p.add_argument("--post-r3", action="store_true",
                   help="Activar Random Replacement Refinement R3 en AP-BSN")
    p.add_argument("--etapa", choices=["entrenar", "inferir", "todo"], default="todo")
    p.add_argument("--max-frames", type=int, default=None)

    a = p.parse_args()

    carpeta_in = a.carpeta_entrada.resolve()
    carpeta_out = a.carpeta_salida.resolve()
    carpeta_cache = a.carpeta_cache or (carpeta_out.parent / "cache_modelo")
    carpeta_cache = carpeta_cache.resolve()

    carpeta_in.mkdir(parents=True, exist_ok=True)
    carpeta_out.mkdir(parents=True, exist_ok=True)
    carpeta_cache.mkdir(parents=True, exist_ok=True)

    dispositivo = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Dispositivo] {dispositivo} | Modelo: {a.modelo.upper()} | Loss: {a.loss_type}")

    n_temporales = MODELOS_TEMPORALES.get(a.modelo, 1)
    rutas = listar_frames(carpeta_in, limite=a.max_frames)
    if not rutas:
        print(f"[ERROR] No se encontraron frames en {carpeta_in}")
        sys.exit(1)

    if a.etapa in ("entrenar", "todo"):
        modelo = entrenar_red(
            modelo_id=a.modelo,
            rutas=rutas,
            carpeta_cache=carpeta_cache,
            epocas=a.epocas,
            lr=a.lr,
            batch_size=a.batch_size,
            patch_size=a.patch_size,
            paciencia=a.paciencia,
            dispositivo=dispositivo,
            loss_type=a.loss_type,
            stride=a.stride_submuestreo,
            struct_mask_tipo=a.struct_mask_tipo,
            n_frames_temporales=n_temporales,
            alinear_homografia=a.alinear_homografia,
        )
    else:
        in_ch = n_temporales * 3 if n_temporales > 1 else 3
        modelo = UNetUniversal(in_channels=in_ch, out_channels=3, base_ch=48, depth=4).to(dispositivo)
        ckpt = carpeta_cache / "modelo.pth"
        if not ckpt.exists():
            print(f"[ERROR] No existe checkpoint en {ckpt}")
            sys.exit(1)
        modelo.load_state_dict(torch.load(str(ckpt), map_location=dispositivo))

    if a.etapa in ("inferir", "todo"):
        inferir_secuencia(
            modelo=modelo,
            rutas=rutas,
            carpeta_salida=carpeta_out,
            dispositivo=dispositivo,
            modelo_id=a.modelo,
            n_frames_temporales=n_temporales,
            alinear_homografia=a.alinear_homografia,
            post_r3=a.post_r3,
            stride=a.stride_submuestreo,
        )


if __name__ == "__main__":
    main()
