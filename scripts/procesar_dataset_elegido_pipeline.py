#!/usr/bin/env python3
"""
Pipeline Completo de Restauración para el Dataset Elegido de la Tesis:
Aplica la cadena metodológica completa sobre las imágenes de Train, Val y Test del dataset seleccionado:
1. Rama 1: Original (Cruda con HUD y Ruido).
2. Rama 2: Sin HUD (Inpainting morfológico fino de telemetría y HUD verde).
3. Rama 3: Denoised UDVD Estándar (Dynamic Kernels 5x5, T=5).
4. Rama 4: Denoised UDVD Mejorado (Dynamic Kernels + Destriping Columnar Anti-FPN).
5. Generación automática de dataset.yaml para cada rama.
"""

import argparse
from datetime import datetime
import gc
import os
from pathlib import Path
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent


# 1. Destriping Columnar Anti-FPN
def aplicar_destriping(img_rgb):
    img_float = img_rgb.astype(np.float32)
    perfil = np.mean(img_float, axis=0, keepdims=True)
    perfil_suave = cv2.GaussianBlur(perfil, (31, 1), 10.0)
    fpn = perfil - perfil_suave
    return np.clip(img_float - fpn, 0.0, 255.0).astype(np.uint8)


# 2. Arquitectura UDVD
class DynamicKernelPredictor(nn.Module):
    def __init__(self, num_frames=5, in_channels=3, kernel_size=5, base_ch=32):
        super().__init__()
        self.num_frames = num_frames
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        in_total = num_frames * in_channels
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
        b, tc, h, w = x_stack.shape
        raw_kernels = self.net(x_stack)
        t_center = self.num_frames // 2
        k_center = (self.kernel_size * self.kernel_size) // 2
        idx_centro = t_center * (self.kernel_size * self.kernel_size) + k_center
        mascara = torch.ones_like(raw_kernels)
        mascara[:, idx_centro : idx_centro + 1, :, :] = 0.0
        raw_kernels = raw_kernels.masked_fill(mascara == 0.0, -1e9)
        kernels = torch.softmax(raw_kernels, dim=1)

        pad = self.kernel_size // 2
        x_pad = torch.nn.functional.pad(x_stack, (pad, pad, pad, pad), mode="reflect")
        frames_unfolded = []
        for t in range(self.num_frames):
            f_t = x_pad[:, t * self.in_channels : (t + 1) * self.in_channels, :, :]
            unfold_t = torch.nn.functional.unfold(f_t, kernel_size=self.kernel_size)
            frames_unfolded.append(unfold_t.view(b, self.in_channels, self.kernel_size * self.kernel_size, h * w))

        all_unfolded = torch.cat(frames_unfolded, dim=2)
        kernels_flat = kernels.view(b, 1, self.num_frames * self.kernel_size * self.kernel_size, h * w)
        filtered = (all_unfolded * kernels_flat).sum(dim=2).view(b, self.in_channels, h, w)
        return filtered


def cargar_modelo_udvd(dispositivo):
    modelo = DynamicKernelPredictor(num_frames=5, in_channels=3, kernel_size=5, base_ch=32).to(dispositivo)
    # Cargar checkpoint entrenado si existe
    ckpt_path = RAIZ / "cache/denoising/video2/udvd/modelo.pth"
    if not ckpt_path.is_file():
        ckpt_path = RAIZ / "cache/denoising/video1/UDVD_SinHUD_K5_lr1e3/modelo.pth"
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=dispositivo)
        modelo.load_state_dict(ckpt.get("estado", ckpt))
        print(f"[+] Pesos UDVD cargados exitosamente desde: {ckpt_path.name}")
    modelo.eval()
    return modelo


def crear_yaml(dir_rama, nombre_yaml, clases):
    data_dict = {
        "path": str(dir_rama.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(clases),
        "names": clases,
    }
    ruta_y = dir_rama / nombre_yaml
    with open(ruta_y, "w", encoding="utf-8") as f:
        yaml.dump(data_dict, f, sort_keys=False)
    return ruta_y


def procesar_split(split, dir_in, ramas, modelo_udvd, dispositivo, batch_size=16):
    dir_imgs = dir_in / split / "images"
    dir_lbls = dir_in / split / "labels"

    if not dir_imgs.is_dir():
        return

    for r_dir in ramas.values():
        (r_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (r_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    archivos = sorted(list(dir_imgs.glob("*.jpg")) + list(dir_imgs.glob("*.png")))
    total_imgs = len(archivos)
    print(f"\nProcesando Split '{split}' ({total_imgs} imágenes)...")

    # Procesar en bloques para acelerar inferencia GPU
    for i in tqdm(range(0, total_imgs, batch_size), desc=f"Split {split}"):
        batch_archivos = archivos[i : i + batch_size]
        
        bgr_list = []
        sin_hud_list = []
        destriped_list = []
        valid_indices = []

        for b_idx, img_p in enumerate(batch_archivos):
            stem = img_p.stem
            lbl_p = dir_lbls / f"{stem}.txt"

            # 1. Copiar etiquetas
            for r_dir in ramas.values():
                dest_l = r_dir / split / "labels" / f"{stem}.txt"
                if lbl_p.is_file() and not dest_l.is_file():
                    shutil.copy2(lbl_p, dest_l)

            # 2. Rama 1: Original
            dest_img_1 = ramas["1_original"] / split / "images" / img_p.name
            if not dest_img_1.is_file():
                shutil.copy2(img_p, dest_img_1)

            # Cargar imagen
            bgr = cv2.imread(str(img_p))
            if bgr is None:
                continue

            # 3. Rama 2: Inpainting Sin HUD
            dest_img_2 = ramas["2_sin_hud"] / split / "images" / img_p.name
            if dest_img_2.is_file():
                img_sin_hud_bgr = cv2.imread(str(dest_img_2))
            else:
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                mask_hud = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([95, 255, 255]))
                mask_hud = cv2.dilate(mask_hud, np.ones((3, 3), np.uint8), iterations=1)
                img_sin_hud_bgr = cv2.inpaint(bgr, mask_hud, 3, cv2.INPAINT_TELEA)
                cv2.imwrite(str(dest_img_2), img_sin_hud_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            rgb_sin_hud = cv2.cvtColor(img_sin_hud_bgr, cv2.COLOR_BGR2RGB)
            rgb_destriped = aplicar_destriping(rgb_sin_hud)

            bgr_list.append(bgr)
            sin_hud_list.append(rgb_sin_hud)
            destriped_list.append(rgb_destriped)
            valid_indices.append((img_p.name, rgb_sin_hud.shape[:2]))

        if not sin_hud_list:
            continue

        # Inferencia UDVD en Batch
        with torch.inference_mode():
            # Pad a multiplo de 16
            h, w = sin_hud_list[0].shape[:2]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            # Preparar Batch Rama 3 (UDVD Standard)
            tensors_std = []
            for rgb_img in sin_hud_list:
                rgb_pad = cv2.copyMakeBorder(rgb_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                t_img = torch.from_numpy(np.ascontiguousarray(rgb_pad)).permute(2, 0, 1).float().div(255.0)
                stack_5 = t_img.repeat(5, 1, 1)  # 15, H, W
                tensors_std.append(stack_5)
            batch_std_tensor = torch.stack(tensors_std, dim=0).to(dispositivo)

            out_std_batch = modelo_udvd(batch_std_tensor)
            out_std_batch = torch.clamp(out_std_batch, 0.0, 1.0).cpu().numpy()

            # Preparar Batch Rama 4 (UDVD Mejorado)
            tensors_enh = []
            for rgb_dest in destriped_list:
                rgb_pad_dest = cv2.copyMakeBorder(rgb_dest, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                t_img_dest = torch.from_numpy(np.ascontiguousarray(rgb_pad_dest)).permute(2, 0, 1).float().div(255.0)
                stack_5_dest = t_img_dest.repeat(5, 1, 1)
                tensors_enh.append(stack_5_dest)
            batch_enh_tensor = torch.stack(tensors_enh, dim=0).to(dispositivo)

            out_enh_batch = modelo_udvd(batch_enh_tensor)
            out_enh_batch = torch.clamp(out_enh_batch, 0.0, 1.0).cpu().numpy()

            # Guardar salidas
            for b_idx, (fname, (orig_h, orig_w)) in enumerate(valid_indices):
                # Rama 3
                dest_img_3 = ramas["3_udvd_standard"] / split / "images" / fname
                if not dest_img_3.is_file():
                    img_std = out_std_batch[b_idx].transpose(1, 2, 0)[:orig_h, :orig_w, :]
                    bgr_std = cv2.cvtColor((img_std * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(dest_img_3), bgr_std, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Rama 4
                dest_img_4 = ramas["4_udvd_mejorado"] / split / "images" / fname
                if not dest_img_4.is_file():
                    img_enh = out_enh_batch[b_idx].transpose(1, 2, 0)[:orig_h, :orig_w, :]
                    bgr_enh = cv2.cvtColor((img_enh * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(dest_img_4), bgr_enh, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    parser = argparse.ArgumentParser(description="Procesar Dataset Elegido con Pipeline UDVD y HUD")
    parser.add_argument("--dataset-in", default="datasets/dataset_preprocesado_11gb/modelo_yolov11_dataset_completo_preprocesado", help="Ruta al dataset de entrada")
    parser.add_argument("--salida-dir", default="datasets/dataset_ablation_final", help="Directorio raíz para las ramas de ablación generadas")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Dispositivo")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamaño de batch para inferencia rápida en GPU")
    args = parser.parse_args()

    dir_in = (RAIZ / args.dataset_in).resolve()
    dir_out_raiz = (RAIZ / args.salida_dir).resolve()

    yaml_in = dir_in / "dataset.yaml"
    if not yaml_in.is_file():
        cands = list(dir_in.glob("**/dataset.yaml"))
        if cands:
            yaml_in = cands[0]
            dir_in = yaml_in.parent
        else:
            print(f"ERROR: No se encontró dataset.yaml en {dir_in}")
            return

    with open(yaml_in, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    clases = config.get("names", {0: "Vehiculos", 1: "Bodegas", 2: "Caminos", 3: "Rios", 4: "Mineria"})

    # Definir 4 ramas
    ramas = {
        "1_original": dir_out_raiz / "1_originales",
        "2_sin_hud": dir_out_raiz / "2_sin_hud",
        "3_udvd_standard": dir_out_raiz / "3_denoised_udvd_standard",
        "4_udvd_mejorado": dir_out_raiz / "4_denoised_udvd_mejorado",
    }

    print("=" * 85)
    print("      PIPELINE DE TRANSFORMACIÓN DE ABLACIÓN (HUD + UDVD ESTÁNDAR / MEJORADO)")
    print(f"Dataset Base: {dir_in}")
    print(f"Destino:      {dir_out_raiz}")
    print(f"Dispositivo:  {args.device} | Batch Size: {args.batch_size}")
    print("=" * 85)

    dispositivo = torch.device(args.device)
    modelo_udvd = cargar_modelo_udvd(dispositivo)

    # Procesar Test, Val y Train
    splits = ["test", "val", "train"]
    for split in splits:
        procesar_split(split, dir_in, ramas, modelo_udvd, dispositivo, batch_size=args.batch_size)

    # Crear data.yaml en cada rama
    for r_name, r_dir in ramas.items():
        y_p = crear_yaml(r_dir, f"dataset_{r_name}.yaml", clases)
        print(f"[OK] YAML generado: {y_p}")

    print("\n" + "=" * 85)
    print("PIPELINE DE ABLACIÓN COMPLETADO EXITOSAMENTE PARA LAS 4 RAMAS.")
    print("=" * 85)


if __name__ == "__main__":
    main()
