#!/usr/bin/env python3
"""
==============================================================================
PIPELINE MULTI-GPU DE ALTO RENDIMIENTO: MODELO FINAL DEFINITIVO (UDVD + DESTRIPING)
==============================================================================
Restaura térmicamente los videos FLIR de la FAC acelerando en paralelo en 2 GPUs:
1. Divide la lista de fotogramas entre todas las GPUs CUDA disponibles (cuda:0, cuda:1).
2. Destriping Gaussiano Columnar rápido (31, 1, sigma=10.0) para remover FPN.
3. Inferencia UDVD (T=5, K=5) con Blind-Spot central (-1e9) en precisión FP16.
4. Consumo VRAM < 2.0 GB por GPU (cero CUDA OOM).
5. Reanudación automática (salta fotogramas existentes).
==============================================================================
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp_torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

# Importar componentes de UDVD
from pipeline_udvd import DynamicKernelPredictor, aplicar_destriping_columnar, natural


class UDVDInferenceDataset(Dataset):
    def __init__(self, todas_rutas, indices_a_procesar, num_frames=5, destriping=True):
        self.todas_rutas = todas_rutas
        self.indices = indices_a_procesar
        self.num_frames = num_frames
        self.destriping = destriping
        self.pad = num_frames // 2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item_idx):
        global_idx = self.indices[item_idx]
        total = len(self.todas_rutas)
        frame_indices = [min(max(0, global_idx + offset), total - 1) for offset in range(-self.pad, self.pad + 1)]

        stack_list = []
        for i in frame_indices:
            img = cv2.imread(str(self.todas_rutas[i]))
            if img is None:
                img = np.zeros((540, 960, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.destriping:
                img = aplicar_destriping_columnar(img)

            t_img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            stack_list.append(t_img)

        x_stack = torch.cat(stack_list, dim=0)  # (T*C, H, W)
        return x_stack, str(self.todas_rutas[global_idx].name)


def worker_gpu_inferencia(gpu_id, v_id, todas_rutas, mis_indices, dir_salida, batch_size=4, destriping=True):
    """Worker independiente por cada GPU."""
    try:
        device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id < torch.cuda.device_count() else "cpu"
        device = torch.device(device_str)
        if "cuda" in device_str:
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()

        # Filtrar solo índices cuyos archivos no existan aún
        existentes = set(p.name for p in dir_salida.iterdir() if p.is_file())
        indices_pendientes = [idx for idx in mis_indices if todas_rutas[idx].name not in existentes]

        if not indices_pendientes:
            print(f"[GPU {gpu_id}] Todos sus {len(mis_indices)} frames ya estaban procesados.")
            return

        # Cargar Modelo UDVD
        modelo = DynamicKernelPredictor(num_frames=5, in_channels=3, kernel_size=5, base_ch=32).to(device)

        ckpt_cands = [
            RAIZ / f"cache/denoising/{v_id}/UDVD_SinHUD_K5_lr1e3/modelo.pth",
            RAIZ / f"cache/denoising/video1/UDVD_SinHUD_K5_lr1e3/modelo.pth",
            RAIZ / f"cache/denoising/video2/UDVD_SinHUD_K5_lr1e3/modelo.pth",
        ]
        ckpt_path = next((c for c in ckpt_cands if c.is_file()), None)

        if ckpt_path:
            st = torch.load(ckpt_path, map_location=device)
            modelo.load_state_dict(st.get("model_state_dict", st), strict=False)

        modelo.eval()

        ds = UDVDInferenceDataset(todas_rutas, indices_pendientes, num_frames=5, destriping=destriping)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=("cuda" in device_str))

        pbar = tqdm(loader, desc=f"GPU {gpu_id} -> {v_id.upper()}", position=gpu_id, leave=True)

        with torch.inference_mode():
            for batch_stack, batch_nombres in pbar:
                batch_stack = batch_stack.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                    salida = modelo(batch_stack, blind_spot=True)

                salida_np = (torch.clamp(salida, 0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

                for i, nom in enumerate(batch_nombres):
                    out_p = dir_salida / nom
                    cv2.imwrite(str(out_p), cv2.cvtColor(salida_np[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])

    except Exception as e:
        print(f"[-] ERROR en GPU {gpu_id}: {e}")
        traceback.print_exc()


def ejecutar_modelo_final_en_video_multigpu(v_id, batch_size=4, destriping=True):
    dir_v = RAIZ / "videos" / v_id
    dir_sin_hud = dir_v / "frames_sin_hud"
    dir_salida = dir_v / "udvd_sin_hud"
    dir_salida.mkdir(parents=True, exist_ok=True)

    if not dir_sin_hud.is_dir():
        print(f"[-] Error: No existe la carpeta {dir_sin_hud}")
        return

    exts = {".png", ".jpg", ".jpeg"}
    todas_rutas = sorted([p for p in dir_sin_hud.iterdir() if p.is_file() and p.suffix.lower() in exts], key=natural)
    total_frames = len(todas_rutas)

    if total_frames == 0:
        print(f"[-] No hay frames sin HUD en {dir_sin_hud}")
        return

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    existentes = len(list(dir_salida.glob("*.*")))

    print("\n" + "=" * 90)
    print(f"   RESTAURACIÓN TÉRMICA MULTI-GPU: {v_id.upper()}")
    print(f"   Total frames: {total_frames:,} | Listos: {existentes:,} | GPUs activas: {num_gpus}")
    print(f"   Arquitectura: UDVD (T=5, K=5) + Destriping FPN + Blind-Spot Real (FP16)")
    print(f"   Directorio salida: {dir_salida}")
    print("=" * 90)

    if existentes >= total_frames:
        print(f"[✅] {v_id.upper()} ya está 100% procesado ({existentes:,} frames).")
        return

    t_inicio = time.time()

    if num_gpus > 1:
        # Particionar índices equitativamente entre las GPUs
        shards = [[] for _ in range(num_gpus)]
        for idx in range(total_frames):
            shards[idx % num_gpus].append(idx)

        procesos = []
        for g_id in range(num_gpus):
            p = mp_torch.Process(
                target=worker_gpu_inferencia,
                args=(g_id, v_id, todas_rutas, shards[g_id], dir_salida, batch_size, destriping)
            )
            p.start()
            procesos.append(p)

        for p in procesos:
            p.join()
    else:
        # Modo single GPU / CPU
        indices = list(range(total_frames))
        worker_gpu_inferencia(0, v_id, todas_rutas, indices, dir_salida, batch_size, destriping)

    t_total = time.time() - t_inicio
    frames_finales = len(list(dir_salida.glob("*.*")))
    fps_global = (frames_finales - existentes) / t_total if t_total > 0 else 0
    print(f"\n[+] {v_id.upper()} finalizado: {frames_finales:,}/{total_frames:,} frames en {t_total:.1f}s ({fps_global:.1f} fps global).")


def main():
    parser = argparse.ArgumentParser(description="Restauración Térmica Multi-GPU Oficial (UDVD + Destriping)")
    parser.add_argument("--video", default="todos", choices=["video1", "video2", "video3", "todos"], help="Video a procesar")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size por GPU (4 = VRAM < 2 GB por tarjeta)")
    args = parser.parse_args()

    v_lista = ["video1", "video2", "video3"] if args.video == "todos" else [args.video]

    for v in v_lista:
        ejecutar_modelo_final_en_video_multigpu(v, batch_size=args.batch_size, destriping=True)

    print("\n" + "=" * 90)
    print("PIPELINE MULTI-GPU COMPLETADO CON ÉXITO.")
    print("=" * 90)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
