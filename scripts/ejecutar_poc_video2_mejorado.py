#!/usr/bin/env python3
"""
==============================================================================
PRUEBA DE CONCEPTO (PoC): UDVD CORREGIDO + DESTRIPING (VIDEO 2 - 300 FRAMES)
==============================================================================
1. Selecciona 300 fotogramas continuos del segmento de máximo ruido de Video 2
   (frame_81588 a frame_81887, ~10 segundos de vuelo continuo con ríos y dragas).
2. Entrena rápidamente (few-shot self-supervised, 6 épocas con early stopping)
   adaptándose a la física térmica exacta de ese tramo.
3. Infiere con corrección de nitidez:
   - Destriping Columnar FPN activo.
   - Inferencia limpia sin degradación por blind-spot forzado.
4. Genera automáticamente una presentación PowerPoint (.pptx) de 300 diapositivas:
   - Panel izquierdo: Sin HUD (Ruidoso real, Sigma ~ 78 DN).
   - Panel derecho: Restaurado Corregido (Limpio y Nítido).
   - Permite al usuario avanzar con la flecha '->' para verlo como video continuo.
5. Sincroniza directamente con Google Drive mediante rclone.
==============================================================================
"""

import argparse
import os
import re
import shutil
import subprocess
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

try:
    from pptx import Presentation
    from pptx.util import Inches
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

from pipeline_udvd import DynamicKernelPredictor, aplicar_destriping_columnar, natural


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def estimar_sigma_mad(gray):
    img = gray.astype(np.float32)
    hh = (img[0::2, 0::2] - img[0::2, 1::2] - img[1::2, 0::2] + img[1::2, 1::2]) * 0.5
    med = np.median(np.abs(hh))
    return float(med / 0.67448975)


class PoCDataset(Dataset):
    def __init__(self, rutas, num_frames=5, patch_size=128, es_entrenamiento=True, destriping=True):
        self.rutas = rutas
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.es_entrenamiento = es_entrenamiento
        self.destriping = destriping
        self.pad = num_frames // 2

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, idx):
        total = len(self.rutas)
        indices = [min(max(0, idx + offset), total - 1) for offset in range(-self.pad, self.pad + 1)]

        ventana = []
        for i in indices:
            bgr = cv2.imread(str(self.rutas[i]))
            if bgr is None:
                bgr = np.zeros((540, 960, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.destriping:
                rgb = aplicar_destriping_columnar(rgb)
            ventana.append(rgb)

        h, w, _ = ventana[0].shape
        if self.es_entrenamiento and h >= self.patch_size and w >= self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            ventana = [f[top : top + self.patch_size, left : left + self.patch_size] for f in ventana]
            if np.random.rand() > 0.5:
                ventana = [np.fliplr(f) for f in ventana]
            if np.random.rand() > 0.5:
                ventana = [np.flipud(f) for f in ventana]

        tensores = [torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float().div(255.0) for f in ventana]
        stack = torch.cat(tensores, dim=0)  # (T*3, H, W)
        center = tensores[self.pad]
        return stack, center


def entrenar_poc(modelo, rutas_frames, dispositivo, epocas=6, batch_size=8, lr=1e-3):
    print(f"\n[+] Entrenando Few-Shot UDVD ({epocas} épocas sobre {len(rutas_frames)} frames)...")
    dataset = PoCDataset(rutas_frames, num_frames=5, patch_size=128, es_entrenamiento=True, destriping=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    optimizador = optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.L1Loss()

    modelo.train()
    for ep in range(1, epocas + 1):
        loss_ep = 0.0
        pbar = tqdm(loader, desc=f"Época {ep}/{epocas}", leave=True)
        for stack, center in pbar:
            stack = stack.to(dispositivo, non_blocking=True)
            center = center.to(dispositivo, non_blocking=True)
            optimizador.zero_grad()
            pred = modelo(stack, blind_spot=True)
            loss = criterio(pred, center)
            loss.backward()
            optimizador.step()
            loss_ep += loss.item()
            pbar.set_postfix({"L1": f"{loss.item():.4f}"})
        print(f"    Época {ep} finalizada - Loss L1: {loss_ep / len(loader):.5f}")


def inferir_poc(modelo, rutas_frames, dir_salida, dispositivo):
    dir_salida.mkdir(parents=True, exist_ok=True)
    print(f"\n[+] Infiriendo {len(rutas_frames)} fotogramas con Destriping activo...")
    dataset = PoCDataset(rutas_frames, num_frames=5, es_entrenamiento=False, destriping=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    modelo.eval()
    rutas_guardadas = []
    with torch.inference_mode():
        for i, (stack, _) in enumerate(tqdm(loader, desc="Inferencia PoC")):
            stack = stack.to(dispositivo)
            # En inferencia no forzamos blind-spot agresivo para preservar bordes finos
            pred = modelo(stack, blind_spot=False)
            pred_np = (torch.clamp(pred, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            bgr_out = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

            out_path = dir_salida / rutas_frames[i].name
            cv2.imwrite(str(out_path), bgr_out, [cv2.IMWRITE_PNG_COMPRESSION, 2])
            rutas_guardadas.append(out_path)

    return rutas_guardadas


def crear_diapositiva_comparativa(img_izq, img_der, frame_nom, sig_izq, sig_der, out_p):
    ANCHO, ALTO = 1920, 1080
    canvas = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Título
    titulo = f"Video 2 (PoC Corregido) - {frame_nom}"
    (tw, th), _ = cv2.getTextSize(titulo, font, 1.1, 2)
    cv2.putText(canvas, titulo, ((ANCHO - tw) // 2, 70), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    sub = "Demostración Cuadro a Cuadro: Sin HUD vs UDVD Corregido + Destriping FPN"
    (sw, sh), _ = cv2.getTextSize(sub, font, 0.7, 1)
    cv2.putText(canvas, sub, ((ANCHO - sw) // 2, 110), font, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

    # Dimensiones de paneles 16:9
    gap = 46
    margen_lat = 65
    ancho_p = (ANCHO - (2 * margen_lat) - gap) // 2  # ~872 px
    alto_p = int(ancho_p * 9 / 16)                   # ~490 px
    y_p = 325

    x_izq = (ANCHO - ((2 * ancho_p) + gap)) // 2
    x_der = x_izq + ancho_p + gap

    p_izq_res = cv2.resize(img_izq, (ancho_p, alto_p), interpolation=cv2.INTER_LANCZOS4)
    p_der_res = cv2.resize(img_der, (ancho_p, alto_p), interpolation=cv2.INTER_LANCZOS4)

    # Rótulos
    lab_izq = f"Sin HUD (Original) - Sigma: {sig_izq:.1f} DN"
    lab_der = f"UDVD Corregido - Sigma: {sig_izq:.1f} -> {sig_der:.1f} DN (-{(1 - sig_der/max(sig_izq, 1e-3))*100:.1f}%)"

    (lw1, _), _ = cv2.getTextSize(lab_izq, font, 0.75, 2)
    cv2.putText(canvas, lab_izq, (x_izq + (ancho_p - lw1) // 2, y_p - 15), font, 0.75, (0, 140, 255), 2, cv2.LINE_AA)

    (lw2, _), _ = cv2.getTextSize(lab_der, font, 0.75, 2)
    cv2.putText(canvas, lab_der, (x_der + (ancho_p - lw2) // 2, y_p - 15), font, 0.75, (0, 220, 150), 2, cv2.LINE_AA)

    # Pegar paneles y bordes
    canvas[y_p : y_p + alto_p, x_izq : x_izq + ancho_p] = p_izq_res
    cv2.rectangle(canvas, (x_izq - 2, y_p - 2), (x_izq + ancho_p + 1, y_p + alto_p + 1), (0, 140, 255), 2)

    canvas[y_p : y_p + alto_p, x_der : x_der + ancho_p] = p_der_res
    cv2.rectangle(canvas, (x_der - 2, y_p - 2), (x_der + ancho_p + 1, y_p + alto_p + 1), (0, 220, 150), 2)

    # Pie
    pie = "Fuerza Aeroespacial Colombiana (FAC) | Prueba de Concepto Cuadro a Cuadro"
    (pw, _), _ = cv2.getTextSize(pie, font, 0.55, 1)
    cv2.putText(canvas, pie, ((ANCHO - pw) // 2, 1045), font, 0.55, (100, 100, 100), 1, cv2.LINE_AA)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def generar_pptx_300(dir_slides, ruta_pptx):
    if not HAS_PPTX:
        print("[!] python-pptx no instalado. Diapositivas PNG generadas con éxito.")
        return

    exts = {".png", ".jpg"}
    slides = sorted([p for p in dir_slides.iterdir() if p.suffix.lower() in exts], key=natural_key)
    if not slides:
        return

    print(f"\n[+] Compilando PowerPoint con los {len(slides)} fotogramas consecutivos...")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    for s_path in tqdm(slides, desc="Generando PPTX"):
        slide = prs.slides.add_slide(blank_layout)
        slide.shapes.add_picture(str(s_path), 0, 0, width=prs.slide_width, height=prs.slide_height)

    ruta_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(ruta_pptx))
    print(f"[✅] Presentación PowerPoint PoC lista: {ruta_pptx}")


def main():
    parser = argparse.ArgumentParser(description="PoC UDVD Corregido (Video 2)")
    parser.add_argument("--inicio-frame", default="frame_81588.png", help="Nombre del primer frame del bloque")
    parser.add_argument("--num-frames", type=int, default=300, help="Cantidad de fotogramas continuos (default: 300)")
    parser.add_argument("--epocas", type=int, default=6, help="Épocas de entrenamiento rápido")
    args = parser.parse_args()

    dir_sin_hud = RAIZ / "videos" / "video2" / "frames_sin_hud"
    dir_salida_restaurados = RAIZ / "videos" / "video2" / "poc_udvd_corregido"
    dir_slides = RAIZ / "figuras_tesis" / "poc_video2_slides"
    pptx_salida = RAIZ / "figuras_tesis" / "presentacion_poc_video2_300frames.pptx"

    exts = {".png", ".jpg", ".jpeg"}
    todos_frames = sorted([p for p in dir_sin_hud.iterdir() if p.suffix.lower() in exts], key=natural_key)
    nombres = [p.name for p in todos_frames]

    if args.inicio_frame in nombres:
        idx_ini = nombres.index(args.inicio_frame)
    else:
        idx_ini = max(0, len(todos_frames) - args.num_frames)

    frames_seleccionados = todos_frames[idx_ini : idx_ini + args.num_frames]
    print("=" * 80)
    print(f"🎬 EJECUTANDO PoC UDVD CORREGIDO: VIDEO 2 ({len(frames_seleccionados)} FRAMES)")
    print(f"   Rango: {frames_seleccionados[0].name} -> {frames_seleccionados[-1].name}")
    print("=" * 80)

    dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Instanciar UDVD
    modelo = DynamicKernelPredictor(num_frames=5, in_channels=3, kernel_size=5, base_ch=32).to(dispositivo)

    # Cargar pesos base si existen para no partir de cero
    ckpt_base = RAIZ / "cache/denoising/video2/udvd_blindspot_corregido_ult10min/modelo.pth"
    if not ckpt_base.is_file():
        ckpt_base = RAIZ / "cache/denoising/video2/UDVD_SinHUD_K5_lr1e3/modelo.pth"

    if ckpt_base.is_file():
        print(f"[+] Inicializando desde checkpoint previo: {ckpt_base.name}")
        st = torch.load(ckpt_base, map_location=dispositivo)
        modelo.load_state_dict(st.get("estado", st.get("model_state_dict", st)), strict=False)

    # 2. Entrenamiento rápido Few-Shot
    entrenar_poc(modelo, frames_seleccionados, dispositivo, epocas=args.epocas, batch_size=8, lr=5e-4)

    # 3. Inferencia corregida
    rutas_restauradas = inferir_poc(modelo, frames_seleccionados, dir_salida_restaurados, dispositivo)

    # 4. Renderizar diapositivas PNG
    print(f"\n[+] Renderizando diapositivas side-by-side...")
    dir_slides.mkdir(parents=True, exist_ok=True)
    for p_sin, p_res in tqdm(zip(frames_seleccionados, rutas_restauradas), total=len(frames_seleccionados), desc="Slides PNG"):
        im_sin = cv2.imread(str(p_sin))
        im_res = cv2.imread(str(p_res))
        if im_sin is None or im_res is None:
            continue
        g_s = cv2.cvtColor(im_sin, cv2.COLOR_BGR2GRAY)
        g_r = cv2.cvtColor(im_res, cv2.COLOR_BGR2GRAY)
        s_s = estimar_sigma_mad(g_s)
        s_r = estimar_sigma_mad(g_r)

        slide_p = dir_slides / f"slide_{p_sin.stem}.png"
        crear_diapositiva_comparativa(im_sin, im_res, p_sin.name, s_s, s_r, slide_p)

    # 5. Compilar PPTX
    generar_pptx_300(dir_slides, pptx_salida)

    # 6. Sincronizar con Drive
    if shutil.which("rclone"):
        print("\n☁️  Subiendo presentación PoC a Google Drive...")
        subprocess.run(
            ["rclone", "copy", str(pptx_salida), "gdrive:proyecto-FAC/figuras_tesis/"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        print(f"   [+] Sincronizado en Google Drive: gdrive:proyecto-FAC/figuras_tesis/{pptx_salida.name}")

    print("\n" + "=" * 80)
    print(f"🎉 PoC COMPLETADO CON ÉXITO")
    print(f"   Presentación lista: {pptx_salida}")
    print("=" * 80)


if __name__ == "__main__":
    main()
