#!/usr/bin/env python3
"""
Genera mosaicos comparativos específicamente para los frames de Video 1 (11min):
[Original con Cajas YOLO] | [Sin HUD (ProPainter)] | [Sin HUD + Denoised (UDVD)]
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent

COLORES_CLASES = {
    0: (0, 255, 255),    # vehicle (amarillo)
    1: (255, 100, 0),    # building (azul claro)
    2: (0, 255, 0),      # road (verde)
    3: (255, 0, 0),      # river (azul)
    4: (0, 0, 255),      # SDZI (rojo)
}

NOMBRES_CLASES = {
    0: "Vehicle",
    1: "Building",
    2: "Road",
    3: "River",
    4: "SDZI",
}


def dibujar_cajas_yolo(img_bgr, ruta_txt):
    """Dibuja las cajas delimitadoras de YOLO con etiquetas sobre la imagen."""
    img_draw = img_bgr.copy()
    h, w = img_draw.shape[:2]

    if not Path(ruta_txt).is_file():
        return img_draw

    with open(ruta_txt, "r", encoding="utf-8") as f:
        for linea in f:
            partes = linea.strip().split()
            if len(partes) < 5:
                continue

            cls_id = int(partes[0])
            xc, yc, bw, bh = map(float, partes[1:5])

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = COLORES_CLASES.get(cls_id, (0, 255, 0))
            nombre = NOMBRES_CLASES.get(cls_id, f"Clase_{cls_id}")

            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_draw, nombre, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return img_draw


def crear_mosaico_comparativo(img_orig_boxes, img_sin_hud, img_udvd, titulo_frame=""):
    """Une horizontalmente las 3 etapas metodológicas."""
    h, w = img_orig_boxes.shape[:2]
    margen_superior = 60
    canvas = np.zeros((h + margen_superior, w * 3, 3), dtype=np.uint8)
    canvas.fill(24)

    canvas[margen_superior:, 0:w] = img_orig_boxes
    canvas[margen_superior:, w:2*w] = img_sin_hud
    canvas[margen_superior:, 2*w:] = img_udvd

    # Líneas divisorias
    cv2.line(canvas, (w, 0), (w, h + margen_superior), (80, 80, 80), 2)
    cv2.line(canvas, (2*w, 0), (2*w, h + margen_superior), (80, 80, 80), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"1. Original Con HUD (YOLO) [{titulo_frame}]", (20, 38), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "2. Preprocesado (Sin HUD - ProPainter)", (w + 20, 38), font, 0.8, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(canvas, "3. Restaurado (Sin HUD + Denoised UDVD)", (2*w + 20, 38), font, 0.8, (100, 200, 255), 2, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generar Mosaicos de Video 1 (11min)")
    parser.add_argument("--dataset-dir", default="datasets/ablation_splits", help="Directorio raíz de datasets")
    parser.add_argument("--salida-dir", default="figuras_tesis/mosaicos_video1", help="Directorio de salida para mosaicos de Video 1")
    parser.add_argument("--num-ejemplos", type=int, default=5, help="Cantidad de ejemplos a generar")
    args = parser.parse_args()

    dir_salida = (RAIZ / args.salida_dir).resolve()
    dir_salida.mkdir(parents=True, exist_ok=True)

    dir_v1_orig = RAIZ / args.dataset_dir / "1_originales/test/images"
    dir_v1_sinhud = RAIZ / args.dataset_dir / "2_sin_hud/test/images"
    dir_v1_udvd = RAIZ / args.dataset_dir / "3_denoised_udvd/test/images"
    dir_v1_labels = RAIZ / args.dataset_dir / "1_originales/test/labels"

    # Si no están en ablation_splits, buscar en dataset_split_completo y mapear directo
    v1_sin_hud_repo = RAIZ / "videos/video1/frames_sin_hud"
    v1_udvd_repo = RAIZ / "videos/video1/expos/udvd_sinhud"
    if not v1_udvd_repo.is_dir():
        v1_udvd_repo = RAIZ / "videos/video1/UDVD_SinHUD_K5_lr1e3"

    # Buscar imágenes de video_11min en test
    if dir_v1_orig.is_dir():
        candidatos = sorted([p for p in dir_v1_orig.iterdir() if "11min" in p.name])
    else:
        candidatos = []

    if not candidatos:
        # Buscar en dataset_split_completo
        cands_base = RAIZ / "datasets/dataset_split_completo/test/images"
        if cands_base.is_dir():
            candidatos = sorted([p for p in cands_base.iterdir() if "11min" in p.name])
            dir_v1_labels = RAIZ / "datasets/dataset_split_completo/test/labels"

    print("=" * 80)
    print(f"GENERANDO MOSAICOS PARA VIDEO 1 ({len(candidatos)} candidatos disponibles en Test)")
    print("=" * 80)

    # Seleccionar N ejemplos distribuidos en el video
    paso = max(1, len(candidatos) // args.num_ejemplos)
    seleccionados = candidatos[::paso][:args.num_ejemplos]

    for p_orig in seleccionados:
        stem = p_orig.stem
        lbl_p = dir_v1_labels / f"{stem}.txt"
        
        # Calcular frame exacto en video 1
        partes = stem.split("_")
        sec_idx = int(partes[-1]) if len(partes) >= 3 and partes[-1].isdigit() else 1
        f_idx = sec_idx * 30

        # Buscar frame en sin_hud y udvd
        p_sh = None
        p_ud = None

        # Opción 1: desde ablation_splits
        if (dir_v1_sinhud / p_orig.name).is_file():
            p_sh = dir_v1_sinhud / p_orig.name
        if (dir_v1_udvd / p_orig.name).is_file():
            p_ud = dir_v1_udvd / p_orig.name

        # Opción 2: buscar directo en videos/video1/ con 5 dígitos
        formatos = [f"frame_{f_idx:05d}.png", f"frame_{f_idx+1:05d}.png", f"frame_{f_idx-1:05d}.png", f"frame_{f_idx:06d}.png"]
        for fmt in formatos:
            if p_sh is None or not p_sh.is_file():
                cand_sh = v1_sin_hud_repo / fmt
                if cand_sh.is_file():
                    p_sh = cand_sh
            if p_ud is None or not p_ud.is_file():
                cand_ud = v1_udvd_repo / fmt
                if cand_ud.is_file():
                    p_ud = cand_ud

        img_o = cv2.imread(str(p_orig))
        if img_o is None:
            continue
        h, w = img_o.shape[:2]

        img_s = cv2.imread(str(p_sh)) if p_sh and p_sh.is_file() else img_o.copy()
        img_u = cv2.imread(str(p_ud)) if p_ud and p_ud.is_file() else img_s.copy()

        if img_s.shape[:2] != (h, w):
            img_s = cv2.resize(img_s, (w, h), interpolation=cv2.INTER_CUBIC)
        if img_u.shape[:2] != (h, w):
            img_u = cv2.resize(img_u, (w, h), interpolation=cv2.INTER_CUBIC)

        img_boxes = dibujar_cajas_yolo(img_o, lbl_p)
        mosaico = crear_mosaico_comparativo(img_boxes, img_s, img_u, titulo_frame=f"{p_orig.name} -> Frame {f_idx}")

        salida_mosaico = dir_salida / f"mosaico_v1_{stem}.png"
        cv2.imwrite(str(salida_mosaico), mosaico, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        print(f"  [+] Generado mosaico Video 1: {salida_mosaico.name} (Frame={f_idx}, SH={p_sh is not None}, UDVD={p_ud is not None})")

    print("\n" + "=" * 80)
    print(f"Mosaicos de Video 1 guardados en: {dir_salida}")
    print("=" * 80)


if __name__ == "__main__":
    main()
