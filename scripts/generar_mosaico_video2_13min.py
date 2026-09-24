#!/usr/bin/env python3
"""
Genera mosaicos comparativos de confirmación para Video 2 (13min / VideoCodefest):
[Original con Cajas YOLO (video_13min)] | [Video 2 Sin HUD (ProPainter)] | [Video 2 Denoised (UDVD)]
Offset exacto: segundo 17:48:26 (frame 59310 de VideoCodefest_008-46min.avi)
"""

import argparse
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


def crear_mosaico_comparativo(img_orig_boxes, img_sin_hud, img_udvd, titulo=""):
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
    cv2.putText(canvas, f"1. Original (Con HUD + Cajas YOLO) [{titulo}]", (20, 38), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "2. Preprocesado (Sin HUD - ProPainter)", (w + 20, 38), font, 0.8, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(canvas, "3. Restaurado (Sin HUD + Denoised UDVD)", (2*w + 20, 38), font, 0.8, (100, 200, 255), 2, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generar Mosaicos de Confirmación Video 2 (13min)")
    parser.add_argument("--dataset-base", default="datasets/dataset_split_completo", help="Ruta al dataset")
    parser.add_argument("--salida-dir", default="figuras_tesis/mosaicos_video2", help="Carpeta de salida")
    args = parser.parse_args()

    dir_base = (RAIZ / args.dataset_base).resolve()
    dir_salida = (RAIZ / args.salida_dir).resolve()
    dir_salida.mkdir(parents=True, exist_ok=True)

    v2_sin_hud = RAIZ / "videos/video2/frames_sin_hud"
    v2_udvd = RAIZ / "videos/video2/expos/udvd_sin_hud"
    if not v2_udvd.is_dir():
        v2_udvd = RAIZ / "videos/video2/udvd_sin_hud"

    # Buscar imágenes de video_13min
    imgs_13min = sorted(list(dir_base.glob("**/video_13min_*.jpg")))
    print("=" * 80)
    print(f"ENCONTRADAS {len(imgs_13min)} IMÁGENES DE video_13min EN EL DATASET")
    print(f"Video 2 Sin HUD: {v2_sin_hud} (Existe: {v2_sin_hud.is_dir()})")
    print(f"Video 2 UDVD:    {v2_udvd} (Existe: {v2_udvd.is_dir()})")
    print("=" * 80)

    # Offset base: 59310 frames (segundo 1977 = 17:48:26)
    FRAME_OFFSET_BASE = 59310

    # Seleccionar 5 ejemplos variados (ej: 050, 100, 200, 300, 400)
    indices_muestra = [50, 100, 200, 300, 400]

    for num in indices_muestra:
        # Buscar el archivo correspondiente
        cands = [p for p in imgs_13min if f"video_13min_{num:03d}.jpg" in p.name]
        if not cands:
            continue
        img_p = cands[0]
        lbl_p = img_p.parent.parent / "labels" / f"{img_p.stem}.txt"

        # Frame exacto en Video 2 (a 30 fps)
        f_idx = FRAME_OFFSET_BASE + (num - 1) * 30

        # Formatos de frame
        p_sh = None
        p_ud = None
        for fmt in [f"frame_{f_idx:05d}.png", f"frame_{f_idx:06d}.png", f"frame_{f_idx+1:05d}.png", f"frame_{f_idx-1:05d}.png"]:
            if (v2_sin_hud / fmt).is_file():
                p_sh = v2_sin_hud / fmt
            if (v2_udvd / fmt).is_file():
                p_ud = v2_udvd / fmt

        img_o = cv2.imread(str(img_p))
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
        mosaico = crear_mosaico_comparativo(img_boxes, img_s, img_u, titulo=f"{img_p.name} -> Frame {f_idx}")

        out_path = dir_salida / f"mosaico_v2_{img_p.stem}.png"
        cv2.imwrite(str(out_path), mosaico, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        print(f"  [+] Generado: {out_path.name} (Frame={f_idx}, SH={p_sh is not None}, UDVD={p_ud is not None})")

    print("\n" + "=" * 80)
    print(f"Mosaicos de Video 2 guardados en: {dir_salida}")
    print("=" * 80)


if __name__ == "__main__":
    main()
