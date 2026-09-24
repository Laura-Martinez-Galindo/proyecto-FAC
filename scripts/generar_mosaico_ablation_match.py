#!/usr/bin/env python3
"""
Genera Mosaicos Visuales Comparativos de Validación para la Tesis:
[Original con Cajas Anotadas de YOLO] | [Sin HUD (ProPainter)] | [Sin HUD + Denoised (UDVD)]
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parent.parent

COLORES_CLASES = {
    0: (0, 255, 255),    # vehicle (amarillo)
    1: (255, 100, 0),    # building (azul claro)
    2: (0, 255, 0),      # road (verde)
    3: (255, 0, 0),      # river (azul)
    4: (0, 0, 255),      # SDZI / alteracion (rojo)
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


def crear_mosaico_comparativo(img_orig_boxes, img_sin_hud, img_udvd, titulo="Comparativa de Deteccion"):
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
    cv2.putText(canvas, "1. Original (Con HUD + Cajas YOLO)", (20, 38), font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "2. Preprocesado (Sin HUD - ProPainter)", (w + 20, 38), font, 0.9, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(canvas, "3. Restaurado (Sin HUD + Denoised UDVD)", (2*w + 20, 38), font, 0.9, (100, 200, 255), 2, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generar Mosaicos de Ablación con Cajas de YOLO")
    parser.add_argument("--img-orig", required=True, help="Ruta a la imagen original")
    parser.add_argument("--label-txt", required=True, help="Ruta al archivo .txt con las etiquetas YOLO")
    parser.add_argument("--img-sin-hud", required=True, help="Ruta a la imagen Sin HUD")
    parser.add_argument("--img-udvd", required=True, help="Ruta a la imagen Denoised UDVD")
    parser.add_argument("--salida", default="figuras_tesis/mosaicos_ablation_match.png", help="Ruta del mosaico final")
    args = parser.parse_args()

    img_o = cv2.imread(str(Path(args.img_orig).resolve()))
    img_s = cv2.imread(str(Path(args.img_sin_hud).resolve()))
    img_u = cv2.imread(str(Path(args.img_udvd).resolve()))

    if img_o is None or img_s is None or img_u is None:
        print("ERROR: No se pudieron leer todas las imágenes.")
        return

    # Ajustar dimensiones si difieren
    h, w = img_o.shape[:2]
    if img_s.shape[:2] != (h, w):
        img_s = cv2.resize(img_s, (w, h))
    if img_u.shape[:2] != (h, w):
        img_u = cv2.resize(img_u, (w, h))

    img_boxes = dibujar_cajas_yolo(img_o, args.label_txt)
    mosaico = crear_mosaico_comparativo(img_boxes, img_s, img_u)

    salida_p = Path(args.salida).resolve()
    salida_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida_p), mosaico, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    print(f"Mosaico comparativo generado exitosamente en: {salida_p}")


if __name__ == "__main__":
    main()
