#!/usr/bin/env python3
"""Genera un collage comparativo de alta calidad con zoom en regiones de interes."""

import argparse
from pathlib import Path
import cv2
import numpy as np


def crear_collage(video_id, frame_idx=500, bbox=None, ruta_salida="comparativa_modelos.png"):
    """
    Genera una imagen con los modelos lado a lado y un recuadro de zoom.
    bbox: [ymin, xmin, ymax, xmax] normalizado o en pixeles.
    """
    ruta_base = Path("videos") / video_id
    
    modelos = [
        ("1. Original", ruta_base / "frames_originales"),
        ("2. Sin HUD (Inpainted)", ruta_base / "frames_sin_hud"),
        ("3. StructN2V Vert (88.4% Nitidez)", ruta_base / "StructN2V_Vert_SinHUD_d3_lr1e3"),
        ("4. Blind2Unblind (BRISQUE 45.3)", ruta_base / "Blind2Unblind_SinHUD_d4_lr1e3"),
        ("5. Neighbor2Neighbor (PIQE 67.3)", ruta_base / "Neigh2Neigh_SinHUD_d4_lr1e3"),
        ("6. FastDVDnet (Temporal T=5)", ruta_base / "FastDVDnet_SinHUD_T5_lr1e3"),
    ]
    
    nombre_frame = f"frame_{frame_idx:06d}.png"
    imagenes_cargadas = []
    
    for nombre, carpeta in modelos:
        ruta_img = carpeta / nombre_frame
        if not ruta_img.exists():
            ruta_img = carpeta / f"frame_{frame_idx:06d}.jpg"
        
        if ruta_img.exists():
            img = cv2.imread(str(ruta_img))
            imagenes_cargadas.append((nombre, img))
        else:
            print(f"Aviso: No se encontro {ruta_img}")

    if not imagenes_cargadas:
        print("No se encontraron imagenes para generar el collage.")
        return

    h, w, c = imagenes_cargadas[0][1].shape
    
    if bbox is None:
        ymin, ymax = int(h * 0.35), int(h * 0.65)
        xmin, xmax = int(w * 0.35), int(w * 0.65)
    else:
        ymin, xmin, ymax, xmax = bbox

    paneles = []
    zooms = []
    
    for nombre, img in imagenes_cargadas:
        img_disp = img.copy()
        
        patch = img[ymin:ymax, xmin:xmax]
        patch_zoom = cv2.resize(patch, (w // 2, h // 2), interpolation=cv2.INTER_LANCZOS4)
        cv2.rectangle(patch_zoom, (0, 0), (patch_zoom.shape[1] - 1, patch_zoom.shape[0] - 1), (0, 255, 255), 3)
        zooms.append(patch_zoom)
        
        cv2.rectangle(img_disp, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        
        header = np.zeros((45, w, 3), dtype=np.uint8) + 30
        cv2.putText(header, nombre, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        
        panel_completo = np.vstack([header, img_disp])
        paneles.append(panel_completo)

    n = len(paneles)
    if n == 6:
        fila1 = np.hstack(paneles[:3])
        fila2 = np.hstack(paneles[3:])
        grilla_modelos = np.vstack([fila1, fila2])
        
        zoom_f1 = np.hstack(zooms[:3])
        zoom_f2 = np.hstack(zooms[3:])
        grilla_zoom = np.vstack([zoom_f1, zoom_f2])
    else:
        grilla_modelos = np.hstack(paneles)
        grilla_zoom = np.hstack(zooms)

    ruta_out = Path(ruta_salida)
    cv2.imwrite(str(ruta_out), grilla_modelos)
    cv2.imwrite(str(ruta_out.with_name(f"zoom_{ruta_out.name}")), grilla_zoom)
    print(f"Collage guardado en: {ruta_out.resolve()}")
    print(f"Zoom guardado en: {ruta_out.with_name(f'zoom_{ruta_out.name}').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera collage comparativo de modelos")
    parser.add_argument("--video", default="video1")
    parser.add_argument("--frame", type=int, default=500)
    parser.add_argument("--salida", default="comparativa_modelos.png")
    args = parser.parse_args()
    
    crear_collage(args.video, args.frame, ruta_salida=args.salida)
