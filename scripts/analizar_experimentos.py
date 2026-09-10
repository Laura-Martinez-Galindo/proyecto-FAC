#!/usr/bin/env python3
"""
Analiza todos los experimentos de denoising (Fase 1 y Fase 2):
1. Lee la tabla depurada de resumen_experimentos.xlsx.
2. Imprime el ranking consolidado y estadisticas para la redaccion de la tesis.
3. Genera mosaicos visuales comparativos de alta definicion (mineria ilegal, rios y selva)
   con las carpetas organizadas dentro de videos/video1/expos/.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import openpyxl

RAIZ = Path(__file__).resolve().parent.parent


def leer_experimentos_excel(ruta_excel):
    """Carga los experimentos desde la tabla depurada de Excel."""
    if not Path(ruta_excel).exists():
        print(f"No se encontro el archivo {ruta_excel}")
        return []

    wb = openpyxl.load_workbook(ruta_excel, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    header = rows[0]
    experimentos = []
    for r in rows[1:]:
        if not any(c is not None for c in r):
            continue
        registro = {str(k): v for k, v in zip(header, r) if k is not None}
        experimentos.append(registro)

    return experimentos


def imprimir_resumen_tesis(registros):
    """Imprime resumen consolidado ordenado por BRISQUE para la redaccion del documento."""
    print("=" * 100)
    print("RESUMEN GLOBAL DE EXPERIMENTOS DEPOSICIÓN / COMPARATIVA TESIS")
    print("=" * 100)
    
    ordenados = sorted(
        [r for r in registros if r.get("BRISQUE media") is not None],
        key=lambda x: float(x.get("BRISQUE media", 999.0))
    )
    
    print(f"{'No.':<3} | {'ID Experimento':<38} | {'Modo':<7} | {'LR':<6} | {'BRISQUE':<8} | {'NIQE':<6} | {'PIQE':<6} | {'Sigma':<6} | {'Nitidez (%)':<10}")
    print("-" * 100)
    for idx, r in enumerate(ordenados, 1):
        exp = str(r.get("ID Experimento", "-"))[:38]
        modo = str(r.get("Modo", "-"))[:7]
        lr = str(r.get("Tasa Aprendizaje (LR)", "-"))[:6]
        b = f"{float(r['BRISQUE media']):.2f}" if r.get("BRISQUE media") is not None else "-"
        n = f"{float(r['NIQE media']):.2f}" if r.get("NIQE media") is not None else "-"
        p = f"{float(r['PIQE media']):.2f}" if r.get("PIQE media") is not None else "-"
        s = f"{float(r['Sigma Ruido media']):.3f}" if r.get("Sigma Ruido media") is not None else "-"
        ret = f"{float(r['Retención Nitidez (%)']):.1f}%" if r.get("Retención Nitidez (%)") is not None else "-"
        print(f"{idx:<3} | {exp:<38} | {modo:<7} | {lr:<6} | {b:<8} | {n:<6} | {p:<6} | {s:<6} | {ret:<10}")
    print("=" * 100)


def resolver_carpeta_frame(base_video, subcarpeta, nombre_frame):
    """Busca un frame en la subcarpeta directa o dentro de expos/."""
    posibles = [
        base_video / subcarpeta / nombre_frame,
        base_video / "expos" / subcarpeta / nombre_frame,
        RAIZ / subcarpeta / nombre_frame,
    ]
    for p in posibles:
        if p.exists():
            return p
        p_jpg = p.with_suffix(".jpg")
        if p_jpg.exists():
            return p_jpg
    return None


def generar_mosaico_cualitativo(video_id, frame_idx=500, salida_png="mosaico_comparativo.png"):
    """Genera mosaico visual comparando los modelos clave con recuadro zoom."""
    base = RAIZ / "videos" / video_id
    
    lista_modelos = [
        ("1. Original Crudo", "frames_originales"),
        ("2. Base Sin HUD", "frames_sin_hud"),
        ("3. Bilateral Temporal", "bilateral_temporal_sinhud"),
        ("4. StructN2V Vertical", "struct_n2v_vert_sinhud"),
        ("5. UDVD (Dynamic Kernels)", "udvd_sinhud"),
        ("6. Ensamble Wavelet (UDVD+B2U)", "ensemble_wavelet_udvd_b2u_sinhud"),
    ]
    
    nombre_archivo = f"frame_{frame_idx:06d}.png"
    imgs = []
    
    for titulo, subcarpeta in lista_modelos:
        ruta_img = resolver_carpeta_frame(base, subcarpeta, nombre_archivo)
        if ruta_img:
            img = cv2.imread(str(ruta_img))
            imgs.append((titulo, img))
        else:
            print(f"Aviso: no se encontro imagen para {titulo} ({subcarpeta})")

    if len(imgs) < 2:
        print("No hay suficientes imagenes para generar el mosaico.")
        return

    h, w, _ = imgs[0][1].shape
    
    # Coordenadas de zoom centrado en la zona de mineria/rio
    ymin, ymax = int(h * 0.40), int(h * 0.70)
    xmin, xmax = int(w * 0.40), int(w * 0.70)
    
    paneles = []
    for titulo, img in imgs:
        disp = img.copy()
        cv2.rectangle(disp, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        
        # Header banner
        header = np.zeros((45, w, 3), dtype=np.uint8) + 25
        cv2.putText(header, titulo, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Zoom patch
        patch = img[ymin:ymax, xmin:xmax]
        patch_zoom = cv2.resize(patch, (int(w * 0.35), int(h * 0.35)), interpolation=cv2.INTER_LANCZOS4)
        cv2.rectangle(patch_zoom, (0, 0), (patch_zoom.shape[1]-1, patch_zoom.shape[0]-1), (0, 255, 255), 2)
        
        zh, zw, _ = patch_zoom.shape
        disp[h - zh - 10 : h - 10, w - zw - 10 : w - 10] = patch_zoom
        
        paneles.append(np.vstack([header, disp]))

    if len(paneles) == 6:
        fila1 = np.hstack(paneles[:3])
        fila2 = np.hstack(paneles[3:])
        mosaico = np.vstack([fila1, fila2])
    else:
        mosaico = np.hstack(paneles)

    ruta_salida = Path(salida_png)
    cv2.imwrite(str(ruta_salida), mosaico)
    print(f"Mosaico visual cualitativo guardado en: {ruta_salida.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisis y consolidacion de experimentos")
    parser.add_argument("--video", default="video1")
    parser.add_argument("--excel", default="resumen_experimentos.xlsx")
    parser.add_argument("--frame", type=int, default=500)
    parser.add_argument("--mosaico", default="mosaico_comparativo.png")
    args = parser.parse_args()

    ruta_excel = RAIZ / "videos" / args.video / args.excel if not Path(args.excel).is_file() else Path(args.excel)
    registros = leer_experimentos_excel(ruta_excel)
    imprimir_resumen_tesis(registros)
    generar_mosaico_cualitativo(args.video, args.frame, args.mosaico)
