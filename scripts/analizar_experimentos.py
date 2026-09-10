#!/usr/bin/env python3
"""
Analiza todos los experimentos de denoising (Fase 1 y Fase 2):
1. Lee la tabla depurada de resumen_experimentos.xlsx.
2. Imprime el ranking consolidado y estadisticas para la redaccion de la tesis.
3. Genera mosaicos visuales comparativos de alta definicion (mineria ilegal, rios y selva)
   para multiples frames clave (ej. 100, 500, 800) y los guarda en figuras_tesis/.
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
        exp = str(r.get("ID Experimento", r.get("Experimento", "-")))[:38]
        modo = str(r.get("Modo", "-"))[:7]
        lr = str(r.get("Tasa Aprendizaje (LR)", r.get("Learning Rate", "-")))[:6]
        b = f"{float(r['BRISQUE']):.2f}" if r.get("BRISQUE") is not None else (f"{float(r['BRISQUE media']):.2f}" if r.get("BRISQUE media") is not None else "-")
        n = f"{float(r['NIQE']):.2f}" if r.get("NIQE") is not None else (f"{float(r['NIQE media']):.2f}" if r.get("NIQE media") is not None else "-")
        p = f"{float(r['PIQE']):.2f}" if r.get("PIQE") is not None else (f"{float(r['PIQE media']):.2f}" if r.get("PIQE media") is not None else "-")
        s = f"{float(r['Sigma Ruido']):.3f}" if r.get("Sigma Ruido") is not None else (f"{float(r['Sigma Ruido media']):.3f}" if r.get("Sigma Ruido media") is not None else "-")
        ret_val = r.get("Retención Nitidez (%)", r.get("Retencion Nitidez media"))
        ret = f"{float(ret_val)*100:.1f}%" if ret_val is not None and float(ret_val) <= 2.0 else (f"{float(ret_val):.1f}%" if ret_val is not None else "-")
        print(f"{idx:<3} | {exp:<38} | {modo:<7} | {lr:<6} | {b:<8} | {n:<6} | {p:<6} | {s:<6} | {ret:<10}")
    print("=" * 100)


def resolver_carpeta_frame(base_video, subcarpeta, frame_idx):
    """Busca un frame en la subcarpeta directa o dentro de expos/ tolerando cualquier formato numérico (5, 6 o 4 digitos)."""
    carpetas_buscar = [
        base_video / subcarpeta,
        base_video / "expos" / subcarpeta,
        RAIZ / subcarpeta,
        RAIZ / "videos" / base_video.name / "expos" / subcarpeta,
    ]

    posibles_nombres = [
        f"frame_{frame_idx:05d}.png",
        f"frame_{frame_idx:05d}.jpg",
        f"frame_{frame_idx:06d}.png",
        f"frame_{frame_idx:06d}.jpg",
        f"frame_{frame_idx:04d}.png",
        f"frame_{frame_idx:04d}.jpg",
        f"frame_{frame_idx}.png",
        f"frame_{frame_idx}.jpg",
    ]

    import re
    for c in carpetas_buscar:
        if not c.is_dir():
            continue
        for n in posibles_nombres:
            p = c / n
            if p.is_file():
                return p

        # Busqueda por numero si tiene sufijos o prefijos particulares
        for p in c.iterdir():
            if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                nums = re.findall(r"\d+", p.stem)
                if nums and int(nums[-1]) == frame_idx:
                    return p
    return None


def generar_mosaico_frame(base, frame_idx, salida_png, modelos_mostrar=None):
    """Genera mosaico visual comparando los modelos clave para un frame especifico."""
    if modelos_mostrar is None:
        modelos_mostrar = [
            ("Original (Con HUD)", "frames_originales"),
            ("Base Sin HUD (Inpainted)", "frames_sin_hud"),
            ("Bilateral 3D (Tradicional)", "bilateral_temporal_sinhud"),
            ("StructN2V Vertical (FPN Free)", "struct_n2v_vert_sinhud"),
            ("UDVD (Dynamic Kernels)", "udvd_sinhud"),
            ("Ensemble Wavelet (UDVD+B2U)", "ensemble_wavelet_udvd_b2u_sinhud"),
        ]

    imgs = []
    
    for titulo, subcarpeta in modelos_mostrar:
        ruta_img = resolver_carpeta_frame(base, subcarpeta, frame_idx)
        if ruta_img:
            img = cv2.imread(str(ruta_img))
            imgs.append((titulo, img))
        else:
            print(f"Aviso: no se encontro frame {frame_idx} para {titulo} en {subcarpeta}")

    if len(imgs) < 2:
        print(f"No hay suficientes imagenes para generar el mosaico del frame {frame_idx}.")
        return False

    h, w, _ = imgs[0][1].shape
    
    # Coordenadas de zoom centrado en la zona de mineria/rio/orilla
    ymin, ymax = int(h * 0.38), int(h * 0.68)
    xmin, xmax = int(w * 0.38), int(w * 0.68)
    
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
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_salida), mosaico)
    print(f"Mosaico frame {frame_idx} guardado en: {ruta_salida.resolve()}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisis y generacion de figuras para la tesis")
    parser.add_argument("--video", default="video1")
    parser.add_argument("--excel", default="resumen_experimentos.xlsx")
    parser.add_argument("--frames", nargs="+", type=int, default=[100, 500, 800], help="Frames clave a renderizar")
    parser.add_argument("--carpeta-figuras", default="figuras_tesis", help="Carpeta de salida para figuras de tesis")
    args = parser.parse_args()

    base = RAIZ / "videos" / args.video
    ruta_excel = base / args.excel if not Path(args.excel).is_file() else Path(args.excel)
    registros = leer_experimentos_excel(ruta_excel)
    if registros:
        imprimir_resumen_tesis(registros)

    dir_figuras = RAIZ / args.carpeta_figuras
    dir_figuras.mkdir(parents=True, exist_ok=True)

    for f_idx in args.frames:
        out_png = dir_figuras / f"mosaico_comparativo_f{f_idx}.png"
        generar_mosaico_frame(base, f_idx, out_png)
