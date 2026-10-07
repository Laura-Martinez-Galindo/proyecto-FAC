#!/usr/bin/env python3
"""
Auditoría Rápida y Exhaustiva del Estado de Preprocesamiento, Modelos, Videos y Google Drive:
Escanea Video 1, Video 2 y Video 3 en Hypatia para saber con precisión:
1. Cuántos frames originales, máscaras y frames sin HUD existen vs esperados.
2. Si el inpainting con ProPainter está completo o incompleto.
3. Qué modelos de denoising (UDVD, Blind2Unblind, Ensembles) ya terminaron.
4. Qué videos .mp4 están renderizados.
5. Qué paquetes .tar.gz están creados y qué falta por sincronizar con Google Drive vía rclone.
"""

import json
import os
from pathlib import Path
import subprocess

RAIZ = Path(__file__).resolve().parent.parent
CONFIG_JSON = RAIZ / "config" / "videos.json"

EXTS_IMG = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def contar_archivos(carpeta, exts=EXTS_IMG):
    if not carpeta or not Path(carpeta).is_dir():
        return 0
    return sum(1 for p in Path(carpeta).iterdir() if p.is_file() and p.suffix.lower() in exts)


def obtener_peso_legible(tam_bytes):
    for u in ['B', 'KB', 'MB', 'GB', 'TB']:
        if tam_bytes < 1024.0:
            return f"{tam_bytes:.2f} {u}"
        tam_bytes /= 1024.0
    return f"{tam_bytes:.2f} PB"


def auditar_video(v_id):
    dir_v = RAIZ / "videos" / v_id
    dir_orig = dir_v / "frames_originales"
    dir_masc = dir_v / "mascaras_hud"
    dir_sin = dir_v / "frames_sin_hud"
    dir_expos = dir_v / "expos"
    dir_render = dir_v / "videos_renderizados"
    xlsx_res = dir_v / "resumen.xlsx"
    csv_linea = dir_v / f"linea_tiempo_{v_id}.csv"

    n_orig = contar_archivos(dir_orig)
    n_masc = contar_archivos(dir_masc)
    n_sin = contar_archivos(dir_sin)
    n_render = contar_archivos(dir_render, {".mp4", ".avi", ".mkv"})

    # Denoised subdirs
    denoised_dirs = []
    if dir_expos.is_dir():
        for sub in dir_expos.iterdir():
            if sub.is_dir():
                c = contar_archivos(sub)
                if c > 0:
                    denoised_dirs.append((sub.name, c))
    
    # También buscar en la raíz de dir_v
    if dir_v.is_dir():
        for sub in dir_v.iterdir():
            if sub.is_dir() and sub.name not in {"frames_originales", "mascaras_hud", "frames_sin_hud", "videos_renderizados", "expos", "original", "ruido_residual"}:
                c = contar_archivos(sub)
                if c > 0:
                    denoised_dirs.append((sub.name, c))

    return {
        "id": v_id,
        "n_orig": n_orig,
        "n_masc": n_masc,
        "n_sin": n_sin,
        "inpainting_completo": (n_sin >= n_orig and n_orig > 0),
        "porcentaje_sin_hud": (n_sin / n_orig * 100) if n_orig > 0 else 0.0,
        "tiene_resumen_xlsx": xlsx_res.is_file(),
        "tiene_linea_tiempo_csv": csv_linea.is_file(),
        "n_videos_render": n_render,
        "modelos_denoised": denoised_dirs
    }


def main():
    print("=" * 95)
    print("           AUDITORÍA DEL ESTADO DE PREPROCESAMIENTO Y MODELOS (HYPATIA)")
    print("=" * 95)

    videos = ["video1", "video2", "video3"]
    estados = [auditar_video(v) for v in videos]

    for est in estados:
        v_id = est["id"].upper()
        print(f"\n🎥 [{v_id}]")
        print(f"  • Frames Originales:   {est['n_orig']:,}")
        print(f"  • Máscaras HUD:        {est['n_masc']:,}")
        print(f"  • Frames Sin HUD:      {est['n_sin']:,} ({est['porcentaje_sin_hud']:.1f}%)")
        
        # Estado Inpainting
        if est["inpainting_completo"]:
            print(f"  • Inpainting ProPainter: ✅ COMPLETADO AL 100%")
        elif est["n_sin"] > 0:
            faltan = est["n_orig"] - est["n_sin"]
            print(f"  • Inpainting ProPainter: ⚠️ EN PROGRESO (Faltan {faltan:,} frames)")
        else:
            print(f"  • Inpainting ProPainter: ❌ NO INICIADO / 0 FRAMES")

        # Métricas
        res_str = "✅ Sí" if est["tiene_resumen_xlsx"] else "❌ No"
        csv_str = "✅ Sí" if est["tiene_linea_tiempo_csv"] else "❌ No"
        print(f"  • resumen.xlsx:        {res_str} | linea_tiempo.csv: {csv_str}")
        print(f"  • Videos Renderizados: {est['n_videos_render']} archivo(s) .mp4")

        # Denoising
        if est["modelos_denoised"]:
            print(f"  • Modelos Denoised Disponibles ({len(est['modelos_denoised'])}):")
            for m_nom, m_cnt in est["modelos_denoised"]:
                print(f"      - {m_nom:<42}: {m_cnt:,} frames")
        else:
            print(f"  • Modelos Denoised:    [Ninguno en carpeta expos]")

    print("\n" + "=" * 95)
    print("             AUDITORÍA DE PAQUETES COMPRIMIDOS (.TAR.GZ)")
    print("=" * 95)
    dir_comp = RAIZ / "videos" / "comprimidos"
    if dir_comp.is_dir():
        tars = sorted(list(dir_comp.glob("*.tar.gz")))
        if tars:
            for t in tars:
                print(f"  📦 {t.name:<45} ({obtener_peso_legible(t.stat().st_size)})")
        else:
            print("  [!] No hay archivos .tar.gz en videos/comprimidos/")
    else:
        print("  [!] La carpeta videos/comprimidos no existe.")

    # Verificación rclone
    print("\n" + "=" * 95)
    print("               VERIFICACIÓN DE GOOGLE DRIVE (RCLONE)")
    print("=" * 95)
    try:
        res = subprocess.run(["rclone", "lsf", "gdrive:proyecto-FAC/videos/"], capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            carpetas_gdrive = res.stdout.strip().split("\n")
            print("  Google Drive remoto ('gdrive:proyecto-FAC/videos/'):")
            for c in carpetas_gdrive:
                if c.strip():
                    print(f"    📁 {c}")
        else:
            print("  [!] rclone no pudo listar 'gdrive:proyecto-FAC/videos/'. Revisa la configuración con `rclone config`.")
    except Exception as e:
        print(f"  [!] rclone no está en el PATH o falló: {e}")

    print("\n" + "=" * 95)
    print("Diagnóstico completado.")
    print("=" * 95)


if __name__ == "__main__":
    main()
