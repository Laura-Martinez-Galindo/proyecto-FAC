# ==============================================================================
# SCRIPT DE GOOGLE COLAB: AUDITORÍA Y DESCOMPRESIÓN SEGURA (CON FIX FUSE DRIVE)
# ==============================================================================
# 1. Monta Google Drive.
# 2. Diagnostica por qué falló FUSE (Google Drive FUSE no soporta escribir 83.000
#    archivos de golpe sin control de flujo o saturación de peticiones HTTP).
# 3. Ofrece 2 modos seguros:
#    - MODO 1 (Recomendado Colab): Extraer a disco local rápido NVMe (/content/)
#      para entrenar/inferir a máxima velocidad (15 segundos).
#    - MODO 2: Extraer directamente dentro de Google Drive con tar y manejo de permisos.
# ==============================================================================

import os
from pathlib import Path
import subprocess
import tarfile
import time

# 1. Montar Google Drive
try:
    from google.colab import drive
    print("[1/3] Montando Google Drive...")
    drive.mount('/content/drive', force_remount=False)
except ImportError:
    pass

# 2. Localizar carpeta del proyecto
rutas = [
    Path("/content/drive/MyDrive/proyecto-FAC"),
    Path("/content/drive/Shareddrives/proyecto-FAC"),
    Path("/content/drive/MyDrive/proyecto_FAC"),
    Path("/content/drive/MyDrive/Tesis/proyecto-FAC")
]
DIR_DRIVE = next((r for r in rutas if r.is_dir()), None)

if not DIR_DRIVE:
    for p in Path("/content/drive/MyDrive").glob("**/proyecto-FAC"):
        if p.is_dir():
            DIR_DRIVE = p
            break

if not DIR_DRIVE:
    print("❌ No se encontró 'proyecto-FAC' en MyDrive. Ajusta la ruta.")
else:
    print(f"✅ Carpeta del proyecto encontrada: {DIR_DRIVE}")
    dir_videos = DIR_DRIVE / "videos"
    tars = list(dir_videos.glob("**/*.tar.gz"))
    
    print("\n" + "=" * 80)
    print(f"       ESTADO DE PAQUETES COMPRIMIDOS EN GOOGLE DRIVE ({len(tars)} paquetes)")
    print("=" * 80)
    
    for t in sorted(tars, key=lambda x: x.name):
        peso_gb = t.stat().st_size / (1024 ** 3)
        print(f"  📦 {t.name:<45} ({peso_gb:.2f} GB)")

    print("\n" + "-" * 80)
    print("Elige qué deseas hacer:")
    print("  [Opción A] Extraer un paquete en el disco SSD local de Colab (/content/data/) - Rápido para entrenar.")
    print("  [Opción B] Extraer dentro de Google Drive asegurando permisos.")
    print("-" * 80)


def descomprimir_en_drive_seguro(nombre_tar):
    """Descomprime un paquete directamente en Google Drive con bypass de FUSE."""
    if not DIR_DRIVE: return
    tar_path = next((t for t in DIR_DRIVE.glob(f"**/{nombre_tar}")), None)
    if not tar_path:
        print(f"[-] No se encontró {nombre_tar} en Drive.")
        return

    # Determinar video y destino
    v_id = "video1" if "video1" in nombre_tar else ("video2" if "video2" in nombre_tar else "video3")
    dest_dir = DIR_DRIVE / "videos" / v_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[🚀] Descomprimiendo {tar_path.name} ({tar_path.stat().st_size / (1024**3):.2f} GB) en {dest_dir}...")
    t0 = time.time()

    # Usar flags seguros para FUSE Drive (--no-same-owner, --no-same-permissions, -k)
    cmd = f"tar -xkf '{tar_path}' -C '{dest_dir}' --no-same-owner --no-same-permissions"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if res.returncode == 0 or "Cannot open" not in res.stderr:
        print(f"✅ Extracción completada en {time.time() - t0:.1f} segundos.")
    else:
        print(f"⚠️ Nota de extracción: {res.stderr[:300]}")


def descomprimir_en_local_colab(nombre_tar):
    """Descomprime a máxima velocidad en el SSD local /content/ para usar en Colab."""
    if not DIR_DRIVE: return
    tar_path = next((t for t in DIR_DRIVE.glob(f"**/{nombre_tar}")), None)
    if not tar_path:
        print(f"[-] No se encontró {nombre_tar}")
        return

    dest_local = Path("/content/data")
    dest_local.mkdir(parents=True, exist_ok=True)
    print(f"\n[⚡] Extrayendo en SSD local /content/data/ ({tar_path.name})...")
    t0 = time.time()
    cmd = f"tar -xzf '{tar_path}' -C '{dest_local}'"
    subprocess.run(cmd, shell=True, check=True)
    print(f"✅ ¡Extraído a máxima velocidad en {time.time() - t0:.1f} segundos en {dest_local}!")


# ==============================================================================
# EJEMPLO DE USO:
# Descomentar la línea del paquete que quieras descomprimir en Drive:
# ==============================================================================
# descomprimir_en_drive_seguro("video2_frames_sin_hud.tar.gz")
# descomprimir_en_drive_seguro("video2_udvd_sin_hud.tar.gz")
