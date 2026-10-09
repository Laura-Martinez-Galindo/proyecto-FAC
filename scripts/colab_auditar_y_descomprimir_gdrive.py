# ==============================================================================
# AUDITORÍA Y DESCOMPRESIÓN DE GOOGLE DRIVE (VIDEO 1, VIDEO 2, VIDEO 3)
# ==============================================================================
# Copia y pega esta celda directamente en tu Google Colab.
# Permite:
# 1. Auditar exactamente qué carpetas y qué paquetes (.tar.gz) están en Drive.
# 2. Descomprimir selectivamente sin saturar el sistema de archivos de Drive.
# ==============================================================================

import os
import shutil
import subprocess
import time
from pathlib import Path

# 1. Montar Google Drive
print("[1/3] Montando Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except ImportError:
    print("ℹ️ Ejecutando en entorno local/servidor.")

# 2. Localizar directorio del proyecto en Drive
posibles_rutas = [
    Path("/content/drive/MyDrive/proyecto-FAC"),
    Path("/content/drive/Shareddrives/proyecto-FAC"),
    Path("/content/drive/MyDrive/proyecto_FAC"),
    Path("/content/drive/MyDrive/Tesis/proyecto-FAC"),
    Path.cwd()
]

DIR_DRIVE = next((p for p in posibles_rutas if p.is_dir()), None)
if not DIR_DRIVE:
    for p in Path("/content/drive/MyDrive").glob("**/proyecto-FAC"):
        if p.is_dir():
            DIR_DRIVE = p
            break

if not DIR_DRIVE:
    raise FileNotFoundError("❌ No se encontró la carpeta 'proyecto-FAC' en tu Google Drive.")

print(f"✅ Carpeta del proyecto encontrada: {DIR_DRIVE}")

# 3. Auditoría completa por Video
print("\n" + "=" * 90)
print("                   AUDITORÍA GENERAL DE ESTADO EN GOOGLE DRIVE")
print("=" * 90)

carpetas_clave = [
    ("Frames Originales", "frames_originales"),
    ("Máscaras HUD", "mascaras_hud"),
    ("Frames Sin HUD", "frames_sin_hud"),
    ("UDVD Restaurado", "udvd_sin_hud"),
    ("Videos Renderizados", "videos_renderizados"),
]

for v_id in ["video1", "video2", "video3"]:
    dir_v = DIR_DRIVE / "videos" / v_id
    print(f"\n📂 [{v_id.upper()}] -> {dir_v}")
    if not dir_v.is_dir():
        print(f"   ⚠️ La carpeta del {v_id} no existe aún en Drive.")
        continue

    # Inspeccionar carpetas descomprimidas
    for nombre_etapa, subcarpeta in carpetas_clave:
        p_sub = dir_v / subcarpeta
        if p_sub.is_dir():
            total = len(list(p_sub.glob("*.*")))
            print(f"   ✅ {nombre_etapa:<22} (Descomprimido): {total:,} archivos")
        else:
            print(f"   ❌ {nombre_etapa:<22} (Descomprimido): NO EXISTE")

    # Inspeccionar paquetes comprimidos .tar.gz
    tars = list(dir_v.glob("**/*.tar.gz")) + list((DIR_DRIVE / "videos" / "comprimidos").glob(f"{v_id}*.tar.gz"))
    if tars:
        print("   📦 Paquetes .tar.gz disponibles:")
        for t in tars:
            peso_gb = t.stat().st_size / (1024 ** 3)
            print(f"      - {t.name:<40} ({peso_gb:.2f} GB)")
    else:
        print("   ⚠️ No hay archivos .tar.gz guardados para este video.")

print("\n" + "=" * 90)
print("             DESCOMPRESIÓN SEGURA DIRECTA EN DRIVE (OPCIONAL)")
print("=" * 90)
print("Para descomprimir un paquete de Video 2 o Video 3 sin error de FUSE:")
print("Ejecuta en una celda siguiente:")
print("descomprimir_paquete('video2_frames_sin_hud.tar.gz')")
print("=" * 90)


def descomprimir_paquete(nombre_tar):
    """Descomprime un paquete de forma segura en Drive usando Python tarfile."""
    tar_path = next((t for t in DIR_DRIVE.glob(f"**/{nombre_tar}")), None)
    if not tar_path:
        print(f"❌ No se encontró '{nombre_tar}' en Drive.")
        return

    v_id = "video1" if "video1" in nombre_tar else ("video2" if "video2" in nombre_tar else "video3")
    dest_dir = DIR_DRIVE / "videos" / v_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[🚀] Descomprimiendo {tar_path.name} ({tar_path.stat().st_size / (1024**3):.2f} GB) en {dest_dir}...")
    t0 = time.time()

    # Usar comando tar con banderas seguras para sistemas FUSE de Drive
    cmd = f"tar --warning=no-unknown-keyword --no-same-owner --no-same-permissions -xzf '{tar_path}' -C '{dest_dir}'"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if res.returncode == 0:
        print(f"✅ ¡Descompresión completada con éxito en {time.time() - t0:.1f} segundos!")
    else:
        print(f"⚠️ Nota durante la extracción: {res.stderr[:300]}")
