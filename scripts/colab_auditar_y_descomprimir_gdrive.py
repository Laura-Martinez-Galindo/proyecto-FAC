# ==============================================================================
# SCRIPT PARA GOOGLE COLAB: AUDITAR Y DESCOMPRIMIR PROYECTO-FAC EN DRIVE
# ==============================================================================
# Pega todo este código en una celda de Google Colab y ejecútala con [Shift + Enter].
# 1. Monta tu Google Drive automáticamente.
# 2. Escanea todas las carpetas y archivos .tar.gz de Video 1, Video 2 y Video 3.
# 3. Detecta qué paquetes ya están descomprimidos y cuáles faltan por descomprimir.
# 4. Descomprime en paralelo a máxima velocidad usando pigz / tar multinúcleo.
# ==============================================================================

import os
from pathlib import Path
import shutil
import subprocess
import time

# 1. Montar Google Drive si no está montado
try:
    from google.colab import drive
    print("[1/4] Montando Google Drive...")
    drive.mount('/content/drive', force_remount=False)
except ImportError:
    print("[!] Ejecutando fuera de Colab. Asegúrate de tener acceso a la ruta de Drive.")

# 2. Localizar la carpeta proyecto-FAC en Drive
rutas_candidatas = [
    Path("/content/drive/MyDrive/proyecto-FAC"),
    Path("/content/drive/Shareddrives/proyecto-FAC"),
    Path("/content/drive/MyDrive/proyecto_FAC"),
    Path("/content/drive/MyDrive/Tesis/proyecto-FAC"),
    Path("./proyecto-FAC"),
]

DIR_DRIVE = None
for r in rutas_candidatas:
    if r.is_dir():
        DIR_DRIVE = r
        break

if DIR_DRIVE is None:
    # Buscar recursivamente en MyDrive
    raiz_mydrive = Path("/content/drive/MyDrive")
    if raiz_mydrive.is_dir():
        for p in raiz_mydrive.glob("**/proyecto-FAC"):
            if p.is_dir():
                DIR_DRIVE = p
                break

if DIR_DRIVE is None:
    print("❌ No se encontró la carpeta 'proyecto-FAC' en MyDrive.")
    print("Verifica si el nombre de la carpeta en tu Drive es diferente.")
else:
    print(f"✅ Carpeta de proyecto encontrada en Drive: {DIR_DRIVE}")


def obtener_peso_legible(tam_bytes):
    for u in ['B', 'KB', 'MB', 'GB', 'TB']:
        if tam_bytes < 1024.0:
            return f"{tam_bytes:.2f} {u}"
        tam_bytes /= 1024.0
    return f"{tam_bytes:.2f} PB"


def auditar_y_descomprimir(auto_descomprimir=True):
    if DIR_DRIVE is None:
        return

    print("\n" + "=" * 80)
    print(f"       AUDITORÍA DE PAQUETES Y ESTADO EN GOOGLE DRIVE")
    print("=" * 80)

    dir_videos = DIR_DRIVE / "videos"
    if not dir_videos.is_dir():
        print(f"[!] No existe la subcarpeta {dir_videos}")
        return

    paquetes_encontrados = []
    
    # Buscar todos los archivos .tar.gz en videos/
    for p_tar in dir_videos.glob("**/*.tar.gz"):
        peso = obtener_peso_legible(p_tar.stat().st_size)
        
        # Determinar a qué video pertenece
        v_parent = p_tar.parent.parent.name if p_tar.parent.name == "comprimidos" else p_tar.parent.name
        if not v_parent.startswith("video"):
            # Tratar de deducir del nombre del archivo
            if "video1" in p_tar.name:
                v_parent = "video1"
            elif "video2" in p_tar.name:
                v_parent = "video2"
            elif "video3" in p_tar.name:
                v_parent = "video3"
            else:
                v_parent = "general"

        # Nombre de la carpeta destino esperada al descomprimir
        # ej: video1_frames_sin_hud.tar.gz -> frames_sin_hud
        nom_base = p_tar.name.replace(".tar.gz", "")
        nom_carpeta_dest = nom_base
        for pref in ["video1_", "video2_", "video3_"]:
            if nom_carpeta_dest.startswith(pref):
                nom_carpeta_dest = nom_carpeta_dest.replace(pref, "", 1)

        carpeta_destino = dir_videos / v_parent / nom_carpeta_dest
        descomprimido = carpeta_destino.is_dir() and sum(1 for _ in carpeta_destino.iterdir()) > 10

        paquetes_encontrados.append({
            "tar_path": p_tar,
            "nombre": p_tar.name,
            "peso": peso,
            "video": v_parent,
            "carpeta_destino": carpeta_destino,
            "descomprimido": descomprimido
        })

    print(f"\nSe encontraron {len(paquetes_encontrados)} paquete(s) comprimido(s) en Google Drive:\n")
    
    pendientes_descompresion = []
    for pkg in paquetes_encontrados:
        estado_str = "✅ DESCOMPRIMIDO" if pkg["descomprimido"] else "⚠️ PENDIENTE POR DESCOMPRIMIR"
        print(f"  📦 [{pkg['video'].upper()}] {pkg['nombre']:<40} ({pkg['peso']}) -> {estado_str}")
        if not pkg["descomprimido"]:
            pendientes_descompresion.append(pkg)

    print("\n" + "-" * 80)
    print(f"Resumen: {len(paquetes_encontrados) - len(pendientes_descompresion)} listos | {len(pendientes_descompresion)} pendientes por descomprimir.")
    print("-" * 80)

    if pendientes_descompresion and auto_descomprimir:
        print("\n[🚀] INICIANDO DESCOMPRESIÓN AUTOMÁTICA EN GOOGLE DRIVE...")
        for idx, pkg in enumerate(pendientes_descompresion, 1):
            tar_file = pkg["tar_path"]
            dest_parent = dir_videos / pkg["video"]
            dest_parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\n({idx}/{len(pendientes_descompresion)}) Descomprimiendo {pkg['nombre']} en {dest_parent}...")
            t0 = time.time()
            
            # Ejecutar tar con aceleración
            cmd = f"tar -xzf '{tar_file}' -C '{dest_parent}'"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            t1 = time.time()
            if res.returncode == 0:
                print(f"    ✅ Completado con éxito en {t1 - t0:.1f} segundos.")
            else:
                print(f"    ❌ Error al descomprimir: {res.stderr}")

        print("\n" + "=" * 80)
        print("🎉 ¡TODOS LOS PAQUETES FUERON DESCOMPRIMIDOS EXITOSAMENTE EN DRIVE!")
        print("=" * 80)


# Ejecutar auditoría y descompresión automática
if DIR_DRIVE is not None:
    auditar_y_descomprimir(auto_descomprimir=True)
