#!/usr/bin/env bash

#SBATCH --job-name=sync_udvd_3v
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=jobs/logs/sync_udvd_3v_%j.out
#SBATCH --error=jobs/logs/sync_udvd_3v_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail
trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs videos/comprimidos

echo "=============================================================================="
echo "🚀 SUBIDA ULTRA-RÁPIDA DE UDVD RESTAURADO A DRIVE (VIDEO 1, 2 Y 3)"
echo "   Nodo: $(hostname)"
echo "   CPUs: 16 hilos en paralelo"
echo "   Fecha: $(date --iso-8601=seconds)"
echo "=============================================================================="

if ! command -v rclone &> /dev/null; then
    echo "[-] ERROR: rclone no está instalado." >&2
    exit 1
fi

RCLONE_FLAGS="--transfers 16 --checkers 32 --drive-chunk-size 256M --drive-upload-cutoff 128M --buffer-size 128M --fast-list --progress"

if command -v pigz &> /dev/null; then
    COMPRESS_CMD="pigz -p 16 -1"
    echo "[+] Usando compresión ultra-rápida 'pigz' con 16 hilos."
else
    COMPRESS_CMD="gzip -1"
    echo "[!] Usando gzip -1."
fi

subir_udvd() {
    local vid="$1"
    local dir_udvd="videos/${vid}/udvd_sin_hud"
    local tar_pkg="videos/comprimidos/${vid}_udvd_sin_hud.tar.gz"
    local dest_drive="gdrive:proyecto-FAC/videos/${vid}/comprimidos/"

    if [[ -d "${dir_udvd}" ]]; then
        local n_frames
        n_frames=$(find "${dir_udvd}" -maxdepth 1 -type f | wc -l)
        echo
        echo "=============================================================================="
        echo "📦 [${vid^^}] Empaquetando ${vid}_udvd_sin_hud.tar.gz (${n_frames:,} frames)..."
        echo "=============================================================================="
        
        local t0=$(date +%s)
        tar -cf - -C "videos/${vid}" "udvd_sin_hud" | ${COMPRESS_CMD} > "${tar_pkg}"
        local t1=$(date +%s)
        local tam=$(du -h "${tar_pkg}" | cut -f1)
        echo "✅ Empaquetado en $((t1 - t0)) seg (${tam})."

        echo "☁️ Subiendo ${vid^^} a Google Drive (${dest_drive})..."
        rclone copy "${tar_pkg}" "${dest_drive}" ${RCLONE_FLAGS}
        echo "✅ ¡${vid^^} UDVD subido exitosamente a Google Drive!"
    else
        echo "[!] No se encontró la carpeta: ${dir_udvd}"
    fi
}

# Subir los 3 videos en orden
subir_udvd "video1"
subir_udvd "video2"
subir_udvd "video3"

echo
echo "=============================================================================="
echo "🎉 ¡LOS 3 PAQUETES UDVD FUERON SUBIDOS A GOOGLE DRIVE EXITOSAMENTE!"
echo "   Fecha fin: $(date --iso-8601=seconds)"
echo "=============================================================================="
