#!/usr/bin/env bash

#SBATCH --job-name=sync_udvd_1_2
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/logs/sync_udvd_%j.out
#SBATCH --error=jobs/logs/sync_udvd_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail
trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs videos/comprimidos

echo "=============================================================================="
echo "🚀 EMPAQUETANDO Y SUBIENDO UDVD RESTAURADO (VIDEO 1 Y VIDEO 2) A DRIVE"
echo "   Nodo: $(hostname)"
echo "   CPUs: 16 (Compresión paralela rápida)"
echo "   Fecha: $(date --iso-8601=seconds)"
echo "=============================================================================="

if ! command -v rclone &> /dev/null; then
    echo "[-] ERROR: rclone no está instalado." >&2
    exit 1
fi

RCLONE_FLAGS="--transfers 32 --checkers 64 --drive-chunk-size 128M --drive-upload-cutoff 32M --buffer-size 64M --fast-list --progress"

if command -v pigz &> /dev/null; then
    COMPRESS_CMD="pigz -p 16 -1"
    echo "[+] Usando pigz con 16 hilos."
else
    COMPRESS_CMD="gzip -1"
    echo "[!] Usando gzip rápido."
fi

subir_udvd_video() {
    local vid="$1"
    local dir_udvd="videos/${vid}/udvd_sin_hud"
    local tar_pkg="videos/comprimidos/${vid}_udvd_sin_hud.tar.gz"
    local dest_drive="gdrive:proyecto-FAC/videos/${vid}/comprimidos/"

    if [[ -d "${dir_udvd}" ]]; then
        local n_frames
        n_frames=$(find "${dir_udvd}" -maxdepth 1 -type f | wc -l)
        echo
        echo "=============================================================================="
        echo "📦 [${vid^^}] Empaquetando ${vid}_udvd_sin_hud.tar.gz (${n_frames:,} frames restaurados)..."
        echo "=============================================================================="
        
        # Eliminar tar previo si existía incompleto para asegurar integridad
        rm -f "${tar_pkg}"
        
        local t0
        t0=$(date +%s)
        tar -cf - -C "videos/${vid}" "udvd_sin_hud" | ${COMPRESS_CMD} > "${tar_pkg}"
        local t1
        t1=$(date +%s)
        local tam
        tam=$(du -h "${tar_pkg}" | cut -f1)
        echo "✅ Empaquetado exitoso en $((t1 - t0))s (${tam})."

        echo "☁️ Subiendo a Google Drive (${dest_drive})..."
        rclone copy "${tar_pkg}" "${dest_drive}" ${RCLONE_FLAGS}
        echo "✅ ¡Subida de ${vid^^} UDVD finalizada con éxito!"
    else
        echo "[!] Carpeta no encontrada: ${dir_udvd}"
    fi
}

# Procesar Video 1 y Video 2
subir_udvd_video "video1"
subir_udvd_video "video2"

echo
echo "=============================================================================="
echo "🎉 PROCESO COMPLETADO: VIDEO 1 Y VIDEO 2 UDVD SUBIDOS A DRIVE: $(date --iso-8601=seconds)"
echo "=============================================================================="
