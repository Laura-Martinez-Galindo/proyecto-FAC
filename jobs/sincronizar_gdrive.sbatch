#!/usr/bin/env bash

#SBATCH --job-name=sync_gdrive
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=jobs/logs/sync_gdrive_%j.out
#SBATCH --error=jobs/logs/sync_gdrive_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs

echo "============================================================"
echo "SINCRONIZACIÓN DE VIDEOS CON GOOGLE DRIVE (RCLONE - CPU)"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

if ! command -v rclone &> /dev/null; then
    echo "ERROR: rclone no está instalado o en el PATH." >&2
    exit 1
fi

if ! rclone listremotes | grep -q "gdrive"; then
    echo "ERROR: El remote 'gdrive' no está configurado en rclone." >&2
    exit 1
fi

sincronizar_video() {
    local vid="$1"
    echo
    echo "--- Sincronizando ${vid} ---"

    if [[ -d "videos/${vid}/frames_originales" ]]; then
        echo "Subiendo frames_originales de ${vid}..."
        rclone copy "videos/${vid}/frames_originales" "gdrive:proyecto-FAC/videos/${vid}/frames_originales" \
            --transfers 16 --checkers 32 --fast-list --progress || true
    fi

    if [[ -d "videos/${vid}/mascaras_hud" ]]; then
        echo "Subiendo mascaras_hud de ${vid}..."
        rclone copy "videos/${vid}/mascaras_hud" "gdrive:proyecto-FAC/videos/${vid}/mascaras_hud" \
            --transfers 16 --checkers 32 --fast-list --progress || true
    fi

    if [[ -d "videos/${vid}/frames_sin_hud" ]]; then
        echo "Subiendo frames_sin_hud de ${vid}..."
        rclone copy "videos/${vid}/frames_sin_hud" "gdrive:proyecto-FAC/videos/${vid}/frames_sin_hud" \
            --transfers 16 --checkers 32 --fast-list --progress || true
    fi

    if [[ -f "videos/${vid}/resumen.xlsx" ]]; then
        echo "Subiendo resumen.xlsx de ${vid}..."
        rclone copy "videos/${vid}/resumen.xlsx" "gdrive:proyecto-FAC/videos/${vid}/" || true
    fi

    if [[ -f "videos/${vid}/resumen_experimentos.xlsx" ]]; then
        echo "Subiendo resumen_experimentos.xlsx de ${vid}..."
        rclone copy "videos/${vid}/resumen_experimentos.xlsx" "gdrive:proyecto-FAC/videos/${vid}/" || true
    fi
}

# 1. Sincronizar Video 1
sincronizar_video "video1"

# 2. Sincronizar Video 2 (rclone omite los que ya subió y retoma solo los faltantes)
sincronizar_video "video2"

echo
echo "============================================================"
echo "SINCRONIZACIÓN COMPLETADA EXITOSAMENTE"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"

