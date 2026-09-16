#!/usr/bin/env bash

#SBATCH --job-name=render_videos
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=jobs/logs/render_videos_%j.out
#SBATCH --error=jobs/logs/render_videos_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video2}"
TIPO="${2:-todos}"  # todos, individual, comparativo_dual, comparativo_2x2
MAX_FRAMES="${3:-}" # Opcional, ej: 18000 o vacio para todos

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs "videos/${VIDEO_ID}/videos_renderizados"

echo "============================================================"
echo "RECONSTRUCCIÓN Y RENDERIZADO DE VIDEOS FLIR (SIDE-BY-SIDE)"
echo "Video: ${VIDEO_ID} | Modo: ${TIPO} | Limite: ${MAX_FRAMES:-Completo}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

PARAM_FRAMES=""
if [ -n "${MAX_FRAMES}" ]; then
    PARAM_FRAMES="--max-frames ${MAX_FRAMES}"
fi

"${RUTA_PYTHON}" scripts/reconstruir_video.py \
    --video "${VIDEO_ID}" \
    --tipo "${TIPO}" \
    --fps 30.0 \
    ${PARAM_FRAMES}

echo
echo "============================================================"
echo "VIDEOS GENERADOS EN: videos/${VIDEO_ID}/videos_renderizados/"
echo "============================================================"
ls -lh "videos/${VIDEO_ID}/videos_renderizados/"
