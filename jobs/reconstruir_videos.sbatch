#!/usr/bin/env bash

#SBATCH --job-name=render_videos
#SBATCH --partition=express,short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=jobs/logs/render_videos_%j.out
#SBATCH --error=jobs/logs/render_videos_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video2}"
TIPO="${2:-todos}"  # todos, individual, comparativo_triple, comparativo_dual, comparativo_2x2
MAX_FRAMES="${3:-}" # Opcional, ej: 18000 o vacio para todos
FORZAR_FLAG="${4:---forzar}" # Por defecto sobrescribe renders antiguos de baja resolucion

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs "videos/${VIDEO_ID}/videos_renderizados"

echo "============================================================"
echo "🎬 RECONSTRUCCIÓN Y RENDERIZADO FULL HD 1080p FLIR"
echo "Video: ${VIDEO_ID} | Modo: ${TIPO} | Limite: ${MAX_FRAMES:-Completo}"
echo "Sobrescribir: ${FORZAR_FLAG}"
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
    ${PARAM_FRAMES} \
    ${FORZAR_FLAG}

echo
echo "============================================================"
echo "☁️ SUBIENDO VIDEOS .MP4 A GOOGLE DRIVE"
echo "============================================================"

DIR_RENDER="videos/${VIDEO_ID}/videos_renderizados"
sync

if command -v rclone &> /dev/null; then
    RCLONE_FLAGS="--transfers 16 --checkers 32 --drive-chunk-size 64M --buffer-size 64M --fast-list --progress"
    
    echo "--- Subiendo videos .mp4 Full HD a Google Drive (reproducibles directamente) ---"
    rclone copy "${DIR_RENDER}" "gdrive:proyecto-FAC/videos/${VIDEO_ID}/videos_renderizados/" ${RCLONE_FLAGS} || true
else
    echo "Advertencia: rclone no esta disponible en PATH para subida automatica."
fi

echo
echo "============================================================"
echo "VIDEOS RENDERIZADOS Y SUBIDOS EXITOSAMENTE A GOOGLE DRIVE"
echo "Carpeta local: ${DIR_RENDER}"
echo "Paquete local: ${TAR_VIDEOS}"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
ls -lh "${DIR_RENDER}"
