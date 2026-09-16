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
echo "EMPAQUETANDO Y SUBIENDO VIDEOS A GOOGLE DRIVE"
echo "============================================================"

DIR_RENDER="videos/${VIDEO_ID}/videos_renderizados"
DIR_COMPRIMIDOS="videos/comprimidos"
TAR_VIDEOS="${DIR_COMPRIMIDOS}/${VIDEO_ID}_videos_renderizados.tar.gz"

mkdir -p "${DIR_COMPRIMIDOS}"

echo "--- 1. Creando paquete comprimido ${TAR_VIDEOS} ---"
tar -cf - -C "videos/${VIDEO_ID}" "videos_renderizados" | gzip -1 > "${TAR_VIDEOS}"

if command -v rclone &> /dev/null; then
    RCLONE_FLAGS="--transfers 32 --checkers 64 --drive-chunk-size 64M --buffer-size 64M --fast-list --progress"
    
    echo "--- 2. Subiendo videos .mp4 individuales a Google Drive (visibles/reproducibles) ---"
    rclone copy "${DIR_RENDER}" "gdrive:proyecto-FAC/videos/${VIDEO_ID}/videos_renderizados/" ${RCLONE_FLAGS} || true
    
    echo "--- 3. Subiendo archivo comprimido ${TAR_VIDEOS} a Google Drive ---"
    rclone copy "${TAR_VIDEOS}" "gdrive:proyecto-FAC/videos/${VIDEO_ID}/comprimidos/" ${RCLONE_FLAGS} || true
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
