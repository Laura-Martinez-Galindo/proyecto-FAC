#!/usr/bin/env bash

#SBATCH --job-name=casos_estudio
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=jobs/logs/casos_estudio_%j.out
#SBATCH --error=jobs/logs/casos_estudio_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="video2"
MODELO_CARPETA="videos/video2/expos/udvd_destriping_t7_ult10min_sin_hud"
NOMBRE_MODELO="UDVD Destriping T=7 [Campeón]"
ULTIMOS_FRAMES="18000"
N_MUESTRAS="20"

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs figuras_tesis/casos_estudio_video2

echo "============================================================"
echo "EXTRACCIÓN AUTOMÁTICA DE CASOS DE ESTUDIO EXTREMOS"
echo "Video: ${VIDEO_ID} | Modelo: ${NOMBRE_MODELO}"
echo "Frames analizados: ${ULTIMOS_FRAMES} | Muestras: 20 Éxitos + 20 Difíciles"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

"${RUTA_PYTHON}" scripts/extraer_casos_estudio.py \
    --video "${VIDEO_ID}" \
    --modelo-carpeta "${MODELO_CARPETA}" \
    --nombre-modelo "${NOMBRE_MODELO}" \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --n-muestras "${N_MUESTRAS}" \
    --carpeta-salida "figuras_tesis/casos_estudio_video2"

echo
echo "============================================================"
echo "CASOS DE ESTUDIO GENERADOS EN: figuras_tesis/casos_estudio_video2/"
echo "============================================================"

# Subir a Google Drive
if command -v rclone &> /dev/null; then
    echo "Subiendo casos de estudio a Google Drive..."
    rclone copy "figuras_tesis/casos_estudio_video2" "gdrive:proyecto-FAC/figuras_tesis/casos_estudio_video2" --transfers 16 --checkers 32 || true
fi

echo "Fin: $(date --iso-8601=seconds)"
