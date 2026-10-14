#!/usr/bin/env bash

#SBATCH --job-name=presentacion_tesis
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=jobs/logs/presentacion_tesis_%j.out
#SBATCH --error=jobs/logs/presentacion_tesis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail
trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video2}"
N_CASOS="${2:-5}"

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs "figuras_tesis/presentacion_${VIDEO_ID}"

echo "=============================================================================="
echo "🎬 GENERACIÓN AUTOMÁTICA DE PRESENTACIÓN PPTX Y CASOS DE ESTUDIO"
echo "   Misión: ${VIDEO_ID} | Casos por categoría: ${N_CASOS}"
echo "   Diseño: Centrado a lo alto (1920x1080 Full HD, estilo mario.pptx)"
echo "   Inicio: $(date --iso-8601=seconds)"
echo "=============================================================================="

# Asegurar python-pptx instalado en el entorno
"${RUTA_PYTHON}" -m pip install python-pptx --quiet 2>/dev/null || true

# Ejecutar generador
"${RUTA_PYTHON}" scripts/generar_presentacion_tesis.py \
    --video "${VIDEO_ID}" \
    --n-casos "${N_CASOS}"

echo
echo "=============================================================================="
echo "☁️ SUBIENDO PRESENTACIÓN Y DIAPOSITIVAS A GOOGLE DRIVE"
echo "=============================================================================="

sync

if command -v rclone &> /dev/null; then
    DIR_LOCAL="figuras_tesis/presentacion_${VIDEO_ID}"
    PPTX_LOCAL="figuras_tesis/presentacion_casos_estudio_${VIDEO_ID}.pptx"

    echo "--- 1. Subiendo carpeta de diapositivas PNG ---"
    rclone copy "${DIR_LOCAL}" "gdrive:proyecto-FAC/figuras_tesis/presentacion_${VIDEO_ID}/" --transfers 16 --checkers 32 || true

    if [[ -f "${PPTX_LOCAL}" ]]; then
        echo "--- 2. Subiendo archivo PowerPoint (.pptx) ---"
        rclone copy "${PPTX_LOCAL}" "gdrive:proyecto-FAC/figuras_tesis/" --transfers 8 --checkers 16 || true
    fi
fi

echo "=============================================================================="
echo "🎉 FINALIZADO CON ÉXITO: $(date --iso-8601=seconds)"
echo "=============================================================================="
