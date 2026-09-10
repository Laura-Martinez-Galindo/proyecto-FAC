#!/usr/bin/env bash

#SBATCH --job-name=wavelet_triple
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/wavelet_triple_%j.out
#SBATCH --error=jobs/logs/wavelet_triple_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video1}"
MAX_FRAMES="${2:-1000}"

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"
EXCEL_EXP="resumen_experimentos.xlsx"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs "videos/${VIDEO_ID}/expos"

echo "============================================================"
echo "EVALUACIÓN RÁPIDA DE TRIPLE FUSIÓN WAVELET"
echo "Modelos: UDVD + StructN2V Vertical + Blind2Unblind"
echo "Video: ${VIDEO_ID} | Frames: ${MAX_FRAMES}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

# Resolver rutas en videos/video1/expos/ o fallback a videos/video1/
resolver_dir() {
    local nombre="$1"
    if [ -d "videos/${VIDEO_ID}/expos/${nombre}" ]; then
        echo "videos/${VIDEO_ID}/expos/${nombre}"
    elif [ -d "videos/${VIDEO_ID}/${nombre}" ]; then
        echo "videos/${VIDEO_ID}/${nombre}"
    elif [ -d "videos/${VIDEO_ID}/expos/${nombre}_lr1e3" ]; then
        echo "videos/${VIDEO_ID}/expos/${nombre}_lr1e3"
    elif [ -d "videos/${VIDEO_ID}/${nombre}_lr1e3" ]; then
        echo "videos/${VIDEO_ID}/${nombre}_lr1e3"
    else
        echo "videos/${VIDEO_ID}/expos/${nombre}"
    fi
}

DIR_UDVD=$(resolver_dir "udvd_sinhud")
DIR_STRUCT=$(resolver_dir "struct_n2v_vert_sinhud")
DIR_B2U=$(resolver_dir "blind2unblind_sinhud")

echo "Usando carpetas base:"
echo "  UDVD: ${DIR_UDVD}"
echo "  StructN2V: ${DIR_STRUCT}"
echo "  Blind2Unblind: ${DIR_B2U}"

# 1. Generar la Triple Fusión Wavelet dentro de expos/
echo
echo "--- Generando ensemble_wavelet_triple_sinhud ---"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "${DIR_UDVD}" "${DIR_STRUCT}" "${DIR_B2U}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "expos/ensemble_wavelet_triple_sinhud"

# 2. Calcular Métricas en GPU
echo
echo "--- Calculando métricas de ensemble_wavelet_triple_sinhud ---"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo ensemble \
    --carpeta-directa "videos/${VIDEO_ID}/expos/ensemble_wavelet_triple_sinhud" \
    --experimento "ensemble_wavelet_triple_sinhud" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Familia=Ensamble, Tipo=Wavelet_Triple, Base=UDVD, Detalles=[StructVert, B2U]"

echo
echo "============================================================"
echo "TRIPLE FUSIÓN WAVELET EVALUADA CON ÉXITO"
echo "Resultados en: videos/${VIDEO_ID}/${EXCEL_EXP}"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
