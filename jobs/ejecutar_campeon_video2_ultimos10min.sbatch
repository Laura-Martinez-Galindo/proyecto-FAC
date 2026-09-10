#!/usr/bin/env bash

#SBATCH --job-name=campeon_video2_10min
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia:2
#SBATCH --time=12:00:00
#SBATCH --output=jobs/logs/campeon_video2_10min_%j.out
#SBATCH --error=jobs/logs/campeon_video2_10min_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="video2"
# Ultimos 10 minutos a 29.97 fps = 17982 frames (~18000 frames)
MAX_FRAMES="${1:-18000}"
EPOCAS="${2:-30}"

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"
EXCEL_EXP="resumen_experimentos.xlsx"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs videos/${VIDEO_ID}/expos

echo "============================================================"
echo "EJECUCIÓN DEL MODELO CAMPEÓN (ENSAMBLE WAVELET UDVD + B2U)"
echo "SOBRE LOS ÚLTIMOS 10 MINUTOS DE VIDEO 2 (${MAX_FRAMES} FRAMES)"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

DIR_UDVD="videos/${VIDEO_ID}/expos/udvd_sinhud"
DIR_B2U="videos/${VIDEO_ID}/expos/blind2unblind_sinhud"
DIR_ENSEMBLE="videos/${VIDEO_ID}/expos/ensemble_wavelet_udvd_b2u_sinhud"

# 1. Entrenar e inferir UDVD en GPU 0
echo
echo "--- [1/3] Ejecutando Campeon Individual: UDVD (Dynamic Kernels) ---"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_udvd.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr 1e-3 \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca 300 \
    --tamano-lote 4 \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_UDVD}" \
    --id-experimento "udvd_sinhud"

echo "Calculando metricas UDVD..."
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo udvd \
    --carpeta-directa "${DIR_UDVD}" \
    --experimento "udvd_sinhud" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Epocas=${EPOCAS}, LR=1e-3, Kernel=5x5, Frames=${MAX_FRAMES}"

# 2. Entrenar e inferir Blind2Unblind en GPU 1
echo
echo "--- [2/3] Ejecutando Campeon de Texturas: Blind2Unblind ---"
CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_blind2unblind.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr 1e-3 \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca 300 \
    --tamano-lote 4 \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_B2U}" \
    --id-experimento "blind2unblind_sinhud"

echo "Calculando metricas Blind2Unblind..."
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo blind2unblind \
    --carpeta-directa "${DIR_B2U}" \
    --experimento "blind2unblind_sinhud" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Epocas=${EPOCAS}, LR=1e-3, Depth=4, Frames=${MAX_FRAMES}"

# 3. Fusión Wavelet Multiescala: Bajas Frecuencias (UDVD) + Altas Frecuencias (B2U)
echo
echo "--- [3/3] Fusión Multiescala Wavelet: UDVD + Blind2Unblind ---"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "${DIR_UDVD}" "${DIR_B2U}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_ENSEMBLE}" \
    --id-experimento "ensemble_wavelet_udvd_b2u_sinhud"

echo "Calculando metricas del Ensamble Ganador..."
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo ensemble \
    --carpeta-directa "${DIR_ENSEMBLE}" \
    --experimento "ensemble_wavelet_udvd_b2u_sinhud" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Metodo=Wavelet_Fusion, Base=UDVD, Textura=Blind2Unblind, Frames=${MAX_FRAMES}"

echo "============================================================"
echo "PROCESO COMPLETADO CON ÉXITO PARA VIDEO 2 (ÚLTIMOS 10 MIN)"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
