#!/usr/bin/env bash

#SBATCH --job-name=ensemble_wavelet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/logs/ensemble_wavelet_%j.out
#SBATCH --error=jobs/logs/ensemble_wavelet_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video2}"
# Por defecto 83,011 frames (todo el video)
MAX_FRAMES="${2:-83011}"
EPOCAS="${3:-50}"
LR="${4:-1e-3}"
MODO="${5:-sin_hud}"

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"
EXCEL_EXP="resumen_experimentos.xlsx"
DIR_EXPOS="videos/${VIDEO_ID}/expos"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs "${DIR_EXPOS}"

echo "============================================================"
echo "EJECUCIÓN DEL ENSAMBLE WAVELET CAMPEÓN (UDVD + B2U)"
echo "Video: ${VIDEO_ID} | Modo: ${MODO} | Frames: ${MAX_FRAMES}"
echo "Hiperparametros: Epocas=${EPOCAS}, LR=${LR}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

DIR_UDVD="${DIR_EXPOS}/udvd_${MODO}"
DIR_B2U="${DIR_EXPOS}/blind2unblind_${MODO}"
DIR_ENSEMBLE="${DIR_EXPOS}/ensemble_wavelet_udvd_b2u_${MODO}"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)

# 1. ENTRENAMIENTO E INFERENCIA DE MODELOS BASE
if [ "$NUM_GPUS" -ge 2 ]; then
    echo "--- [1/2] Detectadas ${NUM_GPUS} GPUs: Ejecutando UDVD (GPU 0) y Blind2Unblind (GPU 1) en paralelo ---"
    CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_udvd.py \
        --video "${VIDEO_ID}" \
        --modo "${MODO}" \
        --lr "${LR}" \
        --epocas "${EPOCAS}" \
        --pasos-por-epoca 300 \
        --batch-size 4 \
        --max-frames "${MAX_FRAMES}" \
        --carpeta-salida "${DIR_UDVD}" \
        --id-experimento "expos/udvd_${MODO}" &

    CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_blind2unblind.py \
        --video "${VIDEO_ID}" \
        --modo "${MODO}" \
        --lr "${LR}" \
        --epocas "${EPOCAS}" \
        --pasos-por-epoca 300 \
        --batch-size 4 \
        --max-frames "${MAX_FRAMES}" \
        --carpeta-salida "${DIR_B2U}" \
        --id-experimento "expos/blind2unblind_${MODO}" &
    wait
else
    echo "--- Detectada 1 GPU: Ejecutando UDVD y Blind2Unblind secuencialmente en GPU 0 ---"
    echo "--- Entrenando e Infiriendo UDVD (GPU 0) ---"
    "${RUTA_PYTHON}" scripts/pipeline_udvd.py \
        --video "${VIDEO_ID}" \
        --modo "${MODO}" \
        --lr "${LR}" \
        --epocas "${EPOCAS}" \
        --pasos-por-epoca 300 \
        --batch-size 4 \
        --max-frames "${MAX_FRAMES}" \
        --carpeta-salida "${DIR_UDVD}" \
        --id-experimento "expos/udvd_${MODO}"

    echo "--- Entrenando e Infiriendo Blind2Unblind (GPU 0) ---"
    "${RUTA_PYTHON}" scripts/pipeline_blind2unblind.py \
        --video "${VIDEO_ID}" \
        --modo "${MODO}" \
        --lr "${LR}" \
        --epocas "${EPOCAS}" \
        --pasos-por-epoca 300 \
        --batch-size 4 \
        --max-frames "${MAX_FRAMES}" \
        --carpeta-salida "${DIR_B2U}" \
        --id-experimento "expos/blind2unblind_${MODO}"
fi

# 2. FUSIÓN MULTIESCALA WAVELET (Genera el Ensamble Campeón)
echo
echo "--- [2/2] Fusión Multiescala Wavelet: UDVD + Blind2Unblind (28 CPUs) ---"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo "${MODO}" \
    --metodo-fusion wavelet \
    --modelos-entrada "${DIR_UDVD}" "${DIR_B2U}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_ENSEMBLE}" \
    --id-experimento "expos/ensemble_wavelet_udvd_b2u_${MODO}"

# 3. CÁLCULO DE MÉTRICAS PARA LOS 3 MODELOS
echo
echo "============================================================"
echo "CALCULANDO MÉTRICAS DE CALIDAD (NIQE, BRISQUE, PIQE, SIGMA)"
echo "============================================================"

echo "1. Métricas UDVD (${DIR_UDVD})..."
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo udvd \
    --carpeta-directa "${DIR_UDVD}" \
    --experimento "udvd_${MODO}" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Epocas=${EPOCAS}, LR=${LR}, Kernel=5x5"

echo "2. Métricas Blind2Unblind (${DIR_B2U})..."
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo blind2unblind \
    --carpeta-directa "${DIR_B2U}" \
    --experimento "blind2unblind_${MODO}" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Epocas=${EPOCAS}, LR=${LR}, Depth=4"

echo "3. Métricas Ensamble Wavelet Ganador (${DIR_ENSEMBLE})..."
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo ensemble \
    --carpeta-directa "${DIR_ENSEMBLE}" \
    --experimento "ensemble_wavelet_udvd_b2u_${MODO}" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Fusion=Wavelet, Base=UDVD, Textura=Blind2Unblind, Frames=${MAX_FRAMES}"

echo "============================================================"
echo "PROCESO COMPLETADO EXITOSAMENTE"
echo "Resultados en: videos/${VIDEO_ID}/${EXCEL_EXP}"
echo "============================================================"

# Sincronizacion automatica con Google Drive si rclone esta configurado
if command -v rclone &> /dev/null; then
    if rclone listremotes | grep -q "gdrive"; then
        echo "Subiendo resultados a Google Drive (proyecto-FAC/videos/${VIDEO_ID})..."
        rclone copy "${DIR_ENSEMBLE}" "gdrive:proyecto-FAC/videos/${VIDEO_ID}/expos/ensemble_wavelet_udvd_b2u_${MODO}" --transfers 8 --checkers 16 || true
        rclone copy "videos/${VIDEO_ID}/${EXCEL_EXP}" "gdrive:proyecto-FAC/videos/${VIDEO_ID}/" || true
    fi
fi
