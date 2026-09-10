#!/usr/bin/env bash

#SBATCH --job-name=ensemble_wavelet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia:2
#SBATCH --time=1-00:00:00
#SBATCH --output=jobs/logs/ensemble_wavelet_%j.out
#SBATCH --error=jobs/logs/ensemble_wavelet_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video2}"
# Por defecto 18,000 frames (ultimos 10 minutos a 29.97 fps)
MAX_FRAMES="${2:-18000}"
EPOCAS="${3:-30}"
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

# 1. Entrenar e inferir UDVD en GPU 0
echo
echo "--- [1/3] Entrenando e Infiriendo UDVD (GPU 0) ---"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_udvd.py \
    --video "${VIDEO_ID}" \
    --modo "${MODO}" \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca 300 \
    --tamano-lote 4 \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_UDVD}" \
    --id-experimento "udvd_${MODO}" &

# 2. Entrenar e inferir Blind2Unblind en GPU 1
echo "--- [2/3] Entrenando e Infiriendo Blind2Unblind (GPU 1) ---"
CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_blind2unblind.py \
    --video "${VIDEO_ID}" \
    --modo "${MODO}" \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca 300 \
    --tamano-lote 4 \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_B2U}" \
    --id-experimento "blind2unblind_${MODO}" &
wait

echo "Calculando metricas individuales..."
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo udvd \
    --carpeta-directa "${DIR_UDVD}" \
    --experimento "udvd_${MODO}" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Epocas=${EPOCAS}, LR=${LR}, Kernel=5x5"

"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo blind2unblind \
    --carpeta-directa "${DIR_B2U}" \
    --experimento "blind2unblind_${MODO}" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Epocas=${EPOCAS}, LR=${LR}, Depth=4"

# 3. Fusión Wavelet Multiescala: Bajas Frecuencias (UDVD) + Altas Frecuencias (B2U)
echo
echo "--- [3/3] Fusión Multiescala Wavelet: UDVD + Blind2Unblind ---"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo "${MODO}" \
    --metodo-fusion wavelet \
    --modelos-entrada "${DIR_UDVD}" "${DIR_B2U}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_ENSEMBLE}" \
    --id-experimento "ensemble_wavelet_udvd_b2u_${MODO}"

echo "Calculando metricas del Ensamble Ganador..."
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
GDRIVE_ID="1RZGdPrUYRHiIQr2UAW7qKP7P7FmH476B"
if command -v rclone &> /dev/null; then
    if rclone listremotes | grep -q "gdrive"; then
        echo "Subiendo resultados a Google Drive..."
        rclone copy "${DIR_ENSEMBLE}" "gdrive:${GDRIVE_ID}/${VIDEO_ID}/expos/ensemble_wavelet_udvd_b2u_${MODO}" --transfers 8 --checkers 16 || true
        rclone copy "videos/${VIDEO_ID}/${EXCEL_EXP}" "gdrive:${GDRIVE_ID}/${VIDEO_ID}/" || true
    fi
fi
