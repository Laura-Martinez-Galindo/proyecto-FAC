#!/usr/bin/env bash

#SBATCH --job-name=campeon_40epocas
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/campeon_40epocas_%j.out
#SBATCH --error=jobs/logs/campeon_40epocas_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video1}"
MAX_FRAMES="${2:-1000}"
EPOCAS="${3:-40}"
PASOS="${4:-300}"

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"
EXCEL_EXP="resumen_experimentos.xlsx"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs

echo "============================================================"
echo "ENTRENAMIENTO DEFINITIVO DEL MODELO CAMPEÓN A ${EPOCAS} ÉPOCAS"
echo "Video: ${VIDEO_ID} | Frames: ${MAX_FRAMES}"
echo "Épocas: ${EPOCAS} | Pasos por época: ${PASOS}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

# Paso 1: Entrenar en paralelo UDVD (GPU 0) y StructN2V Vertical (GPU 1) a 40 epocas
echo
echo "--- [1/3] Entrenando UDVD (GPU 0) y StructN2V Vertical (GPU 1) en paralelo (${EPOCAS} epocas) ---"

CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_udvd.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --kernel-size 5 \
    --temporal-radius 2 \
    --lr 0.001 \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "UDVD_SinHUD_40ep" &
PID_GPU0=$!

CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --algoritmo struct_n2v \
    --struct-n2v vertical \
    --depth 3 \
    --lr 0.001 \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "StructN2V_Vert_SinHUD_40ep" &
PID_GPU1=$!

wait "${PID_GPU0}"
wait "${PID_GPU1}"

# Paso 2: Entrenar Blind2Unblind a 40 epocas
echo
echo "--- [2/3] Entrenando Blind2Unblind (${EPOCAS} epocas) ---"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_blind2unblind.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --depth 4 \
    --lr 0.001 \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "Blind2Unblind_SinHUD_40ep"

# Paso 3: Fusión Wavelet del Ensamble Definitivo
echo
echo "--- [3/3] Generando Ensamble Wavelet de los Campeones a ${EPOCAS} épocas ---"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "videos/${VIDEO_ID}/UDVD_SinHUD_40ep" "videos/${VIDEO_ID}/Blind2Unblind_SinHUD_40ep" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "Ensemble_Wavelet_UDVD_B2U_40ep"

"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "videos/${VIDEO_ID}/UDVD_SinHUD_40ep" "videos/${VIDEO_ID}/StructN2V_Vert_SinHUD_40ep" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "Ensemble_Wavelet_UDVD_StructVert_40ep"

# Paso 4: Calcular métricas y formatear el Excel
echo
echo "--- Evaluando métricas del Ensamble Definitivo ---"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo ensemble \
    --carpeta-directa "videos/${VIDEO_ID}/Ensemble_Wavelet_UDVD_B2U_40ep" \
    --experimento "Ensemble_Wavelet_UDVD_B2U_40ep" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Familia=Ensamble, Tipo=Wavelet, Base=UDVD_40ep, Detalle=B2U_40ep"

CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo ensemble \
    --carpeta-directa "videos/${VIDEO_ID}/Ensemble_Wavelet_UDVD_StructVert_40ep" \
    --experimento "Ensemble_Wavelet_UDVD_StructVert_40ep" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Familia=Ensamble, Tipo=Wavelet, Base=UDVD_40ep, Detalle=StructVert_40ep"

# Paso 5: Formatear y generar el mosaico final
"${RUTA_PYTHON}" scripts/analizar_todos_experimentos.py --video "${VIDEO_ID}" --excel "${EXCEL_EXP}"

echo
echo "============================================================"
echo "ENTRENAMIENTO Y ENSAMBLE A 40 ÉPOCAS FINALIZADO CON ÉXITO"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
