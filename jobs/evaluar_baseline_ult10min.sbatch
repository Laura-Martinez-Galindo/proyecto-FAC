#!/usr/bin/env bash

#SBATCH --job-name=eval_baseline_10min
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia:2
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/eval_baseline_10min_%j.out
#SBATCH --error=jobs/logs/eval_baseline_10min_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

VIDEO_ID="video2"
ULTIMOS_FRAMES="18000"
EXCEL_EXP="resumen_experimentos.xlsx"

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"

cd "${RUTA_PROYECTO}"

echo "============================================================"
echo "EVALUANDO LÍNEAS BASE (ORIGINAL Y SIN HUD) EN ÚLTIMOS 10 MIN"
echo "Video: ${VIDEO_ID} | Frames: ${ULTIMOS_FRAMES}"
echo "============================================================"

# GPU 0: Evaluar Original Crudo (últimos 18.000 frames)
echo "[GPU 0] Calculando métricas de Original Crudo (últimos 18.000 frames)..."
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo original \
    --experimento "original_ult10min" \
    --archivo-resumen "${EXCEL_EXP}" \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --parametros-extra "Original_Crudo_Ultimos_10min" &

# GPU 1: Evaluar Sin HUD ProPainter (últimos 18.000 frames)
echo "[GPU 1] Calculando métricas de Sin HUD ProPainter (últimos 18.000 frames)..."
CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo sin_hud \
    --experimento "sin_hud_ult10min" \
    --archivo-resumen "${EXCEL_EXP}" \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --parametros-extra "Sin_HUD_ProPainter_Ultimos_10min" &

wait

echo "============================================================"
echo "LÍNEAS BASE EVALUADAS CON ÉXITO EN: videos/${VIDEO_ID}/${EXCEL_EXP}"
echo "============================================================"
