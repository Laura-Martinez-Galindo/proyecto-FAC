#!/usr/bin/env bash

#SBATCH --job-name=tuning_ensemble_final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --output=jobs/logs/tuning_ensemble_final_%j.out
#SBATCH --error=jobs/logs/tuning_ensemble_final_%j.err
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
mkdir -p jobs/logs "videos/${VIDEO_ID}/expos"

echo "============================================================"
echo "TUNING DEFINITIVO DEL ENSAMBLE CAMPEÓN (2 GPUs, 140GB RAM)"
echo "Video: ${VIDEO_ID} | Frames: ${MAX_FRAMES}"
echo "Épocas: ${EPOCAS} | Pasos por época: ${PASOS}"
echo "Tasas de Aprendizaje: 3e-4 (menor), 1e-3 (base), 3e-3 (mayor)"
echo "Nodo: ${SLURMD_NODENAME:-local} | GPUs: ${CUDA_VISIBLE_DEVICES:-0,1}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

# ------------------------------------------------------------
# 1. ENTRENAMIENTO DE MODELOS INDIVIDUALES A 40 ÉPOCAS (2 GPUs)
# ------------------------------------------------------------

# Par 1: UDVD lr=3e-4 (GPU 0) y UDVD lr=3e-3 (GPU 1)
echo
echo "--- [1/3] Entrenando UDVD con lr=3e-4 (GPU 0) y lr=3e-3 (GPU 1) a 40 epocas ---"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_udvd.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --kernel-size 5 \
    --temporal-radius 2 \
    --lr 0.0003 \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "expos/udvd_sinhud_lr3e4_ep40" &
PID_1=$!

CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_udvd.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --kernel-size 5 \
    --temporal-radius 2 \
    --lr 0.003 \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "expos/udvd_sinhud_lr3e3_ep40" &
PID_2=$!

wait "${PID_1}"
wait "${PID_2}"

# Par 2: UDVD lr=1e-3 (GPU 0) y StructN2V Vertical lr=1e-3 (GPU 1)
echo
echo "--- [2/3] Entrenando UDVD lr=1e-3 (GPU 0) y StructN2V Vertical lr=1e-3 (GPU 1) ---"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_udvd.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --kernel-size 5 \
    --temporal-radius 2 \
    --lr 0.001 \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "expos/udvd_sinhud_lr1e3_ep40" &
PID_3=$!

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
    --id-experimento "expos/struct_n2v_vert_sinhud_lr1e3_ep40" &
PID_4=$!

wait "${PID_3}"
wait "${PID_4}"

# Par 3: Blind2Unblind lr=1e-3 (GPU 0) y Blind2Unblind lr=3e-4 (GPU 1)
echo
echo "--- [3/3] Entrenando Blind2Unblind lr=1e-3 (GPU 0) y lr=3e-4 (GPU 1) ---"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_blind2unblind.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --depth 4 \
    --lr 0.001 \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "expos/blind2unblind_sinhud_lr1e3_ep40" &
PID_5=$!

CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_blind2unblind.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --depth 4 \
    --lr 0.0003 \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "expos/blind2unblind_sinhud_lr3e4_ep40" &
PID_6=$!

wait "${PID_5}"
wait "${PID_6}"

# ------------------------------------------------------------
# 2. GENERACIÓN DE LOS ENSAMBLES DEFINITIVOS (NOMENCLATURA EXACTA)
# ------------------------------------------------------------
echo
echo "--- Generando Ensambles Definitivos ---"

# Ensamble Wavelet Campeón A: UDVD (Base Temporal) + Blind2Unblind (Detalle)
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "videos/${VIDEO_ID}/expos/udvd_sinhud_lr1e3_ep40" "videos/${VIDEO_ID}/expos/blind2unblind_sinhud_lr1e3_ep40" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "expos/ensemble_wavelet_udvd_b2u_sinhud"

# Ensamble Wavelet Campeón B: UDVD (Base Temporal) + StructN2V Vertical (Desrayado)
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "videos/${VIDEO_ID}/expos/udvd_sinhud_lr1e3_ep40" "videos/${VIDEO_ID}/expos/struct_n2v_vert_sinhud_lr1e3_ep40" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "expos/ensemble_wavelet_udvd_struct_vert_sinhud"

# Ensamble Wavelet Triple: UDVD + StructN2V Vert + Blind2Unblind
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "videos/${VIDEO_ID}/expos/udvd_sinhud_lr1e3_ep40" "videos/${VIDEO_ID}/expos/struct_n2v_vert_sinhud_lr1e3_ep40" "videos/${VIDEO_ID}/expos/blind2unblind_sinhud_lr1e3_ep40" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "expos/ensemble_wavelet_udvd_struct_vert_b2u_sinhud"

# Ensamble Ponderado: UDVD + StructN2V Vert + Blind2Unblind
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion promedio \
    --modelos-entrada "videos/${VIDEO_ID}/expos/udvd_sinhud_lr1e3_ep40" "videos/${VIDEO_ID}/expos/struct_n2v_vert_sinhud_lr1e3_ep40" "videos/${VIDEO_ID}/expos/blind2unblind_sinhud_lr1e3_ep40" \
    --pesos 0.45 0.35 0.20 \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "expos/ensemble_ponderado_udvd_struct_vert_b2u_sinhud"

# ------------------------------------------------------------
# 3. CÁLCULO DE MÉTRICAS EN GPU
# ------------------------------------------------------------
echo
echo "--- Calculando métricas de todos los modelos afinados ---"

MODELOS_EVAL=(
    "udvd_sinhud_lr3e4_ep40"
    "udvd_sinhud_lr1e3_ep40"
    "udvd_sinhud_lr3e3_ep40"
    "struct_n2v_vert_sinhud_lr1e3_ep40"
    "blind2unblind_sinhud_lr1e3_ep40"
    "blind2unblind_sinhud_lr3e4_ep40"
    "ensemble_wavelet_udvd_b2u_sinhud"
    "ensemble_wavelet_udvd_struct_vert_sinhud"
    "ensemble_wavelet_udvd_struct_vert_b2u_sinhud"
    "ensemble_ponderado_udvd_struct_vert_b2u_sinhud"
)

for EXP_NAME in "${MODELOS_EVAL[@]}"; do
    CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
        --video "${VIDEO_ID}" \
        --modelo ensemble \
        --carpeta-directa "videos/${VIDEO_ID}/expos/${EXP_NAME}" \
        --experimento "${EXP_NAME}" \
        --archivo-resumen "${EXCEL_EXP}" \
        --max-frames "${MAX_FRAMES}" \
        --parametros-extra "Epocas=${EPOCAS}, Pasos=${PASOS}, Modo=sinhud"
done

# Generar reporte y mosaico comparativo
"${RUTA_PYTHON}" scripts/analizar_experimentos.py --video "${VIDEO_ID}" --excel "${EXCEL_EXP}" --mosaico "mosaico_comparativo_final.png"

echo
echo "============================================================"
echo "TUNING DEFINITIVO DEL ENSAMBLE COMPLETADO CON ÉXITO"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
