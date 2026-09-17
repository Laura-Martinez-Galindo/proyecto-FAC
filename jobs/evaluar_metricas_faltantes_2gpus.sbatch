#!/bin/bash
#SBATCH --job-name=metricas_faltantes
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=jobs/logs/metricas_faltantes_%j.out
#SBATCH --error=jobs/logs/metricas_faltantes_%j.err

set -e
mkdir -p jobs/logs

echo "=========================================================="
echo "Iniciando Calculo de Metricas FALTANTES en 2 GPUs"
echo "Destino: videos/video2/resumen_experimentos.xlsx"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true

EXCEL="resumen_experimentos.xlsx"

evaluar_si_falta() {
    GPU_ID=$1
    CARPETA=$2
    EXP_NAME=$3
    TIPO_MOD=$4

    if [ -d "$CARPETA" ]; then
        echo "[GPU $GPU_ID] Procesando: $EXP_NAME..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/calcular_metricas.py \
            --video video2 \
            --modelo "$TIPO_MOD" \
            --carpeta-directa "$CARPETA" \
            --experimento "$EXP_NAME" \
            --archivo-resumen "$EXCEL" \
            --ultimos-frames 18000 \
            --omitir-existentes
    fi
}

# --- LOTE EN GPU 0 ---
(
    evaluar_si_falta 0 "videos/video2/expos/fusion_motion_adaptive_udvd_struct_vert" "fusion_motion_adaptive_udvd_struct_vert" "frames2residual"
    evaluar_si_falta 0 "videos/video2/expos/udvd_destriping_t3_ult10min_sin_hud" "udvd_destriping_t3_ult10min_sin_hud" "frames2residual"
    evaluar_si_falta 0 "videos/video2/expos/udvd_destriping_t5_ult10min_sin_hud" "udvd_destriping_t5_ult10min_sin_hud" "frames2residual"
    evaluar_si_falta 0 "videos/video2/frames_originales" "original_ult10min" "original"
) &
PID0=$!

# --- LOTE EN GPU 1 ---
(
    evaluar_si_falta 1 "videos/video2/expos/udvd_t3_ult10min_sin_hud" "udvd_t3_ult10min_sin_hud" "frames2residual"
    evaluar_si_falta 1 "videos/video2/expos/udvd_t5_ult10min_sin_hud" "udvd_t5_ult10min_sin_hud" "frames2residual"
    evaluar_si_falta 1 "videos/video2/expos/ensemble_multitemporal_con_destrip_t1_t3_t5_t7" "ensemble_multitemporal_con_destrip_t1_t3_t5_t7" "frames2residual"
    evaluar_si_falta 1 "videos/video2/frames_sin_hud" "sin_hud_ult10min" "sin_hud"
) &
PID1=$!

wait $PID0
wait $PID1

echo "=========================================================="
echo "Calculo de metricas faltantes completado: $(date)"
echo "Resultados actualizados en: videos/video2/$EXCEL"
echo "=========================================================="
