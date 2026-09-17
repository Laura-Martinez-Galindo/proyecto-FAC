#!/bin/bash
#SBATCH --job-name=metricas_suite
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=jobs/logs/metricas_suite_%j.out
#SBATCH --error=jobs/logs/metricas_suite_%j.err

set -e
mkdir -p jobs/logs

echo "=========================================================="
echo "Iniciando Calculo de Metricas IQA (Suite Ultimos 10 min)"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true

# 1. Fusión Adaptativa Guiada por Movimiento
if [ -d "videos/video2/expos/fusion_motion_adaptive_udvd_struct_vert" ]; then
    echo ">>> Evaluando Fusión Adaptativa..."
    python scripts/calcular_metricas.py \
        --video video2 \
        --modelo frames2residual \
        --carpeta-directa videos/video2/expos/fusion_motion_adaptive_udvd_struct_vert \
        --experimento fusion_motion_adaptive_udvd_struct_vert \
        --archivo-resumen resumen_experimentos.xlsx \
        --ultimos-frames 18000
fi

# 2. UDVD Destriping T=3
if [ -d "videos/video2/expos/udvd_destriping_t3_ult10min_sin_hud" ]; then
    echo ">>> Evaluando UDVD Destriping T=3..."
    python scripts/calcular_metricas.py \
        --video video2 \
        --modelo frames2residual \
        --carpeta-directa videos/video2/expos/udvd_destriping_t3_ult10min_sin_hud \
        --experimento udvd_destriping_t3_ult10min_sin_hud \
        --archivo-resumen resumen_experimentos.xlsx \
        --ultimos-frames 18000
fi

# 3. UDVD Destriping T=5
if [ -d "videos/video2/expos/udvd_destriping_t5_ult10min_sin_hud" ]; then
    echo ">>> Evaluando UDVD Destriping T=5..."
    python scripts/calcular_metricas.py \
        --video video2 \
        --modelo frames2residual \
        --carpeta-directa videos/video2/expos/udvd_destriping_t5_ult10min_sin_hud \
        --experimento udvd_destriping_t5_ult10min_sin_hud \
        --archivo-resumen resumen_experimentos.xlsx \
        --ultimos-frames 18000
fi

# 4. UDVD Puro T=3
if [ -d "videos/video2/expos/udvd_t3_ult10min_sin_hud" ]; then
    echo ">>> Evaluando UDVD T=3 Puro..."
    python scripts/calcular_metricas.py \
        --video video2 \
        --modelo frames2residual \
        --carpeta-directa videos/video2/expos/udvd_t3_ult10min_sin_hud \
        --experimento udvd_t3_ult10min_sin_hud \
        --archivo-resumen resumen_experimentos.xlsx \
        --ultimos-frames 18000
fi

# 5. UDVD Puro T=5
if [ -d "videos/video2/expos/udvd_t5_ult10min_sin_hud" ]; then
    echo ">>> Evaluando UDVD T=5 Puro..."
    python scripts/calcular_metricas.py \
        --video video2 \
        --modelo frames2residual \
        --carpeta-directa videos/video2/expos/udvd_t5_ult10min_sin_hud \
        --experimento udvd_t5_ult10min_sin_hud \
        --archivo-resumen resumen_experimentos.xlsx \
        --ultimos-frames 18000
fi

# 6. Baselines (Original y Sin HUD en los ultimos 18000 frames)
echo ">>> Evaluando Baseline Original (Ultimos 10 min)..."
python scripts/calcular_metricas.py \
    --video video2 \
    --modelo original \
    --carpeta-directa videos/video2/frames_originales \
    --experimento original_ult10min \
    --archivo-resumen resumen_experimentos.xlsx \
    --ultimos-frames 18000

echo ">>> Evaluando Baseline Sin HUD (Ultimos 10 min)..."
python scripts/calcular_metricas.py \
    --video video2 \
    --modelo sin_hud \
    --carpeta-directa videos/video2/frames_sin_hud \
    --experimento sin_hud_ult10min \
    --archivo-resumen resumen_experimentos.xlsx \
    --ultimos-frames 18000

echo "=========================================================="
echo "Calculo de metricas completado exitosamente: $(date)"
echo "Resultados en: videos/video2/resumen_experimentos.xlsx"
echo "=========================================================="
