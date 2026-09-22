#!/usr/bin/env bash
#SBATCH --job-name=udvd_corregido_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/udvd_corregido_v2_%j.out
#SBATCH --error=jobs/logs/udvd_corregido_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail
mkdir -p jobs/logs videos/video2/expos

echo "=========================================================="
echo "Experimento: UDVD Corregido (Blind-Spot Real) - Video 2 (Últimos 10 Min)"
echo "Nodo: $(hostname) | Fecha: $(date)"
echo "=========================================================="

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true

ULTIMOS_FRAMES="18000"
DIR_SALIDA="videos/video2/expos/udvd_blindspot_corregido_ult10min"

# 1. Ejecutar UDVD Corregido sobre los últimos 10 min de Video 2 (Sin HUD)
python scripts/pipeline_udvd.py \
    --video video2 \
    --modo sin_hud \
    --num-frames 5 \
    --kernel-size 5 \
    --lr 0.001 \
    --epocas 15 \
    --pasos-por-epoca 300 \
    --paciencia 6 \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --carpeta-salida "${DIR_SALIDA}" \
    --id-experimento udvd_blindspot_corregido_ult10min

# 2. Calcular Métricas NR-IQA sobre el resultado y consolidar en Excel
python scripts/calcular_metricas.py \
    --video video2 \
    --modelo udvd \
    --carpeta-directa "${DIR_SALIDA}" \
    --experimento udvd_blindspot_corregido_ult10min \
    --archivo-resumen resumen_experimentos.xlsx \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --parametros-extra "UDVD_BlindSpot_Corregido_T5_K5_ult10min"

echo "=========================================================="
echo "Finalizado con éxito: $(date)"
echo "=========================================================="
