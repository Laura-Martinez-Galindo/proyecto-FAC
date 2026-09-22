#!/usr/bin/env bash
#SBATCH --job-name=udvd_corregido_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=jobs/logs/udvd_corregido_v2_%j.out
#SBATCH --error=jobs/logs/udvd_corregido_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail
mkdir -p jobs/logs

echo "=========================================================="
echo "Experimento: UDVD Corregido (Blind-Spot Real) - Video 2"
echo "Nodo: $(hostname) | Fecha: $(date)"
echo "=========================================================="

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true

# 1. Ejecutar UDVD Corregido sobre Video 2 (Sin HUD)
python scripts/pipeline_udvd.py \
    --video video2 \
    --modo sin_hud \
    --num-frames 5 \
    --kernel-size 5 \
    --lr 0.001 \
    --epocas 15 \
    --pasos-por-epoca 300 \
    --paciencia 6 \
    --id-experimento udvd_blindspot_corregido_sin_hud

# 2. Calcular Métricas NR-IQA sobre el resultado
python scripts/calcular_metricas.py \
    --video video2 \
    --modelo udvd \
    --experimento udvd_blindspot_corregido_sin_hud \
    --archivo-resumen resumen_experimentos.xlsx \
    --parametros-extra "UDVD_BlindSpot_T5_K5_lr1e3_Corregido"

echo "=========================================================="
echo "Finalizado con éxito: $(date)"
echo "=========================================================="
