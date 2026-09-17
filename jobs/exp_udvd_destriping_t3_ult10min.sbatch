#!/bin/bash
#SBATCH --job-name=udvd_t3_destrip
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=jobs/logs/exp_udvd_t3_ult10min_%j.out
#SBATCH --error=jobs/logs/exp_udvd_t3_ult10min_%j.err

set -e
mkdir -p jobs/logs

echo "=========================================================="
echo "Iniciando Experimento: UDVD Destriping T=3 (Ultimos 10 min)"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true

python scripts/pipeline_udvd.py \
    --video video2 \
    --modo sin_hud \
    --num-frames 3 \
    --destriping \
    --ultimos-frames 18000 \
    --epocas 10 \
    --batch-size 8 \
    --lr 0.0003 \
    --id-experimento udvd_destriping_t3_ult10min_sin_hud

echo "=========================================================="
echo "Finalizado UDVD T=3: $(date)"
echo "=========================================================="
