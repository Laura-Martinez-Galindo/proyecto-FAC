#!/bin/bash
#SBATCH --job-name=udvd_final_3videos
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=jobs/logs/udvd_final_3videos_%j.out
#SBATCH --error=jobs/logs/udvd_final_3videos_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail
mkdir -p jobs/logs

echo "=========================================================="
echo "RESTAURACIÓN OFICIAL DEFINITIVA (UDVD + DESTRIPING FPN + BLIND-SPOT)"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

nvidia-smi

source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true

VIDEO_SEL="${1:-todos}"

python scripts/ejecutar_pipeline_final_definitivo.py \
    --video "${VIDEO_SEL}" \
    --device cuda:0 \
    --batch-size 16

echo "=========================================================="
echo "Restauración de Videos completada con éxito: $(date)"
echo "=========================================================="
