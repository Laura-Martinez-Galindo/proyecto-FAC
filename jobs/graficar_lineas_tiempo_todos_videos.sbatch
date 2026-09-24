#!/bin/bash
#SBATCH --job-name=lineas_tiempo_3videos
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/lineas_tiempo_3videos_%j.out
#SBATCH --error=jobs/logs/lineas_tiempo_3videos_%j.err

set -e
mkdir -p jobs/logs figuras_tesis/linea_tiempo

echo "=========================================================="
echo "Generacion Turbo de Lineas de Tiempo de Ruido (3 Videos)"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true

python scripts/graficar_lineas_ablation_3estados.py \
    --video todos \
    --stride-v1 15 \
    --stride-v2 30 \
    --stride-v3 60 \
    --batch-size 16 \
    --device cuda:0

echo "=========================================================="
echo "Lineas de Tiempo y Grafica Consolidada generadas con exito."
echo "Fecha: $(date)"
echo "=========================================================="
