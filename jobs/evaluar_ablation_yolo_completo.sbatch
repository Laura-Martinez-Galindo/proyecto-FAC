#!/bin/bash
#SBATCH --job-name=eval_ablation_instancias
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=jobs/logs/eval_ablation_instancias_%j.out
#SBATCH --error=jobs/logs/eval_ablation_instancias_%j.err

set -e
mkdir -p jobs/logs videos

echo "=========================================================="
echo "Evaluacion de Ablacion YOLO: Metricas + Conteo de Instancias"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true

python scripts/evaluar_ablation_yolo_completo.py \
    --device 0 \
    --conf 0.25 \
    --iou 0.50 \
    --salida-excel videos/evaluacion_ablation_yolo_instancias.xlsx

echo "=========================================================="
echo "Evaluacion completada exitosamente."
echo "Fecha: $(date)"
echo "=========================================================="
