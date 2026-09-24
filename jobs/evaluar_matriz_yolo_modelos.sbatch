#!/bin/bash
#SBATCH --job-name=eval_yolo_matriz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/eval_yolo_matriz_%j.out
#SBATCH --error=jobs/logs/eval_yolo_matriz_%j.err

set -e
mkdir -p jobs/logs videos

echo "=========================================================="
echo "Iniciando Evaluacion de Matriz YOLOv11 en 2 GPUs"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true

python -c "import torch; print('CUDA Disponible:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

python scripts/evaluar_metricas_por_clase_y_modelos.py --device 0

echo "=========================================================="
echo "Evaluacion de Matriz YOLO completada con exito."
echo "Fecha: $(date)"
echo "=========================================================="
