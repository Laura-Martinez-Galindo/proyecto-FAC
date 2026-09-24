#!/bin/bash
#SBATCH --job-name=proc_ablation_pipeline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=nodei-gpu-2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/proc_ablation_pipeline_%j.out
#SBATCH --error=jobs/logs/proc_ablation_pipeline_%j.err

set -e
mkdir -p jobs/logs datasets/dataset_ablation_final

echo "=========================================================="
echo "Iniciando Pipeline de Transformacion de Dataset (HUD + UDVD)"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

nvidia-smi

source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true

python scripts/procesar_dataset_elegido_pipeline.py \
    --dataset-in datasets/dataset_preprocesado_11gb/modelo_yolov11_dataset_completo_preprocesado \
    --salida-dir datasets/dataset_ablation_final \
    --workers 8

echo "=========================================================="
echo "Pipeline de Transformacion completado exitosamente."
echo "Fecha: $(date)"
echo "=========================================================="
