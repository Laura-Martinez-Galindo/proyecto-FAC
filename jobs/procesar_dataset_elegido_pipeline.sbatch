#!/bin/bash
#SBATCH --job-name=proc_ablation_pipeline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/proc_ablation_pipeline_%j.out
#SBATCH --error=jobs/logs/proc_ablation_pipeline_%j.err

set -e
mkdir -p jobs/logs datasets/dataset_ablation_final

echo "=========================================================="
echo "Iniciando Pipeline Ultra-Rapido de Transformacion (FP16 + 16 CPUS)"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/procesar_dataset_elegido_pipeline.py \
    --dataset-in datasets/dataset_preprocesado_11gb/modelo_yolov11_dataset_completo_preprocesado \
    --salida-dir datasets/dataset_ablation_final \
    --batch-size 4 \
    --workers 12 \
    --device cuda:0

echo "=========================================================="
echo "Pipeline de Transformacion completado exitosamente."
echo "Fecha: $(date)"
echo "=========================================================="
