#!/bin/bash
#SBATCH --job-name=udvd_final_multigpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=jobs/logs/udvd_final_multigpu_%j.out
#SBATCH --error=jobs/logs/udvd_final_multigpu_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail
mkdir -p jobs/logs

echo "=========================================================="
echo "RESTAURACIÓN OFICIAL MULTI-GPU (2 GPUs PARALELO)"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

nvidia-smi

# Activar entorno Python
if [[ -d "fac_env311" ]]; then
    source fac_env311/bin/activate
elif [[ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    conda activate tesis 2>/dev/null || conda activate base 2>/dev/null || true
fi

VIDEO_SEL="${1:-todos}"

python scripts/ejecutar_pipeline_final_definitivo.py \
    --video "${VIDEO_SEL}" \
    --batch-size 4

echo "=========================================================="
echo "Restauración Multi-GPU completada con éxito: $(date)"
echo "=========================================================="
