#!/usr/bin/env bash

#SBATCH --job-name=poc_udvd_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/poc_udvd_v2_%j.out
#SBATCH --error=jobs/logs/poc_udvd_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail
mkdir -p jobs/logs figuras_tesis

echo "=========================================================="
echo "🎬 PRUEBA DE CONCEPTO (PoC) UDVD CORREGIDO: VIDEO 2"
echo "Nodo: $(hostname) | Fecha: $(date)"
echo "=========================================================="

nvidia-smi

# Activar entorno Python
if [[ -d "fac_env311" ]]; then
    source fac_env311/bin/activate
elif [[ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    conda activate tesis 2>/dev/null || conda activate base 2>/dev/null || true
fi

python scripts/ejecutar_poc_video2_mejorado.py \
    --inicio-frame "frame_81588.png" \
    --num-frames 300 \
    --epocas 6

echo "=========================================================="
echo "PoC completado con éxito: $(date)"
echo "=========================================================="
