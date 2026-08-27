#!/usr/bin/env bash

#SBATCH --job-name=prueba_inferencia_n2n
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/prueba_inferencia_n2n_%j.out
#SBATCH --error=jobs/logs/prueba_inferencia_n2n_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"

cd "${RUTA_PROYECTO}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"${RUTA_PYTHON}" -m py_compile scripts/pipeline_n2n.py

nvidia-smi

"${RUTA_PYTHON}" scripts/pipeline_n2n.py \
    --video video1 \
    --modo original \
    --etapa inferir \
    --max-frames 100 \
    --epocas 1 \
    --pasos-por-epoca 5
