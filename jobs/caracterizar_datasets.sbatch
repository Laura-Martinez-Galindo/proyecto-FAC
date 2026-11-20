#!/usr/bin/env bash
#SBATCH --job-name=caract_datasets
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=jobs/logs/caract_datasets_%j.out
#SBATCH --error=jobs/logs/caract_datasets_%j.err
# Ruido de cada imagen en original -> sin_hud -> denoised (scripts/caracterizar_datasets.py), CPU.
# Uso: sbatch jobs/caracterizar_datasets.sbatch      (IQA="" para no calcular NIQE/BRISQUE)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
python scripts/caracterizar_datasets.py --iqa ${IQA-niqe brisque} --workers 16
tar -cf resultados/caracterizacion_datasets.tar -C resultados caracterizacion_datasets && ls -lh resultados/caracterizacion_datasets.tar
