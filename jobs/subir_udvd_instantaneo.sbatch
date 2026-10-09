#!/usr/bin/env bash

#SBATCH --job-name=sync_instantaneo
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=jobs/logs/sync_instantaneo_%j.out
#SBATCH --error=jobs/logs/sync_instantaneo_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${RUTA_PROYECTO}"

TARGET="${1:-todos}"

echo "=============================================================================="
echo "⚡ LANZANDO SUBIDA INSTANTÁNEA A GOOGLE DRIVE"
echo "   Target: ${TARGET}"
echo "   Fecha: $(date --iso-8601=seconds)"
echo "=============================================================================="

bash scripts/empaquetar_y_subir_drive_rapido.sh "${TARGET}"
