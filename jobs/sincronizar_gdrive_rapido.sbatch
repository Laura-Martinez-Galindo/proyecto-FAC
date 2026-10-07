#!/usr/bin/env bash

#SBATCH --job-name=sync_rapido_gdrive
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=jobs/logs/sync_rapido_%j.out
#SBATCH --error=jobs/logs/sync_rapido_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs videos/comprimidos

TARGET="${1:-todos}"

echo "=============================================================================="
echo "🚀 LANZANDO SINCRONIZACIÓN Y COMPRESIÓN MULTI-HILO A GOOGLE DRIVE"
echo "   Target: ${TARGET}"
echo "   Nodo: $(hostname)"
echo "   CPUs asignadas: 16"
echo "   Fecha: $(date --iso-8601=seconds)"
echo "=============================================================================="

bash scripts/empaquetar_y_subir_drive_rapido.sh "${TARGET}"
