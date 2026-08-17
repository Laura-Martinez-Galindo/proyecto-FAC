#!/bin/bash
#SBATCH --job-name=extraer_frames
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=jobs/logs/extraer_frames_%j.out
#SBATCH --error=jobs/logs/extraer_frames_%j.err

set -euo pipefail

RUTA_PROYECTO="/hpcfs/home/ing_sistemas/lf.martinezg1/proyecto-FAC"

cd "$RUTA_PROYECTO"
source fac_env311/bin/activate

echo "========================================"
echo "Extracción de frames"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodo: ${SLURMD_NODENAME}"
echo "CPU asignadas: ${SLURM_CPUS_PER_TASK}"
echo "Memoria solicitada: 32G"
echo "Fecha de inicio: $(date)"
echo "Argumentos: $*"
echo "========================================"
echo

srun python scripts/extraer_frames.py "$@"

echo
echo "========================================"
echo "Extracción finalizada"
echo "Fecha de finalización: $(date)"
echo "========================================"
