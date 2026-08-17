#!/usr/bin/env bash

#SBATCH --job-name=extraer_prueba
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=Jobs/logs/extraer_frames_prueba-log.o%j
#SBATCH --error=Jobs/logs/extraer_frames_prueba-err.o%j

set -euo pipefail

PYTHON_ENV="/hpcfs/home/ing_sistemas/lf.martinezg1/proyecto-FAC/fac_env311/bin/python"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR no esta definido}"

mkdir -p "Jobs/logs"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "Inicio: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-no_definido}"
echo "Nodo: $(hostname)"
echo "Python: ${PYTHON_ENV}"
echo "============================================================"

if [ ! -x "${PYTHON_ENV}" ]; then
    echo "ERROR: no existe ${PYTHON_ENV}"
    exit 1
fi

"${PYTHON_ENV}" \
-m py_compile \
"Scripts/extraer_frames_prueba.py"

"${PYTHON_ENV}" -u \
"Scripts/extraer_frames_prueba.py"

echo
du -sh \
"Videos/video30min-11to22/frames_prueba"

echo
echo "Finalización correcta: $(date)"
