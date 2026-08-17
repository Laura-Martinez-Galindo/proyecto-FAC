#!/usr/bin/env bash

#SBATCH --job-name=n2n_original
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia:2
#SBATCH --mem=150G
#SBATCH --time=0-08:00:00
#SBATCH --output=Jobs/logs/n2n_original-log.o%j
#SBATCH --error=Jobs/logs/n2n_original-err.o%j

set -euo pipefail

PYTHON_ENV="/hpcfs/home/ing_sistemas/lf.martinezg1/proyecto-FAC/fac_env311/bin/python"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR no esta definido}"

mkdir -p "Jobs/logs"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG=WARN

# Prueba integral corta. Para la corrida final, cambiar a 50.
export N2N_EPOCAS=50

echo "============================================================"
echo "Inicio: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-no_definido}"
echo "Nodo: $(hostname)"
echo "Python: ${PYTHON_ENV}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-no_definido}"
echo "Modo: original"
echo "Épocas solicitadas: ${N2N_EPOCAS}"
echo "============================================================"

if [ ! -x "${PYTHON_ENV}" ]; then
    echo "ERROR: no existe el Python del entorno: ${PYTHON_ENV}"
    exit 1
fi

"${PYTHON_ENV}" -m py_compile \
"Scripts/pipeline_n2n.py"

echo
echo "Preparando parejas de entrenamiento y validación"

"${PYTHON_ENV}" -u \
"Scripts/pipeline_n2n.py" \
--modo original \
--etapa preparar \
--reiniciar

echo
echo "Entrenando N2N original con ${N2N_EPOCAS} épocas"

srun \
--nodes=1 \
--ntasks=2 \
--ntasks-per-node=2 \
--cpus-per-task=8 \
"${PYTHON_ENV}" -u \
"Scripts/pipeline_n2n.py" \
--modo original \
--etapa entrenar

echo
echo "Entrenamiento terminado. Iniciando inferencia en un proceso nuevo."

srun \
--nodes=1 \
--ntasks=1 \
--cpus-per-task=8 \
"${PYTHON_ENV}" -u \
"Scripts/pipeline_n2n.py" \
--modo original \
--etapa inferir

echo
echo "Inferencia terminada. Calculando métricas."

srun \
--nodes=1 \
--ntasks=1 \
--cpus-per-task=8 \
"${PYTHON_ENV}" -u \
"Scripts/pipeline_n2n.py" \
--modo original \
--etapa metricas

echo
echo "============================================================"
echo "Pipeline original terminado correctamente: $(date)"
echo "Épocas: ${N2N_EPOCAS}"
echo "Modelo: Videos/video30min-11to22/n2n/modelo.ckpt"
echo "Frames: Videos/video30min-11to22/n2n/clean"
echo "Métricas: Videos/video30min-11to22/n2n/metricas_frames.csv"
echo "Resumen: Videos/video30min-11to22/resumen.xlsx"
echo "============================================================"
