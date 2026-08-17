#!/usr/bin/env bash

#SBATCH --job-name=n2n_prueba
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia:2
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
#SBATCH --output=Jobs/logs/n2n_prueba-log.o%j
#SBATCH --error=Jobs/logs/n2n_prueba-err.o%j

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

export N2N_EPOCAS=5
export N2N_CARPETA_FUENTE="frames_prueba"
export N2N_CARPETA_SALIDA="n2n_prueba"
export N2N_ETIQUETA_ORIGINAL="frames_prueba_original"
export N2N_ETIQUETA_CLEAN="frames_prueba_n2n"

echo "============================================================"
echo "Inicio: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-no_definido}"
echo "Nodo: $(hostname)"
echo "Python: ${PYTHON_ENV}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-no_definido}"
echo "Fuente: ${N2N_CARPETA_FUENTE}"
echo "Salida: ${N2N_CARPETA_SALIDA}"
echo "Épocas: ${N2N_EPOCAS}"
echo "============================================================"

if [ ! -x "${PYTHON_ENV}" ]; then
    echo "ERROR: no existe ${PYTHON_ENV}"
    exit 1
fi

if [ ! -d \
"Videos/video30min-11to22/${N2N_CARPETA_FUENTE}" ]; then
    echo "ERROR: no existe la carpeta de frames de prueba."
    exit 1
fi

"${PYTHON_ENV}" -m py_compile \
"Scripts/pipeline_n2n.py"

echo
echo "===== PREPARACIÓN ====="

"${PYTHON_ENV}" -u \
"Scripts/pipeline_n2n.py" \
--modo original \
--etapa preparar \
--reiniciar

echo
echo "===== ENTRENAMIENTO ====="

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
echo "===== INFERENCIA ====="

srun \
--nodes=1 \
--ntasks=1 \
--cpus-per-task=8 \
"${PYTHON_ENV}" -u \
"Scripts/pipeline_n2n.py" \
--modo original \
--etapa inferir

echo
echo "===== MÉTRICAS ====="

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
echo "Prueba FPS nativo terminada: $(date)"
echo "Modelo: Videos/video30min-11to22/n2n_prueba/modelo.ckpt"
echo "Clean: Videos/video30min-11to22/n2n_prueba/clean"
echo "Métricas: Videos/video30min-11to22/n2n_prueba/metricas_frames.csv"
echo "Resumen: Videos/video30min-11to22/resumen.xlsx"
echo "============================================================"
