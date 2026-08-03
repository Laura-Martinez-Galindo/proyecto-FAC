#!/usr/bin/env bash
#SBATCH --job-name=n2n_original
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia:2
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
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

echo "Inicio: $(date)"
echo "Python: ${PYTHON_ENV}"
echo "Nodo: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-no_definido}"

"${PYTHON_ENV}" -m py_compile "Scripts/pipeline_n2n_video.py"
"${PYTHON_ENV}" -u "Scripts/pipeline_n2n_video.py" --modo original --etapa preparar --reiniciar

rm -rf "Videos/video30min-11to22/n2n/.careamics_trabajo"
mkdir -p "Videos/video30min-11to22/n2n/.careamics_trabajo"

srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=8 \
"${PYTHON_ENV}" -u "Scripts/pipeline_n2n_video.py" --modo original --etapa entrenar

# Esta etapa empieza en un proceso nuevo, despues de liberar los procesos DDP.
srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
"${PYTHON_ENV}" -u "Scripts/pipeline_n2n_video.py" --modo original --etapa inferir

srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
"${PYTHON_ENV}" -u "Scripts/pipeline_n2n_video.py" --modo original --etapa metricas

echo "Fin: $(date)"