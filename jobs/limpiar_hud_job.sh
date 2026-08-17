#!/usr/bin/env bash

#SBATCH --job-name=limpiar_hud
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia:2
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=limpiar_hud-log.o%j
#SBATCH --error=limpiar_hud-err.o%j

set -euo pipefail

PYTHON_ENV="/hpcfs/home/ing_sistemas/lf.martinezg1/proyecto-FAC/fac_env311/bin/python"

echo "============================================================"
echo "Inicio: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-no_definido}"
echo "Nodo: $(hostname)"
echo "Particion: ${SLURM_JOB_PARTITION:-no_definida}"
echo "Directorio: ${SLURM_SUBMIT_DIR:-no_definido}"
echo "CPU asignadas: ${SLURM_CPUS_PER_TASK:-no_definidas}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-no_definido}"
echo "Python: ${PYTHON_ENV}"
echo "============================================================"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR no esta definido}"

if [ ! -x "${PYTHON_ENV}" ]; then
    echo "ERROR: no existe el Python del entorno: ${PYTHON_ENV}"
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo
echo "Version de Python:"
"${PYTHON_ENV}" --version

echo
echo "Verificacion de PyTorch y GPU:"

"${PYTHON_ENV}" - <<'PY'
import os
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA de PyTorch: {torch.version.cuda}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'no definido')}")
print(f"GPU visibles dentro del job: {torch.cuda.device_count()}")

for indice in range(torch.cuda.device_count()):
    propiedades = torch.cuda.get_device_properties(indice)
    memoria_gb = propiedades.total_memory / 1024**3
    print(f"GPU local {indice}: {propiedades.name} - {memoria_gb:.2f} GB")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA no esta disponible dentro del trabajo")

if torch.cuda.device_count() != 2:
    raise RuntimeError(
        f"Se esperaban 2 GPU asignadas por Slurm, "
        f"pero PyTorch detecto {torch.cuda.device_count()}"
    )
PY

echo
echo "Estado inicial de las GPU:"
nvidia-smi

echo
echo "Iniciando limpieza del HUD con ProPainter"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PYTHON_ENV}" -u Scripts/limpiar_hud.py

echo
echo "Estado final de las GPU:"
nvidia-smi

echo
echo "Verificando resultados:"

CANTIDAD_RESULTADOS=$(
    find Videos/video30min-11to22/frames_sin_hud         -maxdepth 1         -type f         -name "*.png"         | wc -l
)

echo "Frames sin HUD encontrados: ${CANTIDAD_RESULTADOS}"

if [ "${CANTIDAD_RESULTADOS}" -ne 3300 ]; then
    echo "ERROR: se esperaban 3300 frames y se encontraron ${CANTIDAD_RESULTADOS}"
    exit 1
fi

echo
echo "============================================================"
echo "Trabajo terminado correctamente: $(date)"
echo "Resultados: Videos/video30min-11to22/frames_sin_hud"
echo "============================================================"
