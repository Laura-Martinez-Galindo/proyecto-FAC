#!/usr/bin/env bash

#SBATCH --job-name=n2n_infer_orig
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia:1
#SBATCH --mem=120G
#SBATCH --time=0-12:00:00
#SBATCH --output=Jobs/logs/n2n_infer_orig-log.o%j
#SBATCH --error=Jobs/logs/n2n_infer_orig-err.o%j

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

CHECKPOINT="Videos/video30min-11to22/n2n/modelo.ckpt"

echo "============================================================"
echo "Inicio: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-no_definido}"
echo "Nodo: $(hostname)"
echo "Python: ${PYTHON_ENV}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-no_definido}"
echo "Checkpoint: ${CHECKPOINT}"
echo "============================================================"

if [ ! -x "${PYTHON_ENV}" ]; then
    echo "ERROR: no existe el Python del entorno."
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: no existe ${CHECKPOINT}"
    exit 1
fi

"${PYTHON_ENV}" -m py_compile \
"Scripts/pipeline_n2n.py"

echo
echo "Verificando que el checkpoint no contenga NaN ni infinito"

"${PYTHON_ENV}" -u - <<'PY'
from pathlib import Path

import torch

ruta = Path(
    "Videos/video30min-11to22/n2n/modelo.ckpt"
)

checkpoint = torch.load(
    ruta,
    map_location="cpu",
    weights_only=False,
)

estado = checkpoint.get("state_dict", {})
tensores_nan = []
tensores_inf = []
cantidad_parametros = 0

for nombre, tensor in estado.items():
    if not torch.is_tensor(tensor):
        continue

    cantidad_parametros += tensor.numel()

    if torch.isnan(tensor).any().item():
        tensores_nan.append(nombre)

    if torch.isinf(tensor).any().item():
        tensores_inf.append(nombre)

print(f"Época: {checkpoint.get('epoch')}")
print(f"Paso global: {checkpoint.get('global_step')}")
print(f"Parámetros revisados: {cantidad_parametros:,}")
print(f"Tensores con NaN: {len(tensores_nan)}")
print(f"Tensores con infinito: {len(tensores_inf)}")

if tensores_nan:
    print("Primeros tensores con NaN:")

    for nombre in tensores_nan[:20]:
        print(f"  {nombre}")

if tensores_inf:
    print("Primeros tensores con infinito:")

    for nombre in tensores_inf[:20]:
        print(f"  {nombre}")

if tensores_nan or tensores_inf:
    raise RuntimeError(
        "El checkpoint contiene NaN o infinito. "
        "No se ejecutará la inferencia."
    )

print("Checkpoint válido: todos los pesos son finitos.")
PY

echo
echo "Iniciando inferencia sobre todos los frames originales"

"${PYTHON_ENV}" -u \
"Scripts/pipeline_n2n.py" \
--modo original \
--etapa inferir

echo
echo "Iniciando métricas"

"${PYTHON_ENV}" -u \
"Scripts/pipeline_n2n.py" \
--modo original \
--etapa metricas

echo
echo "============================================================"
echo "Finalización correcta: $(date)"
echo "============================================================"
