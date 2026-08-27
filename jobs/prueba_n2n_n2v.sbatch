#!/usr/bin/env bash

#SBATCH --job-name=prueba_n2n_n2v
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=04:00:00
#SBATCH --output=jobs/logs/prueba_n2n_n2v_%j.out
#SBATCH --error=jobs/logs/prueba_n2n_n2v_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; echo "Codigo: ${codigo}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video1}"
RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"

cd "${RUTA_PROYECTO}"

mkdir -p jobs/logs
mkdir -p cache/denoising

for script in pipeline_n2n.py pipeline_n2v.py calcular_metricas.py; do
    if [[ ! -f "scripts/${script}" ]]; then
        echo "ERROR: No se encontro scripts/${script}" >&2
        exit 1
    fi

    "${RUTA_PYTHON}" -m py_compile "scripts/${script}"
done

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export PYTHONUNBUFFERED="1"

echo "============================================================"
echo "PRUEBA N2N Y N2V CON PROCESOS AISLADOS"
echo "============================================================"
echo "Video: ${VIDEO_ID}"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Nodo: ${SLURMD_NODENAME:-local}"
echo "CPU: ${SLURM_CPUS_PER_TASK:-16}"
echo "GPU visibles: ${CUDA_VISIBLE_DEVICES:-desconocidas}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

nvidia-smi

"${RUTA_PYTHON}" - <<'PY'
import careamics
import torch

print(f"CAREamics: {careamics.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU visibles: {torch.cuda.device_count()}")

if careamics.__version__ != "0.3.2":
    raise RuntimeError(
        f"Se esperaba CAREamics 0.3.2 y se encontro "
        f"{careamics.__version__}"
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA no esta disponible")
PY

ejecutar_etapa() {
    local nombre="$1"
    shift

    local inicio
    local fin
    local duracion

    inicio="$(date +%s)"

    echo
    echo "------------------------------------------------------------"
    echo "ETAPA: ${nombre}"
    echo "Inicio: $(date --iso-8601=seconds)"
    echo "Comando: $*"
    echo "------------------------------------------------------------"

    "$@"

    fin="$(date +%s)"
    duracion="$((fin - inicio))"

    echo "Fin: $(date --iso-8601=seconds)"
    echo "Duracion: ${duracion} segundos"
}

ejecutar_etapa \
    "Preparacion Noise2Noise" \
    "${RUTA_PYTHON}" scripts/pipeline_n2n.py \
    --video "${VIDEO_ID}" \
    --modo original \
    --etapa preparar \
    --max-frames 100 \
    --epocas 1 \
    --pasos-por-epoca 5 \
    --reiniciar

ejecutar_etapa \
    "Entrenamiento Noise2Noise" \
    "${RUTA_PYTHON}" scripts/pipeline_n2n.py \
    --video "${VIDEO_ID}" \
    --modo original \
    --etapa entrenar \
    --max-frames 100 \
    --epocas 1 \
    --pasos-por-epoca 5

ejecutar_etapa \
    "Inferencia Noise2Noise" \
    "${RUTA_PYTHON}" scripts/pipeline_n2n.py \
    --video "${VIDEO_ID}" \
    --modo original \
    --etapa inferir \
    --max-frames 100 \
    --epocas 1 \
    --pasos-por-epoca 5

ejecutar_etapa \
    "Metricas Noise2Noise" \
    "${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo n2n

ejecutar_etapa \
    "Preparacion Noise2Void" \
    "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
    --video "${VIDEO_ID}" \
    --modo original \
    --etapa preparar \
    --max-frames 100 \
    --epocas 1 \
    --pasos-por-epoca 5 \
    --reiniciar

ejecutar_etapa \
    "Entrenamiento Noise2Void" \
    "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
    --video "${VIDEO_ID}" \
    --modo original \
    --etapa entrenar \
    --max-frames 100 \
    --epocas 1 \
    --pasos-por-epoca 5

ejecutar_etapa \
    "Inferencia Noise2Void" \
    "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
    --video "${VIDEO_ID}" \
    --modo original \
    --etapa inferir \
    --max-frames 100 \
    --epocas 1 \
    --pasos-por-epoca 5

ejecutar_etapa \
    "Metricas Noise2Void" \
    "${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo n2v

CANTIDAD_N2N="$(
    find "videos/${VIDEO_ID}/n2n" \
        -maxdepth 1 \
        -type f \
        -name "frame_*.png" |
    wc -l
)"

CANTIDAD_N2V="$(
    find "videos/${VIDEO_ID}/n2v" \
        -maxdepth 1 \
        -type f \
        -name "frame_*.png" |
    wc -l
)"

if [[ "${CANTIDAD_N2N}" -ne 100 ]]; then
    echo "ERROR: N2N genero ${CANTIDAD_N2N} frames." >&2
    exit 1
fi

if [[ "${CANTIDAD_N2V}" -ne 100 ]]; then
    echo "ERROR: N2V genero ${CANTIDAD_N2V} frames." >&2
    exit 1
fi

echo
echo "============================================================"
echo "PRUEBA COMPLETADA"
echo "N2N: ${CANTIDAD_N2N} frames"
echo "N2V: ${CANTIDAD_N2V} frames"
echo "Resumen: videos/${VIDEO_ID}/resumen.xlsx"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
