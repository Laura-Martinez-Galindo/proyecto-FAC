#!/usr/bin/env bash

#SBATCH --job-name=denoising_n2n_n2v
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:nvidia:2
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/logs/denoising_%j.out
#SBATCH --error=jobs/logs/denoising_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; echo "Codigo: ${codigo}" >&2; exit "${codigo}"' ERR

if [[ "$#" -ne 2 ]]; then
    echo "Uso: sbatch jobs/denoising_n2n_n2v.sbatch VIDEO_ID original|sin_hud" >&2
    exit 2
fi

VIDEO_ID="$1"
MODO="$2"

if [[ "${MODO}" != "original" && "${MODO}" != "sin_hud" ]]; then
    echo "ERROR: El modo debe ser original o sin_hud." >&2
    exit 2
fi

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"

cd "${RUTA_PROYECTO}"

mkdir -p jobs/logs
mkdir -p cache/denoising

if [[ ! -x "${RUTA_PYTHON}" ]]; then
    echo "ERROR: No se encontro Python en ${RUTA_PYTHON}" >&2
    exit 1
fi

for script in pipeline_n2n.py pipeline_n2v.py calcular_metricas.py; do
    if [[ ! -f "scripts/${script}" ]]; then
        echo "ERROR: No se encontro scripts/${script}" >&2
        exit 1
    fi

    "${RUTA_PYTHON}" -m py_compile "scripts/${script}"
done

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED="1"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

EPOCAS="${EPOCAS:-30}"
PASOS="${PASOS:-500}"

MAX_ARGS=()

if [[ -n "${MAX_FRAMES:-}" ]]; then
    MAX_ARGS=(
        --max-frames
        "${MAX_FRAMES}"
    )
fi

LOG_N2N="jobs/logs/n2n_${VIDEO_ID}_${MODO}_${SLURM_JOB_ID}.log"
LOG_N2V="jobs/logs/n2v_${VIDEO_ID}_${MODO}_${SLURM_JOB_ID}.log"

echo "============================================================"
echo "PIPELINE N2N Y N2V"
echo "============================================================"
echo "Video: ${VIDEO_ID}"
echo "Modo: ${MODO}"
echo "Job: ${SLURM_JOB_ID}"
echo "Nodo: ${SLURMD_NODENAME}"
echo "GPU asignadas: ${CUDA_VISIBLE_DEVICES}"
echo "CPU asignadas: ${SLURM_CPUS_PER_TASK}"
echo "Epocas: ${EPOCAS}"
echo "Pasos por epoca: ${PASOS}"
echo "Maximo de frames: ${MAX_FRAMES:-todos}"
echo "Log N2N: ${LOG_N2N}"
echo "Log N2V: ${LOG_N2V}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

nvidia-smi

ejecutar_n2n() {
    echo "Preparando N2N."

    CUDA_VISIBLE_DEVICES=0 \
    SLURM_CPUS_PER_TASK=8 \
    "${RUTA_PYTHON}" scripts/pipeline_n2n.py \
        --video "${VIDEO_ID}" \
        --modo "${MODO}" \
        --etapa preparar \
        "${MAX_ARGS[@]}" \
        --reiniciar

    echo "Entrenando N2N."

    CUDA_VISIBLE_DEVICES=0 \
    SLURM_CPUS_PER_TASK=8 \
    "${RUTA_PYTHON}" scripts/pipeline_n2n.py \
        --video "${VIDEO_ID}" \
        --modo "${MODO}" \
        --etapa entrenar \
        "${MAX_ARGS[@]}" \
        --epocas "${EPOCAS}" \
        --pasos-por-epoca "${PASOS}"

    echo "Ejecutando inferencia N2N."

    CUDA_VISIBLE_DEVICES=0 \
    SLURM_CPUS_PER_TASK=8 \
    "${RUTA_PYTHON}" scripts/pipeline_n2n.py \
        --video "${VIDEO_ID}" \
        --modo "${MODO}" \
        --etapa inferir \
        "${MAX_ARGS[@]}"

    echo "N2N terminado correctamente."
}

ejecutar_n2v() {
    echo "Preparando N2V."

    CUDA_VISIBLE_DEVICES=1 \
    SLURM_CPUS_PER_TASK=8 \
    "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
        --video "${VIDEO_ID}" \
        --modo "${MODO}" \
        --etapa preparar \
        "${MAX_ARGS[@]}" \
        --reiniciar

    echo "Entrenando N2V."

    CUDA_VISIBLE_DEVICES=1 \
    SLURM_CPUS_PER_TASK=8 \
    "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
        --video "${VIDEO_ID}" \
        --modo "${MODO}" \
        --etapa entrenar \
        "${MAX_ARGS[@]}" \
        --epocas "${EPOCAS}" \
        --pasos-por-epoca "${PASOS}"

    echo "Ejecutando inferencia N2V."

    CUDA_VISIBLE_DEVICES=1 \
    SLURM_CPUS_PER_TASK=8 \
    "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
        --video "${VIDEO_ID}" \
        --modo "${MODO}" \
        --etapa inferir \
        "${MAX_ARGS[@]}"

    echo "N2V terminado correctamente."
}

echo
echo "Iniciando N2N en GPU logica 0."

ejecutar_n2n > "${LOG_N2N}" 2>&1 &

PID_N2N="$!"

echo "Iniciando N2V en GPU logica 1."

ejecutar_n2v > "${LOG_N2V}" 2>&1 &

PID_N2V="$!"

ESTADO_N2N=0
ESTADO_N2V=0

wait "${PID_N2N}" || ESTADO_N2N="$?"
wait "${PID_N2V}" || ESTADO_N2V="$?"

echo
echo "Estado de N2N: ${ESTADO_N2N}"
echo "Estado de N2V: ${ESTADO_N2V}"

if [[ "${ESTADO_N2N}" -ne 0 ]]; then
    echo "ERROR: El pipeline N2N fallo." >&2
    tail -n 120 "${LOG_N2N}" >&2
fi

if [[ "${ESTADO_N2V}" -ne 0 ]]; then
    echo "ERROR: El pipeline N2V fallo." >&2
    tail -n 120 "${LOG_N2V}" >&2
fi

if [[ "${ESTADO_N2N}" -ne 0 || "${ESTADO_N2V}" -ne 0 ]]; then
    exit 1
fi

if [[ "${MODO}" == "original" ]]; then
    MODELO_N2N="n2n"
    MODELO_N2V="n2v"
else
    MODELO_N2N="n2n_sin_hud"
    MODELO_N2V="n2v_sin_hud"
fi

echo
echo "Calculando metricas de ${MODELO_N2N}."

CUDA_VISIBLE_DEVICES=0 \
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo "${MODELO_N2N}"

echo
echo "Calculando metricas de ${MODELO_N2V}."

CUDA_VISIBLE_DEVICES=0 \
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo "${MODELO_N2V}"

echo
echo "============================================================"
echo "PIPELINE COMPLETADO"
echo "Video: ${VIDEO_ID}"
echo "Modo: ${MODO}"
echo "Modelo N2N: ${MODELO_N2N}"
echo "Modelo N2V: ${MODELO_N2V}"
echo "Resumen: videos/${VIDEO_ID}/resumen.xlsx"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
