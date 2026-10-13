#!/usr/bin/env bash

#SBATCH --job-name=metricas_3v_2gpus
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=jobs/logs/metricas_distribucion_%j.out
#SBATCH --error=jobs/logs/metricas_distribucion_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail
trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs figuras_tesis/distribuciones_metricas

echo "=============================================================================="
echo "📊 INICIANDO EVALUACIÓN CIENTÍFICA DE MÉTRICAS Y DISTRIBUCIONES (2 GPUs)"
echo "   Nodo: $(hostname)"
echo "   GPUs asignadas: 2"
echo "   CPUs: 16"
echo "   Fecha: $(date --iso-8601=seconds)"
echo "=============================================================================="

nvidia-smi

# Activar entorno Python
if [[ -d "fac_env311" ]]; then
    source fac_env311/bin/activate
elif [[ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    conda activate tesis 2>/dev/null || conda activate base 2>/dev/null || true
fi

TARGET="${1:-todos}"
PASO="${2:-1}"

python scripts/evaluar_metricas_finales_3videos.py --video "${TARGET}" --paso "${PASO}"

echo
echo "=============================================================================="
echo "🎉 CÁLCULO DE MÉTRICAS Y GRÁFICAS DE DISTRIBUCIÓN FINALIZADO CON ÉXITO!"
echo "   Fecha: $(date --iso-8601=seconds)"
echo "=============================================================================="
