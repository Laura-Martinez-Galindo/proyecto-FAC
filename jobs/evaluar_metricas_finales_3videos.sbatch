#!/usr/bin/env bash

#SBATCH --job-name=metricas_3videos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=jobs/logs/metricas_3videos_%j.out
#SBATCH --error=jobs/logs/metricas_3videos_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs figuras_tesis

echo "=============================================================================="
echo "📊 INICIANDO CÁLCULO DE MÉTRICAS CONSOLIDADAS (3 VIDEOS)"
echo "   Nodo: $(hostname)"
echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'N/A')"
echo "   Fecha: $(date --iso-8601=seconds)"
echo "=============================================================================="

# Activar ambiente virtual
if [[ -d "fac_env311" ]]; then
    source fac_env311/bin/activate
elif [[ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    conda activate tesis 2>/dev/null || conda activate base 2>/dev/null || true
fi

TARGET="${1:-todos}"
PASO="${2:-1}"

python scripts/evaluar_metricas_finales_3videos.py --video "${TARGET}" --device "cuda:0" --paso "${PASO}"

echo
echo "=============================================================================="
echo "🎉 CÁLCULO DE MÉTRICAS FINALIZADO CON ÉXITO: $(date --iso-8601=seconds)"
echo "=============================================================================="
