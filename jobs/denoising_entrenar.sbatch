#!/usr/bin/env bash
#SBATCH --job-name=dn_entrenar
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=jobs/logs/dn_entrenar_%j.out
#SBATCH --error=jobs/logs/dn_entrenar_%j.err

# Un experimento entrenable: entrena con la lista de su familia y limpia las imagenes de
# evaluacion de esa familia (con residuo). Lo normal es lanzarlo con jobs/lanzar_ronda1.sh.
# Variables (obligatorias): FAMILIA, METODO, EXP     (opcionales): ARGS, PASOS (4000)
#   FAMILIA=OPTICO METODO=b2u EXP=b2u_r1 ARGS="--amp" sbatch jobs/denoising_entrenar.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv || true

: "${FAMILIA:?falta FAMILIA}" "${METODO:?falta METODO}" "${EXP:?falta EXP}"
L=resultados/denoising/listas
SAL="cache/denoising/${FAMILIA}/${EXP}"
mkdir -p "${SAL}"
# los argumentos de canales/amp tambien sirven al inferir
INF_ARGS=""
[[ " ${ARGS:-} " == *" --amp "* ]] && INF_ARGS="--amp"

python scripts/denoising.py entrenar --metodo "${METODO}" --familia "${FAMILIA}" \
    --lista "${L}/${FAMILIA}_entrenamiento.csv" --salida "${SAL}" \
    --pasos "${PASOS:-4000}" --instantaneas 2000 ${ARGS:-}

python scripts/denoising.py inferir --checkpoint "${SAL}/modelo.pt" \
    --lista "${L}/${FAMILIA}_evaluacion.csv" --entrada datasets/dataset_sin_hud/images \
    --salida "${SAL}/salidas" --residuo ${INF_ARGS}

echo "Listo: ${SAL}"
