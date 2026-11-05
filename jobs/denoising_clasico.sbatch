#!/usr/bin/env bash
#SBATCH --job-name=dn_clasico
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/dn_clasico_%j.out
#SBATCH --error=jobs/logs/dn_clasico_%j.err

# Metodo clasico (sin entrenamiento) sobre las imagenes de evaluacion de una familia.
# Variables: FAMILIA, METODO (nlm | destripe_nlm), EXP, ARGS
#   FAMILIA=TERMICO METODO=destripe_nlm EXP=destripe_nlm_h6 ARGS="--canales 1 --nlm-h 6" sbatch jobs/denoising_clasico.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate

: "${FAMILIA:?falta FAMILIA}" "${METODO:?falta METODO}" "${EXP:?falta EXP}"
SAL="cache/denoising/${FAMILIA}/${EXP}"
python scripts/denoising.py inferir --metodo "${METODO}" \
    --lista "resultados/denoising/listas/${FAMILIA}_evaluacion.csv" \
    --entrada datasets/dataset_sin_hud/images --salida "${SAL}/salidas" --residuo ${ARGS:-}
echo "Listo: ${SAL}"
