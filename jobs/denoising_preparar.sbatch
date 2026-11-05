#!/usr/bin/env bash
#SBATCH --job-name=dn_preparar
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/dn_preparar_%j.out
#SBATCH --error=jobs/logs/dn_preparar_%j.err

# Antes de la ronda 1:
#   1) vuelve a leer el regimen de cada frame con las plantillas nuevas (HDIR M WH DDE, EXP HST)
#      -> videos/<video>/regimenes_<video>.csv y datasets/dataset_original/regimenes_dataset.csv
#   2) listas por familia (entrenamiento / evaluacion / todas) -> resultados/denoising/listas/
#   3) mascaras del HUD de cada imagen del dataset -> datasets/dataset_sin_hud/mascaras_hud/
# Variables: REGIMENES=0 para saltar el paso 1 (si ya se corrio con las plantillas nuevas)
#            MAX_POR_FAMILIA (1500)  MAX_EVALUACION (120)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate

if [ "${REGIMENES:-1}" = "1" ]; then
  bash jobs/detectar_regimenes.sbatch
fi

python scripts/listas_denoising.py \
    --max-por-familia "${MAX_POR_FAMILIA:-1500}" --max-evaluacion "${MAX_EVALUACION:-120}" \
    --exportar-mascaras datasets/dataset_sin_hud/mascaras_hud

cat resultados/denoising/listas/resumen.csv
