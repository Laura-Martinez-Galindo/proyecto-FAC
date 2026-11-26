#!/usr/bin/env bash
#SBATCH --job-name=ds_variantes
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co
#SBATCH --output=jobs/logs/ds_variantes_%j.out
#SBATCH --error=jobs/logs/ds_variantes_%j.err

# Arma dataset_denoised_iqa: mismas imagenes, nombres y etiquetas que dataset_sin_hud, limpiadas con el
# mejor modelo por NIQE de cada regimen (config/denoising/ganadores_iqa.json). Solo copia imagenes: CPU.
# CON_HUD=1 arma ademas dataset_conhud_grano y dataset_conhud_iqa (solo si ya corrio inferir_con_hud.sbatch).
# Uso:  bash jobs/enviar.sh jobs/construir_variantes_denoising.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source fac_env311/bin/activate
python scripts/construir_dataset_denoised.py --ganadores config/denoising/ganadores_iqa.json \
    --salida datasets/dataset_denoised_iqa --forzar
DS="denoised_iqa"
if [ "${CON_HUD:-0}" = "1" ]; then
  python scripts/construir_dataset_denoised.py --sin-hud datasets/dataset_original --carpeta-salidas salidas_con_hud \
      --salida datasets/dataset_conhud_grano --forzar
  python scripts/construir_dataset_denoised.py --sin-hud datasets/dataset_original --carpeta-salidas salidas_con_hud \
      --ganadores config/denoising/ganadores_iqa.json --salida datasets/dataset_conhud_iqa --forzar
  DS="$DS conhud_grano conhud_iqa"
fi
for d in $DS; do
  echo "dataset_${d}: $(ls datasets/dataset_${d}/images | wc -l) imagenes, faltantes: $(($(wc -l < datasets/dataset_${d}/faltantes.csv) - 1))"
done
