#!/usr/bin/env bash
#SBATCH --job-name=dn_evaluar
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/dn_evaluar_%j.out
#SBATCH --error=jobs/logs/dn_evaluar_%j.err

# Compara TODOS los experimentos que existan en cache/denoising/<FAMILIA>/*/salidas, familia
# por familia, sobre sus imagenes de evaluacion. Resultado por familia en
#   resultados/denoising/<RONDA>/<FAMILIA>/  (metricas_por_imagen.csv, resumen.csv con
#   percentiles, ranking.csv, graficas/<metrica>.png con la distribucion por regimen)
# Variables: RONDA (ronda1), FAMILIAS ("OPTICO_CORRELADO HAZE_DDE BAJA_LUZ TERMICO"), IQA (p.ej. "niqe brisque")

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
RONDA="${RONDA:-ronda1}"

for F in ${FAMILIAS:-OPTICO_CORRELADO HAZE_DDE BAJA_LUZ TERMICO}; do
  EXPS=()
  for d in cache/denoising/${F}/*/salidas; do
    [ -d "$d" ] || continue
    EXPS+=(--experimento "$(basename "$(dirname "$d")")=$d")
  done
  if [ ${#EXPS[@]} -eq 0 ]; then echo "!! ${F}: no hay experimentos"; continue; fi
  echo "=================== ${F} ==================="
  python scripts/evaluar_denoising.py --entrada datasets/dataset_sin_hud/images \
      --lista "resultados/denoising/listas/${F}_evaluacion.csv" \
      --regimenes datasets/dataset_original/regimenes_dataset.csv \
      --mascaras datasets/dataset_sin_hud/mascaras_hud \
      "${EXPS[@]}" --salida "resultados/denoising/${RONDA}/${F}" \
      ${IQA:+--iqa ${IQA}}
done

# un solo archivo para traer al PC
tar -cf "resultados/denoising/${RONDA}.tar" -C resultados/denoising "${RONDA}" listas
# muestras para mirar a ojo: 4 imagenes limpias + su residuo por experimento, y los logs de los carriles
MUESTRAS=()
for F in ${FAMILIAS:-OPTICO_CORRELADO HAZE_DDE BAJA_LUZ TERMICO}; do
  for d in cache/denoising/${F}/*/salidas; do
    [ -d "$d" ] || continue
    while IFS= read -r f; do MUESTRAS+=("$f"); done < <(ls "$d"/*.png 2>/dev/null | head -4; ls "$d"/residuo/*.png 2>/dev/null | head -4)
  done
done
tar -cf "resultados/denoising/${RONDA}_muestras.tar" "${MUESTRAS[@]}" $(ls jobs/logs/dn_gpu*.out 2>/dev/null)
echo "Muestras: resultados/denoising/${RONDA}_muestras.tar"
echo "Para revisar en el PC: resultados/denoising/${RONDA}.tar"
