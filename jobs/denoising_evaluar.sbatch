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
# Antes de evaluar, a cada experimento <E> le crea <E>_ycolor: el brillo de su salida con el color de la
# entrada (scripts/color_original.py), para separar "limpia bien" de "cambia el color".
# Variables: RONDA (ronda1), FAMILIAS (por defecto las de config/denoising/familias.json),
#   IQA (p.ej. "niqe brisque musiq clipiqa"), COLOR_ORIGINAL (1; 0 = no crear las variantes _ycolor),
#   PATRON (que experimentos evaluar, p.ej. "*_r3*"; por defecto todos),
#   COMBINAR (1 = armar ensembles y stacks de los 3 mejores de cada regimen y volver a evaluar)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
RONDA="${RONDA:-ronda1}"
FAMILIAS="${FAMILIAS:-$(python -c "import json;print(' '.join(json.load(open('config/denoising/familias.json'))))")}"

evaluar_todo() {
for F in ${FAMILIAS}; do
  if [ "${COLOR_ORIGINAL:-1}" = "1" ]; then
    for d in cache/denoising/${F}/${PATRON:-*}/salidas; do
      E="$(basename "$(dirname "$d")")"
      [[ "$E" == *_ycolor ]] && continue
      [[ "$E" == ens3* || "$E" == stack* ]] && continue  # combinaciones: se arman de salidas ya corregidas o no
      python scripts/color_original.py --entrada datasets/dataset_sin_hud/images --base "$d" \
          --salida "cache/denoising/${F}/${E}_ycolor/salidas"
    done
  fi
  EXPS=()
  for d in cache/denoising/${F}/${PATRON:-*}/salidas; do
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
      ${IQA:+--iqa ${IQA}} \
      $( [ -d datasets/vecinos_eval/images ] && echo --vecinos-entrada datasets/vecinos_eval/images )
done
}

evaluar_todo
# Ensembles y stacks de los 3 mejores de cada regimen (scripts/combinaciones.py) y nueva evaluacion
if [ "${COMBINAR:-0}" = "1" ]; then
  for F in ${FAMILIAS}; do
    R="resultados/denoising/${RONDA}/${F}/ranking.csv"
    [ -f "$R" ] && python scripts/combinaciones.py --familia "$F" --ranking "$R" || echo "!! ${F}: sin combinaciones"
  done
  echo "=================== EVALUACION CON LAS COMBINACIONES ==================="
  evaluar_todo
fi

# un solo archivo para traer al PC
tar -cf "resultados/denoising/${RONDA}.tar" -C resultados/denoising "${RONDA}" listas
# muestras para mirar a ojo: 4 imagenes limpias + su residuo por experimento, y los logs de los carriles
MUESTRAS=()
for F in ${FAMILIAS}; do
  for d in cache/denoising/${F}/${PATRON:-*}/salidas; do
    [ -d "$d" ] || continue
    while IFS= read -r f; do MUESTRAS+=("$f"); done < <(ls "$d"/*.png 2>/dev/null | head -4; ls "$d"/residuo/*.png 2>/dev/null | head -4)
  done
done
tar -cf "resultados/denoising/${RONDA}_muestras.tar" "${MUESTRAS[@]}" $(ls jobs/logs/dn_gpu*.out 2>/dev/null)
echo "Muestras: resultados/denoising/${RONDA}_muestras.tar"
echo "Para revisar en el PC: resultados/denoising/${RONDA}.tar"
