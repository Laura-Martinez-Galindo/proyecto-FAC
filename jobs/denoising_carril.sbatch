#!/usr/bin/env bash
#SBATCH --job-name=dn_carril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=jobs/logs/dn_carril_%j.out
#SBATCH --error=jobs/logs/dn_carril_%j.err

# Un "carril": corre UNO DETRAS DE OTRO los experimentos de un archivo de carril
# (jobs/carriles/<ronda>_carril<N>.tsv, mismas columnas que config/denoising/ronda1.tsv).
# Existe porque la QOS de la particion gpu solo deja tener 2 jobs enviados a la vez por usuario
# (QOSMaxSubmitJobPerUserLimit): en vez de 9 jobs de GPU se mandan 2 carriles, cada uno con 1 GPU,
# y asi las 2 GPUs trabajan todo el tiempo. Si un experimento falla, se anota y se sigue con el resto.
# Lo arma y lo manda jobs/lanzar_ronda1.sh (la particion, --gres, CPUs y RAM van por linea de comando).
# Variables: CARRIL (archivo), PASOS (4000)
#   Se puede RELANZAR las veces que haga falta: un experimento que ya tiene todas sus salidas se
#   salta, y uno que no alcanza a terminar en el tiempo que le queda al job (columna 6 del carril:
#   minutos estimados) se deja para el siguiente envio. Asi la ronda avanza por tandas.
#   HUMO=1: prueba de humo (30 pasos, 2 imagenes, en cache/denoising_humo) para ver en minutos que
#   todos los metodos cargan, entrenan e infieren antes de gastar horas de GPU.
#   EVALUAR_AL_FINAL=1: al terminar, si ya no queda otro carril de esta ronda corriendo (dn_gpu*),
#   este mismo job corre la evaluacion de todas las familias (jobs/denoising_evaluar.sbatch). Asi la
#   evaluacion tambien queda en la GPU y no hace falta otro job.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true
: "${CARRIL:?falta CARRIL}"
L=resultados/denoising/listas
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
echo "Carril ${CARRIL} | CPUs ${SLURM_CPUS_PER_TASK:-?} | nodo $(hostname)"

# Listas de la ronda (si PREPARAR_LISTAS=1): el primer carril que llega las arma; el otro espera
# el candado y, al ver la marca, no las repite.
if [ "${PREPARAR_LISTAS:-0}" = "1" ]; then
  MARCA="jobs/carriles/.listas_${RONDA:-ronda1}_ok"
  exec 9>"jobs/carriles/.listas_${RONDA:-ronda1}.lock"
  flock 9
  if [ ! -f "$MARCA" ]; then
    echo "== Armando listas por familia (config/denoising/familias.json)"
    MASC=""
    [ -d datasets/dataset_sin_hud/mascaras_hud ] || MASC="--exportar-mascaras datasets/dataset_sin_hud/mascaras_hud"
    python scripts/listas_denoising.py --max-por-familia "${MAX_POR_FAMILIA:-1500}" \
        --max-evaluacion "${MAX_EVALUACION:-120}" ${MASC} && touch "$MARCA"
  else
    echo "== Listas ya armadas por el otro carril"
  fi
  flock -u 9
  [ -f "$MARCA" ] || { echo "!! no se pudieron armar las listas"; exit 1; }
fi

# frames vecinos para las metricas temporales (warping error, uPSNR): una sola vez, con candado
exec 8>"jobs/carriles/.vecinos.lock"
flock 8
FAM1=$(python -c "import json;print(list(json.load(open('config/denoising/familias.json')))[0])")
if [ ! -f "${L}/${FAM1}_vecinos.csv" ]; then
  echo "== Preparando frames vecinos (scripts/preparar_vecinos.py)"
  python scripts/preparar_vecinos.py --n "${N_VECINOS:-20}" || echo "!! no se pudieron preparar los vecinos (sin metricas temporales)"
fi
flock -u 8

resumen=()
BASE_CACHE="cache/denoising"; PASOS_RONDA="${PASOS:-4000}"; MAXIMG=""
if [ "${HUMO:-0}" = "1" ]; then BASE_CACHE="cache/denoising_humo"; PASOS_RONDA=30; MAXIMG="--max-imagenes 2"; fi
minutos_restantes() {
  local fin; fin=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null)
  [[ -z "$fin" || "$fin" == "N/A" ]] && { echo 99999; return; }
  echo $(( ( $(date -d "$fin" +%s) - $(date +%s) ) / 60 ))
}
while IFS=$'\t' read -r EXP FAM MET REC ARGS MIN; do
  [[ -z "${EXP}" || "${EXP}" == \#* ]] && continue
  SAL="${BASE_CACHE}/${FAM}/${EXP}"
  NLISTA=$(( $(wc -l < "${L}/${FAM}_evaluacion.csv") - 1 ))
  [ -n "${MAXIMG}" ] && NLISTA=2
  MI=$(echo " ${ARGS:-} " | grep -o -- "--max-imagenes [0-9]*" | awk '{print $2}')
  [ -n "$MI" ] && [ "$MI" -lt "$NLISTA" ] && NLISTA=$MI
  [ "${HUMO:-0}" = "1" ] && ARGS="${ARGS//cache\/denoising\//cache\/denoising_humo\/}"
  BASE=$(echo " ${ARGS:-} " | grep -oE -- "--(p2n|f2r)-base [^ ]*" | awk "{print \$2}")
  if [ -n "$BASE" ] && [ ! -f "$BASE" ]; then
    echo "== ${FAM}/${EXP}: falta su modelo base (${BASE}); queda para el proximo envio"
    resumen+=("$(printf '%-38s %-10s' "${FAM}/${EXP}" "PENDIENTE")"); continue
  fi
  NHECHAS=$(ls "${SAL}/salidas"/*.png 2>/dev/null | wc -l)
  if [ -f "${SAL}/salidas/inferencia.json" ] && [ "${NHECHAS}" -ge "${NLISTA}" ]; then
    echo "== ${FAM}/${EXP}: ya hecho (${NHECHAS} imagenes), se salta"
    resumen+=("$(printf '%-38s %-10s' "${FAM}/${EXP}" "YA_HECHO")"); continue
  fi
  if [ -n "${MIN:-}" ] && [ "${HUMO:-0}" != "1" ] && [ "$(minutos_restantes)" -lt $(( MIN * 12 / 10 )) ]; then
    echo "== ${FAM}/${EXP}: no alcanza (~${MIN} min, quedan $(minutos_restantes)); queda para el proximo envio"
    resumen+=("$(printf '%-38s %-10s' "${FAM}/${EXP}" "PENDIENTE")"); continue
  fi
  mkdir -p "${SAL}"
  t0=$(date +%s)
  echo
  echo "################ ${FAM}/${EXP} (${MET}, ${REC})  $(date '+%F %T') ################"
  if [ "${REC}" = "gpu" ]; then
    INF_ARGS=""
    [[ " ${ARGS:-} " == *" --amp "* ]] && INF_ARGS="--amp"
    python scripts/denoising.py entrenar --metodo "${MET}" --familia "${FAM}" \
        --lista "${L}/${FAM}_entrenamiento.csv" --salida "${SAL}" \
        --pasos "${PASOS_RONDA}" --instantaneas 2000 ${ARGS:-} \
      && python scripts/denoising.py inferir --checkpoint "${SAL}/modelo.pt" \
        --lista "${L}/${FAM}_evaluacion.csv" --entrada datasets/dataset_sin_hud/images \
        --salida "${SAL}/salidas" --residuo ${INF_ARGS} ${MAXIMG}
  elif [ "${REC}" = "zs" ]; then  # zero-shot: una red por imagen, en la GPU
    python scripts/denoising.py inferir --metodo "${MET}" \
        --lista "${L}/${FAM}_evaluacion.csv" --entrada datasets/dataset_sin_hud/images \
        --salida "${SAL}/salidas" --residuo ${ARGS:-} ${MAXIMG}
  else
    python scripts/denoising.py inferir --metodo "${MET}" \
        --lista "${L}/${FAM}_evaluacion.csv" --entrada datasets/dataset_sin_hud/images \
        --salida "${SAL}/salidas" --residuo ${ARGS:-} ${MAXIMG}
  fi
  codigo=$?
  VL="${L}/${FAM}_vecinos.csv"
  if [ $codigo -eq 0 ] && [ -f "$VL" ] && [ "${REC}" != "zs" ]; then  # salida tambien en t-1 y t+1
    if [ "${REC}" = "gpu" ]; then
      python scripts/denoising.py inferir --checkpoint "${SAL}/modelo.pt" --lista "$VL" \
          --entrada datasets/vecinos_eval/images --salida "${SAL}/vecinos" ${INF_ARGS} ${MAXIMG} \
        || echo "!! ${EXP}: fallo la inferencia en los vecinos (sin metricas temporales)"
    else
      python scripts/denoising.py inferir --metodo "${MET}" --lista "$VL" \
          --entrada datasets/vecinos_eval/images --salida "${SAL}/vecinos" ${ARGS:-} ${MAXIMG} \
        || echo "!! ${EXP}: fallo la inferencia en los vecinos (sin metricas temporales)"
    fi
  fi
  min=$(( ($(date +%s) - t0) / 60 ))
  estado=$([ $codigo -eq 0 ] && echo OK || echo "FALLO($codigo)")
  echo "${FAM}/${EXP}: ${estado} en ${min} min"
  resumen+=("$(printf '%-34s %-10s %4d min' "${FAM}/${EXP}" "${estado}" "${min}")")
done < "${CARRIL}"

echo
echo "=================== RESUMEN DEL CARRIL ==================="
printf '%s\n' "${resumen[@]}"

if [ "${HUMO:-0}" = "1" ]; then
  echo; echo "Prueba de humo terminada: los FALLO de arriba son metodos que hay que corregir antes de la ronda."
  exit 0
fi
if [ "${EVALUAR_AL_FINAL:-0}" = "1" ]; then
  otros=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk -v yo="${SLURM_JOB_ID:-x}" '$2 ~ /^dn_gpu/ && $1 != yo' | wc -l)
  if [ "$otros" -eq 0 ]; then
    echo
    echo "=================== EVALUACION (ultimo carril en terminar) ==================="
    # evaluar las familias ACTUALES (config/denoising/familias.json), no las de un default viejo
    FAMILIAS="${FAMILIAS:-$(python -c "import json;print(' '.join(json.load(open('config/denoising/familias.json'))))")}" \
      bash jobs/denoising_evaluar.sbatch
  else
    echo "Queda otro carril corriendo: la evaluacion la hace el ultimo en terminar."
  fi
fi
