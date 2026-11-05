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

resumen=()
while IFS=$'\t' read -r EXP FAM MET REC ARGS; do
  [[ -z "${EXP}" || "${EXP}" == \#* ]] && continue
  SAL="cache/denoising/${FAM}/${EXP}"
  mkdir -p "${SAL}"
  t0=$(date +%s)
  echo
  echo "################ ${FAM}/${EXP} (${MET}, ${REC})  $(date '+%F %T') ################"
  if [ "${REC}" = "gpu" ]; then
    INF_ARGS=""
    [[ " ${ARGS:-} " == *" --amp "* ]] && INF_ARGS="--amp"
    python scripts/denoising.py entrenar --metodo "${MET}" --familia "${FAM}" \
        --lista "${L}/${FAM}_entrenamiento.csv" --salida "${SAL}" \
        --pasos "${PASOS:-4000}" --instantaneas 2000 ${ARGS:-} \
      && python scripts/denoising.py inferir --checkpoint "${SAL}/modelo.pt" \
        --lista "${L}/${FAM}_evaluacion.csv" --entrada datasets/dataset_sin_hud/images \
        --salida "${SAL}/salidas" --residuo ${INF_ARGS}
  else
    python scripts/denoising.py inferir --metodo "${MET}" \
        --lista "${L}/${FAM}_evaluacion.csv" --entrada datasets/dataset_sin_hud/images \
        --salida "${SAL}/salidas" --residuo ${ARGS:-}
  fi
  codigo=$?
  min=$(( ($(date +%s) - t0) / 60 ))
  estado=$([ $codigo -eq 0 ] && echo OK || echo "FALLO($codigo)")
  echo "${FAM}/${EXP}: ${estado} en ${min} min"
  resumen+=("$(printf '%-34s %-10s %4d min' "${FAM}/${EXP}" "${estado}" "${min}")")
done < "${CARRIL}"

echo
echo "=================== RESUMEN DEL CARRIL ==================="
printf '%s\n' "${resumen[@]}"

if [ "${EVALUAR_AL_FINAL:-0}" = "1" ]; then
  otros=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk -v yo="${SLURM_JOB_ID:-x}" '$2 ~ /^dn_gpu/ && $1 != yo' | wc -l)
  if [ "$otros" -eq 0 ]; then
    echo
    echo "=================== EVALUACION (ultimo carril en terminar) ==================="
    bash jobs/denoising_evaluar.sbatch
  else
    echo "Queda otro carril corriendo: la evaluacion la hace el ultimo en terminar."
  fi
fi
