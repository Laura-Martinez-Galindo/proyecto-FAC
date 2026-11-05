#!/usr/bin/env bash
# Lanza la RONDA 1 completa de denoising, en orden y con dependencias:
#
#   1) dn_preparar   regimenes con las plantillas nuevas + listas por familia + mascaras HUD
#   2) un job por fila de config/denoising/ronda1.tsv (GPU = entrena+infiere, CPU = clasico),
#      todos esperan a (1). Los de GPU piden 1 GPU cada uno: Slurm corre a la vez los que
#      permita tu limite (2 GPUs) y el resto espera turno solo.
#   3) dn_evaluar    compara todo, familia por familia; espera a que terminen TODOS los de (2)
#                    (aunque alguno falle, evalua los que si quedaron).
#
# Los jobs de CPU se mandan a la particion que Slurm estima que arranca antes
# (sbatch --test-only en short/express/medium/long), para que no queden en PD.
#
# Uso (desde la raiz del repo, en el nodo de login):
#   bash jobs/lanzar_ronda1.sh                 # todo
#   PRUEBA=1 bash jobs/lanzar_ronda1.sh        # solo muestra lo que mandaria
#   SOLO=TERMICO bash jobs/lanzar_ronda1.sh    # una familia
#   PREPARAR=0 bash jobs/lanzar_ronda1.sh      # ya existen las listas: no repetir (1)
#   REGIMENES=0 bash jobs/lanzar_ronda1.sh     # repetir listas pero no la lectura de regimenes
#   PASOS=1000 bash jobs/lanzar_ronda1.sh      # presupuesto distinto
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p jobs/logs
TABLA="${TABLA:-config/denoising/ronda1.tsv}"
PARTICIONES_CPU="${PARTICIONES_CPU:-short express medium long}"

# Particion de CPU que arranca antes para (cpus, mem, tiempo)
elegir_particion() {
  local cpus=$1 mem=$2 tiempo=$3 mejor="" mejor_t="9999"
  for p in ${PARTICIONES_CPU}; do
    local t
    t=$(sbatch --test-only -p "$p" -c "$cpus" --mem="$mem" -t "$tiempo" --wrap "true" 2>&1 \
        | grep -o "to start at [0-9T:-]*" | awk '{print $4}')
    [ -n "$t" ] || continue
    if [[ "$t" < "$mejor_t" ]]; then mejor_t=$t; mejor=$p; fi
  done
  echo "${mejor:-short}"
}

enviar() {  # enviar <args de sbatch...>  -> imprime el JobID
  if [ "${PRUEBA:-0}" = "1" ]; then
    echo "  [PRUEBA] sbatch $*" >&2
    echo "PRUEBA$RANDOM"
  else
    sbatch --parsable "$@"
  fi
}

DEP=""
if [ "${PREPARAR:-1}" = "1" ]; then
  P=$(elegir_particion 8 16G 02:00:00)
  ID=$(enviar -p "$P" --export=ALL,REGIMENES="${REGIMENES:-1}" jobs/denoising_preparar.sbatch) || exit 1
  echo "preparar          -> job $ID ($P)"
  DEP="--dependency=afterok:${ID}"
fi

P_CLASICO=$(elegir_particion 4 8G 01:00:00)
IDS=()
while IFS=$'\t' read -r EXP FAM MET REC ARGS; do
  [[ -z "${EXP}" || "${EXP}" == \#* ]] && continue
  [[ -n "${SOLO:-}" && "${FAM}" != "${SOLO}" ]] && continue
  VARS="ALL,FAMILIA=${FAM},METODO=${MET},EXP=${EXP},ARGS=${ARGS:-},PASOS=${PASOS:-4000}"
  NOMBRE="dn_${FAM}_${EXP}"
  if [ "${REC}" = "gpu" ]; then
    ID=$(enviar ${DEP} -J "${NOMBRE}" --export="${VARS}" \
         -o "jobs/logs/${NOMBRE}_%j.out" -e "jobs/logs/${NOMBRE}_%j.err" jobs/denoising_entrenar.sbatch) || exit 1
    echo "$(printf '%-34s' "${FAM}/${EXP}") -> job $ID (gpu)"
  else
    ID=$(enviar ${DEP} -p "${P_CLASICO}" -J "${NOMBRE}" --export="${VARS}" \
         -o "jobs/logs/${NOMBRE}_%j.out" -e "jobs/logs/${NOMBRE}_%j.err" jobs/denoising_clasico.sbatch) || exit 1
    echo "$(printf '%-34s' "${FAM}/${EXP}") -> job $ID (${P_CLASICO})"
  fi
  IDS+=("$ID")
done < "${TABLA}"

if [ ${#IDS[@]} -gt 0 ]; then
  P=$(elegir_particion 8 16G 01:00:00)
  LISTA=$(IFS=:; echo "${IDS[*]}")
  ID=$(enviar -p "$P" --dependency=afterany:"${LISTA}" --export=ALL,FAMILIAS="${SOLO:-OPTICO_DDE OPTICO TERMICO}" \
       jobs/denoising_evaluar.sbatch) || exit 1
  echo "evaluar           -> job $ID ($P)"
fi
echo
echo "Seguimiento:  squeue -u \$USER     |  logs en jobs/logs/   |  bash jobs/recursos_hypatia.sh"
