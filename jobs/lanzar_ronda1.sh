#!/usr/bin/env bash
# Lanza la RONDA 1 completa de denoising, en orden y con dependencias:
#
#   1) dn_preparar   listas por familia (config/denoising/familias.json) + mascaras HUD del dataset
#   2) un job por fila de config/denoising/ronda1.tsv (GPU = entrena+infiere, CPU = clasico),
#      todos esperan a (1). Cada job de GPU pide 1 GPU: Slurm corre a la vez los que permita tu
#      limite (2 GPUs) y el resto espera turno solo.
#   3) dn_evaluar    compara todo, familia por familia; espera a que terminen TODOS los de (2)
#                    (aunque alguno falle, evalua los que si quedaron).
#
# ANTES de mandar nada hace `sbatch --test-only` de CADA job (preparar, GPU, CPU, evaluar) e
# imprime cuando arrancaria. Si alguno no tiene donde correr, no manda ninguno.
# Los de CPU van a la particion que arranca antes (short/express/medium/long/bigmem).
#
# Uso (desde la raiz del repo):
#   PRUEBA=1 bash jobs/lanzar_ronda1.sh        # SOLO el test: muestra la tabla, no manda nada
#   bash jobs/lanzar_ronda1.sh                 # test y, si todo entra, manda todo
#   SOLO=TERMICO bash jobs/lanzar_ronda1.sh    # una familia
#   PREPARAR=0 bash jobs/lanzar_ronda1.sh      # ya existen las listas: no repetir (1)
#   REGIMENES=1 bash jobs/lanzar_ronda1.sh     # volver a leer regimenes antes de las listas
#   PASOS=1000 bash jobs/lanzar_ronda1.sh      # presupuesto distinto
#   GPU_CPUS=16 GPU_MEM=64G bash jobs/lanzar_ronda1.sh   # fijar CPUs/RAM de cada job de GPU
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p jobs/logs
TABLA="${TABLA:-config/denoising/ronda1.tsv}"
PARTICIONES_CPU="${PARTICIONES_CPU:-short express medium long bigmem}"
FAMILIAS=$(python3 -c "import json;print(' '.join(json.load(open('config/denoising/familias.json'))))" 2>/dev/null \
           || echo "OPTICO_CORRELADO HAZE_DDE BAJA_LUZ TERMICO")

inicio() {  # inicio <args de sbatch...> -> fecha estimada de arranque (vacio si no entra)
  sbatch --test-only "$@" 2>&1 | grep -o "to start at [0-9T:-]*" | awk '{print $4}'
}

elegir_particion() {  # elegir_particion <args sin -p> -> "particion fecha"
  local mejor="" mejor_t="9999"
  for p in ${PARTICIONES_CPU}; do
    local t; t=$(inicio -p "$p" "$@")
    [ -n "$t" ] || continue
    if [[ "$t" < "$mejor_t" ]]; then mejor_t=$t; mejor=$p; fi
  done
  [ -n "$mejor" ] && echo "$mejor $mejor_t"
}

# ------------------------------------------------------------------ RECURSOS DE GPU
# Cada experimento de GPU usa 1 GPU; con el limite de 2 GPUs por usuario corren 2 a la vez (las 2
# GPUs ocupadas todo el tiempo). Para que cada uno vaya lo mas rapido posible se le da la mayor
# cantidad de CPUs (lectura de frames en paralelo) y RAM que NO retrase su arranque: se prueba con
# --test-only de mayor a menor y se queda con el primero que arranca a mas tardar MARGEN_MIN minutos
# despues de lo que arrancaria con lo minimo (8 CPUs, 32G). Se puede fijar a mano: GPU_CPUS=16 GPU_MEM=64G
a_epoch() { date -d "$1" +%s 2>/dev/null || echo 0; }
if [ -z "${GPU_CPUS:-}" ] || [ -z "${GPU_MEM:-}" ]; then
  T0=$(inicio -c 8 --mem=32G jobs/denoising_entrenar.sbatch)
  GPU_CPUS=8; GPU_MEM=32G
  if [ -n "$T0" ]; then
    E0=$(a_epoch "$T0")
    for cand in ${CANDIDATOS_GPU:-48:192G 32:128G 24:96G 16:64G 12:48G}; do
      c=${cand%%:*}; m=${cand##*:}
      t=$(inicio -c "$c" --mem="$m" jobs/denoising_entrenar.sbatch)
      [ -n "$t" ] || continue
      if [ $(( $(a_epoch "$t") - E0 )) -le $(( ${MARGEN_MIN:-10} * 60 )) ]; then GPU_CPUS=$c; GPU_MEM=$m; break; fi
    done
  fi
fi
echo "Recursos por experimento de GPU: 1 GPU, ${GPU_CPUS} CPUs, ${GPU_MEM} RAM (2 experimentos a la vez)"
GPU_RES=(-c "${GPU_CPUS}" --mem="${GPU_MEM}")

# ------------------------------------------------------------------ TEST
echo "=================== TEST (sbatch --test-only, no se encola nada) ==================="
FALLAS=0
if [ "${PREPARAR:-1}" = "1" ]; then
  read -r P_PREP T <<<"$(elegir_particion jobs/denoising_preparar.sbatch)"
  printf "%-36s %-8s %s\n" "preparar" "${P_PREP:-NINGUNA}" "${T:-NO ENTRA}"
  [ -n "${P_PREP:-}" ] || FALLAS=1
fi
read -r P_CLASICO T <<<"$(elegir_particion jobs/denoising_clasico.sbatch)"
T_GPU=$(inicio "${GPU_RES[@]}" jobs/denoising_entrenar.sbatch)
read -r P_EVAL T_EVAL <<<"$(elegir_particion jobs/denoising_evaluar.sbatch)"
N=0
while IFS=$'\t' read -r EXP FAM MET REC ARGS; do
  [[ -z "${EXP}" || "${EXP}" == \#* ]] && continue
  [[ -n "${SOLO:-}" && "${FAM}" != "${SOLO}" ]] && continue
  N=$((N + 1))
  if [ "${REC}" = "gpu" ]; then
    printf "%-36s %-8s %s\n" "${FAM}/${EXP}" "gpu" "${T_GPU:-NO ENTRA}"
    [ -n "${T_GPU}" ] || FALLAS=1
  else
    printf "%-36s %-8s %s\n" "${FAM}/${EXP}" "${P_CLASICO:-NINGUNA}" "${T:-NO ENTRA}"
    [ -n "${P_CLASICO:-}" ] || FALLAS=1
  fi
done < "${TABLA}"
printf "%-36s %-8s %s\n" "evaluar" "${P_EVAL:-NINGUNA}" "${T_EVAL:-NO ENTRA}"
[ -n "${P_EVAL:-}" ] || FALLAS=1
echo "(las fechas son si arrancaran YA; los que dependen de otro esperan a que ese termine)"
if [ "$FALLAS" = "1" ]; then echo "!! Hay jobs que no entran en ninguna particion: no se manda nada."; exit 2; fi
if [ "${PRUEBA:-0}" = "1" ]; then echo "[PRUEBA] todo entra; no se mando nada."; exit 0; fi
[ "$N" -gt 0 ] || { echo "No hay experimentos en ${TABLA}"; exit 1; }

# ------------------------------------------------------------------ ENVIO
echo
echo "=================== ENVIO ==================="
DEP=""
if [ "${PREPARAR:-1}" = "1" ]; then
  ID=$(sbatch --parsable -p "$P_PREP" --export=ALL,REGIMENES="${REGIMENES:-0}" jobs/denoising_preparar.sbatch) || exit 1
  echo "preparar                             -> job $ID ($P_PREP)"
  DEP="--dependency=afterok:${ID}"
fi
IDS=()
while IFS=$'\t' read -r EXP FAM MET REC ARGS; do
  [[ -z "${EXP}" || "${EXP}" == \#* ]] && continue
  [[ -n "${SOLO:-}" && "${FAM}" != "${SOLO}" ]] && continue
  VARS="ALL,FAMILIA=${FAM},METODO=${MET},EXP=${EXP},ARGS=${ARGS:-},PASOS=${PASOS:-4000}"
  NOMBRE="dn_${FAM}_${EXP}"
  if [ "${REC}" = "gpu" ]; then
    ID=$(sbatch --parsable ${DEP} "${GPU_RES[@]}" -J "${NOMBRE}" --export="${VARS}" \
         -o "jobs/logs/${NOMBRE}_%j.out" -e "jobs/logs/${NOMBRE}_%j.err" jobs/denoising_entrenar.sbatch) || exit 1
    echo "$(printf '%-36s' "${FAM}/${EXP}") -> job $ID (gpu)"
  else
    ID=$(sbatch --parsable ${DEP} -p "${P_CLASICO}" -J "${NOMBRE}" --export="${VARS}" \
         -o "jobs/logs/${NOMBRE}_%j.out" -e "jobs/logs/${NOMBRE}_%j.err" jobs/denoising_clasico.sbatch) || exit 1
    echo "$(printf '%-36s' "${FAM}/${EXP}") -> job $ID (${P_CLASICO})"
  fi
  IDS+=("$ID")
done < "${TABLA}"
LISTA=$(IFS=:; echo "${IDS[*]}")
ID=$(sbatch --parsable -p "$P_EVAL" --dependency=afterany:"${LISTA}" \
     --export=ALL,FAMILIAS="${SOLO:-$FAMILIAS}" jobs/denoising_evaluar.sbatch) || exit 1
echo "evaluar                              -> job $ID ($P_EVAL)"
echo
echo "Seguimiento:  squeue -u \$USER   (PD con razon Dependency = esperando a otro job, es normal)"
echo "Si 'preparar' falla, los demas quedan esperando para siempre: scancel -u \$USER y revisar jobs/logs/"
