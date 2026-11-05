#!/usr/bin/env bash
# Lanza la RONDA 1 de denoising respetando los limites de Hypatia:
#   * la QOS de la particion gpu deja tener solo 2 jobs ENVIADOS a la vez por usuario
#     (QOSMaxSubmitJobPerUserLimit). Por eso los experimentos de GPU NO van uno por job: se
#     reparten en 2 "carriles" (jobs/denoising_carril.sbatch), cada uno con 1 GPU, que los corren
#     uno detras de otro. Las 2 GPUs trabajan todo el tiempo y solo se envian 2 jobs de GPU.
#   * los metodos clasicos (CPU) van todos en 1 carril de CPU.
#   * al final 1 job de evaluacion (espera a todos, aunque alguno falle).
#
# Orden: [preparar] -> carril GPU 1 + carril GPU 2 + carril CPU -> evaluar
# Antes de enviar NADA hace `sbatch --test-only` de cada job. Si alguno no entra, no envia ninguno.
# Cada carril de GPU recibe el maximo de CPUs/RAM que no retrase su arranque.
#
# Por defecto (TODO_EN_GPU=1) TODO va en los 2 carriles de GPU: tambien los clasicos y, al final,
# la evaluacion (la hace el ultimo carril en terminar). No se usa express ni CPU.
# TODO_EN_GPU=0 vuelve al esquema con carril CPU + job de evaluacion aparte.
#
# Uso (desde la raiz del repo):
#   PRUEBA=1 bash jobs/lanzar_ronda1.sh            # solo el test
#   bash jobs/lanzar_ronda1.sh                     # test y envio
#   PREPARAR=0 DEPENDE_DE=791848 bash jobs/lanzar_ronda1.sh   # las listas ya se estan haciendo en
#                                                  # ese job: no repetir y esperar a que termine
#   SOLO=TERMICO ...  PASOS=1000 ...  CARRILES=2 ...  GPU_CPUS=16 GPU_MEM=64G ...
#   TIEMPO_CARRIL=08:00:00 ...                     # tiempo por carril de GPU (por defecto se estima)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p jobs/logs jobs/carriles
TABLA="${TABLA:-config/denoising/ronda1.tsv}"
RONDA="${RONDA:-ronda1}"
CARRILES="${CARRILES:-2}"
PARTICIONES_CPU="${PARTICIONES_CPU:-short express medium long bigmem}"
FAMILIAS=$(python3 -c "import json;print(' '.join(json.load(open('config/denoising/familias.json'))))" 2>/dev/null \
           || echo "OPTICO_CORRELADO HAZE_DDE BAJA_LUZ TERMICO")

inicio() { sbatch --test-only "$@" 2>&1 | grep -o "to start at [0-9T:-]*" | awk '{print $4}'; }
motivo() { sbatch --test-only "$@" 2>&1 | grep -i "error" | tail -1 | cut -c1-110; }
a_epoch() { date -d "$1" +%s 2>/dev/null || echo 0; }
hhmm() { printf "%02d:%02d:00" $(( $1 / 60 )) $(( $1 % 60 )); }
elegir_particion() {
  local mejor="" mejor_t="9999"
  for p in ${PARTICIONES_CPU}; do
    local t; t=$(inicio -p "$p" "$@")
    [ -n "$t" ] || continue
    if [[ "$t" < "$mejor_t" ]]; then mejor_t=$t; mejor=$p; fi
  done
  [ -n "$mejor" ] && echo "$mejor $mejor_t"
}
minutos_de() {  # estimacion (min) de entrenar 4000 pasos + inferir, para repartir carriles
  case "$1" in
    udvd) echo 75 ;; b2u|apbsn) echo 60 ;; nb2nb|structn2v|n2v) echo 30 ;; *) echo 15 ;;
  esac
}

# ------------------------------------------------------------------ REPARTO EN CARRILES
declare -a CARGA; for ((i=1; i<=CARRILES; i++)); do CARGA[$i]=0; : > "jobs/carriles/${RONDA}_carril${i}.tsv"; done
: > "jobs/carriles/${RONDA}_cpu.tsv"
N_GPU=0; N_CPU=0
while IFS=$'\t' read -r EXP FAM MET REC ARGS; do
  [[ -z "${EXP}" || "${EXP}" == \#* ]] && continue
  [[ -n "${SOLO:-}" && "${FAM}" != "${SOLO}" ]] && continue
  if [ "${REC}" = "gpu" ] || [ "${TODO_EN_GPU:-1}" = "1" ]; then
    m=$(( $(minutos_de "$MET") * ${PASOS:-4000} / 4000 + 5 ))
    [ "${REC}" = "gpu" ] || m=15
    k=1; for ((i=2; i<=CARRILES; i++)); do [ "${CARGA[$i]}" -lt "${CARGA[$k]}" ] && k=$i; done
    printf "%s\t%s\t%s\t%s\t%s\n" "$EXP" "$FAM" "$MET" "$REC" "${ARGS:-}" >> "jobs/carriles/${RONDA}_carril${k}.tsv"
    CARGA[$k]=$(( CARGA[$k] + m )); N_GPU=$((N_GPU + 1))
  else
    printf "%s\t%s\t%s\t%s\t%s\n" "$EXP" "$FAM" "$MET" "$REC" "${ARGS:-}" >> "jobs/carriles/${RONDA}_cpu.tsv"
    N_CPU=$((N_CPU + 1))
  fi
done < "${TABLA}"
MAXC=0; for ((i=1; i<=CARRILES; i++)); do [ "${CARGA[$i]}" -gt "$MAXC" ] && MAXC=${CARGA[$i]}; done
T_CARRIL="${TIEMPO_CARRIL:-$(hhmm $(( MAXC * 14 / 10 > 120 ? MAXC * 14 / 10 : 120 )))}"
T_CPU="$(hhmm $(( N_CPU * 20 + 30 )))"

echo "=================== REPARTO ==================="
for ((i=1; i<=CARRILES; i++)); do
  echo "carril GPU $i (~${CARGA[$i]} min estimados):"
  cut -f1,2,3 "jobs/carriles/${RONDA}_carril${i}.tsv" | sed 's/^/    /'
done
echo "carril CPU (${N_CPU} clasicos):"; cut -f1,2,3 "jobs/carriles/${RONDA}_cpu.tsv" | sed 's/^/    /'
echo "Tiempo pedido por carril de GPU: ${T_CARRIL}"

# ------------------------------------------------------------------ RECURSOS DE GPU
GPU_BASE=(-p gpu --gres=gpu:1 -t "${T_CARRIL}")
# Tope de la QOS de gpu por USUARIO (suma de todos tus jobs): p.ej. MaxTRESPU=cpu=24,mem=...
# Para que los 2 carriles corran A LA VEZ, cada uno pide como maximo tope/CARRILES.
QOS_GPU=$(scontrol show partition gpu 2>/dev/null | grep -o "QoS=[^ ]*" | cut -d= -f2)
TRES=$( [ -n "$QOS_GPU" ] && [ "$QOS_GPU" != "N/A" ] && sacctmgr -nP show qos "$QOS_GPU" format=MaxTRESPU 2>/dev/null | head -1)
TOPE_CPU=$(echo "$TRES" | grep -o "cpu=[0-9]*" | cut -d= -f2)
TOPE_MEM=$(echo "$TRES" | grep -o "mem=[0-9]*[MGT]*" | cut -d= -f2)
echo "QOS de gpu: ${QOS_GPU:-?}  tope por usuario: ${TRES:-no publicado}"
if [ -n "$TOPE_CPU" ]; then
  MAXC=$(( TOPE_CPU / CARRILES ))
  FILTRADOS=""
  for cand in ${CANDIDATOS_GPU:-48:192G 32:128G 24:96G 16:64G 12:48G 8:32G 6:24G 4:16G}; do
    [ "${cand%%:*}" -le "$MAXC" ] && FILTRADOS="$FILTRADOS $cand"
  done
  CANDIDATOS_GPU="$FILTRADOS"
fi
if [ -z "${GPU_CPUS:-}" ] || [ -z "${GPU_MEM:-}" ]; then
  MIN=$(echo ${CANDIDATOS_GPU:-8:32G} | awk '{print $NF}')
  GPU_CPUS=${MIN%%:*}; GPU_MEM=${MIN##*:}
  T0=$(inicio "${GPU_BASE[@]}" -c "$GPU_CPUS" --mem="$GPU_MEM" jobs/denoising_carril.sbatch)
  if [ -n "$T0" ]; then
    E0=$(a_epoch "$T0")
    for cand in ${CANDIDATOS_GPU:-48:192G 32:128G 24:96G 16:64G 12:48G}; do
      c=${cand%%:*}; m=${cand##*:}
      t=$(inicio "${GPU_BASE[@]}" -c "$c" --mem="$m" jobs/denoising_carril.sbatch)
      [ -n "$t" ] || continue
      if [ $(( $(a_epoch "$t") - E0 )) -le $(( ${MARGEN_MIN:-10} * 60 )) ]; then GPU_CPUS=$c; GPU_MEM=$m; break; fi
    done
  fi
fi
GPU_RES=("${GPU_BASE[@]}" -c "${GPU_CPUS}" --mem="${GPU_MEM}")
echo "Cada carril de GPU: 1 GPU, ${GPU_CPUS} CPUs, ${GPU_MEM} RAM, ${T_CARRIL}"

# ------------------------------------------------------------------ TEST
echo
echo "=================== TEST (sbatch --test-only, no se encola nada) ==================="
FALLAS=0
fila() { printf "%-26s %-8s %s\n" "$1" "$2" "$3"; }
if [ "${PREPARAR:-1}" = "1" ]; then
  read -r P_PREP T <<<"$(elegir_particion jobs/denoising_preparar.sbatch)"
  fila "preparar" "${P_PREP:-NINGUNA}" "${T:-NO ENTRA}"; [ -n "${P_PREP:-}" ] || FALLAS=1
fi
if [ "$N_GPU" -gt 0 ]; then
  T_GPU=$(inicio "${GPU_RES[@]}" jobs/denoising_carril.sbatch)
  for ((i=1; i<=CARRILES; i++)); do
    [ -s "jobs/carriles/${RONDA}_carril${i}.tsv" ] || continue
    fila "carril GPU $i" "gpu" "${T_GPU:-NO ENTRA: $(motivo "${GPU_RES[@]}" jobs/denoising_carril.sbatch)}"
  done
  [ -n "${T_GPU}" ] || FALLAS=1
fi
if [ "$N_CPU" -gt 0 ]; then
  read -r P_CPU T <<<"$(elegir_particion -c 4 --mem=8G -t "${T_CPU}" jobs/denoising_carril.sbatch)"
  fila "carril CPU" "${P_CPU:-NINGUNA}" "${T:-NO ENTRA}"; [ -n "${P_CPU:-}" ] || FALLAS=1
fi
if [ "${TODO_EN_GPU:-1}" = "1" ]; then
  fila "evaluar" "gpu" "al final del ultimo carril"
else
  read -r P_EVAL T_EVAL <<<"$(elegir_particion jobs/denoising_evaluar.sbatch)"
  fila "evaluar" "${P_EVAL:-NINGUNA}" "${T_EVAL:-NO ENTRA}"; [ -n "${P_EVAL:-}" ] || FALLAS=1
fi
echo "(fechas si arrancaran ya; los que dependen de otro esperan a que ese termine)"
echo "Jobs de GPU que se van a enviar: $(( N_GPU > 0 ? CARRILES : 0 ))"
if [ "$FALLAS" = "1" ]; then
  echo "!! Hay jobs que no entran: no se envia nada."
  echo "   Si el carril de GPU no entra por tiempo, prueba TIEMPO_CARRIL mas corto o PASOS menor."
  exit 2
fi
if [ "${PRUEBA:-0}" = "1" ]; then echo "[PRUEBA] todo entra; no se envio nada."; exit 0; fi

# Envia probando primero la particion elegida y, si la QOS lo rechaza (p.ej. express solo deja 1 job
# enviado por usuario: QOSMaxSubmitJobPerUserLimit, que --test-only no detecta), las demas de CPU.
enviar_cpu() {  # enviar_cpu <particion_preferida> <args de sbatch...>  -> "ID PARTICION"
  local pref=$1; shift
  for p in $pref ${PARTICIONES_CPU}; do
    local id; id=$(sbatch --parsable -p "$p" "$@" 2>/dev/null) && { echo "$id $p"; return 0; }
  done
  return 1
}

# ------------------------------------------------------------------ ENVIO
echo
echo "=================== ENVIO ==================="
DEP=""
[ -n "${DEPENDE_DE:-}" ] && DEP="--dependency=afterok:${DEPENDE_DE}"
if [ "${PREPARAR:-1}" = "1" ]; then
  read -r ID P_PREP <<<"$(enviar_cpu "$P_PREP" ${DEP} --export=ALL,REGIMENES="${REGIMENES:-0}" jobs/denoising_preparar.sbatch)"
  [ -n "${ID:-}" ] || { echo "!! no se pudo enviar preparar"; exit 1; }
  echo "preparar                   -> job $ID ($P_PREP)"
  DEP="--dependency=afterok:${ID}"
fi
IDS=()
for ((i=1; i<=CARRILES; i++)); do
  f="jobs/carriles/${RONDA}_carril${i}.tsv"
  [ -s "$f" ] || continue
  ID=$(sbatch --parsable ${DEP} "${GPU_RES[@]}" -J "dn_gpu${i}" \
       --export=ALL,CARRIL="$f",PASOS="${PASOS:-4000}",EVALUAR_AL_FINAL="${TODO_EN_GPU:-1}",RONDA="${RONDA}" \
       -o "jobs/logs/dn_gpu${i}_%j.out" -e "jobs/logs/dn_gpu${i}_%j.err" jobs/denoising_carril.sbatch) \
       || { echo "!! no se pudo enviar el carril GPU $i"; exit 1; }
  echo "carril GPU $i               -> job $ID (gpu)"
  IDS+=("$ID")
done
if [ "$N_CPU" -gt 0 ]; then
  read -r ID P_CPU <<<"$(enviar_cpu "$P_CPU" ${DEP} -c 4 --mem=8G -t "${T_CPU}" -J dn_cpu \
       --export=ALL,CARRIL="jobs/carriles/${RONDA}_cpu.tsv" \
       -o "jobs/logs/dn_cpu_%j.out" -e "jobs/logs/dn_cpu_%j.err" jobs/denoising_carril.sbatch)"
  [ -n "${ID:-}" ] || { echo "!! no se pudo enviar el carril CPU"; exit 1; }
  echo "carril CPU                 -> job $ID ($P_CPU)"
  IDS+=("$ID")
fi
if [ "${TODO_EN_GPU:-1}" = "1" ]; then
  echo "evaluar                    -> la hace el ultimo carril de GPU al terminar"
  echo; echo "Seguimiento: squeue -u \$USER  |  progreso: tail -n 20 jobs/logs/dn_gpu1_<job>.out"
  exit 0
fi
LISTA=$(IFS=:; echo "${IDS[*]}")
read -r ID P_EVAL <<<"$(enviar_cpu "$P_EVAL" --dependency=afterany:"${LISTA}" \
     --export=ALL,FAMILIAS="${SOLO:-$FAMILIAS}",RONDA="${RONDA}" jobs/denoising_evaluar.sbatch)"
[ -n "${ID:-}" ] || { echo "!! no se pudo enviar evaluar. A mano:"; echo "   sbatch -p long --dependency=afterany:${LISTA} jobs/denoising_evaluar.sbatch"; exit 1; }
echo "evaluar                    -> job $ID ($P_EVAL)"
echo
echo "Seguimiento: squeue -u \$USER  |  progreso: tail -f jobs/logs/dn_gpu1_<job>.out"
echo "Si 'preparar' falla, los demas quedan esperando: scancel -u \$USER y revisar jobs/logs/"
