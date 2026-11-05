#!/usr/bin/env bash
# Manda un job a la particion que Slurm estima que lo arranca ANTES, probando primero con
# `sbatch --test-only` (no se encola nada en la prueba). Asi no queda en PD por haber elegido mal.
#
# Uso (desde la raiz del repo, en el nodo de login):
#   bash jobs/enviar.sh [opciones de sbatch] jobs/<job>.sbatch
#   bash jobs/enviar.sh -c 8 --mem=12G -t 01:30:00 jobs/detectar_regimenes.sbatch
#   bash jobs/enviar.sh --gres=gpu:1 -c 8 --mem=32G -t 03:00:00 jobs/denoising_entrenar.sbatch
#   PRUEBA=1 bash jobs/enviar.sh ...      # solo muestra la tabla, no manda nada
#   PARTICIONES="short long" bash jobs/enviar.sh ...
#
# Las opciones que pases pisan las #SBATCH del archivo. Si pides GPU (--gres=gpu...) solo se
# prueba la particion gpu. Imprime el JobID al final.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p jobs/logs
if [ $# -lt 1 ]; then sed -n 2,14p "$0"; exit 1; fi
ARGS=("$@")
if [[ " $* " == *"--gres=gpu"* || " $* " == *"--gpus"* ]]; then
  PARTS="${PARTICIONES:-gpu}"
else
  PARTS="${PARTICIONES:-short express medium long bigmem}"
fi

mejor=""; mejor_t="9999-99-99"
printf "%-10s %s\n" "PARTICION" "INICIO ESTIMADO (sbatch --test-only)"
for p in $PARTS; do
  out=$(sbatch --test-only -p "$p" "${ARGS[@]}" 2>&1)
  t=$(echo "$out" | grep -o "to start at [0-9T:-]*" | awk '{print $4}')
  if [ -n "$t" ]; then
    printf "%-10s %s\n" "$p" "$t"
    if [[ "$t" < "$mejor_t" ]]; then mejor_t=$t; mejor=$p; fi
  else
    printf "%-10s no sirve: %s\n" "$p" "$(echo "$out" | tail -1 | cut -c1-90)"
  fi
done
if [ -z "$mejor" ]; then echo "!! Ninguna particion acepta el job con esos recursos."; exit 2; fi
echo "-> elegida: $mejor (arranca ~ $mejor_t)"
if [ "${PRUEBA:-0}" = "1" ]; then echo "[PRUEBA] no se mando nada"; exit 0; fi
sbatch -p "$mejor" "${ARGS[@]}"
