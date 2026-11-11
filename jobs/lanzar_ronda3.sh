#!/usr/bin/env bash
# RONDA 3 de denoising (fase 1): todos los metodos x los 3 regimenes, en las 2 GPUs en paralelo.
# Usa jobs/lanzar_ronda1.sh (2 carriles, uno por GPU, verificacion con --test-only antes de enviar).
#
#   HUMO=1 bash jobs/lanzar_ronda3.sh   # PRIMERO: prueba de humo (30 pasos, 2 imagenes, ~1-2 h):
#                                       # confirma que todos los metodos cargan, entrenan e infieren
#   bash jobs/lanzar_ronda3.sh          # la ronda de verdad
#
# Si la ronda no cabe en el tiempo maximo de la particion, cada carril hace lo que alcance y deja el
# resto PENDIENTE: se vuelve a correr el MISMO comando y sigue donde iba (lo terminado se salta).
# Cada vez que terminan los 2 carriles se evalua lo que haya: resultados/denoising/ronda3.tar.
set -uo pipefail
cd "$(dirname "$0")/.."
export TABLA=config/denoising/ronda3.tsv
export PATRON='*_r3*'
export IQA="${IQA:-niqe brisque musiq clipiqa}"
export MAX_EVALUACION="${MAX_EVALUACION:-300}"
export COMBINAR="${COMBINAR:-1}"   # ensembles y stacks de los 3 mejores, al final de cada evaluacion
if [ "${HUMO:-0}" = "1" ]; then
  export RONDA=ronda3_humo
else
  export RONDA=ronda3
fi
# listas: se arman una sola vez (las de ronda3 sirven tambien para la prueba de humo)
if [ -f jobs/carriles/.listas_ronda3_ok ] || [ -f "jobs/carriles/.listas_${RONDA}_ok" ]; then
  export PREPARAR=0
  echo "Listas ya armadas (evaluacion con hasta ${MAX_EVALUACION} imagenes por regimen)."
else
  export PREPARAR=1
fi
exec bash jobs/lanzar_ronda1.sh
