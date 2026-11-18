#!/usr/bin/env bash
# FASE 2 de denoising: los 3 ganadores de cada regimen (config/denoising/fase2.tsv) reentrenados con mas
# frames y mas pasos, limpiando TODAS las imagenes del dataset de su regimen. Usa jobs/lanzar_ronda1.sh.
#   PRUEBA=1 bash jobs/lanzar_fase2.sh   # solo muestra el reparto y el --test-only, no envia nada
#   bash jobs/lanzar_fase2.sh            # envia los 2 carriles (le llegan correos)
#   RELISTAR=1 bash jobs/lanzar_fase2.sh # borra las listas de la fase 2 y las vuelve a armar
# Si no alcanza el tiempo, se vuelve a correr el MISMO comando: lo terminado se salta y los zero-shot
# siguen en la imagen donde iban. Al terminar: resultados/denoising/fase2.tar
set -uo pipefail
cd "$(dirname "$0")/.."
export TABLA=config/denoising/fase2.tsv
export RONDA=fase2
export PATRON='*_f2*'
export LISTAS=resultados/denoising/listas_fase2
export MARGEN=0                       # sin excluir frames cercanos al dataset
export MAX_POR_FAMILIA="${MAX_POR_FAMILIA:-1000000}"   # TODOS los frames del regimen (el tiempo lo fijan los pasos, no los frames)
export MAX_EVALUACION=0               # todas las imagenes del regimen
export PASOS="${PASOS:-12000}"
export ENTRENAR_EXTRA="${ENTRENAR_EXTRA:---parada-temprana}"   # para si la perdida de validacion deja de bajar
export IQA="${IQA:-niqe brisque musiq clipiqa}"
export COMBINAR="${COMBINAR:-1}"      # ensembles y stacks de los 3, y se vuelven a evaluar
if [ "${RELISTAR:-0}" = "1" ]; then
  rm -f jobs/carriles/.listas_fase2_ok; rm -rf resultados/denoising/listas_fase2
  echo "Listas de la fase 2 borradas: se vuelven a armar con todos los frames."
fi
if [ -f "jobs/carriles/.listas_fase2_ok" ]; then
  export PREPARAR=0; echo "Listas de la fase 2 ya armadas."
else
  export PREPARAR=1
fi
exec bash jobs/lanzar_ronda1.sh
