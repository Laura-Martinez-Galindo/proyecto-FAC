#!/usr/bin/env bash
# FASE 2 alternativa: el mejor metodo por NIQE de cada regimen (config/denoising/fase2_iqa.tsv): ZS-N2N en A
# (se entrena con todos los frames) y ZS-KAN en B y C (zero-shot: una red por imagen). Limpia TODAS las
# imagenes del dataset de su regimen (sin HUD), igual que la fase 2 por grano. Para comparar en YOLO.
#   PRUEBA=1 bash jobs/lanzar_fase2_iqa.sh   # solo muestra el reparto y el --test-only
#   bash jobs/lanzar_fase2_iqa.sh            # envia (2 carriles de GPU como maximo)
# Si no alcanza el tiempo, se vuelve a correr el MISMO comando: lo terminado se salta.
set -uo pipefail
cd "$(dirname "$0")/.."
export TABLA=config/denoising/fase2_iqa.tsv
export RONDA=fase2_iqa
export PATRON='*_f2iqa*'
export LISTAS=resultados/denoising/listas_fase2   # las mismas listas de la fase 2 (ya armadas)
export PREPARAR=0
export MARGEN=0
export MAX_POR_FAMILIA="${MAX_POR_FAMILIA:-1000000}"
export MAX_EVALUACION=0
export PASOS="${PASOS:-12000}"
export ENTRENAR_EXTRA="${ENTRENAR_EXTRA:---parada-temprana}"
export IQA="${IQA:-niqe brisque musiq clipiqa}"
export COMBINAR=0
[ -d "${LISTAS}" ] || { echo "!! falta ${LISTAS}: corre primero la fase 2 normal"; exit 1; }
exec bash jobs/lanzar_ronda1.sh
