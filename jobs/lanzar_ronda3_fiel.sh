#!/usr/bin/env bash
# RONDA 3 FIEL: vuelve a correr, en las mismas condiciones de la ronda 3, los metodos que no estaban
# fieles a su articulo y se pudieron corregir (config/denoising/ronda3_fiel.tsv, experimentos *_r3f).
# Usa las MISMAS listas de la ronda 3 (no las vuelve a armar) y los 2 carriles de GPU de siempre.
#
#   PRUEBA=1 bash jobs/lanzar_ronda3_fiel.sh   # solo muestra el reparto y el sbatch --test-only
#   bash jobs/lanzar_ronda3_fiel.sh            # envia
#
# Si no alcanza el tiempo, se vuelve a correr el MISMO comando: lo terminado se salta.
# Al final, el ultimo carril evalua SOLO los *_r3f en resultados/denoising/ronda3_fiel/<regimen>/ y deja
# resultados/denoising/ronda3_fiel.tar. El ranking final (solo metodos fieles, ronda 3 + ronda 3 fiel)
# se arma despues con scripts/ranking_fiel.py.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f jobs/carriles/.listas_ronda3_ok ] || { echo "!! No estan las listas de la ronda 3 (jobs/carriles/.listas_ronda3_ok)."; exit 1; }
[ -f "${F2F_PESOS:-pesos/dncnn/DnCNN-S-25.pth}" ] || { echo "!! Falta la DnCNN preentrenada: correr antes  bash jobs/preparar_entorno_fiel.sh"; exit 1; }
export TABLA=config/denoising/ronda3_fiel.tsv
export RONDA=ronda3_fiel
export PATRON='*_r3f*'
export IQA="${IQA:-niqe brisque musiq clipiqa}"
export MAX_EVALUACION="${MAX_EVALUACION:-300}"
export PREPARAR=0
export COMBINAR=0
exec bash jobs/lanzar_ronda1.sh
