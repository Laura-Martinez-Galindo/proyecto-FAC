#!/usr/bin/env bash
#SBATCH --job-name=dataset_sin_hud
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=jobs/logs/dataset_sin_hud_%j.out
#SBATCH --error=jobs/logs/dataset_sin_hud_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

# Construye datasets/dataset_sin_hud a partir de datasets/dataset_original
# (mismas imagenes, mismos nombres y mismas etiquetas, pero tomando el frame de
# videos/<video>/frames_sin_hud) y exporta los mosaicos de revision de 3 vistas.
#
# Uso (desde la raiz del repo en Hypatia):
#   sbatch jobs/construir_dataset_sin_hud.sbatch
# Opciones por variables de entorno:
#   MODO=compuesto  (defecto) resolucion original; solo se reemplazan los pixeles del HUD
#   MODO=reescalar  todo el frame sin HUD reescalado a la resolucion original
#   MODO=nativo     frames_sin_hud tal cual (960x540)
#   EXPORTAR_REVISION=0    no genera datasets/revision_3_vistas

set -euo pipefail
trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs

PY="${RUTA_PROYECTO}/fac_env311/bin/python"
[ -x "${PY}" ] || PY=python3

MODO="${MODO:-compuesto}"
if [ "${IGUALAR_RESOLUCION:-0}" = "1" ]; then MODO=reescalar; fi
EXTRA=(--modo "${MODO}")

echo "=============================================================================="
echo " CONSTRUCCION DE dataset_sin_hud"
echo "   Nodo: $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}   Inicio: $(date --iso-8601=seconds)"
echo "=============================================================================="

if [ ! -f datasets/dataset_original/correspondencia.csv ]; then
    echo "No existe datasets/dataset_original/correspondencia.csv. Sube primero dataset_original." >&2
    exit 1
fi

"${PY}" scripts/construir_dataset_sin_hud.py \
    --dataset-original datasets/dataset_original \
    --salida datasets/dataset_sin_hud \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "${EXTRA[@]}"

if [ "${EXPORTAR_REVISION:-1}" = "1" ]; then
    echo
    echo "Exportando mosaicos de revision (origen | dataset_original | dataset_sin_hud)..."
    "${PY}" scripts/revisar_dataset_3_vistas.py \
        --original datasets/dataset_original \
        --sin-hud datasets/dataset_sin_hud \
        --exportar datasets/revision_3_vistas \
        --workers "${SLURM_CPUS_PER_TASK:-8}"
fi

echo
echo "Imagenes en dataset_original: $(ls datasets/dataset_original/images | wc -l)"
echo "Imagenes en dataset_sin_hud:  $(ls datasets/dataset_sin_hud/images | wc -l)"
[ -f datasets/dataset_sin_hud/faltantes.csv ] && echo "Ver problemas en datasets/dataset_sin_hud/faltantes.csv"
echo "Fin: $(date --iso-8601=seconds)"
