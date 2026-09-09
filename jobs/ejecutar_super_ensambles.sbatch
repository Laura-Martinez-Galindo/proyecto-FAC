#!/usr/bin/env bash

#SBATCH --job-name=super_ensambles
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/super_ensambles_%j.out
#SBATCH --error=jobs/logs/super_ensambles_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video1}"
MAX_FRAMES="${2:-1000}"

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"
EXCEL_EXP="resumen_experimentos.xlsx"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs

echo "============================================================"
echo "EJECUCIÓN DE SÚPER ENSAMBLES DE MODELOS CAMPEONES"
echo "Video: ${VIDEO_ID} | Frames: ${MAX_FRAMES}"
echo "Modelos base: UDVD, StructN2V Vertical, Blind2Unblind"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

# Directorios de los 3 campeones
DIR_UDVD="videos/${VIDEO_ID}/UDVD_SinHUD_K5_lr1e3"
DIR_STRUCT="videos/${VIDEO_ID}/StructN2V_Vert_SinHUD_d3_lr1e3"
DIR_B2U="videos/${VIDEO_ID}/Blind2Unblind_SinHUD_d4_lr1e3"

# 1. Ensamble Wavelet 1: Base UDVD (Temporal) + Detalles StructN2V Vertical (Desrayado)
echo
echo "--- [1/4] Ensamble Wavelet: UDVD + StructN2V Vertical ---"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "${DIR_UDVD}" "${DIR_STRUCT}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "SuperEnsemble_Wavelet_UDVD_StructVert"

echo "Calculando metricas SuperEnsemble_Wavelet_UDVD_StructVert..."
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo ensemble \
    --carpeta-directa "videos/${VIDEO_ID}/SuperEnsemble_Wavelet_UDVD_StructVert" \
    --experimento "SuperEnsemble_Wavelet_UDVD_StructVert" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Familia=Ensamble, Tipo=Wavelet_Fusion, Base=UDVD, Detalles=StructN2V_Vert"

# 2. Ensamble Wavelet 2: Base UDVD (Temporal) + Detalles Blind2Unblind (Perceptual)
echo
echo "--- [2/4] Ensamble Wavelet: UDVD + Blind2Unblind ---"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "${DIR_UDVD}" "${DIR_B2U}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "SuperEnsemble_Wavelet_UDVD_B2U"

echo "Calculando metricas SuperEnsemble_Wavelet_UDVD_B2U..."
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo ensemble \
    --carpeta-directa "videos/${VIDEO_ID}/SuperEnsemble_Wavelet_UDVD_B2U" \
    --experimento "SuperEnsemble_Wavelet_UDVD_B2U" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Familia=Ensamble, Tipo=Wavelet_Fusion, Base=UDVD, Detalles=Blind2Unblind"

# 3. Ensamble Ponderado Top 3: UDVD (0.45) + StructN2V Vert (0.35) + Blind2Unblind (0.20)
echo
echo "--- [3/4] Ensamble Promedio Ponderado: UDVD (0.45) + StructN2V (0.35) + B2U (0.20) ---"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion promedio \
    --modelos-entrada "${DIR_UDVD}" "${DIR_STRUCT}" "${DIR_B2U}" \
    --pesos 0.45 0.35 0.20 \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "SuperEnsemble_Ponderado_Top3"

echo "Calculando metricas SuperEnsemble_Ponderado_Top3..."
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo ensemble \
    --carpeta-directa "videos/${VIDEO_ID}/SuperEnsemble_Ponderado_Top3" \
    --experimento "SuperEnsemble_Ponderado_Top3" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Familia=Ensamble, Tipo=Promedio_Ponderado, Pesos=[UDVD:0.45, StructVert:0.35, B2U:0.20]"

# 4. Ensamble Mediana Robusta Top 3: UDVD + StructN2V Vert + Blind2Unblind
echo
echo "--- [4/4] Ensamble Mediana Robusta: UDVD + StructN2V + B2U ---"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion mediana \
    --modelos-entrada "${DIR_UDVD}" "${DIR_STRUCT}" "${DIR_B2U}" \
    --max-frames "${MAX_FRAMES}" \
    --id-experimento "SuperEnsemble_Mediana_Top3"

echo "Calculando metricas SuperEnsemble_Mediana_Top3..."
"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo ensemble \
    --carpeta-directa "videos/${VIDEO_ID}/SuperEnsemble_Mediana_Top3" \
    --experimento "SuperEnsemble_Mediana_Top3" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}" \
    --parametros-extra "Familia=Ensamble, Tipo=Mediana_Robusta, Entradas=[UDVD, StructVert, B2U]"

echo
echo "============================================================"
echo "SÚPER ENSAMBLES COMPLETADOS CON ÉXITO"
echo "Fin: $(date --iso-8601=seconds)"
echo "Resultados consolidados en videos/${VIDEO_ID}/${EXCEL_EXP}"
echo "============================================================"
