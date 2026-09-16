#!/usr/bin/env bash

#SBATCH --job-name=exp_v2_ult10min
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia:2
#SBATCH --time=12:00:00
#SBATCH --output=jobs/logs/exp_v2_ult10min_%j.out
#SBATCH --error=jobs/logs/exp_v2_ult10min_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="video2"
ULTIMOS_FRAMES="18000"
EPOCAS="50"
LR="1e-3"
PASOS="300"
PACIENCIA="6"

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"
EXCEL_EXP="resumen_experimentos.xlsx"
DIR_EXPOS="videos/${VIDEO_ID}/expos"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs "${DIR_EXPOS}"

echo "============================================================"
echo "SUITE EXPERIMENTAL: ULTIMOS 10 MINUTOS VIDEO 2 (FLIR DRIFT)"
echo "Video: ${VIDEO_ID} | Modo: sin_hud | Ultimos Frames: ${ULTIMOS_FRAMES}"
echo "Recursos: 2 GPUs NVIDIA | 128 GB RAM | 28 Cores"
echo "Hiperparametros: Epocas=${EPOCAS}, LR=${LR}, Pasos/Epoca=${PASOS}, Paciencia=${PACIENCIA}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

evaluar_modelo() {
    local mod="$1"
    local dir_out="$2"
    local exp_id="$3"
    local info="$4"

    echo ">>> Calculando metricas IQA para ${exp_id} (${ULTIMOS_FRAMES} frames)..."
    "${RUTA_PYTHON}" scripts/calcular_metricas.py \
        --video "${VIDEO_ID}" \
        --modelo "${mod}" \
        --carpeta-directa "${dir_out}" \
        --experimento "${exp_id}" \
        --archivo-resumen "${EXCEL_EXP}" \
        --ultimos-frames "${ULTIMOS_FRAMES}" \
        --parametros-extra "${info}"
}

# -------------------------------------------------------------------------
# BLOQUE 1: MODELOS DINÁMICOS TEMPORALES UDVD (GPU 0 y GPU 1 en paralelo)
# -------------------------------------------------------------------------
echo
echo "=== BLOQUE 1: ENTRENAMIENTO E INFERENCIA UDVD TEMPORAL (2 GPUs) ==="

DIR_UDVD_T7="${DIR_EXPOS}/udvd_t7_ult10min_sin_hud"
echo "[GPU 0] UDVD Ventana Temporal Expandida T=7..."
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_udvd.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --num-frames 7 \
    --kernel-size 5 \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --paciencia "${PACIENCIA}" \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --carpeta-salida "${DIR_UDVD_T7}" \
    --id-experimento "udvd_t7_ult10min_sin_hud" &

DIR_UDVD_DESTRIP="${DIR_EXPOS}/udvd_destriping_t7_ult10min_sin_hud"
echo "[GPU 1] UDVD con Pre-Filtrado Destriping Columnar FPN + T=7..."
CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_udvd.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --num-frames 7 \
    --destriping \
    --kernel-size 5 \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --paciencia "${PACIENCIA}" \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --carpeta-salida "${DIR_UDVD_DESTRIP}" \
    --id-experimento "udvd_destriping_t7_ult10min_sin_hud" &

wait
echo ">>> Modelos UDVD finalizados con exito."

evaluar_modelo "udvd" "${DIR_UDVD_T7}" "udvd_t7_ult10min_sin_hud" "UDVD_T=7, Epocas=${EPOCAS}, LR=${LR}, Kernel=5x5"
evaluar_modelo "udvd" "${DIR_UDVD_DESTRIP}" "udvd_destriping_t7_ult10min_sin_hud" "UDVD_T=7+Destriping_FPN, Epocas=${EPOCAS}, LR=${LR}"

# -------------------------------------------------------------------------
# BLOQUE 2: MODELOS ESPACIALES AUTOSUPERVISADOS (GPU 0 y GPU 1 en paralelo)
# -------------------------------------------------------------------------
echo
echo "=== BLOQUE 2: ENTRENAMIENTO E INFERENCIA ESPACIAL (StructN2V & N2V2) ==="

DIR_STRUCT_VERT="${DIR_EXPOS}/struct_n2v_vert_ult10min_sin_hud"
echo "[GPU 0] StructN2V Vertical (Mascara lineal anti-FPN columnar)..."
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --paciencia "${PACIENCIA}" \
    --struct-n2v vertical \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --carpeta-salida "${DIR_STRUCT_VERT}" \
    --id-experimento "struct_n2v_vert_ult10min_sin_hud" &

DIR_N2V2="${DIR_EXPOS}/n2v2_ult10min_sin_hud"
echo "[GPU 1] Noise2Void N2V2..."
CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --paciencia "${PACIENCIA}" \
    --use-n2v2 \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --carpeta-salida "${DIR_N2V2}" \
    --id-experimento "n2v2_ult10min_sin_hud" &

wait
echo ">>> Modelos Espaciales finalizados con exito."

evaluar_modelo "n2v" "${DIR_STRUCT_VERT}" "struct_n2v_vert_ult10min_sin_hud" "StructN2V_Vertical, Epocas=${EPOCAS}, LR=${LR}"
evaluar_modelo "n2v" "${DIR_N2V2}" "n2v2_ult10min_sin_hud" "N2V2, Epocas=${EPOCAS}, LR=${LR}"

# -------------------------------------------------------------------------
# BLOQUE 3: ENSAMBLES ROBUSTOS MULTI-ESCALA
# -------------------------------------------------------------------------
echo
echo "=== BLOQUE 3: SUPER ENSAMBLES FRECUENCIALES Y ROBUSTOS ==="

# A. Ensamble Wavelet: UDVD T=7 + StructN2V Vertical (Soft Thresholding)
DIR_ENS_WAV1="${DIR_EXPOS}/ensemble_wavelet_udvd_t7_struct_vert_ult10min"
echo "Ejecutando Ensamble Wavelet (UDVD T=7 + StructN2V Vert)..."
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --umbral-ruido 2.0 \
    --modelos-entrada "${DIR_UDVD_T7}" "${DIR_STRUCT_VERT}" \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --carpeta-salida "${DIR_ENS_WAV1}" \
    --id-experimento "ensemble_wavelet_udvd_t7_struct_vert_ult10min"

evaluar_modelo "ensemble" "${DIR_ENS_WAV1}" "ensemble_wavelet_udvd_t7_struct_vert_ult10min" "Wavelet, Base=UDVD_T7, Textura=StructN2V_Vert, Umbral=2.0"

# B. Ensamble Wavelet: UDVD Destriping + StructN2V Vertical
DIR_ENS_WAV2="${DIR_EXPOS}/ensemble_wavelet_udvd_destrip_struct_vert_ult10min"
echo "Ejecutando Ensamble Wavelet (UDVD Destriping + StructN2V Vert)..."
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --umbral-ruido 2.0 \
    --modelos-entrada "${DIR_UDVD_DESTRIP}" "${DIR_STRUCT_VERT}" \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --carpeta-salida "${DIR_ENS_WAV2}" \
    --id-experimento "ensemble_wavelet_udvd_destrip_struct_vert_ult10min"

evaluar_modelo "ensemble" "${DIR_ENS_WAV2}" "ensemble_wavelet_udvd_destrip_struct_vert_ult10min" "Wavelet, Base=UDVD_Destriping, Textura=StructN2V_Vert, Umbral=2.0"

# C. Ensamble Mediana Robusta Top-3
DIR_ENS_MED="${DIR_EXPOS}/ensemble_mediana_top3_ult10min"
echo "Ejecutando Ensamble Mediana Top-3..."
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion mediana \
    --modelos-entrada "${DIR_UDVD_T7}" "${DIR_UDVD_DESTRIP}" "${DIR_STRUCT_VERT}" \
    --ultimos-frames "${ULTIMOS_FRAMES}" \
    --carpeta-salida "${DIR_ENS_MED}" \
    --id-experimento "ensemble_mediana_top3_ult10min"

evaluar_modelo "ensemble" "${DIR_ENS_MED}" "ensemble_mediana_top3_ult10min" "Mediana_Robusta, Modelos=[UDVD_T7, UDVD_Destrip, StructN2V_Vert]"

echo
echo "============================================================"
echo "TODOS LOS EXPERIMENTOS DE LOS ULTIMOS 10 MINUTOS COMPLETADOS"
echo "Resultados consolidados en: videos/${VIDEO_ID}/${EXCEL_EXP}"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
