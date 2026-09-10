#!/usr/bin/env bash

#SBATCH --job-name=experimentos_sinhud
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia:2
#SBATCH --time=12:00:00
#SBATCH --output=jobs/logs/experimentos_%j.out
#SBATCH --error=jobs/logs/experimentos_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

# Parametros configurables desde la linea de comandos
VIDEO_ID="${1:-video1}"
MAX_FRAMES="${2:-1000}"
EPOCAS="${3:-15}"
LR="${4:-1e-3}"
PASOS="${5:-300}"

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"
EXCEL_EXP="resumen_experimentos.xlsx"
DIR_EXPOS="videos/${VIDEO_ID}/expos"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs "${DIR_EXPOS}"

echo "============================================================"
echo "SUITE COMPLETA DE EXPERIMENTOS DEPOSICIÓN AUTOSUPERVISADA"
echo "Video: ${VIDEO_ID} | Modo: sin_hud | Frames: ${MAX_FRAMES}"
echo "Hiperparametros: Epocas=${EPOCAS}, LR=${LR}, Pasos/Epoca=${PASOS}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

# Función para calcular métricas tras cada corrida
evaluar_modelo() {
    local mod="$1"
    local dir_out="$2"
    local exp_id="$3"
    local info="$4"

    echo "Calculando metricas para ${exp_id}..."
    "${RUTA_PYTHON}" scripts/calcular_metricas.py \
        --video "${VIDEO_ID}" \
        --modelo "${mod}" \
        --carpeta-directa "${dir_out}" \
        --experimento "${exp_id}" \
        --archivo-resumen "${EXCEL_EXP}" \
        --max-frames "${MAX_FRAMES}" \
        --parametros-extra "${info}"
}

# -------------------------------------------------------------------------
# 1. LINEA BASE TRADICIONAL
# -------------------------------------------------------------------------
echo
echo "--- [1/11] Filtro Bilateral Temporal 3D ---"
DIR_BILATERAL="${DIR_EXPOS}/bilateral_temporal_sinhud"
"${RUTA_PYTHON}" scripts/pipeline_temporal_bilateral.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_BILATERAL}" \
    --id-experimento "bilateral_temporal_sinhud"

evaluar_modelo "bilateral" "${DIR_BILATERAL}" "bilateral_temporal_sinhud" "Metodo=Bilateral_3D_Temporal, Frames=${MAX_FRAMES}"

# -------------------------------------------------------------------------
# 2. MODELOS ESPACIALES 2D (Ejecucion en paralelo en 2 GPUs)
# -------------------------------------------------------------------------
echo
echo "--- [2/11] Noise2Noise con Filtro de Movimiento Farnebäck (GPU 0) ---"
DIR_N2N="${DIR_EXPOS}/n2n_sinhud"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_n2n.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --filtrado-movimiento \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_N2N}" \
    --id-experimento "n2n_sinhud" &

echo "--- [3/11] Noise2Void N2V2 (GPU 1) ---"
DIR_N2V2="${DIR_EXPOS}/n2v2_sinhud"
CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --use-n2v2 \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_N2V2}" \
    --id-experimento "n2v2_sinhud" &
wait

evaluar_modelo "n2n" "${DIR_N2N}" "n2n_sinhud" "Epocas=${EPOCAS}, LR=${LR}, FiltroMov=True"
evaluar_modelo "n2v" "${DIR_N2V2}" "n2v2_sinhud" "Epocas=${EPOCAS}, LR=${LR}, N2V2=True"

echo
echo "--- [4/11] StructN2V Vertical 5px (GPU 0 - Desrayador FPN) ---"
DIR_STRUCT_VERT="${DIR_EXPOS}/struct_n2v_vert_sinhud"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --use-n2v2 \
    --struct-n2v vertical \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_STRUCT_VERT}" \
    --id-experimento "struct_n2v_vert_sinhud" &

echo "--- [5/11] StructN2V Horizontal 5px (GPU 1) ---"
DIR_STRUCT_HORIZ="${DIR_EXPOS}/struct_n2v_horiz_sinhud"
CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_n2v.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --use-n2v2 \
    --struct-n2v horizontal \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_STRUCT_HORIZ}" \
    --id-experimento "struct_n2v_horiz_sinhud" &
wait

evaluar_modelo "n2v" "${DIR_STRUCT_VERT}" "struct_n2v_vert_sinhud" "Epocas=${EPOCAS}, LR=${LR}, StructN2V=Vertical_5px"
evaluar_modelo "n2v" "${DIR_STRUCT_HORIZ}" "struct_n2v_horiz_sinhud" "Epocas=${EPOCAS}, LR=${LR}, StructN2V=Horizontal_5px"

echo
echo "--- [6/11] Neighbor2Neighbor (GPU 0) ---"
DIR_NEIGH="${DIR_EXPOS}/neighbor2neighbor_sinhud"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_neighbor2neighbor.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_NEIGH}" \
    --id-experimento "neighbor2neighbor_sinhud" &

echo "--- [7/11] Blind2Unblind (GPU 1) ---"
DIR_B2U="${DIR_EXPOS}/blind2unblind_sinhud"
CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_blind2unblind.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_B2U}" \
    --id-experimento "blind2unblind_sinhud" &
wait

evaluar_modelo "neighbor2neighbor" "${DIR_NEIGH}" "neighbor2neighbor_sinhud" "Epocas=${EPOCAS}, LR=${LR}"
evaluar_modelo "blind2unblind" "${DIR_B2U}" "blind2unblind_sinhud" "Epocas=${EPOCAS}, LR=${LR}"

# -------------------------------------------------------------------------
# 3. MODELOS TEMPORALES Y RECURRENTES
# -------------------------------------------------------------------------
echo
echo "--- [8/11] FastDVDnet Cascada Temporal (GPU 0) ---"
DIR_FASTDVD="${DIR_EXPOS}/fastdvdnet_sinhud"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_fastdvdnet.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_FASTDVD}" \
    --id-experimento "fastdvdnet_sinhud" &

echo "--- [9/11] RViDeNet Recurrente ConvGRU (GPU 1) ---"
DIR_RVIDE="${DIR_EXPOS}/rvidenet_sinhud"
CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_rvidenet.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_RVIDE}" \
    --id-experimento "rvidenet_sinhud" &
wait

evaluar_modelo "fastdvdnet" "${DIR_FASTDVD}" "fastdvdnet_sinhud" "Epocas=${EPOCAS}, LR=${LR}, T=5"
evaluar_modelo "rvidenet" "${DIR_RVIDE}" "rvidenet_sinhud" "Epocas=${EPOCAS}, LR=${LR}, Seq=5"

echo
echo "--- [10/11] UDVD Dynamic Kernels (GPU 0 - Campeón Individual) ---"
DIR_UDVD="${DIR_EXPOS}/udvd_sinhud"
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_udvd.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --lr "${LR}" \
    --epocas "${EPOCAS}" \
    --pasos-por-epoca "${PASOS}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_UDVD}" \
    --id-experimento "udvd_sinhud"

evaluar_modelo "udvd" "${DIR_UDVD}" "udvd_sinhud" "Epocas=${EPOCAS}, LR=${LR}, Kernel=5x5"

# -------------------------------------------------------------------------
# 4. SÚPER ENSAMBLES Y FUSIÓN FRECUENCIAL WAVELET
# -------------------------------------------------------------------------
echo
echo "--- [11/11] Ejecutando Ensambles Multiescala ---"

# A. Ensamble Wavelet Ganador (UDVD + Blind2Unblind)
DIR_WAV_UDVD_B2U="${DIR_EXPOS}/ensemble_wavelet_udvd_b2u_sinhud"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "${DIR_UDVD}" "${DIR_B2U}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_WAV_UDVD_B2U}" \
    --id-experimento "ensemble_wavelet_udvd_b2u_sinhud"

evaluar_modelo "ensemble" "${DIR_WAV_UDVD_B2U}" "ensemble_wavelet_udvd_b2u_sinhud" "Fusion=Wavelet, Base=UDVD, Textura=Blind2Unblind"

# B. Ensamble Wavelet FPN (UDVD + StructN2V Vertical)
DIR_WAV_UDVD_STRUCT="${DIR_EXPOS}/ensemble_wavelet_udvd_struct_vert_sinhud"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "${DIR_UDVD}" "${DIR_STRUCT_VERT}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_WAV_UDVD_STRUCT}" \
    --id-experimento "ensemble_wavelet_udvd_struct_vert_sinhud"

evaluar_modelo "ensemble" "${DIR_WAV_UDVD_STRUCT}" "ensemble_wavelet_udvd_struct_vert_sinhud" "Fusion=Wavelet, Base=UDVD, FPN=StructN2V_Vert"

# C. Ensamble Wavelet Triple (UDVD + StructN2V Vert + Blind2Unblind)
DIR_WAV_TRIPLE="${DIR_EXPOS}/ensemble_wavelet_udvd_struct_vert_b2u_sinhud"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion wavelet \
    --modelos-entrada "${DIR_UDVD}" "${DIR_STRUCT_VERT}" "${DIR_B2U}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_WAV_TRIPLE}" \
    --id-experimento "ensemble_wavelet_udvd_struct_vert_b2u_sinhud"

evaluar_modelo "ensemble" "${DIR_WAV_TRIPLE}" "ensemble_wavelet_udvd_struct_vert_b2u_sinhud" "Fusion=Wavelet, Base=UDVD, FPN=StructVert, Textura=Blind2Unblind"

# D. Ensamble Ponderado Top-3
DIR_POND_TOP3="${DIR_EXPOS}/ensemble_ponderado_udvd_struct_vert_b2u_sinhud"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion promedio \
    --modelos-entrada "${DIR_UDVD}" "${DIR_STRUCT_VERT}" "${DIR_B2U}" \
    --pesos 0.45 0.35 0.20 \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_POND_TOP3}" \
    --id-experimento "ensemble_ponderado_udvd_struct_vert_b2u_sinhud"

evaluar_modelo "ensemble" "${DIR_POND_TOP3}" "ensemble_ponderado_udvd_struct_vert_b2u_sinhud" "Fusion=Ponderado, Pesos=0.45_0.35_0.20"

# E. Ensamble Mediana Top-3
DIR_MED_TOP3="${DIR_EXPOS}/ensemble_mediana_udvd_struct_vert_b2u_sinhud"
"${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
    --video "${VIDEO_ID}" \
    --modo sin_hud \
    --metodo-fusion mediana \
    --modelos-entrada "${DIR_UDVD}" "${DIR_STRUCT_VERT}" "${DIR_B2U}" \
    --max-frames "${MAX_FRAMES}" \
    --carpeta-salida "${DIR_MED_TOP3}" \
    --id-experimento "ensemble_mediana_udvd_struct_vert_b2u_sinhud"

evaluar_modelo "ensemble" "${DIR_MED_TOP3}" "ensemble_mediana_udvd_struct_vert_b2u_sinhud" "Fusion=Mediana, Top3_Modelos"

echo
echo "============================================================"
echo "TODOS LOS EXPERIMENTOS COMPLETADOS CON ÉXITO"
echo "Resultados consolidados en: videos/${VIDEO_ID}/${EXCEL_EXP}"
echo "============================================================"

# Imprimir ranking en los logs
"${RUTA_PYTHON}" scripts/analizar_experimentos.py --video "${VIDEO_ID}" --excel "${EXCEL_EXP}" --frames 560 600 650
