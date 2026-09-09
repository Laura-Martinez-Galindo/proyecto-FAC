#!/usr/bin/env bash

#SBATCH --job-name=grid_search_flir
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=jobs/logs/grid_search_%j.out
#SBATCH --error=jobs/logs/grid_search_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video1}"
MAX_FRAMES="${2:-1000}"
EPOCAS="${3:-15}"
PASOS="${4:-300}"

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"
EXCEL_EXP="resumen_experimentos.xlsx"

cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs cache/denoising

echo "============================================================"
echo "GRID SEARCH ETAPA 1 COMPLETA: 2 GPUs EN PARALELO"
echo "Comparativa: [GPU 0: Original] vs [GPU 1: Sin HUD]"
echo "============================================================"
echo "Video: ${VIDEO_ID}"
echo "Frames para prueba: ${MAX_FRAMES}"
echo "Epocas: ${EPOCAS} | Pasos: ${PASOS}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Nodo: ${SLURMD_NODENAME:-local}"
echo "GPU asignadas: ${CUDA_VISIBLE_DEVICES:-0,1}"
echo "Excel destino: videos/${VIDEO_ID}/${EXCEL_EXP}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

# 0. Metricas Base de Referencia
echo
echo "Calculando metricas base de frames Originales..."
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo original \
    --experimento "Base_Original_${MAX_FRAMES}f" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}"

echo "Calculando metricas base de frames Sin HUD..."
CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo sin_hud \
    --experimento "Base_Sin_HUD_${MAX_FRAMES}f" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}"


ejecutar_un_experimento() {
    local gpu="$1"
    local script="$2"
    local modo="$3"
    local id_exp="$4"
    local params_extra="$5"
    shift 5

    local log_exp="jobs/logs/${id_exp}_${SLURM_JOB_ID:-local}.log"
    echo "Iniciando ${id_exp} en GPU ${gpu} (log: ${log_exp})..."

    (
        CUDA_VISIBLE_DEVICES="${gpu}" "${RUTA_PYTHON}" "${script}" \
            --video "${VIDEO_ID}" \
            --modo "${modo}" \
            --max-frames "${MAX_FRAMES}" \
            --epocas "${EPOCAS}" \
            --pasos-por-epoca "${PASOS}" \
            --id-experimento "${id_exp}" \
            --reiniciar \
            "$@"

        echo "Calculando metricas para ${id_exp} en GPU ${gpu}..."
        CUDA_VISIBLE_DEVICES="${gpu}" "${RUTA_PYTHON}" scripts/calcular_metricas.py \
            --video "${VIDEO_ID}" \
            --modelo "${id_exp}" \
            --experimento "${id_exp}" \
            --archivo-resumen "${EXCEL_EXP}" \
            --max-frames "${MAX_FRAMES}" \
            --parametros-extra "${params_extra}"
        
        echo "COMPLETADO: ${id_exp}"
    ) > "${log_exp}" 2>&1
}


# ============================================================
# LOTE 1: Noise2Noise (Filtro Farneback, LR=1e-3)
# ============================================================
echo
echo "--- Lote 1/13: Noise2Noise ---"
ejecutar_un_experimento 0 "scripts/pipeline_n2n.py" "original" "N2N_Orig_FiltroMov_d4_lr1e3" "Familia=Par_N2N, FiltroMov=True, LR=1e-3, Depth=4" --depth 4 --lr 0.001 --filtrar-movimiento &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_n2n.py" "sin_hud" "N2N_SinHUD_FiltroMov_d4_lr1e3" "Familia=Par_N2N, FiltroMov=True, LR=1e-3, Depth=4" --depth 4 --lr 0.001 --filtrar-movimiento &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 2: StructN2V Horizontal (Mascara Lineal Horizontal)
# ============================================================
echo
echo "--- Lote 2/13: StructN2V Horizontal ---"
ejecutar_un_experimento 0 "scripts/pipeline_n2v.py" "original" "StructN2V_Horiz_Orig_d3_lr1e3" "Familia=Espacial_N2V, Mascara=Horizontal_5px, LR=1e-3, Depth=3" --depth 3 --lr 0.001 --struct-n2v horizontal &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_n2v.py" "sin_hud" "StructN2V_Horiz_SinHUD_d3_lr1e3" "Familia=Espacial_N2V, Mascara=Horizontal_5px, LR=1e-3, Depth=3" --depth 3 --lr 0.001 --struct-n2v horizontal &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 3: StructN2V Vertical (Mascara Lineal Vertical)
# ============================================================
echo
echo "--- Lote 3/13: StructN2V Vertical ---"
ejecutar_un_experimento 0 "scripts/pipeline_n2v.py" "original" "StructN2V_Vert_Orig_d3_lr1e3" "Familia=Espacial_N2V, Mascara=Vertical_5px, LR=1e-3, Depth=3" --depth 3 --lr 0.001 --struct-n2v vertical &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_n2v.py" "sin_hud" "StructN2V_Vert_SinHUD_d3_lr1e3" "Familia=Espacial_N2V, Mascara=Vertical_5px, LR=1e-3, Depth=3" --depth 3 --lr 0.001 --struct-n2v vertical &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 4: StructN2V Cross (Mascara Cruz)
# ============================================================
echo
echo "--- Lote 4/13: StructN2V Cruz ---"
ejecutar_un_experimento 0 "scripts/pipeline_n2v.py" "original" "StructN2V_Cross_Orig_d3_lr1e3" "Familia=Espacial_N2V, Mascara=Cruz_5px, LR=1e-3, Depth=3" --depth 3 --lr 0.001 --struct-n2v cross &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_n2v.py" "sin_hud" "StructN2V_Cross_SinHUD_d3_lr1e3" "Familia=Espacial_N2V, Mascara=Cruz_5px, LR=1e-3, Depth=3" --depth 3 --lr 0.001 --struct-n2v cross &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 5: Blind2Unblind (CVPR 2022)
# ============================================================
echo
echo "--- Lote 5/13: Blind2Unblind ---"
ejecutar_un_experimento 0 "scripts/pipeline_blind2unblind.py" "original" "Blind2Unblind_Orig_d4_lr1e3" "Familia=Espacial_B2U, Mascara=Visibilidad_Estructurada, LR=1e-3, Depth=4" --depth 4 --lr 0.001 &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_blind2unblind.py" "sin_hud" "Blind2Unblind_SinHUD_d4_lr1e3" "Familia=Espacial_B2U, Mascara=Visibilidad_Estructurada, LR=1e-3, Depth=4" --depth 4 --lr 0.001 &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 6: Neighbor2Neighbor (CVPR 2021)
# ============================================================
echo
echo "--- Lote 6/13: Neighbor2Neighbor ---"
ejecutar_un_experimento 0 "scripts/pipeline_neighbor2neighbor.py" "original" "Neigh2Neigh_Orig_d4_lr1e3" "Familia=Espacial_N2N, Subsampler=2x2, LR=1e-3, Depth=4" --depth 4 --lr 0.001 &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_neighbor2neighbor.py" "sin_hud" "Neigh2Neigh_SinHUD_d4_lr1e3" "Familia=Espacial_N2N, Subsampler=2x2, LR=1e-3, Depth=4" --depth 4 --lr 0.001 &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 7: Frames2Residual (Multi-Frame T=5)
# ============================================================
echo
echo "--- Lote 7/13: Frames2Residual (T=5) ---"
ejecutar_un_experimento 0 "scripts/pipeline_frames2residual.py" "original" "Frames2Residual_Orig_T5_lr1e3" "Familia=Temporal_F2R, Ventana=T5, LR=1e-3, Depth=4" --depth 4 --lr 0.001 --num-frames 5 &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_frames2residual.py" "sin_hud" "Frames2Residual_SinHUD_T5_lr1e3" "Familia=Temporal_F2R, Ventana=T5, LR=1e-3, Depth=4" --depth 4 --lr 0.001 --num-frames 5 &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 8: FastDVDnet (Dual-Stage Cascaded UNet, T=5)
# ============================================================
echo
echo "--- Lote 8/13: FastDVDnet (T=5) ---"
ejecutar_un_experimento 0 "scripts/pipeline_fastdvdnet.py" "original" "FastDVDnet_Orig_T5_lr1e3" "Familia=Temporal_FastDVD, Ventana=T5, LR=1e-3, BaseCh=32" --lr 0.001 --base-channels 32 &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_fastdvdnet.py" "sin_hud" "FastDVDnet_SinHUD_T5_lr1e3" "Familia=Temporal_FastDVD, Ventana=T5, LR=1e-3, BaseCh=32" --lr 0.001 --base-channels 32 &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 9: RViDeNet (Recurrente ConvGRU)
# ============================================================
echo
echo "--- Lote 9/13: RViDeNet (Recurrente ConvGRU) ---"
ejecutar_un_experimento 0 "scripts/pipeline_rvidenet.py" "original" "RViDeNet_Orig_Seq5_lr1e3" "Familia=Temporal_RViDeNet, Recurrente=ConvGRU, LR=1e-3, BaseCh=32" --lr 0.001 --base-channels 32 &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_rvidenet.py" "sin_hud" "RViDeNet_SinHUD_Seq5_lr1e3" "Familia=Temporal_RViDeNet, Recurrente=ConvGRU, LR=1e-3, BaseCh=32" --lr 0.001 --base-channels 32 &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 10: UDVD (Dynamic Kernel Denoising, CVPR 2021)
# ============================================================
echo
echo "--- Lote 10/13: UDVD (Dynamic Kernels) ---"
ejecutar_un_experimento 0 "scripts/pipeline_udvd.py" "original" "UDVD_Orig_K5_lr1e3" "Familia=Temporal_UDVD, DynamicKernel=5x5, LR=1e-3" --lr 0.001 --kernel-size 5 &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_udvd.py" "sin_hud" "UDVD_SinHUD_K5_lr1e3" "Familia=Temporal_UDVD, DynamicKernel=5x5, LR=1e-3" --lr 0.001 --kernel-size 5 &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 11: Bilateral Espacio-Temporal (Baseline Tradicional)
# ============================================================
echo
echo "--- Lote 11/13: Filtro Bilateral Espacio-Temporal ---"
(
    CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_temporal_bilateral.py \
        --video "${VIDEO_ID}" \
        --modo "original" \
        --d 7 --sigma-color 25 --sigma-space 25 --num-frames 5 \
        --max-frames "${MAX_FRAMES}" \
        --id-experimento "Bilateral_Temporal_Orig_T5"

    CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
        --video "${VIDEO_ID}" \
        --modelo "Bilateral_Temporal_Orig_T5" \
        --experimento "Bilateral_Temporal_Orig_T5" \
        --archivo-resumen "${EXCEL_EXP}" \
        --max-frames "${MAX_FRAMES}" \
        --parametros-extra "Familia=Tradicional, Metodo=Bilateral_Temporal, d=7, sigma=25, T=5"
) &
PID1=$!

(
    CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_temporal_bilateral.py \
        --video "${VIDEO_ID}" \
        --modo "sin_hud" \
        --d 7 --sigma-color 25 --sigma-space 25 --num-frames 5 \
        --max-frames "${MAX_FRAMES}" \
        --id-experimento "Bilateral_Temporal_SinHUD_T5"

    CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
        --video "${VIDEO_ID}" \
        --modelo "Bilateral_Temporal_SinHUD_T5" \
        --experimento "Bilateral_Temporal_SinHUD_T5" \
        --archivo-resumen "${EXCEL_EXP}" \
        --max-frames "${MAX_FRAMES}" \
        --parametros-extra "Familia=Tradicional, Metodo=Bilateral_Temporal, d=7, sigma=25, T=5"
) &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 12: Ensamble Fusión Wavelet (Multi-Frame + Espacial Nítido)
# ============================================================
echo
echo "--- Lote 12/13: Ensamble Fusión Wavelet ---"
(
    CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
        --video "${VIDEO_ID}" \
        --modo "original" \
        --metodo-fusion "wavelet" \
        --modelos-entrada "Frames2Residual_Orig_T5_lr1e3" "Blind2Unblind_Orig_d4_lr1e3" \
        --max-frames "${MAX_FRAMES}" \
        --id-experimento "Ensemble_Wavelet_Orig"

    CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
        --video "${VIDEO_ID}" \
        --modelo "Ensemble_Wavelet_Orig" \
        --experimento "Ensemble_Wavelet_Orig" \
        --archivo-resumen "${EXCEL_EXP}" \
        --max-frames "${MAX_FRAMES}" \
        --parametros-extra "Familia=Ensamble, Tipo=Wavelet_Fusion, Entradas=[Frames2Residual, Blind2Unblind]"
) &
PID1=$!

(
    CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
        --video "${VIDEO_ID}" \
        --modo "sin_hud" \
        --metodo-fusion "wavelet" \
        --modelos-entrada "Frames2Residual_SinHUD_T5_lr1e3" "Blind2Unblind_SinHUD_d4_lr1e3" \
        --max-frames "${MAX_FRAMES}" \
        --id-experimento "Ensemble_Wavelet_SinHUD"

    CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
        --video "${VIDEO_ID}" \
        --modelo "Ensemble_Wavelet_SinHUD" \
        --experimento "Ensemble_Wavelet_SinHUD" \
        --archivo-resumen "${EXCEL_EXP}" \
        --max-frames "${MAX_FRAMES}" \
        --parametros-extra "Familia=Ensamble, Tipo=Wavelet_Fusion, Entradas=[Frames2Residual, Blind2Unblind]"
) &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# LOTE 13: Ensamble Promedio Ponderado Adaptativo
# ============================================================
echo
echo "--- Lote 13/13: Ensamble Promedio Adaptativo ---"
(
    CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
        --video "${VIDEO_ID}" \
        --modo "original" \
        --metodo-fusion "promedio" \
        --modelos-entrada "FastDVDnet_Orig_T5_lr1e3" "StructN2V_Cross_Orig_d3_lr1e3" "Blind2Unblind_Orig_d4_lr1e3" \
        --pesos 0.4 0.3 0.3 \
        --max-frames "${MAX_FRAMES}" \
        --id-experimento "Ensemble_Promedio_Orig"

    CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
        --video "${VIDEO_ID}" \
        --modelo "Ensemble_Promedio_Orig" \
        --experimento "Ensemble_Promedio_Orig" \
        --archivo-resumen "${EXCEL_EXP}" \
        --max-frames "${MAX_FRAMES}" \
        --parametros-extra "Familia=Ensamble, Tipo=Promedio_Ponderado, Entradas=[FastDVD:0.4, StructCross:0.3, B2U:0.3]"
) &
PID1=$!

(
    CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/pipeline_ensemble.py \
        --video "${VIDEO_ID}" \
        --modo "sin_hud" \
        --metodo-fusion "promedio" \
        --modelos-entrada "FastDVDnet_SinHUD_T5_lr1e3" "StructN2V_Cross_SinHUD_d3_lr1e3" "Blind2Unblind_SinHUD_d4_lr1e3" \
        --pesos 0.4 0.3 0.3 \
        --max-frames "${MAX_FRAMES}" \
        --id-experimento "Ensemble_Promedio_SinHUD"

    CUDA_VISIBLE_DEVICES=1 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
        --video "${VIDEO_ID}" \
        --modelo "Ensemble_Promedio_SinHUD" \
        --experimento "Ensemble_Promedio_SinHUD" \
        --archivo-resumen "${EXCEL_EXP}" \
        --max-frames "${MAX_FRAMES}" \
        --parametros-extra "Familia=Ensamble, Tipo=Promedio_Ponderado, Entradas=[FastDVD:0.4, StructCross:0.3, B2U:0.3]"
) &
PID2=$!
wait $PID1; wait $PID2


echo
echo "============================================================"
echo "GRID SEARCH COMPLETO ETAPA 1 FINALIZADO EXITOSAMENTE"
echo "Resumen consolidado: videos/${VIDEO_ID}/${EXCEL_EXP}"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
