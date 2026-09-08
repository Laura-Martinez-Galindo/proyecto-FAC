#!/usr/bin/env bash

#SBATCH --job-name=grid_search_flir
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia:2
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
echo "GRID SEARCH & ABLATION STUDY: DENOISING FLIR (2 GPUs PARALELO)"
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
echo "Calculando metricas base de frames Originales (primeros ${MAX_FRAMES} frames)..."
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo original \
    --experimento "Base_Original_${MAX_FRAMES}f" \
    --archivo-resumen "${EXCEL_EXP}" \
    --max-frames "${MAX_FRAMES}"

echo "Calculando metricas base de frames Sin HUD..."
CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
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
# PAREJA 1: N2N Filtro Movimiento vs Sin Filtro
# ============================================================
echo
echo "--- Lote 1/7: N2N Filtro Mov vs Sin Filtro ---"
ejecutar_un_experimento 0 "scripts/pipeline_n2n.py" "original" "N2N_Orig_FiltroMov_d4_lr1e4" "depth=4, lr=1e-4, filtro_mov=True" --depth 4 --lr 0.0001 --filtrar-movimiento &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_n2n.py" "original" "N2N_Orig_SinFiltro_d4_lr1e4" "depth=4, lr=1e-4, filtro_mov=False" --depth 4 --lr 0.0001 --sin-filtro-movimiento &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# PAREJA 2: N2N Learning Rates (3e-4 vs 1e-3)
# ============================================================
echo
echo "--- Lote 2/7: N2N Learning Rates ---"
ejecutar_un_experimento 0 "scripts/pipeline_n2n.py" "original" "N2N_Orig_FiltroMov_d4_lr3e4" "depth=4, lr=3e-4, filtro_mov=True" --depth 4 --lr 0.0003 --filtrar-movimiento &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_n2n.py" "original" "N2N_Orig_FiltroMov_d4_lr1e3" "depth=4, lr=1e-3, filtro_mov=True" --depth 4 --lr 0.001 --filtrar-movimiento &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# PAREJA 3: N2N Depth 3 vs N2N sobre frames Sin HUD
# ============================================================
echo
echo "--- Lote 3/7: N2N Depth 3 y N2N Sin HUD ---"
ejecutar_un_experimento 0 "scripts/pipeline_n2n.py" "original" "N2N_Orig_FiltroMov_d3_lr3e4" "depth=3, lr=3e-4, filtro_mov=True" --depth 3 --lr 0.0003 --filtrar-movimiento &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_n2n.py" "sin_hud" "N2N_SinHUD_FiltroMov_d4_lr3e4" "modo=sin_hud, depth=4, lr=3e-4, filtro_mov=True" --depth 4 --lr 0.0003 --filtrar-movimiento &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# PAREJA 4: N2V Clasico vs N2V2
# ============================================================
echo
echo "--- Lote 4/7: N2V Clasico vs N2V2 ---"
ejecutar_un_experimento 0 "scripts/pipeline_n2v.py" "original" "N2V_Orig_Clasico_d3_lr1e4" "algoritmo=N2V1, depth=3, lr=1e-4" --depth 3 --lr 0.0001 --sin-n2v2 &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_n2v.py" "original" "N2V_Orig_N2V2_d3_lr1e4" "algoritmo=N2V2, depth=3, lr=1e-4" --depth 3 --lr 0.0001 --use-n2v2 &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# PAREJA 5: N2V2 Depth 4 vs N2V2 lr 1e-3
# ============================================================
echo
echo "--- Lote 5/7: N2V2 Depth y LR ---"
ejecutar_un_experimento 0 "scripts/pipeline_n2v.py" "original" "N2V_Orig_N2V2_d4_lr3e4" "algoritmo=N2V2, depth=4, lr=3e-4" --depth 4 --lr 0.0003 --use-n2v2 &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_n2v.py" "original" "N2V_Orig_N2V2_d3_lr1e3" "algoritmo=N2V2, depth=3, lr=1e-3" --depth 3 --lr 0.001 --use-n2v2 &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# PAREJA 6: StructN2V vs N2V2 Sin HUD
# ============================================================
echo
echo "--- Lote 6/7: StructN2V y N2V2 Sin HUD ---"
ejecutar_un_experimento 0 "scripts/pipeline_n2v.py" "original" "N2V_Orig_StructHoriz_d3_lr3e4" "algoritmo=StructN2V_horizontal, depth=3, lr=3e-4" --depth 3 --lr 0.0003 --use-n2v2 --struct-n2v horizontal &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_n2v.py" "sin_hud" "N2V_SinHUD_N2V2_d3_lr3e4" "modo=sin_hud, algoritmo=N2V2, depth=3, lr=3e-4" --depth 3 --lr 0.0003 --use-n2v2 &
PID2=$!
wait $PID1; wait $PID2

# ============================================================
# PAREJA 7: Neighbor2Neighbor y Blind2Unblind
# ============================================================
echo
echo "--- Lote 7/7: Neighbor2Neighbor y Blind2Unblind ---"
ejecutar_un_experimento 0 "scripts/pipeline_neighbor2neighbor.py" "original" "Neigh2Neigh_Orig_d4_lr3e4" "algoritmo=Neighbor2Neighbor, depth=4, lr=3e-4" --depth 4 --lr 0.0003 &
PID1=$!
ejecutar_un_experimento 1 "scripts/pipeline_blind2unblind.py" "original" "Blind2Unblind_Orig_d4_lr3e4" "algoritmo=Blind2Unblind, depth=4, lr=3e-4" --depth 4 --lr 0.0003 &
PID2=$!
wait $PID1; wait $PID2


echo
echo "============================================================"
echo "GRID SEARCH EN 2 GPUs COMPLETADO EXITOSAMENTE"
echo "Resumen consolidado: videos/${VIDEO_ID}/${EXCEL_EXP}"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
