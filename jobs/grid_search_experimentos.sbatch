#!/usr/bin/env bash

#SBATCH --job-name=grid_search_flir
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:nvidia:2
#SBATCH --time=2-00:00:00
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
echo "GRID SEARCH & ABLATION STUDY: DENOISING FLIR"
echo "============================================================"
echo "Video: ${VIDEO_ID}"
echo "Frames para prueba: ${MAX_FRAMES}"
echo "Epocas: ${EPOCAS} | Pasos: ${PASOS}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Nodo: ${SLURMD_NODENAME:-local}"
echo "Excel destino: videos/${VIDEO_ID}/${EXCEL_EXP}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

# 0. Metricas Base de Referencia (Original y Sin HUD)
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


ejecutar_experimento() {
    local script="$1"
    local modo="$2"
    local id_exp="$3"
    local params_extra="$4"
    shift 4

    echo
    echo "------------------------------------------------------------"
    echo "EJECUTANDO: ${id_exp} (${script} en modo ${modo})"
    echo "Inicio: $(date --iso-8601=seconds)"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" "${script}" \
        --video "${VIDEO_ID}" \
        --modo "${modo}" \
        --max-frames "${MAX_FRAMES}" \
        --epocas "${EPOCAS}" \
        --pasos-por-epoca "${PASOS}" \
        --id-experimento "${id_exp}" \
        --reiniciar \
        "$@"

    echo "Calculando metricas para ${id_exp}..."
    CUDA_VISIBLE_DEVICES=0 "${RUTA_PYTHON}" scripts/calcular_metricas.py \
        --video "${VIDEO_ID}" \
        --modelo "${id_exp}" \
        --experimento "${id_exp}" \
        --archivo-resumen "${EXCEL_EXP}" \
        --max-frames "${MAX_FRAMES}" \
        --parametros-extra "${params_extra}"

    echo "Completado: ${id_exp}"
}

# ============================================================
# 1. BLOQUE NOISE2NOISE (N2N)
# ============================================================

# N2N con filtro estricto vs sin filtro
ejecutar_experimento "scripts/pipeline_n2n.py" "original" "N2N_Orig_FiltroMov_d4_lr1e4" "depth=4, lr=1e-4, filtro_mov=True" --depth 4 --lr 0.0001 --filtrar-movimiento
ejecutar_experimento "scripts/pipeline_n2n.py" "original" "N2N_Orig_SinFiltro_d4_lr1e4" "depth=4, lr=1e-4, filtro_mov=False" --depth 4 --lr 0.0001 --sin-filtro-movimiento

# N2N variacion de Learning Rate
ejecutar_experimento "scripts/pipeline_n2n.py" "original" "N2N_Orig_FiltroMov_d4_lr3e4" "depth=4, lr=3e-4, filtro_mov=True" --depth 4 --lr 0.0003 --filtrar-movimiento
ejecutar_experimento "scripts/pipeline_n2n.py" "original" "N2N_Orig_FiltroMov_d4_lr1e3" "depth=4, lr=1e-3, filtro_mov=True" --depth 4 --lr 0.001 --filtrar-movimiento

# N2N variacion de Depth
ejecutar_experimento "scripts/pipeline_n2n.py" "original" "N2N_Orig_FiltroMov_d3_lr3e4" "depth=3, lr=3e-4, filtro_mov=True" --depth 3 --lr 0.0003 --filtrar-movimiento

# N2N sobre frames Sin HUD (Orden B: HUD primero -> Denoising)
ejecutar_experimento "scripts/pipeline_n2n.py" "sin_hud" "N2N_SinHUD_FiltroMov_d4_lr3e4" "modo=sin_hud, depth=4, lr=3e-4, filtro_mov=True" --depth 4 --lr 0.0003 --filtrar-movimiento


# ============================================================
# 2. BLOQUE NOISE2VOID (N2V / N2V2 / StructN2V)
# ============================================================

# N2V clasico vs N2V2
ejecutar_experimento "scripts/pipeline_n2v.py" "original" "N2V_Orig_Clasico_d3_lr1e4" "algoritmo=N2V1, depth=3, lr=1e-4" --depth 3 --lr 0.0001 --sin-n2v2
ejecutar_experimento "scripts/pipeline_n2v.py" "original" "N2V_Orig_N2V2_d3_lr1e4" "algoritmo=N2V2, depth=3, lr=1e-4" --depth 3 --lr 0.0001 --use-n2v2

# N2V2 variacion de Depth y LR
ejecutar_experimento "scripts/pipeline_n2v.py" "original" "N2V_Orig_N2V2_d4_lr3e4" "algoritmo=N2V2, depth=4, lr=3e-4" --depth 4 --lr 0.0003 --use-n2v2
ejecutar_experimento "scripts/pipeline_n2v.py" "original" "N2V_Orig_N2V2_d3_lr1e3" "algoritmo=N2V2, depth=3, lr=1e-3" --depth 3 --lr 0.001 --use-n2v2

# StructN2V (para ruido en franjas horizontales tipico de FLIR)
ejecutar_experimento "scripts/pipeline_n2v.py" "original" "N2V_Orig_StructHoriz_d3_lr3e4" "algoritmo=StructN2V_horizontal, depth=3, lr=3e-4" --depth 3 --lr 0.0003 --use-n2v2 --struct-n2v horizontal

# N2V2 sobre frames Sin HUD
ejecutar_experimento "scripts/pipeline_n2v.py" "sin_hud" "N2V_SinHUD_N2V2_d3_lr3e4" "modo=sin_hud, algoritmo=N2V2, depth=3, lr=3e-4" --depth 3 --lr 0.0003 --use-n2v2


# ============================================================
# 3. BLOQUE NEIGHBOR2NEIGHBOR (CVPR 2021)
# ============================================================
ejecutar_experimento "scripts/pipeline_neighbor2neighbor.py" "original" "Neigh2Neigh_Orig_d4_lr3e4" "algoritmo=Neighbor2Neighbor, depth=4, lr=3e-4" --depth 4 --lr 0.0003
ejecutar_experimento "scripts/pipeline_neighbor2neighbor.py" "sin_hud" "Neigh2Neigh_SinHUD_d4_lr3e4" "modo=sin_hud, algoritmo=Neighbor2Neighbor, depth=4, lr=3e-4" --depth 4 --lr 0.0003


# ============================================================
# 4. BLOQUE BLIND2UNBLIND (CVPR 2022)
# ============================================================
ejecutar_experimento "scripts/pipeline_blind2unblind.py" "original" "Blind2Unblind_Orig_d4_lr3e4" "algoritmo=Blind2Unblind, depth=4, lr=3e-4" --depth 4 --lr 0.0003
ejecutar_experimento "scripts/pipeline_blind2unblind.py" "sin_hud" "Blind2Unblind_SinHUD_d4_lr3e4" "modo=sin_hud, algoritmo=Blind2Unblind, depth=4, lr=3e-4" --depth 4 --lr 0.0003


echo
echo "============================================================"
echo "GRID SEARCH COMPLETADO EXITOSAMENTE"
echo "Resumen consolidado: videos/${VIDEO_ID}/${EXCEL_EXP}"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
