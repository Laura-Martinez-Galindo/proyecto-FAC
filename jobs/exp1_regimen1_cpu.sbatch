#!/bin/bash
#SBATCH --job-name=EXP1_cpu_short
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/exp1_cpu_%j.out
#SBATCH --error=logs/slurm/exp1_cpu_%j.err

# ==============================================================================
# EXP-1 EN PARTICIÓN SHORT (16 CPUs INTEL XEON, ARRANQUE INMEDIATO)
# Como son solo 300 frames con patches 128x128, cada época en 16 cores
# toma ~4 segundos. El experimento completo con los 5 modelos tarda ~12 min.
# ==============================================================================

set -euo pipefail
date
echo "=== EXP-1 en CPU (Partición short) iniciado en $(hostname) ==="
lscpu | grep -E "Model name|CPU\(s\):"

source /home/lf.martinezg1/miniconda3/bin/activate fac_env311

RAIZ="/home/lf.martinezg1/proyecto-FAC"
SCRIPTS="$RAIZ/scripts"
LOG_DIR="$RAIZ/logs/slurm"
mkdir -p "$LOG_DIR"

DIR_V3="$RAIZ/videos/video3/expos_regimenes/EXP-1_v3_001800-002099"
INPUT_V3="$DIR_V3/input"
mkdir -p "$INPUT_V3"

echo ">>> [1/3] Preparando y auditando 300 frames de Video 3..."
python "$SCRIPTS/preparar_bloque_entrada.py" \
    --video video3 \
    --inicio-frame frame_001800.png \
    --n-frames 300 \
    --carpeta-salida "$INPUT_V3" \
    --filtrar-menu-central

echo ">>> [2/3] Entrenando e infiriendo modelos en 16 hilos CPU..."

# 1. UDVD
echo "  -> [1/5] UDVD..."
python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo udvd \
    --carpeta-entrada "$INPUT_V3" \
    --carpeta-salida "$DIR_V3/udvd/denoised" \
    --carpeta-cache "$DIR_V3/udvd/cache" \
    --epocas 8 --batch-size 8 --patch-size 128 --paciencia 3

# 2. P2N (CVPR 2025 - Resolución 100% completa)
echo "  -> [2/5] P2N..."
python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo p2n \
    --carpeta-entrada "$INPUT_V3" \
    --carpeta-salida "$DIR_V3/p2n/denoised" \
    --carpeta-cache "$DIR_V3/p2n/cache" \
    --epocas 8 --batch-size 8 --patch-size 128 --paciencia 3

# 3. MM-BSN (CVPRW 2023 - Máscaras contra ISP)
echo "  -> [3/5] MM-BSN..."
python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo mm_bsn \
    --carpeta-entrada "$INPUT_V3" \
    --carpeta-salida "$DIR_V3/mm_bsn/denoised" \
    --carpeta-cache "$DIR_V3/mm_bsn/cache" \
    --stride-submuestreo 2 \
    --epocas 8 --batch-size 8 --patch-size 128 --paciencia 3

# 4. NB2NB (CVPR 2021)
echo "  -> [4/5] NB2NB..."
python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo nb2nb \
    --carpeta-entrada "$INPUT_V3" \
    --carpeta-salida "$DIR_V3/nb2nb/denoised" \
    --carpeta-cache "$DIR_V3/nb2nb/cache" \
    --epocas 8 --batch-size 8 --patch-size 128 --paciencia 3

# 5. Combo: P2N -> F2R
echo "  -> [5/5] Combo P2N -> F2R..."
mkdir -p "$DIR_V3/combo_p2n_f2r/denoised"
python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo f2r \
    --carpeta-entrada "$DIR_V3/p2n/denoised" \
    --carpeta-salida "$DIR_V3/combo_p2n_f2r/denoised" \
    --carpeta-cache "$DIR_V3/combo_p2n_f2r/cache" \
    --epocas 6 --batch-size 8 --patch-size 128 --paciencia 3

echo ">>> [3/3] Calculando métricas científicas completas..."
python "$SCRIPTS/calcular_metricas_experimento.py" \
    --dir-input "$INPUT_V3" \
    --dir-exp "$DIR_V3" \
    --exp-id "EXP-1" \
    --etiqueta "v3_001800-002099" \
    --modelos "udvd,p2n,mm_bsn,nb2nb,combo_p2n_f2r"

echo ""
echo "=== EXP-1 (CPU SHORT) COMPLETADO EXITOSAMENTE ==="
date
