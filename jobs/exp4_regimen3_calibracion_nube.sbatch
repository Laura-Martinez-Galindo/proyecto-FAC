#!/bin/bash
#SBATCH --job-name=EXP4_regimen3_CALIBRACION_NUBE
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/exp4_regimen3_%j.out
#SBATCH --error=logs/slurm/exp4_regimen3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@unal.edu.co

# ==============================================================================
# EXP-4 — Régimen 3: Laboratorio de Calibración en Nube Densa (Pure Noise Test)
# Bloque: Video1 frames 004622 a 004921 (300 frames en nube densa)
# Escenario: Terreno 100% cubierto por nube, sin ríos ni estructuras.
# Objetivo: Medir la capacidad de supresión de ruido puro hacia sigma_MAD -> 0
#           y verificar la huella espectral FFT / autocorrelación del ISP.
#
# Modelos evaluados:
#   udvd, p2n, mm_bsn, ap_bsn, nb2nb, f2r
#
# Distribución en 2 GPUs:
#   GPU 0: udvd, mm_bsn, nb2nb
#   GPU 1: p2n, ap_bsn, f2r
# ==============================================================================

set -euo pipefail
date
echo "=== EXP-4 iniciado en $(hostname) ==="

source /home/lf.martinezg1/miniconda3/bin/activate fac_env311

RAIZ="/home/lf.martinezg1/proyecto-FAC"
SCRIPTS="$RAIZ/scripts"
LOG_DIR="$RAIZ/logs/slurm"
mkdir -p "$LOG_DIR"

DIR_V1="$RAIZ/videos/video1/expos_regimenes/EXP-4_v1_004622-004921_nube"
INPUT_V1="$DIR_V1/input"
mkdir -p "$INPUT_V1"

python "$SCRIPTS/preparar_bloque_entrada.py" \
    --video video1 \
    --inicio-frame frame_04622.png \
    --n-frames 300 \
    --carpeta-salida "$INPUT_V1" \
    --filtrar-menu-central 2>&1 | tee "$LOG_DIR/exp4_prep_$SLURM_JOB_ID.log"

# Ronda 1: UDVD (GPU 0) y P2N (GPU 1)
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo udvd \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/udvd/denoised" \
    --carpeta-cache "$DIR_V1/udvd/cache" \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp4_udvd_$SLURM_JOB_ID.log" 2>&1 &
PID_U=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo p2n \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/p2n/denoised" \
    --carpeta-cache "$DIR_V1/p2n/cache" \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp4_p2n_$SLURM_JOB_ID.log" 2>&1 &
PID_P=$!

wait $PID_U $PID_P

# Ronda 2: MM-BSN (GPU 0) y AP-BSN (GPU 1)
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo mm_bsn \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/mm_bsn/denoised" \
    --carpeta-cache "$DIR_V1/mm_bsn/cache" \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp4_mm_bsn_$SLURM_JOB_ID.log" 2>&1 &
PID_M=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo ap_bsn \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/ap_bsn/denoised" \
    --carpeta-cache "$DIR_V1/ap_bsn/cache" \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp4_ap_bsn_$SLURM_JOB_ID.log" 2>&1 &
PID_A=$!

wait $PID_M $PID_A

# Ronda 3: NB2NB (GPU 0) y F2R (GPU 1)
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo nb2nb \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/nb2nb/denoised" \
    --carpeta-cache "$DIR_V1/nb2nb/cache" \
    --epocas 10 --batch-size 8 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp4_nb2nb_$SLURM_JOB_ID.log" 2>&1 &
PID_N=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo f2r \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/f2r/denoised" \
    --carpeta-cache "$DIR_V1/f2r/cache" \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp4_f2r_$SLURM_JOB_ID.log" 2>&1 &
PID_F=$!

wait $PID_N $PID_F

# ═════════════════════════════════════════════════════════════════════════════
# MÉTRICAS Y TABLA DE CALIBRACIÓN
# ═════════════════════════════════════════════════════════════════════════════
python "$SCRIPTS/calcular_metricas_experimento.py" \
    --dir-input "$INPUT_V1" \
    --dir-exp "$DIR_V1" \
    --exp-id "EXP-4" \
    --etiqueta "v1_004622-004921_nube" \
    --modelos "udvd,p2n,mm_bsn,ap_bsn,nb2nb,f2r" \
    > "$LOG_DIR/exp4_metricas_$SLURM_JOB_ID.log" 2>&1

python "$SCRIPTS/consolidar_resultados_exp.py" \
    --exp-id "EXP-4" \
    --dirs "$DIR_V1" \
    --salida "$RAIZ/videos/video1/expos_regimenes/resultados_EXP-4_nube.xlsx"

echo ""
echo "=== EXP-4 COMPLETADO EXITOSAMENTE ==="
date
