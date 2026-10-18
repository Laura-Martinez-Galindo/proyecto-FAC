#!/bin/bash
#SBATCH --job-name=EXP1_regimen1_HDEO_DDE
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/exp1_regimen1_%j.out
#SBATCH --error=logs/slurm/exp1_regimen1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@unal.edu.co

# ==============================================================================
# EXP-1 — Régimen 1: HDEO + DDE (Alta altitud 15,895 FT, objetos diminutos, DVR)
# Bloques:
#   1. Video3 (frame_001800 al frame_002099 - 300 frames)
#   2. Video1 (frame_19500 al frame_19756 - 257 frames)
# Modelos:
#   1. UDVD (Baseline)
#   2. P2N (CVPR 2025 - Resolución 100% completa para objetos pequeños)
#   3. MM-BSN (CVPRW 2023 - Máscaras para romper halo DDE y bloques DVR)
#   4. NB2NB (CVPR 2021 - Submuestreo 2x2)
#   5. Combo: P2N -> F2R (Espacial puro full-res + Fusión temporal)
#
# Distribución paralela en 2 GPUs:
#   GPU 0: UDVD, MM-BSN, Combo P2N->F2R
#   GPU 1: P2N, NB2NB
# ==============================================================================

set -euo pipefail
date
echo "=== EXP-1 iniciado en $(hostname) ==="

source /home/lf.martinezg1/miniconda3/bin/activate fac_env311

RAIZ="/home/lf.martinezg1/proyecto-FAC"
SCRIPTS="$RAIZ/scripts"
LOG_DIR="$RAIZ/logs/slurm"
mkdir -p "$LOG_DIR"

# ═════════════════════════════════════════════════════════════════════════════
# BLOQUE 1: Video 3 — frame_001800 a frame_002099 (300 frames)
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> [BLOQUE 1/2] Video 3: frame_001800 a frame_002099 <<<"
DIR_V3="$RAIZ/videos/video3/expos_regimenes/EXP-1_v3_001800-002099"
INPUT_V3="$DIR_V3/input"
mkdir -p "$INPUT_V3"

python "$SCRIPTS/preparar_bloque_entrada.py" \
    --video video3 \
    --inicio-frame frame_001800.png \
    --n-frames 300 \
    --carpeta-salida "$INPUT_V3" \
    --filtrar-menu-central 2>&1 | tee "$LOG_DIR/exp1_v3_prep_$SLURM_JOB_ID.log"

# Lanzar UDVD (GPU 0) y P2N (GPU 1) en paralelo
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo udvd \
    --carpeta-entrada "$INPUT_V3" \
    --carpeta-salida "$DIR_V3/udvd/denoised" \
    --carpeta-cache "$DIR_V3/udvd/cache" \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp1_v3_udvd_$SLURM_JOB_ID.log" 2>&1 &
PID_UDVD=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo p2n \
    --carpeta-entrada "$INPUT_V3" \
    --carpeta-salida "$DIR_V3/p2n/denoised" \
    --carpeta-cache "$DIR_V3/p2n/cache" \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp1_v3_p2n_$SLURM_JOB_ID.log" 2>&1 &
PID_P2N=$!

wait $PID_UDVD $PID_P2N
echo "[✓] V3: UDVD y P2N finalizados"

# Lanzar MM-BSN (GPU 0) y NB2NB (GPU 1) en paralelo
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo mm_bsn \
    --carpeta-entrada "$INPUT_V3" \
    --carpeta-salida "$DIR_V3/mm_bsn/denoised" \
    --carpeta-cache "$DIR_V3/mm_bsn/cache" \
    --stride-submuestreo 2 \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp1_v3_mm_bsn_$SLURM_JOB_ID.log" 2>&1 &
PID_MMBSN=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo nb2nb \
    --carpeta-entrada "$INPUT_V3" \
    --carpeta-salida "$DIR_V3/nb2nb/denoised" \
    --carpeta-cache "$DIR_V3/nb2nb/cache" \
    --epocas 10 --batch-size 8 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp1_v3_nb2nb_$SLURM_JOB_ID.log" 2>&1 &
PID_NB2NB=$!

wait $PID_MMBSN $PID_NB2NB
echo "[✓] V3: MM-BSN y NB2NB finalizados"

# Combo: P2N -> F2R (toma la salida de P2N y aplica fusión temporal F2R en GPU 0)
echo "[...] V3: Ejecutando Combo P2N -> F2R"
mkdir -p "$DIR_V3/combo_p2n_f2r/denoised"
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo f2r \
    --carpeta-entrada "$DIR_V3/p2n/denoised" \
    --carpeta-salida "$DIR_V3/combo_p2n_f2r/denoised" \
    --carpeta-cache "$DIR_V3/combo_p2n_f2r/cache" \
    --epocas 8 --batch-size 6 --patch-size 128 --paciencia 3 --gpu 0 \
    > "$LOG_DIR/exp1_v3_combo_p2n_f2r_$SLURM_JOB_ID.log" 2>&1
echo "[✓] V3: Combo P2N -> F2R finalizado"

# Métricas Bloque V3
python "$SCRIPTS/calcular_metricas_experimento.py" \
    --dir-input "$INPUT_V3" \
    --dir-exp "$DIR_V3" \
    --exp-id "EXP-1" \
    --etiqueta "v3_001800-002099" \
    --modelos "udvd,p2n,mm_bsn,nb2nb,combo_p2n_f2r" \
    > "$LOG_DIR/exp1_v3_metricas_$SLURM_JOB_ID.log" 2>&1

# ═════════════════════════════════════════════════════════════════════════════
# BLOQUE 2: Video 1 — frame_19500 a frame_19756 (257 frames)
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> [BLOQUE 2/2] Video 1: frame_19500 a frame_19756 <<<"
DIR_V1="$RAIZ/videos/video1/expos_regimenes/EXP-1_v1_019500-019756"
INPUT_V1="$DIR_V1/input"
mkdir -p "$INPUT_V1"

python "$SCRIPTS/preparar_bloque_entrada.py" \
    --video video1 \
    --inicio-frame frame_19500.png \
    --n-frames 257 \
    --carpeta-salida "$INPUT_V1" \
    --filtrar-menu-central 2>&1 | tee "$LOG_DIR/exp1_v1_prep_$SLURM_JOB_ID.log"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo udvd \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/udvd/denoised" \
    --carpeta-cache "$DIR_V1/udvd/cache" \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp1_v1_udvd_$SLURM_JOB_ID.log" 2>&1 &
PID_V1_UDVD=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo p2n \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/p2n/denoised" \
    --carpeta-cache "$DIR_V1/p2n/cache" \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp1_v1_p2n_$SLURM_JOB_ID.log" 2>&1 &
PID_V1_P2N=$!

wait $PID_V1_UDVD $PID_V1_P2N

CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo mm_bsn \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/mm_bsn/denoised" \
    --carpeta-cache "$DIR_V1/mm_bsn/cache" \
    --stride-submuestreo 2 \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp1_v1_mm_bsn_$SLURM_JOB_ID.log" 2>&1 &
PID_V1_MMBSN=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo nb2nb \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/nb2nb/denoised" \
    --carpeta-cache "$DIR_V1/nb2nb/cache" \
    --epocas 10 --batch-size 8 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp1_v1_nb2nb_$SLURM_JOB_ID.log" 2>&1 &
PID_V1_NB2NB=$!

wait $PID_V1_MMBSN $PID_V1_NB2NB

# Combo P2N -> F2R en V1
mkdir -p "$DIR_V1/combo_p2n_f2r/denoised"
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo f2r \
    --carpeta-entrada "$DIR_V1/p2n/denoised" \
    --carpeta-salida "$DIR_V1/combo_p2n_f2r/denoised" \
    --carpeta-cache "$DIR_V1/combo_p2n_f2r/cache" \
    --epocas 8 --batch-size 6 --patch-size 128 --paciencia 3 --gpu 0 \
    > "$LOG_DIR/exp1_v1_combo_p2n_f2r_$SLURM_JOB_ID.log" 2>&1

# Métricas Bloque V1
python "$SCRIPTS/calcular_metricas_experimento.py" \
    --dir-input "$INPUT_V1" \
    --dir-exp "$DIR_V1" \
    --exp-id "EXP-1" \
    --etiqueta "v1_019500-019756" \
    --modelos "udvd,p2n,mm_bsn,nb2nb,combo_p2n_f2r" \
    > "$LOG_DIR/exp1_v1_metricas_$SLURM_JOB_ID.log" 2>&1

# ═════════════════════════════════════════════════════════════════════════════
# CONSOLIDACIÓN Y ELECCIÓN DEL GANADOR
# ═════════════════════════════════════════════════════════════════════════════
python "$SCRIPTS/consolidar_resultados_exp.py" \
    --exp-id "EXP-1" \
    --dirs "$DIR_V3" "$DIR_V1" \
    --salida "$RAIZ/videos/video3/expos_regimenes/resultados_EXP-1.xlsx"

echo ""
echo "=== EXP-1 COMPLETADO EXITOSAMENTE ==="
date
