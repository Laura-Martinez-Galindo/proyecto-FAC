#!/bin/bash
#SBATCH --job-name=EXP2_regimen2_HDEO_HAZE
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/exp2_regimen2_%j.out
#SBATCH --error=logs/slurm/exp2_regimen2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@unal.edu.co

# ==============================================================================
# EXP-2 — Régimen 2: HDEO + DDE + HAZE (Bruma estirada, grumos, sigma ~30-95 DN)
# Bloques:
#   1. Video2 (frame_77550 al frame_77849 - 300 frames, tramo crítico con HAZE)
#   2. Video1 (frame_14900 al frame_15199 - 300 frames, bruma amazónica)
#
# Modelos y Ablaciones:
#   1. UDVD (Baseline)
#   2. AP-BSN (Submuestreo asimétrico s=3 + refinamiento R3)
#   3. MM-BSN (Máscaras multi-geometría contra grumos del ISP)
#   4. F2R (Desacoplamiento residual temporal)
#   5. Combo 1 (Espacial -> Temporal): MM-BSN -> F2R
#   6. Combo 2 (Temporal -> Espacial): F2R -> MM-BSN [Ablación de orden]
#   7. Ablación de Función de Pérdida: MM-BSN con L1 vs L2 (colas pesadas)
#
# Distribución paralela en 2 GPUs:
#   GPU 0: UDVD, MM-BSN (L1), Combo 1 (MM-BSN -> F2R)
#   GPU 1: AP-BSN, F2R, Combo 2 (F2R -> MM-BSN), MM-BSN (L2)
# ==============================================================================

set -euo pipefail
date
echo "=== EXP-2 iniciado en $(hostname) ==="

source /home/lf.martinezg1/miniconda3/bin/activate fac_env311

RAIZ="/home/lf.martinezg1/proyecto-FAC"
SCRIPTS="$RAIZ/scripts"
LOG_DIR="$RAIZ/logs/slurm"
mkdir -p "$LOG_DIR"

# ═════════════════════════════════════════════════════════════════════════════
# BLOQUE 1: Video 2 — frame_77550 a frame_77849 (300 frames)
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> [BLOQUE 1/2] Video 2: frame_77550 a frame_77849 (HAZE crítico) <<<"
DIR_V2="$RAIZ/videos/video2/expos_regimenes/EXP-2_v2_077550-077849"
INPUT_V2="$DIR_V2/input"
mkdir -p "$INPUT_V2"

python "$SCRIPTS/preparar_bloque_entrada.py" \
    --video video2 \
    --inicio-frame frame_77550.png \
    --n-frames 300 \
    --carpeta-salida "$INPUT_V2" \
    --filtrar-menu-central 2>&1 | tee "$LOG_DIR/exp2_v2_prep_$SLURM_JOB_ID.log"

# Paralelo 1: UDVD (GPU 0) y AP-BSN con R3 (GPU 1)
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo udvd \
    --carpeta-entrada "$INPUT_V2" \
    --carpeta-salida "$DIR_V2/udvd/denoised" \
    --carpeta-cache "$DIR_V2/udvd/cache" \
    --epocas 12 --batch-size 4 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp2_v2_udvd_$SLURM_JOB_ID.log" 2>&1 &
PID_UDVD=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo ap_bsn \
    --carpeta-entrada "$INPUT_V2" \
    --carpeta-salida "$DIR_V2/ap_bsn/denoised" \
    --carpeta-cache "$DIR_V2/ap_bsn/cache" \
    --stride-submuestreo 3 \
    --post-r3 \
    --epocas 12 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp2_v2_ap_bsn_$SLURM_JOB_ID.log" 2>&1 &
PID_APBSN=$!

wait $PID_UDVD $PID_APBSN
echo "[✓] V2: UDVD y AP-BSN finalizados"

# Paralelo 2: MM-BSN (GPU 0) y F2R (GPU 1)
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo mm_bsn \
    --carpeta-entrada "$INPUT_V2" \
    --carpeta-salida "$DIR_V2/mm_bsn/denoised" \
    --carpeta-cache "$DIR_V2/mm_bsn/cache" \
    --loss-type l1 \
    --epocas 12 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp2_v2_mm_bsn_$SLURM_JOB_ID.log" 2>&1 &
PID_MMBSN=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo f2r \
    --carpeta-entrada "$INPUT_V2" \
    --carpeta-salida "$DIR_V2/f2r/denoised" \
    --carpeta-cache "$DIR_V2/f2r/cache" \
    --epocas 12 --batch-size 4 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp2_v2_f2r_$SLURM_JOB_ID.log" 2>&1 &
PID_F2R=$!

wait $PID_MMBSN $PID_F2R
echo "[✓] V2: MM-BSN y F2R finalizados"

# Paralelo 3: ABLACIÓN DE ORDEN (Combo 1 en GPU 0 vs Combo 2 en GPU 1)
echo "[...] V2: Ejecutando Ablación de Orden Espacial->Temporal vs Temporal->Espacial"
mkdir -p "$DIR_V2/combo_mmbsn_f2r/denoised" "$DIR_V2/combo_f2r_mmbsn/denoised"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo f2r \
    --carpeta-entrada "$DIR_V2/mm_bsn/denoised" \
    --carpeta-salida "$DIR_V2/combo_mmbsn_f2r/denoised" \
    --carpeta-cache "$DIR_V2/combo_mmbsn_f2r/cache" \
    --epocas 8 --batch-size 4 --patch-size 128 --paciencia 3 --gpu 0 \
    > "$LOG_DIR/exp2_v2_combo1_$SLURM_JOB_ID.log" 2>&1 &
PID_C1=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo mm_bsn \
    --carpeta-entrada "$DIR_V2/f2r/denoised" \
    --carpeta-salida "$DIR_V2/combo_f2r_mmbsn/denoised" \
    --carpeta-cache "$DIR_V2/combo_f2r_mmbsn/cache" \
    --epocas 8 --batch-size 6 --patch-size 128 --paciencia 3 --gpu 1 \
    > "$LOG_DIR/exp2_v2_combo2_$SLURM_JOB_ID.log" 2>&1 &
PID_C2=$!

wait $PID_C1 $PID_C2
echo "[✓] V2: Combos de ablación de orden finalizados"

# Paralelo 4: ABLACIÓN L1 vs L2 en MM-BSN (GPU 1)
CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo mm_bsn \
    --carpeta-entrada "$INPUT_V2" \
    --carpeta-salida "$DIR_V2/ablation_mmbsn_l2/denoised" \
    --carpeta-cache "$DIR_V2/ablation_mmbsn_l2/cache" \
    --loss-type l2 \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp2_v2_ablation_l2_$SLURM_JOB_ID.log" 2>&1

# Métricas Bloque V2
python "$SCRIPTS/calcular_metricas_experimento.py" \
    --dir-input "$INPUT_V2" \
    --dir-exp "$DIR_V2" \
    --exp-id "EXP-2" \
    --etiqueta "v2_077550-077849" \
    --modelos "udvd,ap_bsn,mm_bsn,f2r,combo_mmbsn_f2r,combo_f2r_mmbsn,ablation_mmbsn_l2" \
    > "$LOG_DIR/exp2_v2_metricas_$SLURM_JOB_ID.log" 2>&1

# ═════════════════════════════════════════════════════════════════════════════
# BLOQUE 2: Video 1 — frame_14900 a frame_15199 (300 frames)
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> [BLOQUE 2/2] Video 1: frame_14900 a frame_15199 <<<"
DIR_V1="$RAIZ/videos/video1/expos_regimenes/EXP-2_v1_014900-015199"
INPUT_V1="$DIR_V1/input"
mkdir -p "$INPUT_V1"

python "$SCRIPTS/preparar_bloque_entrada.py" \
    --video video1 \
    --inicio-frame frame_14900.png \
    --n-frames 300 \
    --carpeta-salida "$INPUT_V1" \
    --filtrar-menu-central 2>&1 | tee "$LOG_DIR/exp2_v1_prep_$SLURM_JOB_ID.log"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo udvd \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/udvd/denoised" \
    --carpeta-cache "$DIR_V1/udvd/cache" \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp2_v1_udvd_$SLURM_JOB_ID.log" 2>&1 &
PID_V1_UDVD=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo ap_bsn \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/ap_bsn/denoised" \
    --carpeta-cache "$DIR_V1/ap_bsn/cache" \
    --stride-submuestreo 3 --post-r3 \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp2_v1_apbsn_$SLURM_JOB_ID.log" 2>&1 &
PID_V1_APBSN=$!

wait $PID_V1_UDVD $PID_V1_APBSN

CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo mm_bsn \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/mm_bsn/denoised" \
    --carpeta-cache "$DIR_V1/mm_bsn/cache" \
    --loss-type l1 \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp2_v1_mmbsn_$SLURM_JOB_ID.log" 2>&1 &
PID_V1_MMBSN=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo f2r \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/f2r/denoised" \
    --carpeta-cache "$DIR_V1/f2r/cache" \
    --epocas 10 --batch-size 6 --patch-size 128 --paciencia 4 --gpu 1 \
    > "$LOG_DIR/exp2_v1_f2r_$SLURM_JOB_ID.log" 2>&1 &
PID_V1_F2R=$!

wait $PID_V1_MMBSN $PID_V1_F2R

# Combo 1 y Combo 2 en V1
mkdir -p "$DIR_V1/combo_mmbsn_f2r/denoised" "$DIR_V1/combo_f2r_mmbsn/denoised"
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo f2r \
    --carpeta-entrada "$DIR_V1/mm_bsn/denoised" \
    --carpeta-salida "$DIR_V1/combo_mmbsn_f2r/denoised" \
    --carpeta-cache "$DIR_V1/combo_mmbsn_f2r/cache" \
    --epocas 8 --batch-size 6 --patch-size 128 --paciencia 3 --gpu 0 \
    > "$LOG_DIR/exp2_v1_combo1_$SLURM_JOB_ID.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo mm_bsn \
    --carpeta-entrada "$DIR_V1/f2r/denoised" \
    --carpeta-salida "$DIR_V1/combo_f2r_mmbsn/denoised" \
    --carpeta-cache "$DIR_V1/combo_f2r_mmbsn/cache" \
    --epocas 8 --batch-size 6 --patch-size 128 --paciencia 3 --gpu 1 \
    > "$LOG_DIR/exp2_v1_combo2_$SLURM_JOB_ID.log" 2>&1 &
wait

# Métricas Bloque V1
python "$SCRIPTS/calcular_metricas_experimento.py" \
    --dir-input "$INPUT_V1" \
    --dir-exp "$DIR_V1" \
    --exp-id "EXP-2" \
    --etiqueta "v1_014900-015199" \
    --modelos "udvd,ap_bsn,mm_bsn,f2r,combo_mmbsn_f2r,combo_f2r_mmbsn" \
    > "$LOG_DIR/exp2_v1_metricas_$SLURM_JOB_ID.log" 2>&1

# ═════════════════════════════════════════════════════════════════════════════
# CONSOLIDACIÓN Y ELECCIÓN DEL GANADOR
# ═════════════════════════════════════════════════════════════════════════════
python "$SCRIPTS/consolidar_resultados_exp.py" \
    --exp-id "EXP-2" \
    --dirs "$DIR_V2" "$DIR_V1" \
    --salida "$RAIZ/videos/video2/expos_regimenes/resultados_EXP-2.xlsx"

echo ""
echo "=== EXP-2 COMPLETADO EXITOSAMENTE ==="
date
