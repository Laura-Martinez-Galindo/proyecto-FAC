#!/bin/bash
#SBATCH --job-name=EXP3_regimen4_HDIR_FPN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=logs/slurm/exp3_regimen4_%j.out
#SBATCH --error=logs/slurm/exp3_regimen4_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@unal.edu.co

# ==============================================================================
# EXP-3 — Régimen 4: Infrarrojo Térmico Puro (HDIR C WH + EXP AUT + FPN Striping)
# Bloque: Video1 tramo térmico final (frame_19760 en adelante, ΔRGB = 0.0)
#
# Modelos:
#   1. udvd_sin_homografia  : Baseline UDVD directo (deja siluetas por movimiento a 180 KTS)
#   2. udvd_con_homografia  : UDVD con alineación homográfica RANSAC previa
#   3. struct_n2v_lineal    : StructN2V con máscara vertical 9x1 (destruye striping columnar)
#   4. struct_n2v_isotrop   : N2V2 con máscara puntual 1x1 [Ablación C2]
#   5. f2r                  : Frames2Residual puro
#   6. combo_termico_optimo : StructN2V (9x1) -> Homografía RANSAC -> F2R
#
# Distribución paralela en 2 GPUs:
#   GPU 0: udvd_sin_homografia, struct_n2v_lineal, combo_termico_optimo
#   GPU 1: udvd_con_homografia, struct_n2v_isotrop, f2r
# ==============================================================================

set -euo pipefail
date
echo "=== EXP-3 iniciado en $(hostname) ==="

source /home/lf.martinezg1/miniconda3/bin/activate fac_env311

RAIZ="/home/lf.martinezg1/proyecto-FAC"
SCRIPTS="$RAIZ/scripts"
LOG_DIR="$RAIZ/logs/slurm"
mkdir -p "$LOG_DIR"

DIR_V1="$RAIZ/videos/video1/expos_regimenes/EXP-3_v1_termico_hdir"
INPUT_V1="$DIR_V1/input"
mkdir -p "$INPUT_V1"

# 1. Preparar bloque térmico (exigir_termico valida que sea infrarrojo puro InSb)
python "$SCRIPTS/preparar_bloque_entrada.py" \
    --video video1 \
    --inicio-frame frame_19760.png \
    --n-frames 50 \
    --exigir-termico \
    --carpeta-salida "$INPUT_V1" 2>&1 | tee "$LOG_DIR/exp3_prep_$SLURM_JOB_ID.log"

N_FRAMES=$(ls "$INPUT_V1"/*.png 2>/dev/null | wc -l)
echo "Total frames térmicos disponibles: $N_FRAMES"

# Paralelo 1: UDVD sin homografía (GPU 0) vs UDVD con homografía RANSAC (GPU 1)
# Demuestra matemáticamente el efecto de alinear los 180 KTS antes del denoiser
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo udvd \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/udvd_sin_homografia/denoised" \
    --carpeta-cache "$DIR_V1/udvd_sin_homografia/cache" \
    --epocas 15 --batch-size 4 --patch-size 128 --paciencia 5 --gpu 0 \
    > "$LOG_DIR/exp3_udvd_sin_homo_$SLURM_JOB_ID.log" 2>&1 &
PID_U0=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo udvd \
    --alinear-homografia \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/udvd_con_homografia/denoised" \
    --carpeta-cache "$DIR_V1/udvd_con_homografia/cache" \
    --epocas 15 --batch-size 4 --patch-size 128 --paciencia 5 --gpu 1 \
    > "$LOG_DIR/exp3_udvd_con_homo_$SLURM_JOB_ID.log" 2>&1 &
PID_U1=$!

wait $PID_U0 $PID_U1
echo "[✓] UDVD (sin y con homografía) finalizados"

# Paralelo 2: StructN2V 9x1 vertical (GPU 0) vs N2V2 1x1 isotrópica (GPU 1)
# Demuestra cómo la máscara 9x1 destruye el striping FPN columnar del detector InSb
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo struct_n2v \
    --struct-mask-tipo lineal \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/struct_n2v_lineal_9x1/denoised" \
    --carpeta-cache "$DIR_V1/struct_n2v_lineal_9x1/cache" \
    --epocas 15 --batch-size 4 --patch-size 128 --paciencia 5 --gpu 0 \
    > "$LOG_DIR/exp3_n2v_lineal_$SLURM_JOB_ID.log" 2>&1 &
PID_N0=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo struct_n2v \
    --struct-mask-tipo isotropica \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/struct_n2v_isotropica_1x1/denoised" \
    --carpeta-cache "$DIR_V1/struct_n2v_isotropica_1x1/cache" \
    --epocas 15 --batch-size 4 --patch-size 128 --paciencia 5 --gpu 1 \
    > "$LOG_DIR/exp3_n2v_iso_$SLURM_JOB_ID.log" 2>&1 &
PID_N1=$!

wait $PID_N0 $PID_N1
echo "[✓] StructN2V (lineal 9x1 vs isotrópica 1x1) finalizados"

# Paralelo 3: F2R puro (GPU 1)
CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo f2r \
    --carpeta-entrada "$INPUT_V1" \
    --carpeta-salida "$DIR_V1/f2r/denoised" \
    --carpeta-cache "$DIR_V1/f2r/cache" \
    --epocas 15 --batch-size 4 --patch-size 128 --paciencia 5 --gpu 1 \
    > "$LOG_DIR/exp3_f2r_$SLURM_JOB_ID.log" 2>&1 &
PID_F2R=$!

# Paralelo 3 (GPU 0): COMBO TÉRMICO ÓPTIMO
# StructN2V (elimina FPN columnar) -> Homografía RANSAC -> F2R (elimina grano cuántico EXP AUT)
echo "[...] Ejecutando Combo Térmico Óptimo: StructN2V -> RANSAC -> F2R"
mkdir -p "$DIR_V1/combo_struct_ransac_f2r/denoised"
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/correr_modelo_experimento.py" \
    --modelo f2r \
    --alinear-homografia \
    --carpeta-entrada "$DIR_V1/struct_n2v_lineal_9x1/denoised" \
    --carpeta-salida "$DIR_V1/combo_struct_ransac_f2r/denoised" \
    --carpeta-cache "$DIR_V1/combo_struct_ransac_f2r/cache" \
    --epocas 12 --batch-size 4 --patch-size 128 --paciencia 4 --gpu 0 \
    > "$LOG_DIR/exp3_combo_optimo_$SLURM_JOB_ID.log" 2>&1 &
PID_COMBO=$!

wait $PID_F2R $PID_COMBO
echo "[✓] F2R y Combo Térmico Óptimo finalizados"

# ═════════════════════════════════════════════════════════════════════════════
# MÉTRICAS Y ELECCIÓN DEL GANADOR
# ═════════════════════════════════════════════════════════════════════════════
python "$SCRIPTS/calcular_metricas_experimento.py" \
    --dir-input "$INPUT_V1" \
    --dir-exp "$DIR_V1" \
    --exp-id "EXP-3" \
    --etiqueta "v1_termico_hdir" \
    --modelos "udvd_sin_homografia,udvd_con_homografia,struct_n2v_lineal_9x1,struct_n2v_isotropica_1x1,f2r,combo_struct_ransac_f2r" \
    > "$LOG_DIR/exp3_metricas_$SLURM_JOB_ID.log" 2>&1

python "$SCRIPTS/consolidar_resultados_exp.py" \
    --exp-id "EXP-3" \
    --dirs "$DIR_V1" \
    --salida "$RAIZ/videos/video1/expos_regimenes/resultados_EXP-3_termico.xlsx"

echo ""
echo "=== EXP-3 COMPLETADO EXITOSAMENTE ==="
date
