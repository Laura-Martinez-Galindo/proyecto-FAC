#!/bin/bash
#SBATCH --job-name=caracterizar_v1
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/caracterizar_v1_%j.out
#SBATCH --error=logs/slurm/caracterizar_v1_%j.err

# ==============================================================================
# CARACTERIZACIÓN Y CONTEO DE REGÍMENES FLIR PARA VIDEO 1
# Partición short (CPU - Arranque inmediato sin colas de GPU)
# ==============================================================================

set -euo pipefail
date
echo "=== Caracterización de Regímenes de Video 1 iniciada en $(hostname) ==="

source /home/lf.martinezg1/miniconda3/bin/activate fac_env311

RAIZ="/home/lf.martinezg1/proyecto-FAC"
SCRIPTS="$RAIZ/scripts"
LOG_DIR="$RAIZ/logs/slurm"
mkdir -p "$LOG_DIR"

FRAMES_V1="$RAIZ/videos/video1/frames_originales"
SALIDA_CSV="$RAIZ/videos/video1/conteo_regimenes_video1.csv"

echo "Carpeta de frames : $FRAMES_V1"
echo "Archivo de salida : $SALIDA_CSV"

python "$SCRIPTS/caracterizar_regimenes_video.py" \
    --carpeta "$FRAMES_V1" \
    --salida "$SALIDA_CSV"

echo "=== Caracterización finalizada con éxito ==="
date
