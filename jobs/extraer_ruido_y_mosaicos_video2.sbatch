#!/usr/bin/env bash
#SBATCH --job-name=ruido_mosaicos_v2
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=jobs/logs/ruido_mosaicos_v2_%j.out
#SBATCH --error=jobs/logs/ruido_mosaicos_v2_%j.err

set -euo pipefail
mkdir -p jobs/logs figuras_tesis/linea_tiempo/mosaicos_ruido_video2 videos/video2/ruido_residual

echo "=========================================================="
echo "Extracción de Ruido, Mosaicos y Línea de Tiempo - Video 2"
echo "Nodo: $(hostname) | Fecha: $(date)"
echo "=========================================================="

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source fac_env311/bin/activate

# Ejecuta la extracción de ruido residual, los 10 mosaicos trípticos y la línea de tiempo frame-a-frame
python scripts/extraer_ruido_y_mosaicos.py \
    --video video2 \
    --carpeta-original videos/video2/frames_sin_hud \
    --carpeta-denoised videos/video2/expos/ensemble_wavelet_triple_sinhud \
    --modelo-nombre "Ensemble_Wavelet_Triple" \
    --salida-ruido videos/video2/ruido_residual \
    --salida-mosaicos figuras_tesis/linea_tiempo/mosaicos_ruido_video2 \
    --salida-grafica figuras_tesis/linea_tiempo/linea_tiempo_ruido_video2.png \
    --salida-csv videos/video2/ritmo_temporal_video2.csv

echo "=========================================================="
echo "Extracción completada exitosamente: $(date)"
echo "=========================================================="
