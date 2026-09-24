#!/bin/bash
#SBATCH --job-name=mosaicos_ruido_v2
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/mosaicos_ruido_v2_%j.out
#SBATCH --error=jobs/logs/mosaicos_ruido_v2_%j.err

set -e
mkdir -p jobs/logs figuras_tesis/mosaicos_ruido_video2

echo "=========================================================="
echo "Extraccion Turbo de Ruido y Mosaicos 2x2 - Video 2"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

source fac_env311/bin/activate

python scripts/extraer_ruidos_y_mosaicos_video2_turbo.py --cores 16

echo "=========================================================="
echo "Extraccion y Mosaicos completados con exito."
echo "Fecha: $(date)"
echo "=========================================================="
