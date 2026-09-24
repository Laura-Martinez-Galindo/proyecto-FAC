#!/bin/bash
#SBATCH --job-name=ruido_timeline_3videos
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=jobs/logs/ruido_timeline_3videos_%j.out
#SBATCH --error=jobs/logs/ruido_timeline_3videos_%j.err

set -e
mkdir -p jobs/logs figuras_tesis/linea_tiempo

echo "=========================================================="
echo "Extraccion de Mapas de Ruido y Linea de Tiempo (3 Videos)"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "=========================================================="

source fac_env311/bin/activate

python scripts/ejecutar_extraccion_ruido_3videos.py --workers 16

echo "=========================================================="
echo "Extraccion completada exitosamente."
echo "Fecha: $(date)"
echo "=========================================================="
