#!/usr/bin/env bash

#SBATCH --job-name=sync_v3_orig
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=jobs/logs/subir_v3_orig_%j.out
#SBATCH --error=jobs/logs/subir_v3_orig_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail
trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs videos/comprimidos

echo "=============================================================================="
echo "🚀 SUBIENDO FRAMES ORIGINALES DE VIDEO 3 A GOOGLE DRIVE (SLURM HPC)"
echo "   Nodo: $(hostname)"
echo "   CPUs: 16"
echo "   Inicio: $(date --iso-8601=seconds)"
echo "=============================================================================="

tar -cf - -C "videos/video3" "frames_originales" | rclone rcat "gdrive:proyecto-FAC/videos/video3/comprimidos/video3_frames_originales.tar" --drive-chunk-size 256M --transfers 16 --progress

echo
echo "=============================================================================="
echo "🎉 ¡SUBIDA DE VIDEO 3 FRAMES ORIGINALES COMPLETADA! $(date --iso-8601=seconds)"
echo "=============================================================================="
