#!/usr/bin/env bash
#SBATCH --job-name=udvd_oficial_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=jobs/logs/udvd_oficial_v2_%j.out
#SBATCH --error=jobs/logs/udvd_oficial_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

# Prueba del UDVD OFICIAL (Sheth et al., ICCV 2021) en el video 2:
#   1) entrena sobre los frames sin HUD del tramo 33:00-46:10 (frames 59.300-83.011)
#   2) limpia las imagenes del dataset que caen en ese tramo y guarda el residuo
#      (ruidoso - limpio). Si el residuo muestra bordes/casas, el modelo esta
#      quitando senal; si se ve como grano sin forma, esta quitando ruido.
# Variables opcionales: PASOS (6000), BATCH (16), PARCHE (128), CARPETA (frames_sin_hud)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true
nvidia-smi || true

CARPETA="${CARPETA:-videos/video2/frames_sin_hud}"
SAL="cache/udvd_oficial/video2_ult13min"

python scripts/udvd_oficial.py entrenar \
    --carpeta "${CARPETA}" --video video2 --rango 59300:83011 \
    --salida "${SAL}" --pasos "${PASOS:-6000}" --batch "${BATCH:-16}" --parche "${PARCHE:-128}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" --amp

python scripts/udvd_oficial.py inferir \
    --checkpoint "${SAL}/modelo.pt" --carpeta "${CARPETA}" \
    --lista-csv datasets/dataset_original/correspondencia.csv --video video2 \
    --salida "${SAL}/salidas" --residuo --amp

echo "Listo: ${SAL}/salidas (limpias) y ${SAL}/salidas/residuo (ruido estimado x4)"
