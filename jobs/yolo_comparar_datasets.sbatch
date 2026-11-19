#!/usr/bin/env bash
#SBATCH --job-name=yolo_datasets
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co
#SBATCH --output=jobs/logs/yolo_datasets_%j.out
#SBATCH --error=jobs/logs/yolo_datasets_%j.err

# Mismo YOLO del articulo IET (yolo11m, configuraciones heavy_aug y warm_restart) entrenado y
# probado con el MISMO split sobre original, sin_hud y denoised (scripts/yolo_comparar_datasets.py).
# Antes, una vez en el login (los nodos de GPU no tienen internet):
#   source fac_env311/bin/activate && python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
# Uso:  SPLIT=config/splits/<split>.csv sbatch jobs/yolo_comparar_datasets.sbatch
#   (CSV con columnas imagen,split usando los nombres de dataset_original; el split sin fuga de Jorge)
# Opcional: CONFIGS="heavy_aug" DATASETS="original sin_hud denoised" EPOCAS=150 PACIENCIA=20

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
: "${SPLIT:?falta SPLIT=ruta/al/split.csv}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/yolo_comparar_datasets.py --split "${SPLIT}" \
    --configs ${CONFIGS:-heavy_aug warm_restart} --datasets ${DATASETS:-original sin_hud denoised} \
    --epocas "${EPOCAS:-150}" --paciencia "${PACIENCIA:-20}" --device 0,1
S="resultados/yolo_datasets/$(basename "${SPLIT}" .csv)"
tar -cf "${S}.tar" --exclude='*.pt' "${S}" && ls -lh "${S}.tar"
