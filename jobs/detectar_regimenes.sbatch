#!/usr/bin/env bash
#SBATCH --job-name=regimenes_hud
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/regimenes_hud_%j.out
#SBATCH --error=jobs/logs/regimenes_hud_%j.err

# PASO 1 del plan de denoising: regimen (sensor + filtros) de cada frame, leido del HUD.
#   1) Videos completos (1 de cada PASO frames) -> videos/<video>/regimenes_<video>.csv
#      + _intervalos.csv + _resumen.json + carpeta _dudosos con los recortes ilegibles
#   2) Dataset -> datasets/dataset_original/regimenes_dataset.csv (los ilegibles toman el
#      regimen del frame legible mas cercano del video)
# Variables: PASO (15 = 2 lecturas por segundo)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
PASO="${PASO:-15}"
W="${SLURM_CPUS_PER_TASK:-8}"

for V in video1 video2 video3; do
  MAX=0; [ "$V" = "video3" ] && MAX=23315     # video3: solo la primera vuelta del clip
  python scripts/detectar_regimen_hud.py carpeta --carpeta "videos/$V/frames_originales" --video "$V" \
      --paso "$PASO" --maximo-frame "$MAX" --workers "$W" --salida "videos/$V/regimenes_$V.csv"
done

python scripts/detectar_regimen_hud.py dataset --dataset datasets/dataset_original --workers "$W" \
    --tabla-videos videos/video1/regimenes_video1.csv videos/video2/regimenes_video2.csv videos/video3/regimenes_video3.csv \
    --salida datasets/dataset_original/regimenes_dataset.csv
