#!/usr/bin/env bash
#SBATCH --job-name=cap_preparar
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co
#SBATCH --output=jobs/logs/cap_preparar_%j.out
#SBATCH --error=jobs/logs/cap_preparar_%j.err

# Sesion desatendida para jobs/captioning_preparar.sh (descarga del modelo de lenguaje visual), porque
# en el login la descarga se muere por memoria.   sbatch jobs/captioning_preparar.sbatch
# Si el nodo no tiene internet, falla en segundos con un mensaje claro (no gasta las 6 horas).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
echo "Nodo: $(hostname)"
if ! timeout 20 curl -sSI https://huggingface.co >/dev/null 2>&1; then
  echo "!! Este nodo NO tiene internet: no se puede descargar el modelo desde aqui."
  exit 2
fi
bash jobs/captioning_preparar.sh
