#!/usr/bin/env bash
#SBATCH --job-name=visor
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co
#SBATCH --output=jobs/logs/visor_%j.out
#SBATCH --error=jobs/logs/visor_%j.err

# Visor HTML de resultados (limpieza por regimen, deteccion YOLO, captioning). Solo lee imagenes y
# las convierte a JPG: no usa GPU (no haria nada mas rapido). Resultado: resultados/visor.tar
# Uso:  bash jobs/enviar.sh jobs/generar_visor.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
python scripts/generar_visor.py --n-limpieza "${N:-12}" --workers "${SLURM_CPUS_PER_TASK:-8}"
tar -cf resultados/visor.tar -C resultados visor
du -sh resultados/visor resultados/visor.tar
echo "Para traer al PC: resultados/visor.tar  (descomprimir y abrir visor/index.html)"
