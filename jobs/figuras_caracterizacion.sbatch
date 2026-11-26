#!/usr/bin/env bash
#SBATCH --job-name=figs_caract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co
#SBATCH --output=jobs/logs/figs_caract_%j.out
#SBATCH --error=jobs/logs/figs_caract_%j.err

# Figuras de caracterizacion de los 3 videos (intensidades, forma del ruido, sigma, sal y pimienta,
# NIQE y BRISQUE) para la presentacion. Resultado: resultados/figuras_caracterizacion.tar
# Uso:  bash jobs/enviar.sh --gres=gpu:1 jobs/figuras_caracterizacion.sbatch   (GPU para NIQE/BRISQUE; lo demas en 8 CPU)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
python scripts/figuras_caracterizacion.py --por-video "${POR_VIDEO:-120}" --workers "${SLURM_CPUS_PER_TASK:-8}"
tar -cf resultados/figuras_caracterizacion.tar -C resultados figuras_caracterizacion
echo "Para traer al PC: resultados/figuras_caracterizacion.tar"
