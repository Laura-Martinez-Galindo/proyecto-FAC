#!/usr/bin/env bash
#SBATCH --job-name=ruido_regimenes
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/ruido_regimenes_%j.out
#SBATCH --error=jobs/logs/ruido_regimenes_%j.err

# Caracteriza el ruido de cada regimen en cada video (scripts/caracterizar_ruido.py) para decidir
# con datos que regimenes se agrupan. Usa videos/<video>/regimenes_<video>.csv: correrlo DESPUES
# de jobs/detectar_regimenes.sbatch (con las plantillas nuevas).
# Resultado: resultados/ruido_regimenes/  y  resultados/ruido_regimenes.tar para traer al PC.
# Variables: POR_GRUPO (60 frames por video|regimen)
#
# Uso:  bash jobs/enviar.sh -c 8 --mem=12G -t 01:00:00 jobs/caracterizar_ruido.sbatch
#   o, para que espere a la lectura de regimenes:
#       bash jobs/enviar.sh -c 8 --mem=12G -t 01:00:00 --dependency=afterok:<JOBID> jobs/caracterizar_ruido.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate

python scripts/caracterizar_ruido.py --por-grupo "${POR_GRUPO:-60}" --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --salida resultados/ruido_regimenes

# todo lo necesario para revisarlo en el PC, en un solo archivo
tar -cf resultados/ruido_regimenes.tar -C resultados ruido_regimenes \
    -C .. $(ls videos/*/regimenes_video*.csv videos/*/regimenes_video*_intervalos.csv \
               videos/*/regimenes_video*_resumen.json videos/*/regimenes_video*_tramos.jpg 2>/dev/null)
echo "Para revisar en el PC: resultados/ruido_regimenes.tar"
