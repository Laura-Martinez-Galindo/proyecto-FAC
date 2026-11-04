#!/usr/bin/env bash
#SBATCH --job-name=contar_v3
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/contar_v3_%j.out
#SBATCH --error=jobs/logs/contar_v3_%j.err

# Cuenta los frames de video3 por carpeta y verifica que las vueltas 2-8 son copia de la 1.
# Uso:
#   sbatch jobs/contar_frames_video3.sbatch                       # solo cuenta (no toca nada)
#   MOVER_A=cache/video3_vueltas_2a8 sbatch jobs/contar_frames_video3.sbatch
#       -> mueve los frames > 23.315 de TODAS las carpetas de videos/video3 a esa carpeta
#   CARPETAS="frames_originales mascaras_hud" MOVER_A=... sbatch ...   # solo esas carpetas

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
PY="fac_env311/bin/python"; [ -x "$PY" ] || PY=python3

ARGS=()
if [ -n "${MOVER_A:-}" ]; then
  ARGS+=(--limpiar --mover-a "${MOVER_A}")
  if [ -n "${CARPETAS:-}" ]; then ARGS+=(--carpetas ${CARPETAS}); fi
fi
"$PY" scripts/contar_frames_video3.py "${ARGS[@]}"
