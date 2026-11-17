#!/usr/bin/env bash
#SBATCH --job-name=captioning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co
#SBATCH --output=jobs/logs/captioning_%j.out
#SBATCH --error=jobs/logs/captioning_%j.err

# Dense video captioning por ventanas (scripts/captioning_ventanas.py) sobre un tramo de un video.
# Antes, una vez en login:  bash jobs/captioning_preparar.sh
# Variables (todas opcionales):
#   VIDEO (video2)  VERSIONES ("sin_hud")  MODELO (Qwen/Qwen2.5-VL-7B-Instruct)
#   PROMPTS ("abierto guiado lista")  INICIO (1980)  FIN (2770)   -> min 33 a 46 del video2, donde
#   estan 753 de las imagenes anotadas, para poder contrastar con las etiquetas
#   VENTANA (24 s)  FRAMES (8 por ventana)  LADO (640 px)
#   HUMO=1: 3 ventanas con el prompt "lista", para ver en minutos que el modelo carga y responde
# Resultado: resultados/captioning/<video>/<modelo>/<version>_<prompt>/ y todo junto en
#   resultados/captioning.tar (solo texto: pesa poco)

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # los nodos de GPU no tienen internet
VIDEO="${VIDEO:-video2}"
MODELO="${MODELO:-Qwen/Qwen2.5-VL-7B-Instruct}"
PROMPTS="${PROMPTS:-abierto guiado lista}"
VERSIONES="${VERSIONES:-sin_hud}"
EXTRA=()
BASE="resultados/captioning"
if [ "${HUMO:-0}" = "1" ]; then
  PROMPTS="lista"; EXTRA+=(--max-ventanas 3); BASE="resultados/captioning_humo"
fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
FALLOS=0
for V in ${VERSIONES}; do
  for P in ${PROMPTS}; do
    SAL="${BASE}/${VIDEO}/$(basename "${MODELO}")/${V}_${P}"
    echo "################ ${VIDEO} ${V} ${P}  $(date '+%F %T') ################"
    python scripts/captioning_ventanas.py --video "${VIDEO}" --version "${V}" --modelo "${MODELO}" \
        --prompt "${P}" --inicio-s "${INICIO:-1980}" --fin-s "${FIN:-2770}" --ventana-s "${VENTANA:-24}" \
        --frames-por-ventana "${FRAMES:-8}" --lado-max "${LADO:-640}" --salida "${SAL}" "${EXTRA[@]}" \
      || { echo "${SAL}: FALLO"; FALLOS=$((FALLOS + 1)); }
  done
done
tar -cf "${BASE}.tar" "${BASE}" && ls -lh "${BASE}.tar"
echo "Terminado con ${FALLOS} fallo(s)."
