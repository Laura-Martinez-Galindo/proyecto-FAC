#!/usr/bin/env bash
#SBATCH --job-name=cap_mineria
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co
#SBATCH --output=jobs/logs/cap_mineria_%j.out
#SBATCH --error=jobs/logs/cap_mineria_%j.err

# Experimento 2 de dense video captioning, orientado a MINERIA ILEGAL (zero-shot, sin entrenar).
# Diferencias con el experimento 1 (que fue sobre el video2, donde no hay ninguna etiqueta SDZI):
#  * Tramos con zonas SDZI anotadas: video1 (60 imagenes SDZI) y video3 (86); el video2 queda como
#    control negativo (pueblo y vias, para medir falsas alarmas).
#  * Prompt "mineria": 8 indicadores de mineria aluvial + 2 de contexto, evidencia por frame y ubicacion,
#    y despues el puntaje 0-10 con una escala explicita. "mineria_corto": mismo objetivo sin la lista
#    (ablacion: cuanto aporta la lista).
#  * Ventanas de 16 s con paso de 8 s (solapadas, mejor ubicacion en el tiempo), 6 frames a 896 px
#    (mas detalle para objetos pequenos que 8 a 640).
#  * Verificacion: si el puntaje es >= 5, segunda mirada a 1280 px de los frames citados como evidencia.
#  * Con HUD y sin HUD, para medir el efecto del HUD.
#  * Se guarda una hoja de contacto por ventana (mosaicos/) para revisar a ojo que vio el modelo.
# Analisis despues, en el PC: scripts/analizar_captioning_mineria.py (cruza cada ventana con las
# etiquetas del dataset que caen en ella).
# Variables: MODELO, TRAMOS ("video1:0:700 video3:100:720 video2:1980:2770"), PROMPTS ("mineria mineria_corto"),
#   VERSIONES ("sin_hud original"), VENTANA (16) PASO (8) FRAMES (6) LADO (896) VERIFICAR (5)
#   HUMO=1: 3 ventanas del video1 donde hay SDZI (3:28), solo sin_hud + mineria, para probar en minutos.
# Resultado: resultados/captioning_mineria.tar (texto + hojas de contacto pequenas)

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODELO="${MODELO:-Qwen/Qwen2.5-VL-7B-Instruct}"
TRAMOS="${TRAMOS:-video1:0:700 video3:100:720 video2:1980:2770}"
PROMPTS="${PROMPTS:-mineria mineria_corto}"
VERSIONES="${VERSIONES:-sin_hud original}"
BASE="resultados/captioning_mineria"
EXTRA=()
if [ "${HUMO:-0}" = "1" ]; then
  TRAMOS="video1:200:240"; PROMPTS="mineria"; VERSIONES="sin_hud"; EXTRA+=(--max-ventanas 3)
  BASE="resultados/captioning_mineria_humo"
fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
FALLOS=0
for P in ${PROMPTS}; do
  for V in ${VERSIONES}; do
    for T in ${TRAMOS}; do
      IFS=: read -r VID INI FIN <<< "${T}"
      SAL="${BASE}/${VID}/$(basename "${MODELO}")/${V}_${P}"
      OPC=(--max-tokens 700 --mosaicos)
      [ "${P}" = "mineria" ] && OPC+=(--verificar "${VERIFICAR:-5}")
      echo "################ ${VID} ${INI}-${FIN}s ${V} ${P}  $(date '+%F %T') ################"
      python scripts/captioning_ventanas.py --video "${VID}" --version "${V}" --modelo "${MODELO}" \
          --prompt "${P}" --inicio-s "${INI}" --fin-s "${FIN}" --ventana-s "${VENTANA:-16}" --paso-s "${PASO:-8}" \
          --frames-por-ventana "${FRAMES:-6}" --lado-max "${LADO:-896}" --salida "${SAL}" "${OPC[@]}" "${EXTRA[@]}" \
        || { echo "${SAL}: FALLO"; FALLOS=$((FALLOS + 1)); }
      tar -cf "${BASE}.tar" "${BASE}"   # se actualiza en cada tramo: sirve aunque el job no termine
    done
  done
done
ls -lh "${BASE}.tar"
echo "Terminado con ${FALLOS} fallo(s)."
