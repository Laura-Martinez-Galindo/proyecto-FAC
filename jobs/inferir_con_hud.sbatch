#!/usr/bin/env bash
#SBATCH --job-name=dn_conhud
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co
#SBATCH --output=jobs/logs/dn_conhud_%j.out
#SBATCH --error=jobs/logs/dn_conhud_%j.err

# OPCIONAL, solo para VER como quedan: aplica los modelos ya elegidos a las imagenes del dataset ORIGINAL
# (con HUD). No reentrena nada: usa los mismos modelos (grano: F2F, CVF-SID, PUCA; NIQE: ZS-N2N, ZS-KAN).
# Salida: cache/denoising/<regimen>/<exp>/salidas_con_hud (y <exp>_ycolor/salidas_con_hud)
# Uso:  bash jobs/enviar.sh --gres=gpu:1 jobs/inferir_con_hud.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate
module load cuda/12.1 2>/dev/null || true
L=resultados/denoising/listas_fase2
FALLOS=0
#           regimen        experimento  metodo  ycolor
for linea in "HDEO_DDE_HAZE f2f_f2 f2f 1" "HDEO_HAZE cvfsid_f2 cvfsid 0" "HDEO_DDE puca_f2 puca 0" \
             "HDEO_DDE_HAZE zsn2n_f2iqa zsn2n 1" "HDEO_HAZE zskan_f2iqa zskan 1" "HDEO_DDE zskan_f2iqa zskan 1"; do
  read -r F E M Y <<< "$linea"
  C="cache/denoising/${F}/${E}"
  echo "######## ${F} ${E}  $(date '+%F %T')"
  if [ "$M" = "zskan" ]; then   # zero-shot: no hay checkpoint, se entrena una red por imagen
    python scripts/denoising.py inferir --metodo zskan --lista "${L}/${F}_evaluacion.csv" \
        --entrada datasets/dataset_original/images --salida "${C}/salidas_con_hud" \
      || { echo "!! fallo ${E}"; FALLOS=$((FALLOS+1)); continue; }
  else
    [ -f "${C}/modelo.pt" ] || { echo "!! falta ${C}/modelo.pt (se salta)"; FALLOS=$((FALLOS+1)); continue; }
    python scripts/denoising.py inferir --checkpoint "${C}/modelo.pt" --lista "${L}/${F}_evaluacion.csv" \
        --entrada datasets/dataset_original/images --salida "${C}/salidas_con_hud" --vecinos original \
      || { echo "!! fallo ${E}"; FALLOS=$((FALLOS+1)); continue; }
  fi
  if [ "$Y" = "1" ]; then
    python scripts/color_original.py --entrada datasets/dataset_original/images --base "${C}/salidas_con_hud" \
        --salida "${C}_ycolor/salidas_con_hud"
  fi
done
echo "Fallos: ${FALLOS}"
