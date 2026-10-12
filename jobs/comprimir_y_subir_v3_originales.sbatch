#!/usr/bin/env bash

#SBATCH --job-name=comp_subir_v3_orig
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/logs/comp_subir_v3_orig_%j.out
#SBATCH --error=jobs/logs/comp_subir_v3_orig_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail
trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs videos/comprimidos

ARCHIVO_TAR="videos/comprimidos/video3_frames_originales.tar.gz"

echo "=============================================================================="
echo "📦 FASE 1: COMPRESIÓN MULTIHILO A DISCO (16 CPUs con pigz)"
echo "   Archivo destino: ${ARCHIVO_TAR}"
echo "   Inicio: $(date --iso-8601=seconds)"
echo "=============================================================================="

# Detectar herramienta de compresión paralela
if command -v pigz &> /dev/null; then
    COMP_CMD="pigz -p 16 -6"
else
    COMP_CMD="gzip -6"
fi

# Empaquetar y comprimir directamente a disco local
t0=$(date +%s)
tar -cf - -C "videos/video3" "frames_originales" | ${COMP_CMD} > "${ARCHIVO_TAR}"
t1=$(date +%s)
tamano=$(du -h "${ARCHIVO_TAR}" | cut -f1)

echo "✅ Compresión finalizada con éxito en $((t1 - t0)) segundos."
echo "   Tamaño final reducido: ${tamano}"

echo
echo "=============================================================================="
echo "☁️ FASE 2: SUBIDA BLINDADA A GOOGLE DRIVE (CON REANUDACIÓN Y ANTI-403)"
echo "   Inicio: $(date --iso-8601=seconds)"
echo "=============================================================================="

# Flags de rclone blindados contra el Error 403:
# --tpslimit 10: limita peticiones/segundo para que Google NUNCA bloquee por cuota
# --retries 30: si hay micro-corte, reintenta hasta 30 veces sin abortar
# --low-level-retries 50: reanuda el chunk exacto donde iba
rclone copy "${ARCHIVO_TAR}" "gdrive:proyecto-FAC/videos/video3/comprimidos/" \
    --drive-chunk-size 128M \
    --transfers 8 \
    --checkers 16 \
    --tpslimit 10 \
    --retries 30 \
    --low-level-retries 50 \
    --progress

echo
echo "=============================================================================="
echo "🎉 ¡VIDEO 3 FRAMES ORIGINALES COMPRIMIDO Y SUBIDO A DRIVE CON ÉXITO!"
echo "   Fin: $(date --iso-8601=seconds)"
echo "=============================================================================="
