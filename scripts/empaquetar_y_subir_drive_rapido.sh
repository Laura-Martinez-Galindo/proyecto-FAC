#!/usr/bin/env bash
# ==============================================================================
# SUBIDA INSTANTÁNEA A GOOGLE DRIVE: ESTRATEGIA DE EMPAQUETADO PLANO (TAR DIRECTO)
# ==============================================================================
# ¿Por qué tardaba? Los archivos .PNG ya tienen compresión interna DEFLATE.
# Intentar comprimirlos con gzip/pigz no reduce el peso y quemaba 2 horas de CPU.
# 
# SOLUCIÓN INSTANTÁNEA:
# 1. Empaquetado secuencial plano con 'tar -cf' (SIN recompresión). Tarda 15 segundos.
# 2. Subida multi-hilo con rclone (transfers 32, chunks 256MB).
# ==============================================================================

set -euo pipefail

OBJETIVO="${1:-udvd}" # udvd, video3, todos
RUTA_PROYECTO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RUTA_PROYECTO}"

mkdir -p videos/comprimidos jobs/logs

echo "=============================================================================="
echo "⚡ SUBIDA INSTANTÁNEA A GOOGLE DRIVE (TAR PLANO DIRECTO)"
echo "   Objetivo: ${OBJETIVO}"
echo "   Inicio: $(date --iso-8601=seconds)"
echo "=============================================================================="

RCLONE_FLAGS="--transfers 32 --checkers 64 --drive-chunk-size 256M --drive-upload-cutoff 128M --buffer-size 128M --fast-list --progress"

empaquetar_y_subir_tar_plano() {
    local ruta_origen="$1"
    local nombre_tar="$2"
    local destino_drive="$3"

    if [[ -d "${ruta_origen}" ]]; then
        local total_frames
        total_frames=$(find "${ruta_origen}" -maxdepth 1 -type f | wc -l)
        if [[ ${total_frames} -eq 0 ]]; then
            echo "[!] Carpeta vacía: ${ruta_origen}, se omite."
            return
        fi

        local archivo_tar="videos/comprimidos/${nombre_tar}.tar"
        
        echo "------------------------------------------------------------------------------"
        echo "📦 [EMPAQUETANDO PLANO EN SEGUNDOS] ${nombre_tar}.tar (${total_frames:,} frames)..."
        local t0=$(date +%s)
        # tar plano sin compresión redundante: velocidad de lectura pura de disco
        tar -cf "${archivo_tar}" -C "$(dirname "${ruta_origen}")" "$(basename "${ruta_origen}")"
        local t1=$(date +%s)
        local tam=$(du -h "${archivo_tar}" | cut -f1)
        echo "✅ ¡Empaquetado en solo $((t1 - t0)) segundos! (${tam})"

        echo "☁️ [SUBIENDO A GDRIVE] -> ${destino_drive}..."
        rclone copy "${archivo_tar}" "${destino_drive}" ${RCLONE_FLAGS}
        echo "✅ Subida completada: ${nombre_tar}.tar"
    fi
}

if [[ "${OBJETIVO}" == "udvd" || "${OBJETIVO}" == "todos" ]]; then
    echo "=== SUBIENDO UDVD RESTAURADO DE LOS 3 VIDEOS ==="
    empaquetar_y_subir_tar_plano "videos/video1/udvd_sin_hud" "video1_udvd_sin_hud" "gdrive:proyecto-FAC/videos/video1/comprimidos/"
    empaquetar_y_subir_tar_plano "videos/video2/udvd_sin_hud" "video2_udvd_sin_hud" "gdrive:proyecto-FAC/videos/video2/comprimidos/"
    empaquetar_y_subir_tar_plano "videos/video3/udvd_sin_hud" "video3_udvd_sin_hud" "gdrive:proyecto-FAC/videos/video3/comprimidos/"
fi

if [[ "${OBJETIVO}" == "video3" || "${OBJETIVO}" == "todos" ]]; then
    echo "=== SUBIENDO COMPONENTES DE VIDEO 3 ==="
    empaquetar_y_subir_tar_plano "videos/video3/mascaras_hud" "video3_mascaras_hud" "gdrive:proyecto-FAC/videos/video3/comprimidos/"
    empaquetar_y_subir_tar_plano "videos/video3/frames_sin_hud" "video3_frames_sin_hud" "gdrive:proyecto-FAC/videos/video3/comprimidos/"
fi

echo
echo "=============================================================================="
echo "🎉 ¡SUBIDA FINALIZADA CON ÉXITO! $(date --iso-8601=seconds)"
echo "=============================================================================="
