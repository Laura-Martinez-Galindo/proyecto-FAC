#!/usr/bin/env bash
# ==============================================================================
# SUBIDA INSTANTÁNEA A GOOGLE DRIVE (TAR PLANO SIN COMPRESIÓN REDUNDANTE)
# ==============================================================================

RUTA_PROYECTO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RUTA_PROYECTO}"

OBJETIVO="${1:-udvd}"

mkdir -p videos/comprimidos jobs/logs

echo "=============================================================================="
echo "⚡ SUBIDA INSTANTÁNEA A GOOGLE DRIVE (TAR DIRECTO)"
echo "   Objetivo: ${OBJETIVO}"
echo "   Directorio: ${RUTA_PROYECTO}"
echo "   Inicio: $(date --iso-8601=seconds)"
echo "=============================================================================="

if ! command -v rclone &> /dev/null; then
    echo "[-] ERROR: rclone no está disponible en PATH." >&2
    exit 1
fi

RCLONE_FLAGS="--transfers 16 --checkers 32 --drive-chunk-size 256M --drive-upload-cutoff 128M --buffer-size 128M --fast-list --progress"

empaquetar_y_subir_tar_plano() {
    local ruta_origen="$1"
    local nombre_tar="$2"
    local destino_drive="$3"

    if [ -d "${ruta_origen}" ]; then
        local total_frames
        total_frames=$(find "${ruta_origen}" -maxdepth 1 -type f | wc -l)
        if [ "${total_frames}" -eq 0 ]; then
            echo "[!] Carpeta vacía: ${ruta_origen}, se omite."
            return
        fi

        local archivo_tar="videos/comprimidos/${nombre_tar}.tar"
        
        echo "------------------------------------------------------------------------------"
        echo "📦 [EMPAQUETANDO PLANO] ${nombre_tar}.tar (${total_frames} frames)..."
        local t0
        t0=$(date +%s)
        
        # tar plano sin compresión redundante (lectura de disco pura en segundos)
        tar -cf "${archivo_tar}" -C "$(dirname "${ruta_origen}")" "$(basename "${ruta_origen}")"
        
        local t1
        t1=$(date +%s)
        local tam
        tam=$(du -h "${archivo_tar}" | cut -f1)
        echo "✅ Empaquetado en $((t1 - t0)) segundos (${tam})."

        echo "☁️ [SUBIENDO A GDRIVE] -> ${destino_drive}..."
        rclone copy "${archivo_tar}" "${destino_drive}" ${RCLONE_FLAGS}
        echo "✅ ¡Subida completada: ${nombre_tar}.tar!"
    else
        echo "[!] No existe directorio: ${ruta_origen}"
    fi
}

if [ "${OBJETIVO}" = "udvd" ] || [ "${OBJETIVO}" = "todos" ]; then
    echo "=== 1. SUBIENDO UDVD RESTAURADO DE LOS 3 VIDEOS ==="
    empaquetar_y_subir_tar_plano "videos/video1/udvd_sin_hud" "video1_udvd_sin_hud" "gdrive:proyecto-FAC/videos/video1/comprimidos/"
    empaquetar_y_subir_tar_plano "videos/video2/udvd_sin_hud" "video2_udvd_sin_hud" "gdrive:proyecto-FAC/videos/video2/comprimidos/"
    empaquetar_y_subir_tar_plano "videos/video3/udvd_sin_hud" "video3_udvd_sin_hud" "gdrive:proyecto-FAC/videos/video3/comprimidos/"
fi

if [ "${OBJETIVO}" = "video3" ] || [ "${OBJETIVO}" = "todos" ]; then
    echo "=== 2. SUBIENDO COMPONENTES DE VIDEO 3 ==="
    empaquetar_y_subir_tar_plano "videos/video3/mascaras_hud" "video3_mascaras_hud" "gdrive:proyecto-FAC/videos/video3/comprimidos/"
    empaquetar_y_subir_tar_plano "videos/video3/frames_sin_hud" "video3_frames_sin_hud" "gdrive:proyecto-FAC/videos/video3/comprimidos/"
fi

echo
echo "=============================================================================="
echo "🎉 ¡SUBIDA FINALIZADA CON ÉXITO! $(date --iso-8601=seconds)"
echo "=============================================================================="
