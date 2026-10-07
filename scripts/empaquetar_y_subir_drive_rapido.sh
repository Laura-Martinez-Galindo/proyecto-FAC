#!/usr/bin/env bash
# ==============================================================================
# SINCRONIZACIÓN Y COMPRESIÓN ULTRA-RÁPIDA A GOOGLE DRIVE (VIDEO 1, 2, 3)
# ==============================================================================
# Uso:
#   bash scripts/empaquetar_y_subir_drive_rapido.sh [video1|video2|video3|todos]
# ==============================================================================

set -euo pipefail

OBJETIVO="${1:-todos}"
RUTA_PROYECTO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RUTA_PROYECTO}"

mkdir -p videos/comprimidos jobs/logs

echo "=============================================================================="
echo "🚀 INICIO DE COMPRESIÓN Y SUBIDA ULTRA-RÁPIDA A GOOGLE DRIVE"
echo "   Objetivo: ${OBJETIVO}"
echo "   Hora inicio: $(date --iso-8601=seconds)"
echo "=============================================================================="

if ! command -v rclone &> /dev/null; then
    echo "[-] ERROR: rclone no está disponible en el PATH." >&2
    exit 1
fi

RCLONE_FLAGS="--transfers 32 --checkers 64 --drive-chunk-size 128M --drive-upload-cutoff 32M --buffer-size 64M --fast-list --progress"

# Detectar si pigz (gzip paralelo multihilo) está disponible
if command -v pigz &> /dev/null; then
    COMPRESS_CMD="pigz -p 16 -1"
    echo "[+] Utilizando 'pigz' con 16 hilos para compresión ultra-rápida."
else
    COMPRESS_CMD="gzip -1"
    echo "[!] 'pigz' no encontrado, utilizando gzip estándar en modo rápido (-1)."
fi

empaquetar_y_subir_carpeta() {
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

        local archivo_tar="videos/comprimidos/${nombre_tar}.tar.gz"
        if [[ ! -f "${archivo_tar}" ]]; then
            echo "------------------------------------------------------------------------------"
            echo "📦 [EMPAQUETANDO] ${nombre_tar}.tar.gz (${total_frames} frames)..."
            local t_inicio
            t_inicio=$(date +%s)
            tar -cf - -C "$(dirname "${ruta_origen}")" "$(basename "${ruta_origen}")" | ${COMPRESS_CMD} > "${archivo_tar}"
            local t_fin
            t_fin=$(date +%s)
            local tamano
            tamano=$(du -h "${archivo_tar}" | cut -f1)
            echo "✅ Empaquetado en $((t_fin - t_inicio)) seg (${tamano})."
        else
            echo "ℹ️ El paquete ${archivo_tar} ya existe."
        fi

        echo "☁️ [SUBIENDO A GDRIVE] -> ${destino_drive}..."
        rclone copy "${archivo_tar}" "${destino_drive}" ${RCLONE_FLAGS}
        echo "✅ Subida completada: ${nombre_tar}.tar.gz"
    fi
}

procesar_video() {
    local vid="$1"
    echo
    echo "=============================================================================="
    echo "📁 PROCESANDO VUELO: ${vid^^}"
    echo "=============================================================================="

    local dir_vid="videos/${vid}"
    local drive_vid="gdrive:proyecto-FAC/videos/${vid}"
    local drive_comp="gdrive:proyecto-FAC/videos/${vid}/comprimidos/"

    # 1. Empaquetar y subir los 4 datasets principales
    empaquetar_y_subir_carpeta "${dir_vid}/frames_originales" "${vid}_frames_originales" "${drive_comp}"
    empaquetar_y_subir_carpeta "${dir_vid}/mascaras_hud" "${vid}_mascaras_hud" "${drive_comp}"
    empaquetar_y_subir_carpeta "${dir_vid}/frames_sin_hud" "${vid}_frames_sin_hud" "${drive_comp}"
    empaquetar_y_subir_carpeta "${dir_vid}/udvd_sin_hud" "${vid}_udvd_sin_hud" "${drive_comp}"

    # 2. Subir videos renderizados MP4 oficiales (directos sin comprimir para ver en Drive)
    if [[ -d "${dir_vid}/videos_renderizados" ]]; then
        echo "🎬 Subiendo videos renderizados .mp4 oficiales..."
        rclone copy "${dir_vid}/videos_renderizados" "${drive_vid}/videos_renderizados" ${RCLONE_FLAGS}
    fi

    # 3. Subir excels y CSVs de métricas
    echo "📊 Subiendo tablas de métricas y línea de tiempo..."
    rclone copy "${dir_vid}/resumen.xlsx" "${drive_vid}/" 2>/dev/null || true
    rclone copy "${dir_vid}/resumen_experimentos.xlsx" "${drive_vid}/" 2>/dev/null || true
    rclone copy "${dir_vid}/linea_tiempo_${vid}.csv" "${drive_vid}/" 2>/dev/null || true
    rclone copy "${dir_vid}/metricas_consolidadas.csv" "${drive_vid}/" 2>/dev/null || true
}

# Ejecutar según argumento
if [[ "${OBJETIVO}" == "todos" ]]; then
    procesar_video "video1"
    procesar_video "video2"
    procesar_video "video3"
else
    procesar_video "${OBJETIVO}"
fi

# Subir figuras consolidadas de la tesis y reportes generales
echo
echo "=============================================================================="
echo "📑 SUBIENDO FIGURAS Y REPORTES CONSOLIDADOS DE TESIS"
echo "=============================================================================="
rclone copy "figuras_tesis" "gdrive:proyecto-FAC/figuras_tesis" ${RCLONE_FLAGS} 2>/dev/null || true

echo
echo "=============================================================================="
echo "🎉 SINCRONIZACIÓN Y SUBIDA ULTRA-RÁPIDA FINALIZADA CON ÉXITO: $(date --iso-8601=seconds)"
echo "=============================================================================="
