#!/usr/bin/env bash

#SBATCH --job-name=sync_gdrive
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=jobs/logs/sync_gdrive_%j.out
#SBATCH --error=jobs/logs/sync_gdrive_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; exit "${codigo}"' ERR

RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${RUTA_PROYECTO}"
mkdir -p jobs/logs

echo "============================================================"
echo "SINCRONIZACIÓN DE ALTA VELOCIDAD CON GOOGLE DRIVE (CPU)"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

if ! command -v rclone &> /dev/null; then
    echo "ERROR: rclone no está instalado o en el PATH." >&2
    exit 1
fi

RCLONE_FLAGS="--transfers 32 --checkers 64 --drive-chunk-size 64M --drive-upload-cutoff 32M --buffer-size 64M --fast-list --progress"

# 1. Función para empaquetar y subir en .tar.gz (Ultra-rápido)
empaquetar_y_subir() {
    local ruta_origen="$1"
    local nombre_paquete="$2"
    local ruta_destino_gdrive="$3"

    if [[ -d "${ruta_origen}" ]]; then
        local ruta_tar="videos/comprimidos/${nombre_paquete}.tar.gz"
        mkdir -p "videos/comprimidos"

        if [[ ! -f "${ruta_tar}" ]]; then
            echo "--- Empaquetando ${nombre_paquete}.tar.gz desde ${ruta_origen} ---"
            tar -cf - -C "$(dirname "${ruta_origen}")" "$(basename "${ruta_origen}")" | gzip -1 > "${ruta_tar}"
        fi

        echo "Subiendo ${nombre_paquete}.tar.gz a ${ruta_destino_gdrive}..."
        rclone copy "${ruta_tar}" "${ruta_destino_gdrive}" ${RCLONE_FLAGS} || true
    fi
}

echo "============================================================"
echo "FASE 1: SINCRONIZACIÓN DE VIDEO 1 (EXCELS + PAQUETES TAR.GZ)"
echo "============================================================"

# Excels de Video 1
rclone copy "videos/video1/resumen.xlsx" "gdrive:proyecto-FAC/videos/video1/" 2>/dev/null || true
rclone copy "videos/video1/resumen_experimentos.xlsx" "gdrive:proyecto-FAC/videos/video1/" 2>/dev/null || true

# Paquetes base Video 1
empaquetar_y_subir "videos/video1/frames_originales" "video1_frames_originales" "gdrive:proyecto-FAC/videos/video1/comprimidos/"
empaquetar_y_subir "videos/video1/mascaras_hud" "video1_mascaras_hud" "gdrive:proyecto-FAC/videos/video1/comprimidos/"
empaquetar_y_subir "videos/video1/frames_sin_hud" "video1_frames_sin_hud" "gdrive:proyecto-FAC/videos/video1/comprimidos/"

# Modelos específicos solicitados de Video 1 (UDVD, Blind2Unblind y Ensamble Wavelet)
empaquetar_y_subir "videos/video1/expos/udvd_sinhud" "video1_udvd_sinhud" "gdrive:proyecto-FAC/videos/video1/comprimidos/"
empaquetar_y_subir "videos/video1/UDVD_SinHUD_K5_lr1e3" "video1_udvd_sinhud" "gdrive:proyecto-FAC/videos/video1/comprimidos/"
empaquetar_y_subir "videos/video1/expos/blind2unblind_sinhud" "video1_blind2unblind_sinhud" "gdrive:proyecto-FAC/videos/video1/comprimidos/"
empaquetar_y_subir "videos/video1/Blind2Unblind_SinHUD_d4_lr1e3" "video1_blind2unblind_sinhud" "gdrive:proyecto-FAC/videos/video1/comprimidos/"
empaquetar_y_subir "videos/video1/SuperEnsemble_Wavelet_UDVD_B2U" "video1_ensemble_wavelet" "gdrive:proyecto-FAC/videos/video1/comprimidos/"
empaquetar_y_subir "videos/video1/Ensemble_Wavelet_SinHUD" "video1_ensemble_wavelet" "gdrive:proyecto-FAC/videos/video1/comprimidos/"
empaquetar_y_subir "videos/video1/expos/ensemble_wavelet_triple_sinhud" "video1_ensemble_wavelet" "gdrive:proyecto-FAC/videos/video1/comprimidos/"

echo
echo "============================================================"
echo "FASE 2: SINCRONIZACIÓN DE VIDEO 2 (PAQUETES TAR.GZ + EXCELS)"
echo "============================================================"

# Excels de Video 2
rclone copy "videos/video2/resumen.xlsx" "gdrive:proyecto-FAC/videos/video2/" 2>/dev/null || true
rclone copy "videos/video2/resumen_experimentos.xlsx" "gdrive:proyecto-FAC/videos/video2/" 2>/dev/null || true
rclone copy "videos/video2/resultados_video2.xlsx" "gdrive:proyecto-FAC/videos/video2/" 2>/dev/null || true

# Paquetes de Video 2 (en tar.gz para subida inmediata en minutos)
empaquetar_y_subir "videos/video2/mascaras_hud" "video2_mascaras_hud" "gdrive:proyecto-FAC/videos/video2/comprimidos/"
empaquetar_y_subir "videos/video2/frames_sin_hud" "video2_frames_sin_hud" "gdrive:proyecto-FAC/videos/video2/comprimidos/"
empaquetar_y_subir "videos/video2/expos/udvd_sin_hud" "video2_udvd_sin_hud" "gdrive:proyecto-FAC/videos/video2/comprimidos/"
empaquetar_y_subir "videos/video2/expos/blind2unblind_sin_hud" "video2_blind2unblind_sin_hud" "gdrive:proyecto-FAC/videos/video2/comprimidos/"
empaquetar_y_subir "videos/video2/expos/ensemble_wavelet_udvd_b2u_sin_hud" "video2_ensemble_wavelet" "gdrive:proyecto-FAC/videos/video2/comprimidos/"

# Videos Renderizados (.mp4 y comparativos)
rclone copy "videos/video1/videos_renderizados" "gdrive:proyecto-FAC/videos/video1/videos_renderizados" ${RCLONE_FLAGS} 2>/dev/null || true
rclone copy "videos/video2/videos_renderizados" "gdrive:proyecto-FAC/videos/video2/videos_renderizados" ${RCLONE_FLAGS} 2>/dev/null || true

echo
echo "============================================================"
echo "FASE 3: TERMINAR SUBIDA INDIVIDUAL DE FRAMES ORIGINALES (VIDEO 2)"
echo "============================================================"
if [[ -d "videos/video2/frames_originales" ]]; then
    echo "Retomando subida individual de frames_originales (desde el 66%)..."
    rclone copy "videos/video2/frames_originales" "gdrive:proyecto-FAC/videos/video2/frames_originales" ${RCLONE_FLAGS} || true
fi

echo
echo "============================================================"
echo "SINCRONIZACIÓN FINALIZADA CON ÉXITO: $(date --iso-8601=seconds)"
echo "============================================================"

echo
echo "============================================================"
echo "SINCRONIZACIÓN TOTAL COMPLETADA CON ÉXITO"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"

