#!/usr/bin/env bash

#SBATCH --job-name=preprocesamiento_hud
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia:2
#SBATCH --time=1-00:00:00
#SBATCH --output=jobs/logs/preprocesamiento_%x_%j.out
#SBATCH --error=jobs/logs/preprocesamiento_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; echo "Codigo de salida: ${codigo}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video1}"
RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_ENTORNO="${RUTA_PROYECTO}/fac_env311"
RUTA_PYTHON="${RUTA_ENTORNO}/bin/python"
RUTA_CONFIGURACION="${RUTA_PROYECTO}/config/videos.json"

cd "${RUTA_PROYECTO}"

if [[ ! -d "${RUTA_PROYECTO}/scripts" || ! -d "${RUTA_PROYECTO}/config" ]]; then
    echo "ERROR: ${RUTA_PROYECTO} no parece ser la raiz del proyecto." >&2
    echo "Envie el job desde la raiz de proyecto-FAC." >&2
    exit 1
fi

if [[ ! -x "${RUTA_PYTHON}" ]]; then
    echo "ERROR: No se encontro Python en ${RUTA_PYTHON}" >&2
    exit 1
fi

if [[ ! -f "${RUTA_CONFIGURACION}" ]]; then
    echo "ERROR: No se encontro ${RUTA_CONFIGURACION}" >&2
    exit 1
fi

for script in extraer_frames.py calcular_metricas.py segmentar_hud.py limpiar_hud.py; do
    if [[ ! -f "${RUTA_PROYECTO}/scripts/${script}" ]]; then
        echo "ERROR: No se encontro scripts/${script}" >&2
        exit 1
    fi
done

if [[ ! -f "${RUTA_PROYECTO}/ProPainter/inference_propainter.py" ]]; then
    echo "ERROR: No se encontro ProPainter/inference_propainter.py" >&2
    exit 1
fi

"${RUTA_PYTHON}" -m json.tool "${RUTA_CONFIGURACION}" > /dev/null

for script in extraer_frames.py calcular_metricas.py segmentar_hud.py limpiar_hud.py; do
    "${RUTA_PYTHON}" -m py_compile "${RUTA_PROYECTO}/scripts/${script}"
done

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export PYTHONUNBUFFERED="1"

echo "============================================================"
echo "PREPROCESAMIENTO DE VIDEO FLIR"
echo "============================================================"
echo "Video: ${VIDEO_ID}"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Nodo: ${SLURMD_NODENAME:-local}"
echo "Directorio: ${RUTA_PROYECTO}"
echo "Python: ${RUTA_PYTHON}"
echo "CPU asignadas: ${SLURM_CPUS_PER_TASK:-16}"
echo "GPU visibles: ${CUDA_VISIBLE_DEVICES:-asignadas_por_slurm}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

echo
echo "Informacion de las GPU asignadas:"
nvidia-smi || true

echo
echo "Verificacion de PyTorch:"

"${RUTA_PYTHON}" - <<'PY'
import torch

print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU visibles para PyTorch: {torch.cuda.device_count()}")

for indice in range(torch.cuda.device_count()):
    propiedades = torch.cuda.get_device_properties(indice)
    memoria_gb = propiedades.total_memory / 1024 ** 3
    print(f"GPU {indice}: {propiedades.name}, {memoria_gb:.2f} GB")
PY

consultar_fase() {
    local fase="$1"

    "${RUTA_PYTHON}" - "${VIDEO_ID}" "${fase}" <<'PY'
import json
import sys

from pathlib import Path
from openpyxl import load_workbook

video_id = sys.argv[1]
fase = sys.argv[2]
raiz = Path.cwd()
ruta_configuracion = raiz / "config" / "videos.json"

with ruta_configuracion.open("r", encoding="utf-8") as archivo:
    configuracion = json.load(archivo)

if video_id not in configuracion:
    print("no|El video no esta registrado en config/videos.json")
    raise SystemExit(0)

video = configuracion[video_id]


def resolver_ruta(ruta_configurada):
    if not ruta_configurada:
        return None

    ruta = Path(ruta_configurada).expanduser()

    if not ruta.is_absolute():
        ruta = raiz / ruta

    return ruta.resolve()


def contar_frames(carpeta):
    if carpeta is None or not carpeta.is_dir():
        return 0

    extensiones = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sum(1 for ruta in carpeta.iterdir() if ruta.is_file() and ruta.suffix.lower() in extensiones)


def obtener_ruta_resumen():
    ruta_video = resolver_ruta(video.get("ruta"))

    if ruta_video is None:
        return None

    if ruta_video.parent.name != "original":
        return None

    return ruta_video.parent.parent / "resumen.xlsx"


def revisar_fila_excel(nombre_modelo):
    ruta_resumen = obtener_ruta_resumen()

    if ruta_resumen is None:
        return False, "No se pudo determinar la ruta de resumen.xlsx"

    if not ruta_resumen.is_file():
        return False, f"No existe {ruta_resumen}"

    try:
        libro = load_workbook(ruta_resumen, read_only=True, data_only=True)
    except Exception as error:
        return False, f"No se pudo abrir resumen.xlsx: {error}"

    try:
        hoja = libro["Métricas"] if "Métricas" in libro.sheetnames else libro.active

        for fila in hoja.iter_rows(min_row=2, values_only=True):
            if not fila:
                continue

            modelo = fila[0] if len(fila) > 0 else None

            if modelo is None:
                continue

            if str(modelo).strip().lower() != nombre_modelo.strip().lower():
                continue

            frames = fila[2] if len(fila) > 2 else None
            niqe_media = fila[3] if len(fila) > 3 else None
            brisque_media = fila[8] if len(fila) > 8 else None

            valida = frames not in (None, 0) and niqe_media is not None and brisque_media is not None

            if valida:
                return True, f"resumen.xlsx contiene una fila valida para {nombre_modelo}"

            return False, f"La fila de {nombre_modelo} existe, pero esta incompleta"

        return False, f"resumen.xlsx no contiene una fila para {nombre_modelo}"
    finally:
        libro.close()


if fase == "extraccion":
    datos = video.get("extraccion", {})
    carpeta = resolver_ruta(datos.get("carpeta_salida"))
    esperados = int(datos.get("frames_extraidos", 0) or 0)
    encontrados = contar_frames(carpeta)
    estado = datos.get("estado")
    completa = estado == "completada" and esperados > 0 and encontrados == esperados
    razon = f"estado={estado}, esperados={esperados}, encontrados={encontrados}"
    print(f"{'si' if completa else 'no'}|{razon}")

elif fase == "metricas_original":
    completa, razon = revisar_fila_excel("Original")
    print(f"{'si' if completa else 'no'}|{razon}")

elif fase == "hud":
    datos = video.get("hud", {})
    carpeta = resolver_ruta(datos.get("carpeta_salida"))
    esperados = int(datos.get("mascaras_generadas", datos.get("frames_procesados", 0)) or 0)
    encontrados = contar_frames(carpeta)
    estado = datos.get("estado")
    completa = estado == "completada" and esperados > 0 and encontrados == esperados
    razon = f"estado={estado}, esperados={esperados}, encontrados={encontrados}"
    print(f"{'si' if completa else 'no'}|{razon}")

elif fase == "limpieza_hud":
    datos = video.get("hud", {}).get("limpieza", {})
    carpeta = resolver_ruta(datos.get("carpeta_salida"))
    esperados = int(datos.get("frames_procesados", 0) or 0)
    encontrados = contar_frames(carpeta)
    estado = datos.get("estado")
    completa = estado == "completada" and esperados > 0 and encontrados == esperados
    razon = f"estado={estado}, esperados={esperados}, encontrados={encontrados}"
    print(f"{'si' if completa else 'no'}|{razon}")

elif fase == "metricas_sin_hud":
    completa, razon = revisar_fila_excel("Sin HUD")
    print(f"{'si' if completa else 'no'}|{razon}")

else:
    print(f"no|Fase no reconocida: {fase}")
PY
}

ejecutar_etapa() {
    local nombre="$1"
    shift
    local inicio
    local fin
    local duracion

    inicio="$(date +%s)"

    echo
    echo "------------------------------------------------------------"
    echo "ETAPA: ${nombre}"
    echo "Inicio: $(date --iso-8601=seconds)"
    echo "Comando: $*"
    echo "------------------------------------------------------------"

    "$@"

    fin="$(date +%s)"
    duracion="$((fin - inicio))"

    echo "Fin: $(date --iso-8601=seconds)"
    echo "Duracion: ${duracion} segundos"
}

ejecutar_si_falta() {
    local fase="$1"
    local nombre="$2"
    shift 2
    local resultado
    local estado
    local detalle

    resultado="$(consultar_fase "${fase}")"
    estado="${resultado%%|*}"
    detalle="${resultado#*|}"

    echo
    echo "Verificacion de ${nombre}: ${detalle}"

    if [[ "${estado}" == "si" ]]; then
        echo "OMITIDA: ${nombre} ya esta completada y verificada."
    else
        ejecutar_etapa "${nombre}" "$@"
    fi
}

TIEMPO_TOTAL_INICIO="$(date +%s)"

ejecutar_si_falta "extraccion" "Extraccion de frames" "${RUTA_PYTHON}" scripts/extraer_frames.py --video "${VIDEO_ID}"

ejecutar_si_falta "metricas_original" "Metricas de frames originales" "${RUTA_PYTHON}" scripts/calcular_metricas.py --video "${VIDEO_ID}" --modelo original

ejecutar_si_falta "hud" "Segmentacion del HUD" "${RUTA_PYTHON}" scripts/segmentar_hud.py --video "${VIDEO_ID}"

ejecutar_si_falta "limpieza_hud" "Eliminacion del HUD con ProPainter" "${RUTA_PYTHON}" scripts/limpiar_hud.py --video "${VIDEO_ID}"

ejecutar_si_falta "metricas_sin_hud" "Metricas de frames sin HUD" "${RUTA_PYTHON}" scripts/calcular_metricas.py --video "${VIDEO_ID}" --modelo sin_hud

TIEMPO_TOTAL_FIN="$(date +%s)"
TIEMPO_TOTAL="$((TIEMPO_TOTAL_FIN - TIEMPO_TOTAL_INICIO))"
HORAS="$((TIEMPO_TOTAL / 3600))"
MINUTOS="$(((TIEMPO_TOTAL % 3600) / 60))"
SEGUNDOS="$((TIEMPO_TOTAL % 60))"

echo
echo "============================================================"
echo "PREPROCESAMIENTO COMPLETADO"
echo "Video: ${VIDEO_ID}"
echo "Fin: $(date --iso-8601=seconds)"
echo "Tiempo total: ${TIEMPO_TOTAL} segundos"
printf "Duracion total: %02d:%02d:%02d\n" "${HORAS}" "${MINUTOS}" "${SEGUNDOS}"
echo "Resumen: videos/${VIDEO_ID}/resumen.xlsx"
echo "============================================================"

# --- Sincronizacion automatica con Google Drive (si rclone esta configurado) ---
GDRIVE_ID="1RZGdPrUYRHiIQr2UAW7qKP7P7FmH476B"
if command -v rclone &> /dev/null; then
    if rclone listremotes | grep -q "gdrive"; then
        echo
        echo "Sincronizando frames de ${VIDEO_ID} con Google Drive (Carpeta ID: ${GDRIVE_ID})..."
        rclone copy "videos/${VIDEO_ID}/frames_originales" "gdrive:${GDRIVE_ID}/${VIDEO_ID}/frames_originales" --transfers 8 --checkers 16 || true
        rclone copy "videos/${VIDEO_ID}/mascaras_hud" "gdrive:${GDRIVE_ID}/${VIDEO_ID}/mascaras_hud" --transfers 8 --checkers 16 || true
        rclone copy "videos/${VIDEO_ID}/frames_sin_hud" "gdrive:${GDRIVE_ID}/${VIDEO_ID}/frames_sin_hud" --transfers 8 --checkers 16 || true
        rclone copy "videos/${VIDEO_ID}/resumen.xlsx" "gdrive:${GDRIVE_ID}/${VIDEO_ID}/" || true
        echo "Sincronizacion con Google Drive completada exitosamente."
    fi
fi