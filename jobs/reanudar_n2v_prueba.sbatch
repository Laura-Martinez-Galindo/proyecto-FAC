#!/usr/bin/env bash

#SBATCH --job-name=reanudar_n2v
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=04:00:00
#SBATCH --output=jobs/logs/reanudar_n2v_prueba_%j.out
#SBATCH --error=jobs/logs/reanudar_n2v_prueba_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; echo "Codigo: ${codigo}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video1}"
RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"
CHECKPOINT="${RUTA_PROYECTO}/cache/denoising/${VIDEO_ID}/n2v/modelo.ckpt"

cd "${RUTA_PROYECTO}"

mkdir -p jobs/logs

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export PYTHONUNBUFFERED="1"

for script in pipeline_n2v.py predecir_lote_n2v.py calcular_metricas.py; do
    if [[ ! -f "scripts/${script}" ]]; then
        echo "ERROR: No se encontro scripts/${script}" >&2
        exit 1
    fi

    "${RUTA_PYTHON}" -m py_compile "scripts/${script}"
done

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: No existe el checkpoint N2V: ${CHECKPOINT}" >&2
    exit 1
fi

echo "============================================================"
echo "REANUDACION DE LA PRUEBA N2V"
echo "============================================================"
echo "Video: ${VIDEO_ID}"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Nodo: ${SLURMD_NODENAME:-local}"
echo "CPU: ${SLURM_CPUS_PER_TASK:-16}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-desconocida}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

nvidia-smi

echo
echo "Iniciando inferencia N2V sobre 100 frames."

"${RUTA_PYTHON}" scripts/pipeline_n2v.py \
    --video "${VIDEO_ID}" \
    --modo original \
    --etapa inferir \
    --max-frames 100 \
    --epocas 1 \
    --pasos-por-epoca 5

CANTIDAD_N2V="$(
    "${RUTA_PYTHON}" - "${VIDEO_ID}" <<'PY'
import sys
from pathlib import Path

video_id = sys.argv[1]
carpeta = Path("videos") / video_id / "n2v"

if carpeta.is_dir():
    cantidad = sum(
        1
        for ruta in carpeta.iterdir()
        if ruta.is_file()
        and ruta.name.startswith("frame_")
        and ruta.suffix.lower() == ".png"
    )
else:
    cantidad = 0

print(cantidad)
PY
)"

echo
echo "Frames N2V encontrados: ${CANTIDAD_N2V}"

if [[ "${CANTIDAD_N2V}" -ne 100 ]]; then
    echo "ERROR: Se esperaban 100 frames N2V y se encontraron ${CANTIDAD_N2V}" >&2
    exit 1
fi

echo
echo "Calculando NIQE y BRISQUE para N2V."

"${RUTA_PYTHON}" scripts/calcular_metricas.py \
    --video "${VIDEO_ID}" \
    --modelo n2v

echo
echo "============================================================"
echo "METRICAS DE LA PRUEBA N2V"
echo "============================================================"

"${RUTA_PYTHON}" - "${VIDEO_ID}" <<'PY'
import sys
from pathlib import Path

from openpyxl import load_workbook

video_id = sys.argv[1]
ruta_resumen = Path("videos") / video_id / "resumen.xlsx"

if not ruta_resumen.is_file():
    raise FileNotFoundError(
        f"No existe el resumen: {ruta_resumen}"
    )

libro = load_workbook(
    ruta_resumen,
    read_only=True,
    data_only=True,
)

try:
    if "Métricas" in libro.sheetnames:
        hoja = libro["Métricas"]
    else:
        hoja = libro.active

    encabezados = [
        celda.value
        for celda in hoja[1]
    ]

    fila_encontrada = None

    for fila in hoja.iter_rows(
        min_row=2,
        values_only=True,
    ):
        if not fila:
            continue

        modelo = fila[0]

        if modelo is None:
            continue

        if str(modelo).strip().lower() == "noise2void":
            fila_encontrada = fila
            break

    if fila_encontrada is None:
        raise RuntimeError(
            "No se encontro la fila Noise2Void en resumen.xlsx"
        )

    for encabezado, valor in zip(
        encabezados,
        fila_encontrada,
    ):
        print(f"{encabezado}: {valor}")
finally:
    libro.close()
PY

echo
echo "============================================================"
echo "PRUEBA N2V COMPLETADA"
echo "Frames N2V: ${CANTIDAD_N2V}"
echo "Resumen: videos/${VIDEO_ID}/resumen.xlsx"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
