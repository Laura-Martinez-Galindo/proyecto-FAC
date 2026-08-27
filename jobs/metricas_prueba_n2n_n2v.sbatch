#!/usr/bin/env bash

#SBATCH --job-name=metricas_n2n_n2v
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/metricas_n2n_n2v_%j.out
#SBATCH --error=jobs/logs/metricas_n2n_n2v_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lf.martinezg1@uniandes.edu.co

set -euo pipefail

trap 'codigo=$?; echo "ERROR en linea ${LINENO}: ${BASH_COMMAND}" >&2; echo "Codigo: ${codigo}" >&2; exit "${codigo}"' ERR

VIDEO_ID="${1:-video1}"
RUTA_PROYECTO="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUTA_PYTHON="${RUTA_PROYECTO}/fac_env311/bin/python"
RUTA_SCRIPT="${RUTA_PROYECTO}/scripts/calcular_metricas.py"
RUTA_RESUMEN="${RUTA_PROYECTO}/videos/${VIDEO_ID}/resumen.xlsx"

cd "${RUTA_PROYECTO}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export PYTHONUNBUFFERED="1"

if [[ ! -x "${RUTA_PYTHON}" ]]; then
    echo "ERROR: No se encontro Python en ${RUTA_PYTHON}" >&2
    exit 1
fi

if [[ ! -f "${RUTA_SCRIPT}" ]]; then
    echo "ERROR: No se encontro ${RUTA_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${RUTA_PROYECTO}/config/videos.json" ]]; then
    echo "ERROR: No se encontro config/videos.json" >&2
    exit 1
fi

"${RUTA_PYTHON}" -m py_compile "${RUTA_SCRIPT}"
"${RUTA_PYTHON}" -m json.tool config/videos.json > /dev/null

echo "============================================================"
echo "METRICAS DE LA PRUEBA N2N Y N2V"
echo "============================================================"
echo "Video: ${VIDEO_ID}"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Nodo: ${SLURMD_NODENAME:-local}"
echo "CPU asignadas: ${SLURM_CPUS_PER_TASK:-8}"
echo "GPU asignada: ${CUDA_VISIBLE_DEVICES:-desconocida}"
echo "Inicio: $(date --iso-8601=seconds)"
echo "============================================================"

nvidia-smi

echo
echo "Verificando resultados registrados."

"${RUTA_PYTHON}" - "${VIDEO_ID}" <<'PY'
import json
import sys
from pathlib import Path

video_id = sys.argv[1]
raiz = Path.cwd()

with (raiz / "config" / "videos.json").open(
    "r",
    encoding="utf-8",
) as archivo:
    configuracion = json.load(archivo)

if video_id not in configuracion:
    raise RuntimeError(
        f"El video {video_id} no esta registrado."
    )

modelos = configuracion[video_id].get("modelos", {})

for modelo in ("n2n", "n2v"):
    if modelo not in modelos:
        raise RuntimeError(
            f"El modelo {modelo} no esta registrado."
        )

    datos = modelos[modelo]

    if datos.get("estado") != "completada":
        raise RuntimeError(
            f"El modelo {modelo} no tiene estado completada."
        )

    ruta = Path(datos["carpeta_salida"])

    if not ruta.is_absolute():
        ruta = raiz / ruta

    if not ruta.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de {modelo}: {ruta}"
        )

    cantidad = sum(
        1
        for archivo_resultado in ruta.iterdir()
        if archivo_resultado.is_file()
        and archivo_resultado.name.startswith("frame_")
        and archivo_resultado.suffix.lower() == ".png"
    )

    if cantidad != 100:
        raise RuntimeError(
            f"Se esperaban 100 frames de {modelo} "
            f"y se encontraron {cantidad}."
        )

    print(
        f"{modelo}: estado completada, "
        f"carpeta={ruta}, frames={cantidad}"
    )
PY

ejecutar_metricas() {
    local modelo="$1"
    local inicio
    local fin
    local duracion

    inicio="$(date +%s)"

    echo
    echo "------------------------------------------------------------"
    echo "CALCULANDO METRICAS: ${modelo}"
    echo "Inicio: $(date --iso-8601=seconds)"
    echo "------------------------------------------------------------"

    "${RUTA_PYTHON}" "${RUTA_SCRIPT}" \
        --video "${VIDEO_ID}" \
        --modelo "${modelo}"

    fin="$(date +%s)"
    duracion="$((fin - inicio))"

    echo "Metricas de ${modelo} terminadas."
    echo "Duracion: ${duracion} segundos"
}

ejecutar_metricas "n2n"
ejecutar_metricas "n2v"

if [[ ! -f "${RUTA_RESUMEN}" ]]; then
    echo "ERROR: No se genero ${RUTA_RESUMEN}" >&2
    exit 1
fi

echo
echo "============================================================"
echo "RESULTADOS GUARDADOS EN RESUMEN.XLSX"
echo "============================================================"

"${RUTA_PYTHON}" - "${RUTA_RESUMEN}" <<'PY'
import sys
from pathlib import Path

from openpyxl import load_workbook

ruta_resumen = Path(sys.argv[1])

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

    buscados = {
        "noise2noise",
        "noise2void",
    }

    encontrados = set()

    for fila in hoja.iter_rows(
        min_row=2,
        values_only=True,
    ):
        if not fila:
            continue

        modelo = fila[0]

        if modelo is None:
            continue

        modelo_normalizado = str(modelo).strip().lower()

        if modelo_normalizado not in buscados:
            continue

        encontrados.add(modelo_normalizado)

        print()
        print("=" * 70)
        print(modelo)
        print("=" * 70)

        for encabezado, valor in zip(
            encabezados,
            fila,
        ):
            print(f"{encabezado}: {valor}")

    faltantes = buscados - encontrados

    if faltantes:
        raise RuntimeError(
            "No se encontraron todas las filas esperadas: "
            + ", ".join(sorted(faltantes))
        )
finally:
    libro.close()
PY

echo
echo "============================================================"
echo "METRICAS COMPLETADAS"
echo "Resumen: ${RUTA_RESUMEN}"
echo "Fin: $(date --iso-8601=seconds)"
echo "============================================================"
