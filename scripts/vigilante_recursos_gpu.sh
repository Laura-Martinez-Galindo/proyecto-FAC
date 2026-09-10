#!/usr/bin/env bash

# ==============================================================================
# vigilante_recursos_gpu.sh
# Monitorea continuamente los nodos GPU de Hypatia (nodei-gpu-1, nodei-gpu-2).
# Cuando un nodo tenga >= 2 GPUs libres y >= 40 GB de RAM libres:
# 1. Calcula dinámicamente los recursos óptimos (RAM libre - 10GB, CPUs libres - 2).
# 2. Lanza inmediatamente jobs/preprocesamiento.sbatch para video2 en ese nodo.
# 3. Notifica e informa al correo del usuario y termina.
# ==============================================================================

set -euo pipefail

VIDEO_ID="${1:-video2}"
CORREO="lf.martinezg1@uniandes.edu.co"
INTERVALO_SEGUNDOS=30
MIN_GPUS=2
MIN_RAM_GB=40

echo "======================================================================"
echo "VIGILANTE DE RECURSOS GPU INICIADO"
echo "Objetivo: Detectar nodo con >= ${MIN_GPUS} GPUs y >= ${MIN_RAM_GB} GB RAM libres"
echo "Video a procesar: ${VIDEO_ID}"
echo "Correo destino: ${CORREO}"
echo "Frecuencia de escaneo: cada ${INTERVALO_SEGUNDOS} segundos"
echo "Inicio: $(date --iso-8601=seconds)"
echo "======================================================================"

while true; do
    FECHA_ACTUAL="$(date +'%Y-%m-%d %H:%M:%S')"
    NODO_ELEGIDO=""
    GPUS_LIBRES_ELEGIDO=0
    RAM_LIBRE_GB_ELEGIDO=0
    CPUS_LIBRES_ELEGIDO=0

    for NODO in nodei-gpu-1 nodei-gpu-2; do
        INFO_NODO="$(scontrol show node "${NODO}" 2>/dev/null || true)"
        if [[ -z "${INFO_NODO}" ]]; then
            continue
        fi

        # 1. CPUs
        CPU_TOT=$(echo "${INFO_NODO}" | grep -o 'CPUTot=[0-9]*' | head -1 | cut -d= -f2 || echo 0)
        CPU_ALLOC=$(echo "${INFO_NODO}" | grep -o 'CPUAlloc=[0-9]*' | head -1 | cut -d= -f2 || echo 0)
        CPU_LIBRES=$(( CPU_TOT - CPU_ALLOC ))

        # 2. Memoria RAM (en MB)
        REAL_MEM=$(echo "${INFO_NODO}" | grep -o 'RealMemory=[0-9]*' | head -1 | cut -d= -f2 || echo 0)
        ALLOC_MEM=$(echo "${INFO_NODO}" | grep -o 'AllocMem=[0-9]*' | head -1 | cut -d= -f2 || echo 0)
        RAM_LIBRE_MB=$(( REAL_MEM - ALLOC_MEM ))
        RAM_LIBRE_GB=$(( RAM_LIBRE_MB / 1024 ))

        # 3. GPUs
        GPU_TOT=$(echo "${INFO_NODO}" | grep -o 'Gres=gpu:[^ ]*' | head -1 | grep -o '[0-9]*$' || echo 3)
        if [[ -z "${GPU_TOT}" || "${GPU_TOT}" -eq 0 ]]; then
            GPU_TOT=3
        fi

        GPU_ALLOC=$(echo "${INFO_NODO}" | grep -o 'gres/gpu=[0-9]*' | head -1 | cut -d= -f2 || echo 0)
        if [[ -z "${GPU_ALLOC}" ]]; then
            GPU_ALLOC=0
        fi
        GPU_LIBRES=$(( GPU_TOT - GPU_ALLOC ))

        echo "[${FECHA_ACTUAL}] ${NODO} -> GPUs libres: ${GPU_LIBRES}/${GPU_TOT} | RAM libre: ${RAM_LIBRE_GB} GB | CPUs libres: ${CPU_LIBRES}"

        if [[ ${GPU_LIBRES} -ge ${MIN_GPUS} && ${RAM_LIBRE_GB} -ge ${MIN_RAM_GB} ]]; then
            NODO_ELEGIDO="${NODO}"
            GPUS_LIBRES_ELEGIDO="${GPU_LIBRES}"
            RAM_LIBRE_GB_ELEGIDO="${RAM_LIBRE_GB}"
            CPUS_LIBRES_ELEGIDO="${CPU_LIBRES}"
            break
        fi
    done

    if [[ -n "${NODO_ELEGIDO}" ]]; then
        echo
        echo "======================================================================"
        echo "¡RECURSOS DETECTADOS EN ${NODO_ELEGIDO}!"
        echo "GPUs libres: ${GPUS_LIBRES_ELEGIDO} | RAM libre: ${RAM_LIBRE_GB_ELEGIDO} GB | CPUs libres: ${CPUS_LIBRES_ELEGIDO}"
        echo "======================================================================"

        MEM_ASIGNAR=$(( RAM_LIBRE_GB_ELEGIDO - 10 ))
        if [[ ${MEM_ASIGNAR} -lt 32 ]]; then
            MEM_ASIGNAR=32
        fi

        CPUS_ASIGNAR=$(( CPUS_LIBRES_ELEGIDO - 2 ))
        if [[ ${CPUS_ASIGNAR} -lt 6 ]]; then
            CPUS_ASIGNAR=6
        fi
        if [[ ${CPUS_ASIGNAR} -gt 16 ]]; then
            CPUS_ASIGNAR=16
        fi

        echo "Lanzando preprocesamiento.sbatch con:"
        echo "  - Nodo: ${NODO_ELEGIDO}"
        echo "  - GPUs: 2"
        echo "  - Memoria: ${MEM_ASIGNAR}G"
        echo "  - CPUs: ${CPUS_ASIGNAR}"
        echo

        CMD="sbatch --nodelist=${NODO_ELEGIDO} --gres=gpu:2 --mem=${MEM_ASIGNAR}G --cpus-per-task=${CPUS_ASIGNAR} --mail-type=BEGIN,END,FAIL --mail-user=${CORREO} jobs/preprocesamiento.sbatch ${VIDEO_ID}"
        echo "Ejecutando: ${CMD}"
        JOB_OUTPUT=$(${CMD})
        echo "${JOB_OUTPUT}"

        echo
        echo "======================================================================"
        echo "Trabajo enviado satisfactoriamente. El vigilante termina su ejecucion."
        echo "Slurm enviara la notificacion oficial por correo a ${CORREO} al iniciar."
        echo "======================================================================"
        exit 0
    fi

    sleep "${INTERVALO_SEGUNDOS}"
done
