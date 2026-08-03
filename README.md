# Proyecto FAC

Pipeline reproducible para extracción de frames, eliminación de HUD y restauración Noise2Noise específica por video.

## Estructura principal

```text
proyecto-FAC/
├── fac_env311/                    # Entorno Python local, ignorado por Git
├── Jobs/
│   ├── logs/
│   ├── limpiar_hud_job.sh
│   ├── pipeline_n2n_original.sh
│   └── pipeline_n2n_sin_hud.sh
├── ProPainter/                    # Dependencia pesada, ignorada por Git
├── Scripts/
│   ├── pipeline_n2n.py
│   ├── limpiar_hud.py
│   ├── extraer_frames.py
│   └── otros scripts del proyecto
├── Videos/
│   └── video30min-11to22/
│       ├── frames/
│       ├── frames_sin_hud/
│       ├── masks/
│       ├── n2n/
│       │   ├── train/input y train/target
│       │   ├── valid/input y valid/target
│       │   ├── clean/
│       │   ├── parejas.csv
│       │   ├── modelo.ckpt
│       │   └── metricas_frames.csv
│       ├── n2n_sin_hud/
│       └── resumen.xlsx
├── .gitignore
├── README.md
└── main.ipynb
```

## Entorno Python

Los jobs no dependen de que el entorno esté activado en la terminal. Usan directamente:

```text
/hpcfs/home/ing_sistemas/lf.martinezg1/proyecto-FAC/fac_env311/bin/python
```

Para una sesión interactiva:

```bash
source fac_env311/bin/activate
```

## Etapas internas del pipeline

Los archivos `.sh` ejecutan automáticamente estas fases del mismo script:

1. `preparar`: crea parejas consecutivas, excluye transiciones y movimiento extremo, y divide 95 % train y 5 % valid.
2. `entrenar`: entrena con dos GPU y actualiza `modelo.ckpt` al finalizar cada época.
3. `inferir`: inicia un proceso nuevo con una GPU, cero workers y procesa un frame por llamada.
4. `metricas`: calcula métricas antes/después y actualiza `resumen.xlsx`.

## Revisar recursos

```bash
for nodo in nodei-gpu-1 nodei-gpu-2; do
    echo "================ ${nodo} ================"
    scontrol show node "${nodo}" | grep -E \
    'NodeName=|State=|CPUAlloc=|CPUTot=|RealMemory=|AllocMem=|FreeMem=|CfgTRES=|AllocTRES=|Gres='
done
```

Cada job solicita dos GPU, 16 CPU y 150 GB de RAM.

## Validar antes de ejecutar

Desde la raíz de `proyecto-FAC`:

```bash
fac_env311/bin/python -m py_compile Scripts/pipeline_n2n.py
bash -n Jobs/pipeline_n2n_original.sh
bash -n Jobs/pipeline_n2n_sin_hud.sh
sbatch --test-only Jobs/pipeline_n2n_original.sh
sbatch --test-only Jobs/pipeline_n2n_sin_hud.sh
```

## Ejecutar original y después sin HUD

```bash
JOB_ORIGINAL=$(sbatch --parsable Jobs/pipeline_n2n_original.sh)
JOB_SIN_HUD=$(sbatch --dependency="afterok:${JOB_ORIGINAL}" --parsable Jobs/pipeline_n2n_sin_hud.sh)

printf "JOB_ORIGINAL=%s\nJOB_SIN_HUD=%s\n" \
"${JOB_ORIGINAL}" "${JOB_SIN_HUD}" \
> Jobs/jobs_n2n_actuales.txt

squeue -j "${JOB_ORIGINAL},${JOB_SIN_HUD}" \
-o "%.12i %.18j %.10T %.12M %.35R"
```

## Seguir la salida

```bash
source Jobs/jobs_n2n_actuales.txt

tail -n 100 -F \
"Jobs/logs/n2n_original-log.o${JOB_ORIGINAL}" \
"Jobs/logs/n2n_original-err.o${JOB_ORIGINAL}"
```

## Resultados

`parejas.csv` documenta el frame input, frame target, el TIFF compartido por CAREamics, el conjunto, diferencia visual, movimiento y exclusiones.

`resumen.xlsx` contiene cuatro filas cuando ambos experimentos terminan:

1. `frames_original`
2. `frames_n2n_original`
3. `frames_sin_hud`
4. `frames_n2n_sin_hud`

Las flechas de los encabezados indican la dirección deseable. NIQE y BRISQUE más bajos suelen ser mejores. La retención de nitidez se interpreta mejor cerca de 1. Las métricas sin referencia no demuestran por sí solas que todo el residual eliminado sea ruido.
