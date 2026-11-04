# Plan de denoising: de los videos a `dataset_denoised`

Objetivo: entregar tres versiones del MISMO dataset (mismos nombres, mismas etiquetas):
`dataset_original` (listo) → `dataset_sin_hud` (job listo) → `dataset_denoised` (este plan).
Las particiones train/val/test NO son parte de este plan: las define otro integrante sobre estos datasets.

## Paso 1. Régimen de cada frame (sensor + filtros), leído del HUD

`sbatch jobs/detectar_regimenes.sbatch` → `videos/<video>/regimenes_<video>.csv` (todos los frames,
1 de cada 15) y `datasets/dataset_original/regimenes_dataset.csv` (las 1.116 imágenes).
Se guarda como **tabla por frame**: `video, frame, regimen, sensor, filtros, foco, exposicion`.

Cómo sabemos que no falta ningún régimen: el detector compara cada línea del HUD con plantillas y,
si una línea no se parece a ninguna, la marca DESCONOCIDO y guarda el recorte en `*_dudosos/`.
Así aparecieron `HDIR C WH DDE` (video2) y `SWIR DDE` (video3). Se revisan los dudosos, se agrega
la plantilla (`config/ejemplos_plantillas_hud.csv`) y se vuelve a correr hasta que no queden nuevos.

Regímenes encontrados en el dataset (lectura local, 29 ilegibles pendientes de la tabla de video):

| Régimen | Video | Imágenes | Ruido esperado |
|---|---|---|---|
| HDEO_DDE_HAZE | video2 | 820 | óptico, grumos por DDE + HAZE + H.264 |
| HDEO_DDE | video3 | 162 | óptico, grumos por DDE |
| HDEO_HAZE | video1 | 73 | óptico, amplificado en sombras |
| HDIR_C_WH | video1 | 15 | térmico, rayas (FPN) |
| HDLL_DDE | video2 | 13 | baja luz, mucho ruido de disparo + DDE |
| HDIR_C_WH_DDE | video2 | 3 | térmico + DDE |
| SWIR_DDE | video3 | 1 | infrarrojo onda corta + DDE |

## Paso 2. Familias de entrenamiento

No se encadenan modelos "uno por filtro": el ruido de DDE + HAZE no es la suma de dos ruidos
independientes. Se entrena **un modelo por familia** (combinación que cambia el tipo de ruido) y se
evalúa por régimen:

| Familia | Regímenes | Por qué juntos |
|---|---|---|
| OPTICO_DDE | HDEO_DDE_HAZE, HDEO_DDE, HDLL_DDE, SWIR_DDE | DDE domina: ruido en grumos |
| OPTICO | HDEO_HAZE | ruido de sensor sin realce de bordes |
| TERMICO | HDIR_C_WH, HDIR_C_WH_DDE | rayas de columna + ruido térmico, 1 canal |

Si en la ronda 1 una familia muestra diferencias grandes entre sus regímenes, se separa.

## Paso 3. Datos de los experimentos (lo mínimo suficiente)

* **Entrenamiento:** hasta 1.500 frames centrales por familia, tomados de los videos completos
  (no del dataset), repartidos en todos sus tramos. Se **excluyen ±90 frames (3 s) alrededor de cada
  imagen del dataset** (`--excluir-csv datasets/dataset_original/correspondencia.csv`). Así el
  denoiser nunca ve los frames que va a limpiar: no depende de cómo se haga el split después.
* **Evaluación:** las imágenes del dataset de esa familia (todas las de las familias chicas; 120 al
  azar de OPTICO_DDE en la ronda 1).

## Paso 4. Candidatos por familia (ronda 1)

| Familia | Candidatos |
|---|---|
| OPTICO_DDE | UDVD oficial (5 frames), AP-BSN (pensado para ruido correlacionado), Blind2Unblind, NLM (clásico, línea base) |
| OPTICO | UDVD oficial, Blind2Unblind, Neighbor2Neighbor, NLM |
| TERMICO | UDVD oficial 1 canal, StructN2V vertical (rayas), destriping + NLM |

Mismo presupuesto para todos (p.ej. 4.000 pasos). Con 2 GPUs se corren 2 entrenamientos a la vez.
Ronda 2: solo el ganador de cada familia, barrido corto de lr {1e-4, 3e-4}, parche {96, 128},
pasos {4k, 8k}. Todos los scripts reciben esos valores por parámetro y guardan su configuración
junto al checkpoint (`cache/<modelo>/<familia>/<experimento>/modelo.pt + configuracion.json`).

## Paso 5. Métricas (`scripts/evaluar_denoising.py`)

Por imagen, y resumidas con percentiles (p05, p25, p50, p75, p95) y gráficas de distribución:
`sigma_ruido` (Immerkær, sin bordes), `ruido_quitado`, `retencion_bordes`, `corr_residuo_bordes`
(¿lo que se quitó tiene forma de la escena?), `autocorr_residuo`, y opcionalmente NIQE/BRISQUE.
Ranking por familia: descarta si `retencion_bordes` p25 < 0,8 o `corr_residuo_bordes` p50 > 0,15,
y ordena los que quedan por `sigma_ruido` mediano.

## Paso 6. Generar `dataset_denoised`

Para cada imagen: régimen → familia → checkpoint ganador → limpiar. Los modelos de video (UDVD)
necesitan los frames vecinos ±2 del mismo video; los de imagen (B2U, AP-BSN, StructN2V) solo la
imagen. Mismo nombre y etiquetas que `dataset_original`, con `correspondencia.csv` + columnas
`familia, modelo, checkpoint`.

## Paso 7. Impacto en YOLO

Lo corre quien haga los splits, con las tres versiones y la misma partición.
