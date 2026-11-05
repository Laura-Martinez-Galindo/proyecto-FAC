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

Resultado en Hypatia (job 791539, 0 frames sin resolver). Revisión visual de recortes de cada régimen
raro: `SWIR C WH DDE` (video2) es real (1 lectura); en video3 el térmico dice **`HDIR M WH DDE` con
`EXP HST`**, que no tenía plantilla y se leía como `C WH DDE` + exposición desconocida. Se agregaron
las dos plantillas (`config/ejemplos_hud/`, ahora autocontenidas: solo el recorte del HUD) y
`jobs/denoising_preparar.sbatch` vuelve a leer todo antes de la ronda 1.

| Régimen | Video | Frames del video | Libres (±90 fuera) | Imágenes dataset | Ruido esperado |
|---|---|---|---|---|---|
| HDEO_DDE_HAZE | video2 | 69.344 | 41.171 | 823 | óptico, grumos por DDE + HAZE + H.264 |
| HDEO_DDE | video3 | 20.800 | 10.125 | 162 | óptico, grumos por DDE |
| HDEO_DDE | video2 | 3.555 | 3.555 | 0 | óptico, grumos por DDE |
| HDLL_DDE | video2 | 7.392 | 6.842 | 13 | baja luz, mucho ruido de disparo + DDE |
| SWIR_DDE | video3 / video2 | 615 / 24 | 368 / 12 | 1 / 0 | infrarrojo onda corta + DDE |
| HDEO_HAZE | video1 | 17.889 | 11.847 | 97 | óptico, amplificado en sombras |
| HDIR_C_WH | video1 | 1.463 | 831 | 17 | térmico, rayas (FPN) |
| HDIR_C_WH_DDE | video2 | 1.149 | 1.110 | 3 | térmico + DDE |
| HDIR_M_WH_DDE | video3 | 1.533 | 1.533 | 0 | térmico + DDE (otro modo de AGC) |
| HDEO_C_WH, SWIR_C_WH_DDE | v1, v2 | 30, 15 | — | 0 | lecturas sueltas: no se usan |

"Frames del video" cuenta solo frames cuya lectura anterior y siguiente del HUD coinciden y con
vecinos ±2 del mismo régimen (no se usan las transiciones).

## Paso 2. Familias de entrenamiento

No se encadenan modelos "uno por filtro": el ruido de DDE + HAZE no es la suma de dos ruidos
independientes. Se entrena **un modelo por familia** (combinación que cambia el tipo de ruido) y se
evalúa por régimen:

| Familia | Regímenes | Por qué juntos |
|---|---|---|
| OPTICO_DDE | HDEO_DDE_HAZE, HDEO_DDE, HDLL_DDE, SWIR_DDE | DDE domina: ruido en grumos |
| OPTICO | HDEO_HAZE | ruido de sensor sin realce de bordes |
| TERMICO | HDIR_C_WH, HDIR_C_WH_DDE, HDIR_M_WH_DDE | rayas de columna + ruido térmico, 1 canal |

Definidas en `config/denoising/familias.json`. Si una imagen del dataset tiene un régimen que no está
en ninguna familia, `listas_denoising.py` lo avisa. Si en la ronda 1 una familia muestra diferencias
grandes entre sus regímenes, se separa.

## Paso 3. Datos de los experimentos (lo mínimo suficiente) — `scripts/listas_denoising.py`

* **Entrenamiento:** 1.500 frames por familia de los videos completos (no del dataset). Cada régimen
  recibe su parte proporcional pero al menos 150; dentro del régimen se toman equiespaciados (quedan
  en todos sus tramos). Se **excluyen ±90 frames (3 s) alrededor de cada imagen del dataset**: el
  denoiser nunca ve los frames que va a limpiar, sin importar cómo se haga el split después.
  Se leen los frames originales a resolución completa y los parches se sortean **fuera del HUD**
  (≥ 8 px de la máscara), así la red no aprende ni el texto ni el relleno de ProPainter.
* **Evaluación:** imágenes del dataset (sin HUD) de la familia: todas en OPTICO (97) y TERMICO (20);
  120 estratificadas en OPTICO_DDE (77 HDEO_DDE_HAZE, 29 HDEO_DDE, 13 HDLL_DDE, 1 SWIR_DDE).
* Salidas: `resultados/denoising/listas/<FAMILIA>_{entrenamiento,evaluacion,todas}.csv` + `resumen.csv`.

## Paso 4. Ronda 1 — `config/denoising/ronda1.tsv` (una fila = un experimento)

Todos los métodos están en `scripts/denoising.py` con la misma interfaz (`entrenar` / `inferir`) y
todos sus hiperparámetros por parámetro (`python scripts/denoising.py entrenar -h`). Redes copiadas
de los repos oficiales con su licencia (`scripts/fac_denoise/licencias/`); pérdidas reimplementadas
siguiendo el código oficial.

| Familia | Experimentos |
|---|---|
| OPTICO_DDE | UDVD oficial (5 frames), AP-BSN (config. SIDD: ruido real correlacionado), Blind2Unblind, NLM h=4 y h=8 |
| OPTICO | UDVD oficial, Blind2Unblind, Neighbor2Neighbor, NLM h=4 y h=8 |
| TERMICO | UDVD 1 canal, StructN2V (máscara vertical de 9 px, para rayas de columna), NLM h=6, destriping + NLM h=6 |

Mismo presupuesto: 4.000 pasos (copia en 2.000), parche/batch/lr del artículo de cada método. Si el
batch no cabe en la GPU se baja a la mitad automáticamente (queda en `configuracion.json`).
Cada experimento deja `cache/denoising/<FAMILIA>/<experimento>/`: `modelo.pt`, `modelo_p2000.pt`,
`configuracion.json`, `entrenamiento.csv` (curva de pérdida) y `salidas/` (+ `salidas/residuo/`).

Lanzar todo, en orden y con dependencias: `bash jobs/lanzar_ronda1.sh` (ver el encabezado del script).

Ronda 2: solo el ganador de cada familia, barrido corto de lr {1e-4, 3e-4}, parche {96, 128},
pasos {4k, 8k}: es otra tabla `ronda2.tsv` con el mismo lanzador (`TABLA=...`).

## Paso 5. Métricas (`scripts/evaluar_denoising.py`, job `dn_evaluar`)

Por imagen, y resumidas con percentiles (p05, p25, p50, p75, p95) y gráficas de distribución:
`sigma_ruido` (Immerkær, sin bordes), `ruido_quitado`, `retencion_bordes`, `corr_residuo_bordes`
(¿lo que se quitó tiene forma de la escena?), `autocorr_residuo`, y opcionalmente NIQE/BRISQUE.
Ranking por familia: descarta si `retencion_bordes` p25 < 0,8 o `corr_residuo_bordes` p50 > 0,15,
y ordena los que quedan por `sigma_ruido` mediano. Los píxeles del HUD se excluyen
(`datasets/dataset_sin_hud/mascaras_hud`). Resultado por familia en `resultados/denoising/ronda1/<FAMILIA>/`
y todo junto en `resultados/denoising/ronda1.tar` para revisarlo en el PC.

## Paso 6. Generar `dataset_denoised`

Para cada imagen: régimen → familia → checkpoint ganador → limpiar. Los modelos de video (UDVD)
necesitan los frames vecinos ±2 del mismo video; los de imagen (B2U, AP-BSN, StructN2V) solo la
imagen. Mismo nombre y etiquetas que `dataset_original`, con `correspondencia.csv` + columnas
`familia, modelo, checkpoint`.

## Paso 7. Impacto en YOLO

Lo corre quien haga los splits, con las tres versiones y la misma partición.
