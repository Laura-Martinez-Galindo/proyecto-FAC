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

## Paso 2. Familias por régimen (decididas midiendo el ruido)

`scripts/caracterizar_ruido.py` (job `caracterizar_ruido.sbatch`) midió 60 frames de cada régimen en
cada video: nivel de ruido, correlación del ruido con el vecino a 1 y 2 px y por dirección, rayas,
% de sal y pimienta, bloques de compresión y espectro (`resultados/ruido_regimenes/`). Las familias
se definen por **régimen del HUD** (así un video nuevo se asigna leyendo el HUD) agrupando los que
tienen el mismo **tipo** de ruido, en línea con la investigación de ruido por régimen:

| Familia | Regímenes | Imágenes | Tipo de ruido medido |
|---|---|---|---|
| OPTICO_CORRELADO | HDEO_HAZE, HDEO_DDE, SWIR_DDE | 260 | en grumos (corr. 0,4-0,55 a 1 px, 0,1-0,35 a 2 px), sin impulsos; bloques en video3 |
| HAZE_DDE | HDEO_DDE_HAZE | 823 | grano fino (corr. ~0 a 2 px) + 3,6 % sal y pimienta: la "cascada" HAZE→DDE |
| BAJA_LUZ | HDLL_DDE | 13 | anisotrópico: correlado en vertical (0,48), no en horizontal (0,05) |
| TERMICO | HDIR_C_WH, HDIR_C_WH_DDE, HDIR_M_WH_DDE | 20 | 1 canal; rayas horizontales y 13 % sal y pimienta en video2 |

Hallazgos: el mismo régimen puede tener distinto **nivel** de ruido según el video (HDEO_DDE: 4,5×
más en video2), por eso la familia se define por tipo y no por nivel; HAZE solo no es ruidoso, HAZE+DDE
sí; SWIR no mostró píxeles defectuosos; video3 tiene los bloques de compresión más marcados
(perfil Baseline). Con 3 videos no se puede separar del todo el efecto del filtro del efecto del video.
Definidas en `config/denoising/familias.json`.

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
todos sus hiperparámetros por parámetro. Modelos elegidos según el tipo de ruido (matriz de decisión
de la investigación de denoising autosupervisado):

| Familia | Experimentos | Por qué |
|---|---|---|
| OPTICO_CORRELADO | UDVD, AP-BSN, NLM | ruido en grumos: AP-BSN rompe la correlación con PD; UDVD usa los frames vecinos |
| HAZE_DDE | UDVD, AP-BSN, Neighbor2Neighbor (L1), Blind2Unblind (L1), mediana+NLM | grano casi independiente + impulsos: L1 converge a la mediana |
| BAJA_LUZ | UDVD, StructN2V vertical, Neighbor2Neighbor, NLM | ruido en una dirección: máscara orientada |
| TERMICO | UDVD 1 canal, StructN2V horizontal, destriping horizontal + NLM | rayas horizontales |

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

## Ronda 2 (7 nov 2026): solo HDEO_DDE_HAZE (video2)

Mirando recortes ampliados de la ronda 1 (`_claude_informe/zoom_*.png`):

* **Video1 (HDEO_HAZE, sigma 0,4) y video3 (HDEO_DDE, sigma 0,24) van SIN denoising.** Su ruido esta
  por debajo del nivel de cuantizacion de 8 bits (~0,3): no se ve. Lo que tienen son bloques de
  compresion H.264, y las redes que los quitan (AP-BSN, MM-BSN) tambien desenfocan detalle real.
* **Video2 (74 % del dataset) si tiene ruido visible** (grano + sal y pimienta). AP-BSN lo limpia
  (grano 6,7 -> 1,5) pero corre el color (croma 0,82); NB2NB y B2U casi no limpiaron y lavaron el
  color; UDVD mete una rejilla de cuadritos.

Cambios:

1. `scripts/color_original.py`: brillo (Y) de la red + color (Cr, Cb) de la entrada suavizado. La
   evaluacion crea `<exp>_ycolor` de todos los experimentos automaticamente.
2. `scripts/evaluar_denoising.py`, metricas nuevas: `grano` (el ruido de 2-4 px que si se ve; el
   ranking ordena por esta), `croma_ratio`, `delta_e_suave`, `bloques_8px` y
   `corr_residuo_temporal` (ruido vs textura con el frame vecino alineado por flujo optico).
   Los bordes se calculan sobre la imagen con mediana 5x5 (el grano ya no cuenta como borde).
   El ranking descarta solo por color; retencion y corr. con bordes quedan como avisos.
3. `config/denoising/ronda2.tsv`: MM-BSN, AP-BSN con 8000 pasos, NB2NB y B2U sin AMP, Mediana+NLM h12.

```
RONDA=ronda2 TABLA=config/denoising/ronda2.tsv SOLO=HDEO_DDE_HAZE PREPARAR=0 bash jobs/lanzar_ronda1.sh
```

## Ronda 3 (11 nov 2026): todos los metodos, por regimen (fase 1 = seleccion)

Objetivo (propuesta, objetivo especifico 2): metodos autosupervisados espaciales y temporales, con
reduccion >= 8 % de NIQE y BRISQUE frente al video sin procesar. Se elige el mejor modelo DE CADA
regimen por separado (HDEO_DDE_HAZE, HDEO_HAZE, HDEO_DDE).

* **Fase 1 (esta ronda):** cada metodo se entrena con la misma muestra de frames de su regimen
  (1.500), los mismos 4.000 pasos, sin AMP, con los hiperparametros de su articulo, y se evalua en
  hasta 300 imagenes del dataset por regimen. Ranking por % de reduccion de NIQE y BRISQUE.
* **Fase 2:** los 3 mejores de cada regimen (y sus combinaciones: ensembles y stacks) se
  reentrenan con TODOS los frames del regimen (incluidos los del dataset; en autosupervisado es
  valido) y limpian el dataset completo -> dataset_denoised.

Tanda 1 (config/denoising/ronda3.tsv, 60 experimentos = 20 metodos x 3 regimenes):
* Codigo oficial (scripts/fac_denoise/terceros): PUCA, LG-BPN, U-BSN, TBSN; ya integrados: AP-BSN,
  MM-BSN, UDVD, NB2NB, B2U.
* Reimplementados del articulo: Noise2Self, Noise2Same, R2R (sigma estimada), ZS-N2N, N2V, StructN2V.
* Clasicos: BM3D (paquete oficial), filtro guiado, mediana adaptativa, NLM, mediana+NLM.

Tanda 2 (scripts/fac_denoise/metodos_tanda2.py, 30 experimentos mas): CVF-SID, Noise2Score, N2V2,
Self2Self y Cut2Self (red de referencia), GR2R, Positive2Negative* (ajusta la UNet de NB2NB del mismo
regimen) y, zero-shot (una red por imagen, evaluados en 100 o 30 imagenes): ZS-N2N, Noise2Detail y DIP.
Pendientes de la tanda 2: SASS, MASH*, TM-BSN, NSP, ZS-KAN*, SelfFormer*, SwinIA*, Filter2Noise,
Self2Self+; PN2V no aplica (modelo de ruido por histograma para imagenes en gris de microscopia).
Tanda 3 (pendiente, temporales): MF2F, TAP, STBN, PCST, RDRF*, SAVeD, F2R (con guia autosupervisada).
(* = sin codigo publico: reimplementacion desde el articulo.) Metricas temporales (warping error,
uPSNR) en la tanda 3. Los ensembles y stacks se arman sobre los ganadores.

Uso en Hypatia:
```
bash jobs/preparar_entorno_ronda3.sh      # una vez, en login: instala einops/bm3d/pyiqa y baja pesos
HUMO=1 bash jobs/lanzar_ronda3.sh         # prueba de humo (~1-2 h): que todo carga y corre
bash jobs/lanzar_ronda3.sh                # la ronda; si no alcanza el tiempo, relanzar igual
```
