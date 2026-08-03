# %% 0. Imports y configuración
import os
import re
import shutil

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from tqdm import tqdm


# Ruta principal del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parent.parent

# Ruta del video asociado con los frames
RUTA_VIDEO = RUTA_PROYECTO / "Videos" / "video30min-11to22.mp4"

# Modo de construcción del dataset
# "original" utiliza Videos/<video>/frames y genera Videos/<video>/n2n
# "sin_hud" utiliza Videos/<video>/frames_sin_hud y genera Videos/<video>/n2n_sin_hud
MODO = "sin_hud"

# Carpetas de análisis y exportación
CARPETA_VIDEO = RUTA_VIDEO.parent / RUTA_VIDEO.stem
CARPETA_ANALISIS = CARPETA_VIDEO / "frames"
CARPETA_FUENTE = CARPETA_VIDEO / ("frames" if MODO == "original" else "frames_sin_hud")
CARPETA_SALIDA = CARPETA_VIDEO / ("n2n" if MODO == "original" else "n2n_sin_hud")
CARPETA_TEMPORAL = CARPETA_VIDEO / f".{CARPETA_SALIDA.name}_temporal"

# Proporciones objetivo
PROPORCION_TRAIN = 0.80
PROPORCION_VALID = 0.10
PROPORCION_TEST = 0.10

# Semilla reproducible
SEMILLA = 42

# Resolución máxima usada solamente durante el análisis
LADO_ANALISIS = 480
MARGEN_ROI = 0.08

# Histogramas utilizados para comparar frames y escenas
BINS_H = 32
BINS_S = 32
BINS_GRIS = 64

# Detección adaptativa de cortes
FACTOR_MAD_CORTE = 6.0
UMBRAL_MINIMO_HS = 0.30
UMBRAL_MINIMO_INTENSIDAD = 0.12
UMBRAL_MINIMO_PUNTAJE = 3.5

# Un corte debe presentar cambio relevante en color o intensidad
PESO_HS = 0.65
PESO_INTENSIDAD = 0.35

# Evita crear escenas extremadamente cortas
MIN_FRAMES_ESCENA = 8
MIN_FRAMES_ENTRE_CORTES = 8

# Frames excluidos alrededor de cada corte
MARGEN_TRANSICION_ANTES = 2
MARGEN_TRANSICION_DESPUES = 2

# Escenas con menos frames válidos no producen parejas
MIN_FRAMES_VALIDOS_ESCENA = 2

# Agrupamiento de escenas visualmente similares
# Un valor menor crea más grupos; uno mayor une más escenas
UMBRAL_AGRUPAMIENTO_ESCENAS = 0.23

# Cantidad máxima de frames muestreados para describir cada escena
FRAMES_DESCRIPTOR_ESCENA = 12

# Configuración TIFF
COMPRESION_TIFF = "deflate"

# Cantidad de trabajadores
# En SLURM toma automáticamente --cpus-per-task
NUM_TRABAJADORES = int(
    os.environ.get("SLURM_CPUS_PER_TASK")
    or os.environ.get("SLURM_CPUS_ON_NODE")
    or min(16, os.cpu_count() or 1)
)

# Control de resultados anteriores
SOBRESCRIBIR_DATASET = True


# %% 1. Funciones auxiliares
def clave_natural(ruta):
    return [int(parte) if parte.isdigit() else parte.lower() for parte in re.split(r"(\d+)", ruta.name)]


def mediana_mad(valores):
    valores = np.asarray(valores, dtype=np.float64)
    valores = valores[np.isfinite(valores)]

    if valores.size == 0:
        return 0.0, 0.0

    mediana = float(np.median(valores))
    mad = float(np.median(np.abs(valores - mediana)))

    return mediana, mad


def normalizar_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norma = float(np.linalg.norm(vector))

    return vector / norma if norma > 0 else vector


def preparar_frame(ruta_frame):
    frame = cv2.imread(str(ruta_frame), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"No se pudo leer el frame: {ruta_frame}")

    alto, ancho = frame.shape[:2]
    escala = min(1.0, LADO_ANALISIS / max(alto, ancho))

    if escala < 1.0:
        frame = cv2.resize(frame, (round(ancho * escala), round(alto * escala)), interpolation=cv2.INTER_AREA)

    alto, ancho = frame.shape[:2]
    margen_y = min(round(alto * MARGEN_ROI), max(0, alto // 2 - 1))
    margen_x = min(round(ancho * MARGEN_ROI), max(0, ancho // 2 - 1))

    if margen_y > 0 and margen_x > 0:
        frame = frame[margen_y:alto - margen_y, margen_x:ancho - margen_x]

    return frame


def calcular_descriptor_frame(argumentos):
    posicion, ruta_frame = argumentos
    frame = preparar_frame(ruta_frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    histograma_hs = cv2.calcHist([hsv], [0, 1], None, [BINS_H, BINS_S], [0, 180, 0, 256]).astype(np.float32)
    histograma_gris = cv2.calcHist([gris], [0], None, [BINS_GRIS], [0, 256]).astype(np.float32)
    cv2.normalize(histograma_hs, histograma_hs, alpha=1.0, norm_type=cv2.NORM_L1)
    cv2.normalize(histograma_gris, histograma_gris, alpha=1.0, norm_type=cv2.NORM_L1)

    return {
        "posicion": posicion,
        "frame": ruta_frame.name,
        "histograma_hs": histograma_hs,
        "histograma_gris": histograma_gris,
        "intensidad_media": float(gris.mean()),
        "contraste": float(gris.std()),
        "fraccion_negra": float((gris <= 10).mean()),
        "fraccion_blanca": float((gris >= 245).mean()),
    }


def calcular_cambio_consecutivo(argumentos):
    posicion, descriptor_anterior, descriptor_actual = argumentos
    distancia_hs = float(
        cv2.compareHist(
            descriptor_anterior["histograma_hs"],
            descriptor_actual["histograma_hs"],
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )
    distancia_gris = float(
        cv2.compareHist(
            descriptor_anterior["histograma_gris"],
            descriptor_actual["histograma_gris"],
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )
    diferencia_intensidad = abs(
        descriptor_actual["intensidad_media"] - descriptor_anterior["intensidad_media"]
    ) / 255.0

    return {
        "posicion": posicion,
        "frame_anterior": descriptor_anterior["frame"],
        "frame": descriptor_actual["frame"],
        "distancia_hs": distancia_hs,
        "distancia_gris": distancia_gris,
        "diferencia_intensidad": float(diferencia_intensidad),
    }


def construir_escenas(rutas_frames, df_diagnostico):
    cortes = df_diagnostico.index[df_diagnostico["corte_escena"]].tolist()
    cortes_filtrados = []
    ultimo_corte = 0

    for corte in cortes:
        if corte < MIN_FRAMES_ESCENA:
            continue

        if len(rutas_frames) - corte < MIN_FRAMES_ESCENA:
            continue

        if corte - ultimo_corte < MIN_FRAMES_ENTRE_CORTES:
            continue

        cortes_filtrados.append(corte)
        ultimo_corte = corte

    limites = [0, *cortes_filtrados, len(rutas_frames)]
    escenas = []

    for numero, (inicio, fin) in enumerate(zip(limites[:-1], limites[1:]), start=1):
        indices_validos = list(range(inicio, fin))

        if inicio > 0:
            indices_validos = [
                indice
                for indice in indices_validos
                if indice >= inicio + MARGEN_TRANSICION_DESPUES
            ]

        if fin < len(rutas_frames):
            indices_validos = [
                indice
                for indice in indices_validos
                if indice < fin - MARGEN_TRANSICION_ANTES
            ]

        escenas.append(
            {
                "escena": f"escena_{numero:04d}",
                "indice_inicial": inicio,
                "indice_final": fin - 1,
                "frame_inicial": rutas_frames[inicio].name,
                "frame_final": rutas_frames[fin - 1].name,
                "cantidad_frames": fin - inicio,
                "indices_validos": indices_validos,
                "cantidad_frames_validos": len(indices_validos),
                "cantidad_parejas": max(0, len(indices_validos) - 1),
            }
        )

    return escenas, cortes_filtrados


def describir_escena(escena, descriptores):
    indices = escena["indices_validos"]

    if not indices:
        indices = list(range(escena["indice_inicial"], escena["indice_final"] + 1))

    cantidad = min(FRAMES_DESCRIPTOR_ESCENA, len(indices))
    posiciones = np.linspace(0, len(indices) - 1, cantidad).round().astype(int)
    seleccion = [indices[posicion] for posicion in np.unique(posiciones)]
    histogramas_hs = np.stack([descriptores[indice]["histograma_hs"].ravel() for indice in seleccion])
    histogramas_gris = np.stack([descriptores[indice]["histograma_gris"].ravel() for indice in seleccion])
    descriptor_hs = normalizar_vector(np.mean(histogramas_hs, axis=0))
    descriptor_gris = normalizar_vector(np.mean(histogramas_gris, axis=0))

    return normalizar_vector(np.concatenate([descriptor_hs * 0.75, descriptor_gris * 0.25]))


def agrupar_escenas(escenas, descriptores):
    escenas_validas = [
        escena
        for escena in escenas
        if escena["cantidad_frames_validos"] >= MIN_FRAMES_VALIDOS_ESCENA
    ]

    if not escenas_validas:
        raise RuntimeError("No quedaron escenas con suficientes frames válidos")

    matriz = np.stack([describir_escena(escena, descriptores) for escena in escenas_validas])

    if len(escenas_validas) == 1:
        etiquetas = np.ones(1, dtype=int)
    else:
        distancias = pdist(matriz, metric="cosine")

        if not np.all(np.isfinite(distancias)):
            raise RuntimeError("Se encontraron distancias no finitas al agrupar escenas")

        arbol = linkage(distancias, method="average")
        etiquetas = fcluster(arbol, t=UMBRAL_AGRUPAMIENTO_ESCENAS, criterion="distance")

    for escena, grupo in zip(escenas_validas, etiquetas):
        escena["grupo_visual"] = int(grupo)

    return escenas_validas


def asignar_grupos(escenas):
    grupos = {}

    for escena in escenas:
        grupo = escena["grupo_visual"]
        grupos.setdefault(grupo, []).append(escena)

    resumen_grupos = []

    for grupo, escenas_grupo in grupos.items():
        parejas = sum(escena["cantidad_parejas"] for escena in escenas_grupo)
        frames_validos = sum(escena["cantidad_frames_validos"] for escena in escenas_grupo)
        peso = max(parejas, frames_validos)

        resumen_grupos.append({
            "grupo_visual": grupo,
            "escenas": escenas_grupo,
            "cantidad_escenas": len(escenas_grupo),
            "cantidad_parejas": parejas,
            "cantidad_frames_validos": frames_validos,
            "peso": peso,
        })

    conjuntos = ("train", "valid", "test")
    proporciones = {
        "train": PROPORCION_TRAIN,
        "valid": PROPORCION_VALID,
        "test": PROPORCION_TEST,
    }
    peso_total = sum(grupo["peso"] for grupo in resumen_grupos)

    if peso_total <= 0:
        raise RuntimeError("Los grupos visuales no contienen frames ni parejas utilizables")

    objetivos = {
        conjunto: peso_total * proporciones[conjunto]
        for conjunto in conjuntos
    }

    generador = np.random.default_rng(SEMILLA)
    mejor_asignacion = None
    mejor_acumulados = None
    mejor_error = np.inf

    def calcular_error(acumulados):
        errores_relativos = [
            abs(acumulados[conjunto] - objetivos[conjunto]) / max(objetivos[conjunto], 1.0)
            for conjunto in conjuntos
        ]
        penalizacion_vacio = sum(
            100.0
            for conjunto in conjuntos
            if acumulados[conjunto] == 0
        )

        return sum(error ** 2 for error in errores_relativos) + penalizacion_vacio

    # Se prueban varios órdenes reproducibles porque el problema de asignar grupos
    # indivisibles a proporciones exactas es combinatorio.
    for intento in range(2000):
        if intento == 0:
            orden = sorted(
                resumen_grupos,
                key=lambda grupo: grupo["peso"],
                reverse=True,
            )
        else:
            orden = resumen_grupos.copy()
            generador.shuffle(orden)

        acumulados = {conjunto: 0 for conjunto in conjuntos}
        asignacion = {}

        for posicion, grupo in enumerate(orden):
            grupos_restantes = len(orden) - posicion
            conjuntos_vacios = [
                conjunto
                for conjunto in conjuntos
                if acumulados[conjunto] == 0
            ]

            if conjuntos_vacios and grupos_restantes <= len(conjuntos_vacios):
                candidatos = conjuntos_vacios
            else:
                candidatos = conjuntos

            mejor_conjunto = None
            mejor_error_local = np.inf

            for conjunto in candidatos:
                acumulados_prueba = acumulados.copy()
                acumulados_prueba[conjunto] += grupo["peso"]
                error = calcular_error(acumulados_prueba)

                # Mientras todavía se construye la solución se prioriza el déficit
                # absoluto para respetar 80/10/10 y no tratar los tres conjuntos
                # como si debieran tener el mismo tamaño.
                deficit = objetivos[conjunto] - acumulados[conjunto]
                error -= 1e-9 * deficit

                if error < mejor_error_local:
                    mejor_error_local = error
                    mejor_conjunto = conjunto

            asignacion[grupo["grupo_visual"]] = mejor_conjunto
            acumulados[mejor_conjunto] += grupo["peso"]

        # Optimización local mediante movimientos individuales.
        mejoro = True

        while mejoro:
            mejoro = False
            error_actual = calcular_error(acumulados)

            for grupo in resumen_grupos:
                identificador = grupo["grupo_visual"]
                origen = asignacion[identificador]

                for destino in conjuntos:
                    if destino == origen:
                        continue

                    grupos_en_origen = sum(
                        conjunto == origen
                        for conjunto in asignacion.values()
                    )

                    if grupos_en_origen <= 1:
                        continue

                    acumulados_prueba = acumulados.copy()
                    acumulados_prueba[origen] -= grupo["peso"]
                    acumulados_prueba[destino] += grupo["peso"]
                    error_prueba = calcular_error(acumulados_prueba)

                    if error_prueba + 1e-12 < error_actual:
                        asignacion[identificador] = destino
                        acumulados = acumulados_prueba
                        error_actual = error_prueba
                        mejoro = True
                        origen = destino

        # Optimización adicional mediante intercambio de dos grupos.
        mejoro = True

        while mejoro:
            mejoro = False
            error_actual = calcular_error(acumulados)

            for indice_a, grupo_a in enumerate(resumen_grupos):
                id_a = grupo_a["grupo_visual"]
                conjunto_a = asignacion[id_a]

                for grupo_b in resumen_grupos[indice_a + 1:]:
                    id_b = grupo_b["grupo_visual"]
                    conjunto_b = asignacion[id_b]

                    if conjunto_a == conjunto_b:
                        continue

                    acumulados_prueba = acumulados.copy()
                    acumulados_prueba[conjunto_a] += grupo_b["peso"] - grupo_a["peso"]
                    acumulados_prueba[conjunto_b] += grupo_a["peso"] - grupo_b["peso"]
                    error_prueba = calcular_error(acumulados_prueba)

                    if error_prueba + 1e-12 < error_actual:
                        asignacion[id_a] = conjunto_b
                        asignacion[id_b] = conjunto_a
                        acumulados = acumulados_prueba
                        error_actual = error_prueba
                        mejoro = True
                        break

                if mejoro:
                    break

        error_final = calcular_error(acumulados)

        if error_final < mejor_error:
            mejor_error = error_final
            mejor_asignacion = asignacion.copy()
            mejor_acumulados = acumulados.copy()

    if mejor_asignacion is None:
        raise RuntimeError("No se pudo encontrar una asignación válida de grupos")

    for escena in escenas:
        escena["conjunto"] = mejor_asignacion[escena["grupo_visual"]]

    for grupo in resumen_grupos:
        grupo["conjunto"] = mejor_asignacion[grupo["grupo_visual"]]

    print()
    print("Distribución interna de grupos:")
    print(f"Train: {mejor_acumulados['train']} / {peso_total} = {mejor_acumulados['train'] / peso_total:.2%}")
    print(f"Valid: {mejor_acumulados['valid']} / {peso_total} = {mejor_acumulados['valid'] / peso_total:.2%}")
    print(f"Test:  {mejor_acumulados['test']} / {peso_total} = {mejor_acumulados['test'] / peso_total:.2%}")
    print(f"Error de asignación: {mejor_error:.8f}")

    return resumen_grupos, mejor_acumulados, objetivos


def guardar_tiff(argumentos):
    ruta_origen, ruta_destino = argumentos
    imagen = cv2.imread(str(ruta_origen), cv2.IMREAD_UNCHANGED)

    if imagen is None:
        raise RuntimeError(f"No se pudo leer la imagen: {ruta_origen}")

    ruta_destino.parent.mkdir(parents=True, exist_ok=True)

    if imagen.ndim == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    tifffile.imwrite(ruta_destino, imagen, compression=COMPRESION_TIFF)

    return ruta_destino


def crear_estructura_dataset(carpeta):
    for conjunto in ("train", "valid"):
        (carpeta / conjunto / "input").mkdir(parents=True, exist_ok=True)
        (carpeta / conjunto / "target").mkdir(parents=True, exist_ok=True)

    (carpeta / "test" / "original").mkdir(parents=True, exist_ok=True)
    (carpeta / "test" / "clean").mkdir(parents=True, exist_ok=True)
    (carpeta / "diagnostics").mkdir(parents=True, exist_ok=True)


# %% 2. Separación del dataset
def main():
    if MODO not in {"original", "sin_hud"}:
        raise ValueError('MODO debe ser "original" o "sin_hud"')

    if not CARPETA_ANALISIS.exists():
        raise FileNotFoundError(f"No se encontró la carpeta utilizada para analizar escenas: {CARPETA_ANALISIS}")

    if not CARPETA_FUENTE.exists():
        raise FileNotFoundError(f"No se encontró la carpeta fuente del modo {MODO}: {CARPETA_FUENTE}")

    rutas_analisis = sorted(
        (ruta for ruta in CARPETA_ANALISIS.iterdir() if ruta.suffix.lower() == ".png"),
        key=clave_natural,
    )
    rutas_fuente = sorted(
        (ruta for ruta in CARPETA_FUENTE.iterdir() if ruta.suffix.lower() == ".png"),
        key=clave_natural,
    )

    if len(rutas_analisis) < 3:
        raise RuntimeError("Se necesitan al menos tres frames para realizar la separación")

    mapa_fuente = {ruta.name: ruta for ruta in rutas_fuente}
    faltantes = [ruta.name for ruta in rutas_analisis if ruta.name not in mapa_fuente]

    if faltantes:
        raise RuntimeError(
            f"Faltan {len(faltantes)} frames en {CARPETA_FUENTE}. "
            f"Primeros faltantes: {faltantes[:20]}"
        )

    if not np.isclose(PROPORCION_TRAIN + PROPORCION_VALID + PROPORCION_TEST, 1.0):
        raise ValueError("Las proporciones de train, valid y test deben sumar 1")

    cv2.setNumThreads(1)

    if CARPETA_TEMPORAL.exists():
        shutil.rmtree(CARPETA_TEMPORAL)

    crear_estructura_dataset(CARPETA_TEMPORAL)

    print(f"Modo: {MODO}")
    print(f"Frames para análisis de escenas: {CARPETA_ANALISIS}")
    print(f"Frames que se exportarán: {CARPETA_FUENTE}")
    print(f"Dataset de salida: {CARPETA_SALIDA}")
    print(f"Frames encontrados: {len(rutas_analisis)}")
    print(f"Trabajadores utilizados: {NUM_TRABAJADORES}")
    print(f"Proporción objetivo: {PROPORCION_TRAIN:.0%} train, {PROPORCION_VALID:.0%} valid, {PROPORCION_TEST:.0%} test")

    argumentos_frames = list(enumerate(rutas_analisis))

    with ThreadPoolExecutor(max_workers=NUM_TRABAJADORES) as ejecutor:
        descriptores = list(
            tqdm(
                ejecutor.map(calcular_descriptor_frame, argumentos_frames),
                total=len(argumentos_frames),
                desc="Analizando frames",
                unit="frame",
                mininterval=2.0,
            )
        )

    argumentos_cambios = [
        (posicion, descriptores[posicion - 1], descriptores[posicion])
        for posicion in range(1, len(descriptores))
    ]

    with ThreadPoolExecutor(max_workers=NUM_TRABAJADORES) as ejecutor:
        cambios = list(
            tqdm(
                ejecutor.map(calcular_cambio_consecutivo, argumentos_cambios),
                total=len(argumentos_cambios),
                desc="Comparando frames consecutivos",
                unit="par",
                mininterval=2.0,
            )
        )

    df_diagnostico = pd.DataFrame(
        [
            {
                "posicion": descriptor["posicion"],
                "frame": descriptor["frame"],
                "intensidad_media": descriptor["intensidad_media"],
                "contraste": descriptor["contraste"],
                "fraccion_negra": descriptor["fraccion_negra"],
                "fraccion_blanca": descriptor["fraccion_blanca"],
            }
            for descriptor in descriptores
        ]
    ).merge(pd.DataFrame(cambios), on=["posicion", "frame"], how="left")

    mediana_hs, mad_hs = mediana_mad(df_diagnostico["distancia_hs"])
    mediana_intensidad, mad_intensidad = mediana_mad(df_diagnostico["diferencia_intensidad"])
    escala_hs = max(1.4826 * mad_hs, 1e-6)
    escala_intensidad = max(1.4826 * mad_intensidad, 1e-6)
    umbral_hs = max(UMBRAL_MINIMO_HS, mediana_hs + FACTOR_MAD_CORTE * escala_hs)
    umbral_intensidad = max(
        UMBRAL_MINIMO_INTENSIDAD,
        mediana_intensidad + FACTOR_MAD_CORTE * escala_intensidad,
    )

    df_diagnostico["z_hs"] = (
        (df_diagnostico["distancia_hs"] - mediana_hs) / escala_hs
    ).fillna(0.0)
    df_diagnostico["z_intensidad"] = (
        (df_diagnostico["diferencia_intensidad"] - mediana_intensidad)
        / escala_intensidad
    ).fillna(0.0)
    df_diagnostico["puntaje_cambio"] = (
        PESO_HS * df_diagnostico["z_hs"]
        + PESO_INTENSIDAD * df_diagnostico["z_intensidad"]
    )
    df_diagnostico["corte_candidato"] = (
        (
            (df_diagnostico["distancia_hs"] >= umbral_hs)
            | (df_diagnostico["diferencia_intensidad"] >= umbral_intensidad)
        )
        & (df_diagnostico["puntaje_cambio"] >= UMBRAL_MINIMO_PUNTAJE)
    )
    df_diagnostico["corte_escena"] = df_diagnostico["corte_candidato"]
    df_diagnostico.loc[0, ["corte_candidato", "corte_escena"]] = False

    escenas, cortes = construir_escenas(rutas_analisis, df_diagnostico)
    cortes_validos = set(cortes)
    df_diagnostico["corte_escena"] = df_diagnostico["posicion"].isin(cortes_validos)
    df_diagnostico["excluido_transicion"] = False
    df_diagnostico["escena"] = ""
    df_diagnostico["grupo_visual"] = pd.Series([pd.NA] * len(df_diagnostico), dtype="Int64")
    df_diagnostico["conjunto"] = ""

    escenas_validas = agrupar_escenas(escenas, descriptores)
    resumen_grupos, acumulados, objetivos = asignar_grupos(escenas_validas)

    for escena in escenas:
        inicio = escena["indice_inicial"]
        fin = escena["indice_final"]
        df_diagnostico.loc[inicio:fin, "escena"] = escena["escena"]
        indices_validos = set(escena["indices_validos"])

        for indice in range(inicio, fin + 1):
            if indice not in indices_validos:
                df_diagnostico.loc[indice, "excluido_transicion"] = True

        if "grupo_visual" in escena:
            df_diagnostico.loc[inicio:fin, "grupo_visual"] = escena["grupo_visual"]
            df_diagnostico.loc[inicio:fin, "conjunto"] = escena["conjunto"]

    parejas = []
    exportaciones = []

    for escena in escenas_validas:
        conjunto = escena["conjunto"]
        indices = escena["indices_validos"]

        if conjunto in {"train", "valid"}:
            for indice_entrada, indice_objetivo in zip(indices[:-1], indices[1:]):
                ruta_entrada_analisis = rutas_analisis[indice_entrada]
                ruta_objetivo_analisis = rutas_analisis[indice_objetivo]
                ruta_entrada = mapa_fuente[ruta_entrada_analisis.name]
                ruta_objetivo = mapa_fuente[ruta_objetivo_analisis.name]
                nombre_salida = f"{ruta_entrada.stem}.tif"
                destino_entrada = CARPETA_TEMPORAL / conjunto / "input" / nombre_salida
                destino_objetivo = CARPETA_TEMPORAL / conjunto / "target" / nombre_salida
                exportaciones.extend([(ruta_entrada, destino_entrada), (ruta_objetivo, destino_objetivo)])
                parejas.append(
                    {
                        "conjunto": conjunto,
                        "grupo_visual": escena["grupo_visual"],
                        "escena": escena["escena"],
                        "frame_input": ruta_entrada.name,
                        "frame_target": ruta_objetivo.name,
                        "archivo_tiff": nombre_salida,
                        "indice_input": indice_entrada,
                        "indice_target": indice_objetivo,
                    }
                )
        else:
            for indice in indices:
                ruta_analisis = rutas_analisis[indice]
                ruta_origen = mapa_fuente[ruta_analisis.name]
                destino = CARPETA_TEMPORAL / "test" / "original" / f"{ruta_origen.stem}.tif"
                exportaciones.append((ruta_origen, destino))

    with ThreadPoolExecutor(max_workers=NUM_TRABAJADORES) as ejecutor:
        list(
            tqdm(
                ejecutor.map(guardar_tiff, exportaciones),
                total=len(exportaciones),
                desc="Escribiendo TIFF",
                unit="archivo",
                mininterval=2.0,
            )
        )

    df_parejas = pd.DataFrame(parejas)
    df_escenas = pd.DataFrame(
        [
            {
                key: value
                for key, value in escena.items()
                if key != "indices_validos"
            }
            for escena in escenas
        ]
    )
    df_grupos = pd.DataFrame(
        [
            {
                "grupo_visual": grupo["grupo_visual"],
                "cantidad_escenas": grupo["cantidad_escenas"],
                "cantidad_parejas": grupo["cantidad_parejas"],
                "cantidad_frames_validos": grupo["cantidad_frames_validos"],
                "peso": grupo["peso"],
                "conjunto": grupo["escenas"][0]["conjunto"],
                "escenas": ";".join(escena["escena"] for escena in grupo["escenas"]),
            }
            for grupo in resumen_grupos
        ]
    )

    resumen = []

    for conjunto in ("train", "valid", "test"):
        escenas_conjunto = [
            escena
            for escena in escenas_validas
            if escena["conjunto"] == conjunto
        ]
        cantidad_parejas = (
            int((df_parejas["conjunto"] == conjunto).sum())
            if not df_parejas.empty and conjunto in {"train", "valid"}
            else 0
        )
        cantidad_frames = (
            len(list((CARPETA_TEMPORAL / "test" / "original").glob("*.tif")))
            if conjunto == "test"
            else cantidad_parejas
        )
        resumen.append(
            {
                "conjunto": conjunto,
                "grupos_visuales": len({escena["grupo_visual"] for escena in escenas_conjunto}),
                "escenas": len(escenas_conjunto),
                "frames_o_parejas": cantidad_frames,
                "parejas": cantidad_parejas,
            }
        )

    df_resumen = pd.DataFrame(resumen)
    total_distribucion = df_resumen["frames_o_parejas"].sum()
    df_resumen["proporcion_obtenida"] = (
        df_resumen["frames_o_parejas"] / total_distribucion
        if total_distribucion
        else 0.0
    )
    df_resumen["proporcion_objetivo"] = [
        PROPORCION_TRAIN,
        PROPORCION_VALID,
        PROPORCION_TEST,
    ]

    df_diagnostico.to_csv(
        CARPETA_TEMPORAL / "diagnostics" / "diagnostico_frames.csv",
        index=False,
    )
    df_escenas.to_csv(
        CARPETA_TEMPORAL / "diagnostics" / "escenas.csv",
        index=False,
    )
    df_grupos.to_csv(
        CARPETA_TEMPORAL / "diagnostics" / "grupos_visuales.csv",
        index=False,
    )
    df_resumen.to_csv(
        CARPETA_TEMPORAL / "diagnostics" / "resumen_separacion.csv",
        index=False,
    )

    if not df_parejas.empty:
        df_parejas[df_parejas["conjunto"] == "train"].to_csv(
            CARPETA_TEMPORAL / "train" / "parejas.csv",
            index=False,
        )
        df_parejas[df_parejas["conjunto"] == "valid"].to_csv(
            CARPETA_TEMPORAL / "valid" / "parejas.csv",
            index=False,
        )

    parametros = pd.DataFrame(
        [
            {"parametro": "modo", "valor": MODO},
            {"parametro": "semilla", "valor": SEMILLA},
            {"parametro": "trabajadores", "valor": NUM_TRABAJADORES},
            {"parametro": "umbral_hs_calculado", "valor": umbral_hs},
            {"parametro": "umbral_intensidad_calculado", "valor": umbral_intensidad},
            {"parametro": "umbral_puntaje_cambio", "valor": UMBRAL_MINIMO_PUNTAJE},
            {"parametro": "umbral_agrupamiento_escenas", "valor": UMBRAL_AGRUPAMIENTO_ESCENAS},
            {"parametro": "margen_transicion_antes", "valor": MARGEN_TRANSICION_ANTES},
            {"parametro": "margen_transicion_despues", "valor": MARGEN_TRANSICION_DESPUES},
            {"parametro": "proporcion_train", "valor": PROPORCION_TRAIN},
            {"parametro": "proporcion_valid", "valor": PROPORCION_VALID},
            {"parametro": "proporcion_test", "valor": PROPORCION_TEST},
        ]
    )
    parametros.to_csv(
        CARPETA_TEMPORAL / "diagnostics" / "parametros.csv",
        index=False,
    )

    cantidades_esperadas = {
        "train_input": int((df_parejas["conjunto"] == "train").sum()) if not df_parejas.empty else 0,
        "train_target": int((df_parejas["conjunto"] == "train").sum()) if not df_parejas.empty else 0,
        "valid_input": int((df_parejas["conjunto"] == "valid").sum()) if not df_parejas.empty else 0,
        "valid_target": int((df_parejas["conjunto"] == "valid").sum()) if not df_parejas.empty else 0,
        "test_original": int(df_resumen.loc[df_resumen["conjunto"] == "test", "frames_o_parejas"].iloc[0]),
    }
    cantidades_obtenidas = {
        "train_input": len(list((CARPETA_TEMPORAL / "train" / "input").glob("*.tif"))),
        "train_target": len(list((CARPETA_TEMPORAL / "train" / "target").glob("*.tif"))),
        "valid_input": len(list((CARPETA_TEMPORAL / "valid" / "input").glob("*.tif"))),
        "valid_target": len(list((CARPETA_TEMPORAL / "valid" / "target").glob("*.tif"))),
        "test_original": len(list((CARPETA_TEMPORAL / "test" / "original").glob("*.tif"))),
    }

    if cantidades_esperadas != cantidades_obtenidas:
        raise RuntimeError(
            f"Las cantidades exportadas no coinciden. "
            f"Esperadas: {cantidades_esperadas}. Obtenidas: {cantidades_obtenidas}"
        )

    grupos_por_conjunto = {
        conjunto: set(df_grupos.loc[df_grupos["conjunto"] == conjunto, "grupo_visual"])
        for conjunto in ("train", "valid", "test")
    }

    if (
        grupos_por_conjunto["train"] & grupos_por_conjunto["valid"]
        or grupos_por_conjunto["train"] & grupos_por_conjunto["test"]
        or grupos_por_conjunto["valid"] & grupos_por_conjunto["test"]
    ):
        raise RuntimeError("Se detectó fuga de grupos visuales entre conjuntos")

    if CARPETA_SALIDA.exists():
        if not SOBRESCRIBIR_DATASET:
            raise FileExistsError(f"El dataset ya existe: {CARPETA_SALIDA}")

        shutil.rmtree(CARPETA_SALIDA)

    CARPETA_TEMPORAL.rename(CARPETA_SALIDA)

    print()
    print("Separación terminada")
    print(f"Modo procesado: {MODO}")
    print(f"Cortes de escena aceptados: {len(cortes)}")
    print(f"Escenas detectadas: {len(escenas)}")
    print(f"Escenas utilizables: {len(escenas_validas)}")
    print(f"Grupos visuales: {len(resumen_grupos)}")
    print(f"Frames excluidos cerca de transiciones: {int(df_diagnostico['excluido_transicion'].sum())}")
    print()
    print(df_resumen.to_string(index=False, formatters={"proporcion_obtenida": "{:.2%}".format, "proporcion_objetivo": "{:.2%}".format}))
    print()
    print(f"Dataset guardado en: {CARPETA_SALIDA}")
    print(f"Diagnósticos guardados en: {CARPETA_SALIDA / 'diagnostics'}")
    print(f"Test clean quedó vacío para recibir los resultados de inferencia: {CARPETA_SALIDA / 'test' / 'clean'}")


if __name__ == "__main__":
    main()
