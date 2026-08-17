# %% 0. Imports y configuracion
import argparse
import fcntl
import gc
import json
import os
import re
import shutil
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import pandas as pd
import pyiqa
import tifffile
import torch
from careamics import CAREamist
from careamics.config import create_n2n_config
from careamics.lightning.data.grouped_index_sampler import GroupedIndexSampler
from pytorch_lightning import Callback
from scipy import stats
from skimage.restoration import estimate_sigma
from tqdm import tqdm

# %% 1. Parametros
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_VIDEO = RUTA_PROYECTO / "Videos" / "video30min-11to22"

SEMILLA = 42
PORCENTAJE_VALIDACION = 0.05
EPOCAS = int(os.environ.get("N2N_EPOCAS", "50"))
PASOS_POR_EPOCA = 2000
TAMANO_PARCHE = (128, 128)
TAMANO_LOTE = 8
PROFUNDIDAD_UNET = 4
CANALES_INICIALES = 48
CONEXION_RESIDUAL = True
USAR_BATCH_NORM = False
CANALES_INDEPENDIENTES = False
NUM_TRABAJADORES_DATOS = 4
EJES = "YXC"
NUM_CANALES = 3
TAMANO_TILE = (128, 128)
SOLAPAMIENTO_TILE = (48, 48)
COMPRESION_TIFF = "deflate"


# Compatibilidad de CAREamics 0.3.x con DistributedSamplerWrapper de Lightning.
def _longitud_grouped_index_sampler(self):
    return sum(len(grupo) for grupo in self.grouped_indices)


if not hasattr(GroupedIndexSampler, "__len__"):
    GroupedIndexSampler.__len__ = _longitud_grouped_index_sampler


# %% 2. Funciones generales
def clave_natural(ruta):
    return [
        int(parte) if parte.isdigit() else parte.lower()
        for parte in re.split(r"(\d+)", ruta.name)
    ]


def listar_imagenes(carpeta):
    extensiones = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    return sorted(
        (ruta for ruta in carpeta.iterdir() if ruta.suffix.lower() in extensiones),
        key=clave_natural,
    )


def convertir_uint8(imagen):
    imagen = np.asarray(imagen)

    while imagen.ndim > 3 and imagen.shape[0] == 1:
        imagen = imagen[0]

    if (
        imagen.ndim == 3
        and imagen.shape[0] in (1, 3, 4)
        and imagen.shape[-1] not in (1, 3, 4)
    ):
        imagen = np.moveaxis(imagen, 0, -1)

    if np.issubdtype(imagen.dtype, np.floating):
        minimo = float(np.nanmin(imagen))
        maximo = float(np.nanmax(imagen))
        if minimo >= -0.1 and maximo <= 1.5:
            imagen = imagen * 255.0

    imagen = np.nan_to_num(imagen, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(imagen, 0, 255).round().astype(np.uint8)


def leer_rgb(ruta):
    if ruta.suffix.lower() in {".tif", ".tiff"}:
        imagen = tifffile.imread(ruta)
    else:
        imagen_bgr = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
        if imagen_bgr is None:
            raise RuntimeError(f"No se pudo leer: {ruta}")
        if imagen_bgr.ndim == 2:
            imagen = np.repeat(imagen_bgr[:, :, None], 3, axis=2)
        elif imagen_bgr.shape[2] == 4:
            imagen = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGRA2RGB)
        else:
            imagen = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

    imagen = convertir_uint8(imagen)
    if imagen.ndim == 2:
        imagen = np.repeat(imagen[:, :, None], 3, axis=2)
    if imagen.shape[-1] == 4:
        imagen = imagen[:, :, :3]
    if imagen.ndim != 3 or imagen.shape[-1] != 3:
        raise RuntimeError(f"Forma RGB no valida en {ruta}: {imagen.shape}")
    return imagen


def guardar_tiff(ruta, imagen):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(ruta, convertir_uint8(imagen), compression=COMPRESION_TIFF)


def obtener_rutas(modo):
    fuente_personalizada = os.environ.get(
        "N2N_CARPETA_FUENTE"
    )
    salida_personalizada = os.environ.get(
        "N2N_CARPETA_SALIDA"
    )

    if fuente_personalizada or salida_personalizada:
        if not (
            fuente_personalizada
            and salida_personalizada
        ):
            raise RuntimeError(
                "N2N_CARPETA_FUENTE y "
                "N2N_CARPETA_SALIDA deben definirse juntas."
            )

        return (
            CARPETA_VIDEO / fuente_personalizada,
            CARPETA_VIDEO / salida_personalizada,
        )

    if modo == "original":
        return (
            CARPETA_VIDEO / "frames",
            CARPETA_VIDEO / "n2n",
        )

    return (
        CARPETA_VIDEO / "frames_sin_hud",
        CARPETA_VIDEO / "n2n_sin_hud",
    )


def limpiar_memoria():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# %% 3. Preparacion de parejas
def puntuar_pareja(ruta_input, ruta_target):
    input_gris = cv2.cvtColor(leer_rgb(ruta_input), cv2.COLOR_RGB2GRAY)
    target_gris = cv2.cvtColor(leer_rgb(ruta_target), cv2.COLOR_RGB2GRAY)

    alto = 180
    ancho = max(1, round(input_gris.shape[1] * alto / input_gris.shape[0]))
    input_gris = cv2.resize(
        input_gris, (ancho, alto), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    target_gris = cv2.resize(
        target_gris, (ancho, alto), interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    diferencia_media = float(np.mean(np.abs(input_gris - target_gris)))
    desplazamiento, respuesta = cv2.phaseCorrelate(input_gris, target_gris)
    movimiento_estimado = float(np.hypot(*desplazamiento))
    return diferencia_media, movimiento_estimado, float(respuesta)


def calcular_umbral_robusto(valores, percentil=97.5):
    valores = np.asarray(valores, dtype=float)
    mediana = float(np.median(valores))
    mad = float(np.median(np.abs(valores - mediana)))
    umbral_mad = mediana + 8.0 * 1.4826 * max(mad, 1e-6)
    return min(float(np.percentile(valores, percentil)), umbral_mad)


def preparar_dataset(modo, reiniciar):
    carpeta_fuente, carpeta_salida = obtener_rutas(modo)
    frames = listar_imagenes(carpeta_fuente)

    if len(frames) < 3:
        raise RuntimeError(f"No hay suficientes frames en {carpeta_fuente}")

    if reiniciar:
        for nombre in (
            "train",
            "valid",
            "clean",
            ".careamics_trabajo",
            ".inferencia_temporal",
        ):
            ruta = carpeta_salida / nombre
            if ruta.exists():
                shutil.rmtree(ruta)
        for nombre in (
            "modelo.ckpt",
            "configuracion.json",
            "historial_entrenamiento.csv",
            "parejas.csv",
            "preparacion.json",
            "metricas_frames.csv",
            "tiempo_entrenamiento_minutos.txt",
            "tiempo_inferencia_minutos.txt",
        ):
            (carpeta_salida / nombre).unlink(missing_ok=True)

    temporal = carpeta_salida / ".temporal_preparacion"
    if temporal.exists():
        shutil.rmtree(temporal)

    for conjunto in ("train", "valid"):
        (temporal / conjunto / "input").mkdir(parents=True)
        (temporal / conjunto / "target").mkdir(parents=True)

    registros = []
    for indice in tqdm(
        range(len(frames) - 1),
        desc=f"Analizando parejas {modo}",
        unit="pareja",
    ):
        frame_input = frames[indice]
        frame_target = frames[indice + 1]
        diferencia, movimiento, respuesta = puntuar_pareja(
            frame_input, frame_target
        )
        registros.append(
            {
                "indice": indice,
                "frame_input": frame_input.name,
                "frame_target": frame_target.name,
                "archivo_tiff": f"{frame_input.stem}.tif",
                "diferencia_media": diferencia,
                "movimiento_estimado": movimiento,
                "respuesta_fase": respuesta,
            }
        )

    parejas = pd.DataFrame(registros)
    umbral_diferencia = calcular_umbral_robusto(parejas["diferencia_media"])
    umbral_movimiento = calcular_umbral_robusto(
        parejas["movimiento_estimado"]
    )
    parejas["excluida_transicion"] = (
        parejas["diferencia_media"] > umbral_diferencia
    )
    parejas["excluida_movimiento"] = (
        parejas["movimiento_estimado"] > umbral_movimiento
    )
    parejas["valida"] = ~(
        parejas["excluida_transicion"] | parejas["excluida_movimiento"]
    )

    indices_validos = parejas.index[parejas["valida"]].tolist()
    cada_n = max(2, round(1.0 / PORCENTAJE_VALIDACION))
    indices_validacion = set(indices_validos[cada_n - 1 :: cada_n])

    parejas["conjunto"] = "excluida"
    parejas.loc[indices_validos, "conjunto"] = "train"
    parejas.loc[list(indices_validacion), "conjunto"] = "valid"

    filas_validas = parejas[parejas["valida"]]
    for _, fila in tqdm(
        filas_validas.iterrows(),
        total=len(filas_validas),
        desc=f"Escribiendo TIFF {modo}",
        unit="pareja",
    ):
        conjunto = fila["conjunto"]
        nombre_tiff = fila["archivo_tiff"]
        guardar_tiff(
            temporal / conjunto / "input" / nombre_tiff,
            leer_rgb(carpeta_fuente / fila["frame_input"]),
        )
        guardar_tiff(
            temporal / conjunto / "target" / nombre_tiff,
            leer_rgb(carpeta_fuente / fila["frame_target"]),
        )

    for conjunto in ("train", "valid"):
        destino = carpeta_salida / conjunto
        if destino.exists():
            shutil.rmtree(destino)
        shutil.move(str(temporal / conjunto), str(destino))

    shutil.rmtree(temporal)
    (carpeta_salida / "clean").mkdir(parents=True, exist_ok=True)

    columnas = [
        "frame_input",
        "frame_target",
        "archivo_tiff",
        "conjunto",
        "diferencia_media",
        "movimiento_estimado",
        "respuesta_fase",
        "excluida_transicion",
        "excluida_movimiento",
        "valida",
    ]
    parejas[columnas].to_csv(carpeta_salida / "parejas.csv", index=False)

    resumen = {
        "modo": modo,
        "frames_fuente": len(frames),
        "parejas_totales": len(parejas),
        "parejas_train": int((parejas["conjunto"] == "train").sum()),
        "parejas_valid": int((parejas["conjunto"] == "valid").sum()),
        "parejas_excluidas": int((parejas["conjunto"] == "excluida").sum()),
        "porcentaje_validacion": PORCENTAJE_VALIDACION,
        "umbral_diferencia_media": umbral_diferencia,
        "umbral_movimiento_estimado": umbral_movimiento,
    }
    (carpeta_salida / "preparacion.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(resumen, indent=2, ensure_ascii=False), flush=True)


# %% 4. Configuracion CAREamics
def crear_configuracion(nombre_experimento, entrenamiento):
    config = create_n2n_config(
        experiment_name=nombre_experimento,
        data_type="tiff",
        axes=EJES,
        patch_size=TAMANO_PARCHE,
        batch_size=TAMANO_LOTE,
        num_epochs=EPOCAS if entrenamiento else 1,
        n_channels_in=NUM_CANALES,
        n_channels_out=NUM_CANALES,
    )

    workers = NUM_TRABAJADORES_DATOS if entrenamiento else 0
    config.data_config.in_memory = False
    config.data_config.num_workers = workers
    config.data_config.train_dataloader_params["num_workers"] = workers
    config.data_config.val_dataloader_params["num_workers"] = workers
    config.data_config.pred_dataloader_params["num_workers"] = 0
    config.data_config.pred_dataloader_params["persistent_workers"] = False
    config.data_config.pred_dataloader_params["prefetch_factor"] = None
    config.algorithm_config.model.depth = PROFUNDIDAD_UNET
    config.algorithm_config.model.num_channels_init = CANALES_INICIALES
    config.algorithm_config.model.residual = CONEXION_RESIDUAL
    config.algorithm_config.model.use_batch_norm = USAR_BATCH_NORM
    config.algorithm_config.model.independent_channels = CANALES_INDEPENDIENTES

    if entrenamiento:
        config.training_config.trainer_params.update(
            {
                "accelerator": "gpu",
                "devices": 2,
                "strategy": "ddp_find_unused_parameters_true",
                "precision": "32-true",
                "benchmark": True,
                "deterministic": False,
                "gradient_clip_val": 1.0,
                "gradient_clip_algorithm": "norm",
                "limit_train_batches": PASOS_POR_EPOCA,
                "enable_progress_bar": False,
                "log_every_n_steps": 50,
                "num_sanity_val_steps": 2,
            }
        )
        config.training_config.checkpoint_params.update(
            {"every_n_epochs": 1, "save_last": True, "save_top_k": -1}
        )
    else:
        config.training_config.trainer_params.update(
            {
                "accelerator": "gpu",
                "devices": 1,
                "strategy": "auto",
                "precision": "32-true",
                "enable_progress_bar": False,
            }
        )

    return config


class HistorialPerdida(Callback):
    def __init__(self, ruta_csv, ruta_modelo):
        super().__init__()
        self.ruta_csv = Path(ruta_csv)
        self.ruta_modelo = Path(ruta_modelo)
        self.registros = []
        self.inicio_epoca = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.inicio_epoca = time.time()
        if trainer.is_global_zero:
            print(
                f"Epoca {trainer.current_epoch + 1}/{trainer.max_epochs} iniciada",
                flush=True,
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        metricas = trainer.callback_metrics

        def obtener_valor(nombres):
            for nombre in nombres:
                if nombre in metricas:
                    valor = metricas[nombre]
                    if isinstance(valor, torch.Tensor):
                        valor = valor.detach().cpu().item()
                    return float(valor)
            return np.nan

        duracion_minutos = (
            (time.time() - self.inicio_epoca) / 60.0
            if self.inicio_epoca is not None
            else np.nan
        )
        registro = {
            "epoca": int(trainer.current_epoch),
            "train_loss": obtener_valor(("train_loss_epoch", "train_loss")),
            "val_loss": obtener_valor(("val_loss", "validation_loss")),
            "duracion_minutos": duracion_minutos,
        }

        perdidas = [
            registro["train_loss"],
            registro["val_loss"],
        ]

        if not all(np.isfinite(valor) for valor in perdidas):
            raise RuntimeError(
                "Entrenamiento detenido porque train_loss o val_loss "
                "contiene NaN o infinito. Se conserva el último "
                "checkpoint válido."
            )
        self.registros.append(registro)
        pd.DataFrame(self.registros).to_csv(self.ruta_csv, index=False)
        print(
            f"Epoca {registro['epoca'] + 1}/{trainer.max_epochs} terminada | "
            f"train_loss={registro['train_loss']:.6f} | "
            f"val_loss={registro['val_loss']:.6f} | "
            f"duracion={duracion_minutos:.2f} min",
            flush=True,
        )


def entrenar(modo):
    _, carpeta_salida = obtener_rutas(modo)
    rango_global = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    proceso_principal = rango_global == 0
    carpeta_trabajo = carpeta_salida / ".careamics_trabajo"
    carpeta_trabajo.mkdir(parents=True, exist_ok=True)

    config = crear_configuracion(f"n2n_{modo}", entrenamiento=True)
    if proceso_principal:
        contenido = (
            config.model_dump_json(indent=2)
            if hasattr(config, "model_dump_json")
            else config.json(indent=2)
        )
        (carpeta_salida / "configuracion.json").write_text(
            contenido, encoding="utf-8"
        )

    directorio_anterior = Path.cwd()
    os.chdir(carpeta_trabajo)
    inicio = time.time()
    try:
        careamist = CAREamist(config=config)
        careamist.trainer.callbacks.append(
            HistorialPerdida(
                carpeta_salida / "historial_entrenamiento.csv",
                carpeta_salida / "modelo.ckpt",
            )
        )
        careamist.train(
            train_data=str(carpeta_salida / "train" / "input"),
            train_data_target=str(carpeta_salida / "train" / "target"),
            val_data=str(carpeta_salida / "valid" / "input"),
            val_data_target=str(carpeta_salida / "valid" / "target"),
        )
    finally:
        os.chdir(directorio_anterior)

    if proceso_principal:
        duracion_minutos = (time.time() - inicio) / 60.0

        candidatos_last = sorted(
            carpeta_trabajo.rglob("*last.ckpt"),
            key=lambda ruta_checkpoint: ruta_checkpoint.stat().st_mtime,
        )

        if candidatos_last:
            checkpoint_final = candidatos_last[-1]
        else:
            candidatos = sorted(
                carpeta_trabajo.rglob("*.ckpt"),
                key=lambda ruta_checkpoint: ruta_checkpoint.stat().st_mtime,
            )

            if not candidatos:
                raise RuntimeError(
                    f"No se encontro ningun checkpoint en {carpeta_trabajo}"
                )

            checkpoint_final = candidatos[-1]

        shutil.copy2(
            checkpoint_final,
            carpeta_salida / "modelo.ckpt",
        )

        (carpeta_salida / "tiempo_entrenamiento_minutos.txt").write_text(
            f"{duracion_minutos:.6f}\n",
            encoding="utf-8",
        )

        shutil.rmtree(
            carpeta_trabajo,
            ignore_errors=True,
        )

        print(
            f"Modelo final: {carpeta_salida / 'modelo.ckpt'}",
            flush=True,
        )
        print(
            f"Entrenamiento: {duracion_minutos:.2f} min",
            flush=True,
        )


# %% 5. Inferencia
def extraer_predicciones(resultado):
    if isinstance(resultado, tuple):
        resultado = resultado[0]

    if isinstance(resultado, torch.Tensor):
        resultado = resultado.detach().cpu().numpy()

    if isinstance(resultado, np.ndarray):
        if resultado.ndim == 3:
            predicciones = [resultado]
        elif resultado.ndim == 4:
            predicciones = [
                resultado[indice]
                for indice in range(resultado.shape[0])
            ]
        else:
            raise RuntimeError(
                "Forma de predicción no reconocida: "
                f"{resultado.shape}"
            )
    elif isinstance(resultado, list):
        predicciones = []

        for prediccion in resultado:
            if isinstance(prediccion, torch.Tensor):
                prediccion = (
                    prediccion.detach().cpu().numpy()
                )

            predicciones.append(prediccion)
    else:
        raise RuntimeError(
            "Tipo de predicción no reconocido: "
            f"{type(resultado)}"
        )

    predicciones_uint8 = []

    for posicion, prediccion in enumerate(predicciones):
        prediccion = np.asarray(prediccion)

        while (
            prediccion.ndim > 3
            and prediccion.shape[0] == 1
        ):
            prediccion = prediccion[0]

        if not np.isfinite(prediccion).all():
            cantidad_nan = int(
                np.isnan(prediccion).sum()
            )
            cantidad_inf = int(
                np.isinf(prediccion).sum()
            )

            raise RuntimeError(
                f"La predicción {posicion} contiene "
                f"NaN={cantidad_nan} e "
                f"infinito={cantidad_inf}."
            )

        predicciones_uint8.append(
            convertir_uint8(prediccion)
        )

    return predicciones_uint8


def inferir(modo):
    carpeta_fuente, carpeta_salida = obtener_rutas(
        modo
    )
    checkpoint = carpeta_salida / "modelo.ckpt"
    carpeta_clean = carpeta_salida / "clean"
    carpeta_temporal = (
        carpeta_salida / ".inferencia_temporal"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No existe: {checkpoint}"
        )

    if carpeta_temporal.exists():
        shutil.rmtree(carpeta_temporal)

    carpeta_temporal.mkdir(parents=True)
    carpeta_clean.mkdir(
        parents=True,
        exist_ok=True,
    )

    limpiar_memoria()

    config = crear_configuracion(
        f"n2n_{modo}_inferencia",
        entrenamiento=False,
    )
    config.data_config.num_workers = 0
    config.data_config.pred_dataloader_params[
        "num_workers"
    ] = 0
    config.data_config.pred_dataloader_params[
        "persistent_workers"
    ] = False
    config.data_config.pred_dataloader_params[
        "prefetch_factor"
    ] = None

    careamist = CAREamist(config=config)
    frames = listar_imagenes(carpeta_fuente)

    tamano_bloque = 16
    cantidad_bloques = (
        len(frames) + tamano_bloque - 1
    ) // tamano_bloque

    inicio = time.time()

    for indice_bloque in tqdm(
        range(cantidad_bloques),
        desc=f"Inferencia {modo}",
        unit="bloque",
    ):
        inicio_bloque = (
            indice_bloque * tamano_bloque
        )
        fin_bloque = min(
            inicio_bloque + tamano_bloque,
            len(frames),
        )
        frames_bloque = frames[
            inicio_bloque:fin_bloque
        ]

        frames_pendientes = [
            frame
            for frame in frames_bloque
            if not (
                carpeta_clean
                / f"{frame.stem}.png"
            ).exists()
        ]

        if not frames_pendientes:
            continue

        carpeta_bloque = (
            carpeta_temporal
            / f"bloque_{indice_bloque:05d}"
        )

        if carpeta_bloque.exists():
            shutil.rmtree(carpeta_bloque)

        carpeta_bloque.mkdir(parents=True)

        for frame in frames_pendientes:
            guardar_tiff(
                carpeta_bloque
                / f"{frame.stem}.tif",
                leer_rgb(frame),
            )

        resultado = careamist.predict(
            pred_data=str(carpeta_bloque),
            checkpoint=str(checkpoint),
            tile_size=TAMANO_TILE,
            tile_overlap=SOLAPAMIENTO_TILE,
        )
        predicciones = extraer_predicciones(
            resultado
        )

        if (
            len(predicciones)
            != len(frames_pendientes)
        ):
            raise RuntimeError(
                "CAREamics devolvió "
                f"{len(predicciones)} predicciones "
                f"para {len(frames_pendientes)} "
                "frames."
            )

        for frame, prediccion in zip(
            frames_pendientes,
            predicciones,
        ):
            ruta_salida = (
                carpeta_clean
                / f"{frame.stem}.png"
            )

            guardada = cv2.imwrite(
                str(ruta_salida),
                cv2.cvtColor(
                    prediccion,
                    cv2.COLOR_RGB2BGR,
                ),
            )

            if not guardada:
                raise RuntimeError(
                    "No se pudo guardar: "
                    f"{ruta_salida}"
                )

        shutil.rmtree(carpeta_bloque)

        del resultado
        del predicciones

        limpiar_memoria()

    duracion_minutos = (
        time.time() - inicio
    ) / 60.0

    shutil.rmtree(
        carpeta_temporal,
        ignore_errors=True,
    )

    cantidad_clean = len(
        listar_imagenes(carpeta_clean)
    )

    if cantidad_clean != len(frames):
        raise RuntimeError(
            f"Se esperaban {len(frames)} "
            f"frames clean y se encontraron "
            f"{cantidad_clean}."
        )

    (
        carpeta_salida
        / "tiempo_inferencia_minutos.txt"
    ).write_text(
        f"{duracion_minutos:.6f}\n",
        encoding="utf-8",
    )

    print(
        f"Inferencia: {duracion_minutos:.2f} min",
        flush=True,
    )


# %% 6. Metricas
def calcular_metricas_cpu(ruta_original, ruta_clean):
    original = leer_rgb(ruta_original)
    clean = leer_rgb(ruta_clean)
    gris_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gris_clean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
    residual = gris_original - gris_clean

    sigma_original = float(
        estimate_sigma(
            gris_original / 255.0, channel_axis=None, average_sigmas=True
        )
        * 255.0
    )
    sigma_clean = float(
        estimate_sigma(gris_clean / 255.0, channel_axis=None, average_sigmas=True)
        * 255.0
    )
    nitidez_original = float(cv2.Laplacian(gris_original, cv2.CV_32F).var())
    nitidez_clean = float(cv2.Laplacian(gris_clean, cv2.CV_32F).var())

    izquierda = residual[:, :-1].ravel()
    derecha = residual[:, 1:].ravel()
    autocorrelacion = (
        float(np.corrcoef(izquierda, derecha)[0, 1])
        if np.std(izquierda) > 0 and np.std(derecha) > 0
        else np.nan
    )

    return {
        "frame": ruta_original.name,
        "ruido_sigma_original": sigma_original,
        "ruido_sigma_clean": sigma_clean,
        "reduccion_sigma": sigma_original - sigma_clean,
        "residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
        "residual_media_abs": float(np.mean(np.abs(residual))),
        "residual_curtosis": float(
            stats.kurtosis(residual.ravel(), fisher=True, bias=False)
        ),
        "residual_autocorrelacion_x": autocorrelacion,
        "nitidez_original": nitidez_original,
        "nitidez_clean": nitidez_clean,
        "retencion_nitidez": nitidez_clean / max(nitidez_original, 1e-8),
        "contraste_original": float(gris_original.std()),
        "contraste_clean": float(gris_clean.std()),
    }


def calcular_metricas(modo):
    carpeta_fuente, carpeta_salida = obtener_rutas(modo)
    carpeta_clean = carpeta_salida / "clean"
    originales = listar_imagenes(carpeta_fuente)
    mapa_clean = {ruta.stem: ruta for ruta in listar_imagenes(carpeta_clean)}

    registros = []
    for original in tqdm(originales, desc=f"Metricas CPU {modo}", unit="frame"):
        if original.stem not in mapa_clean:
            raise RuntimeError(f"Falta clean para {original.name}")
        registros.append(calcular_metricas_cpu(original, mapa_clean[original.stem]))

    metricas = pd.DataFrame(registros)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    niqe = pyiqa.create_metric("niqe", device=dispositivo)
    brisque = pyiqa.create_metric("brisque", device=dispositivo)
    registros_no_referencia = []

    with torch.inference_mode():
        for original in tqdm(
            originales, desc=f"NIQE y BRISQUE {modo}", unit="frame"
        ):
            clean = mapa_clean[original.stem]
            try:
                registros_no_referencia.append(
                    {
                        "frame": original.name,
                        "niqe_original": float(niqe(str(original)).item()),
                        "niqe_clean": float(niqe(str(clean)).item()),
                        "brisque_original": float(brisque(str(original)).item()),
                        "brisque_clean": float(brisque(str(clean)).item()),
                    }
                )
            except Exception as error:
                registros_no_referencia.append(
                    {
                        "frame": original.name,
                        "niqe_original": np.nan,
                        "niqe_clean": np.nan,
                        "brisque_original": np.nan,
                        "brisque_clean": np.nan,
                        "error_metricas": str(error),
                    }
                )

    metricas = metricas.merge(
        pd.DataFrame(registros_no_referencia), on="frame", how="left"
    )
    metricas.to_csv(carpeta_salida / "metricas_frames.csv", index=False)

    tiempo_entrenamiento = float(
        (carpeta_salida / "tiempo_entrenamiento_minutos.txt").read_text().strip()
    )
    tiempo_inferencia = float(
        (carpeta_salida / "tiempo_inferencia_minutos.txt").read_text().strip()
    )

    etiqueta_original = os.environ.get(
        "N2N_ETIQUETA_ORIGINAL",
        (
            "frames_original"
            if modo == "original"
            else "frames_sin_hud"
        ),
    )
    etiqueta_clean = os.environ.get(
        "N2N_ETIQUETA_CLEAN",
        (
            "frames_n2n_original"
            if modo == "original"
            else "frames_n2n_sin_hud"
        ),
    )
    filas = pd.DataFrame(
        [
            {
                "conjunto": etiqueta_original,
                "cantidad_frames": len(metricas),
                "NIQE (menor)": metricas["niqe_original"].mean(),
                "BRISQUE (menor)": metricas["brisque_original"].mean(),
                "ruido_sigma (menor)": metricas["ruido_sigma_original"].mean(),
                "nitidez_laplaciana (comparar)": metricas["nitidez_original"].mean(),
                "contraste (comparar)": metricas["contraste_original"].mean(),
                "reduccion_sigma (mayor)": np.nan,
                "residual_RMS (diagnostico)": np.nan,
                "autocorrelacion_residual_abs (menor)": np.nan,
                "retencion_nitidez (cercano a 1)": np.nan,
                "tiempo_entrenamiento_minutos (menor)": np.nan,
                "tiempo_inferencia_minutos (menor)": np.nan,
            },
            {
                "conjunto": etiqueta_clean,
                "cantidad_frames": len(metricas),
                "NIQE (menor)": metricas["niqe_clean"].mean(),
                "BRISQUE (menor)": metricas["brisque_clean"].mean(),
                "ruido_sigma (menor)": metricas["ruido_sigma_clean"].mean(),
                "nitidez_laplaciana (comparar)": metricas["nitidez_clean"].mean(),
                "contraste (comparar)": metricas["contraste_clean"].mean(),
                "reduccion_sigma (mayor)": metricas["reduccion_sigma"].mean(),
                "residual_RMS (diagnostico)": metricas["residual_rms"].mean(),
                "autocorrelacion_residual_abs (menor)": metricas[
                    "residual_autocorrelacion_x"
                ].abs().mean(),
                "retencion_nitidez (cercano a 1)": metricas["retencion_nitidez"].mean(),
                "tiempo_entrenamiento_minutos (menor)": tiempo_entrenamiento,
                "tiempo_inferencia_minutos (menor)": tiempo_inferencia,
            },
        ]
    )

    ruta_resumen = CARPETA_VIDEO / "resumen.xlsx"
    ruta_lock = CARPETA_VIDEO / ".resumen.lock"
    with ruta_lock.open("w") as archivo_bloqueo:
        fcntl.flock(archivo_bloqueo.fileno(), fcntl.LOCK_EX)
        existente = (
            pd.read_excel(ruta_resumen, engine="openpyxl")
            if ruta_resumen.exists()
            else pd.DataFrame()
        )
        if not existente.empty and "conjunto" in existente.columns:
            existente = existente[~existente["conjunto"].isin(filas["conjunto"])]
        resumen = pd.concat([existente, filas], ignore_index=True)
        orden = [
            "frames_original",
            "frames_n2n_original",
            "frames_sin_hud",
            "frames_n2n_sin_hud",
        ]
        resumen["_orden"] = resumen["conjunto"].map(
            {nombre: indice for indice, nombre in enumerate(orden)}
        )
        resumen = resumen.sort_values("_orden").drop(columns="_orden")
        temporal = CARPETA_VIDEO / ".resumen_temporal.xlsx"
        resumen.to_excel(temporal, index=False, engine="openpyxl")
        temporal.replace(ruta_resumen)
        fcntl.flock(archivo_bloqueo.fileno(), fcntl.LOCK_UN)

    print(f"Metricas por frame: {carpeta_salida / 'metricas_frames.csv'}")
    print(f"Resumen: {ruta_resumen}")


# %% 7. Ejecucion
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", choices=("original", "sin_hud"), required=True)
    parser.add_argument(
        "--etapa",
        choices=("preparar", "entrenar", "inferir", "metricas"),
        required=True,
    )
    parser.add_argument("--reiniciar", action="store_true")
    argumentos = parser.parse_args()

    if argumentos.etapa == "preparar":
        preparar_dataset(argumentos.modo, argumentos.reiniciar)
    elif argumentos.etapa == "entrenar":
        entrenar(argumentos.modo)
    elif argumentos.etapa == "inferir":
        inferir(argumentos.modo)
    else:
        calcular_metricas(argumentos.modo)


if __name__ == "__main__":
    main()