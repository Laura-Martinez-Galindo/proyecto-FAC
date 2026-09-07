#!/usr/bin/env python3
"""Pipeline final CAREamics 0.3.2: preparacion, entrenamiento e inferencia."""
import argparse
import fcntl
import gc
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import tifffile
import torch
from careamics import CAREamist
from careamics.config.factories import create_advanced_n2v_config
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
RUTA_JSON = RAIZ / "config" / "videos.json"
EJES = "YXC"
N_CANALES = 3
PARCHES = (128, 128)
LOTE = 8
TILE = (256, 256)
OVERLAP = (48, 48)
AUMENTOS = ["x_flip", "y_flip", "rotate_90"]
SEMILLA = 42


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--modo", choices=("original", "sin_hud"), required=True)
    p.add_argument("--etapa", choices=("preparar", "entrenar", "inferir", "todo"), default="todo")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--pasos-por-epoca", type=int, default=500)
    p.add_argument("--reiniciar", action="store_true")
    return p.parse_args()


def cargar_json():
    with RUTA_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def guardar_json_bloqueado(datos):
    lock = RUTA_JSON.with_suffix(".lock")
    with lock.open("w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        actual = cargar_json()
        for video, contenido in datos.items():
            actual[video] = contenido
        tmp = RUTA_JSON.with_suffix(".tmp")
        tmp.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(RUTA_JSON)
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def absoluta(valor):
    p = Path(valor).expanduser()
    return (RAIZ / p).resolve() if not p.is_absolute() else p.resolve()


def relativa(p):
    try:
        return str(p.resolve().relative_to(RAIZ))
    except ValueError:
        return str(p.resolve())


def natural(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def listar(carpeta, limite=None):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    rutas = sorted((p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in exts), key=natural)
    return rutas[:limite] if limite else rutas


def leer_rgb(p):
    if p.suffix.lower() in {".tif", ".tiff"}:
        a = tifffile.imread(p)
    else:
        bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise RuntimeError(f"No se pudo leer {p}")
        if bgr.ndim == 2:
            a = np.repeat(bgr[:, :, None], 3, axis=2)
        elif bgr.shape[2] == 4:
            a = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
        else:
            a = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.shape[-1] == 4:
        a = a[:, :, :3]
    return np.clip(a, 0, 255).astype(np.uint8)


def rutas_base(a, cfg, sufijo):
    v = cfg[a.video]
    fuente_cfg = v.get("extraccion", {}) if a.modo == "original" else v.get("hud", {}).get("limpieza", {})
    if fuente_cfg.get("estado") != "completada":
        raise RuntimeError(f"La fuente {a.modo} no esta completada")
    fuente = absoluta(fuente_cfg["carpeta_salida"])
    base_video = absoluta(v["ruta"]).parent.parent
    nombre = sufijo if a.modo == "original" else f"{sufijo}_sin_hud"
    cache = RAIZ / "cache" / "denoising" / a.video / nombre
    return {
        "fuente": fuente, "salida": base_video / nombre, "cache": cache,
        "train": cache / "dataset" / "train", "val": cache / "dataset" / "val",
        "predict_input": cache / "dataset" / "predict",
        "work": cache / "trabajo", "ckpt": cache / "modelo.ckpt",
        "config": cache / "configuracion.json", "pred_tmp": cache / "predicciones_tiff",
        "nombre": nombre,
    }


def preparar_comun(a, r):
    frames = listar(r["fuente"], a.max_frames)
    if len(frames) < 10:
        raise RuntimeError("Se requieren al menos 10 frames")
    if a.reiniciar:
        shutil.rmtree(r["cache"], ignore_errors=True)
        shutil.rmtree(r["salida"], ignore_errors=True)
    shutil.rmtree(r["cache"] / "dataset", ignore_errors=True)
    r["train"].mkdir(parents=True, exist_ok=True)
    r["val"].mkdir(parents=True, exist_ok=True)
    r["predict_input"].mkdir(parents=True, exist_ok=True)

    cada = max(2, round(1 / 0.05))

    for i, frame in enumerate(
        tqdm(
            frames,
            desc="Preparando TIFF N2V",
            unit="frame",
        )
    ):
        imagen = leer_rgb(frame)
        nombre = f"{frame.stem}.tif"
        destino = (
            r["val"]
            if (i + 1) % cada == 0
            else r["train"]
        )

        tifffile.imwrite(
            destino / nombre,
            imagen,
            compression="deflate",
        )

        tifffile.imwrite(
            r["predict_input"] / nombre,
            imagen,
            compression="deflate",
        )
    meta = {"frames": len(frames), "train": len(list(r["train"].glob("*.tif"))), "val": len(list(r["val"].glob("*.tif")))}
    (r["cache"] / "preparacion.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


def checkpoint_final(r):
    candidatos = sorted(r["work"].rglob("*last.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidatos:
        candidatos = sorted(r["work"].rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidatos:
        raise RuntimeError("CAREamics no genero checkpoint")
    shutil.copy2(candidatos[-1], r["ckpt"])


def guardar_prediccion_png(file_path, img, *args, **kwargs):
    """Escribe directamente una prediccion CAREamics como PNG RGB."""
    destino = Path(file_path)

    if destino.suffix.lower() != ".png":
        destino = destino.with_suffix(".png")

    destino.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    imagen = np.asarray(img)
    forma_original = imagen.shape

    while (
        imagen.ndim > 3
        and imagen.shape[0] == 1
    ):
        imagen = imagen[0]

    if (
        imagen.ndim == 3
        and imagen.shape[0] in (1, 3, 4)
        and imagen.shape[-1] not in (1, 3, 4)
    ):
        imagen = np.moveaxis(
            imagen,
            0,
            -1,
        )

    if imagen.ndim == 2:
        imagen = np.repeat(
            imagen[:, :, None],
            3,
            axis=2,
        )

    if (
        imagen.ndim == 3
        and imagen.shape[-1] == 1
    ):
        imagen = np.repeat(
            imagen,
            3,
            axis=2,
        )

    if (
        imagen.ndim == 3
        and imagen.shape[-1] == 4
    ):
        imagen = imagen[:, :, :3]

    if (
        imagen.ndim != 3
        or imagen.shape[-1] != 3
    ):
        raise RuntimeError(
            f"Forma de prediccion no reconocida: "
            f"original={forma_original}, "
            f"transformada={imagen.shape}"
        )

    imagen = np.nan_to_num(
        imagen,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    imagen = np.clip(
        imagen,
        0,
        255,
    ).round().astype(np.uint8)

    temporal = destino.with_suffix(".png.tmp")

    correcta, codificada = cv2.imencode(
        ".png",
        cv2.cvtColor(
            imagen,
            cv2.COLOR_RGB2BGR,
        ),
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3,
        ],
    )

    if not correcta:
        raise RuntimeError(
            f"No se pudo codificar {destino}"
        )

    codificada.tofile(temporal)
    temporal.replace(destino)


def actualizar_registro(a, r, etiqueta, inicio, cantidad):
    """Actualiza solamente el modelo actual bajo bloqueo de archivo."""
    ruta_lock = RUTA_JSON.with_suffix(".lock")

    registro = {
        "nombre": etiqueta,
        "estado": "completada",
        "modo": a.modo,
        "carpeta_salida": relativa(
            r["salida"]
        ),
        "checkpoint": relativa(
            r["ckpt"]
        ),
        "configuracion": relativa(
            r["config"]
        ),
        "frames_procesados": cantidad,
        "fecha_ejecucion": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "tiempo_total_minutos": (
            time.monotonic() - inicio
        ) / 60.0,
    }

    with ruta_lock.open("w") as archivo_lock:
        fcntl.flock(
            archivo_lock.fileno(),
            fcntl.LOCK_EX,
        )

        configuracion = cargar_json()

        configuracion[a.video].setdefault(
            "modelos",
            {},
        )

        configuracion[a.video]["modelos"][
            r["nombre"]
        ] = registro

        temporal = RUTA_JSON.with_suffix(
            ".json.tmp"
        )

        temporal.write_text(
            json.dumps(
                configuracion,
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        temporal.replace(RUTA_JSON)

        fcntl.flock(
            archivo_lock.fileno(),
            fcntl.LOCK_UN,
        )


def crear_config(a, entrenamiento):
    """Crea la configuracion avanzada de Noise2Void."""
    trabajadores = (
        max(
            1,
            min(
                8,
                int(
                    os.environ.get(
                        "SLURM_CPUS_PER_TASK",
                        "8",
                    )
                ),
            ),
        )
        if entrenamiento
        else 4
    )

    trainer_params = {
        "accelerator": "gpu",
        "devices": 1,
        "strategy": "auto",
        "precision": "16-mixed",
        "enable_progress_bar": True,
        "log_every_n_steps": 10,
        "benchmark": True,
        "deterministic": False,
    }

    return create_advanced_n2v_config(
        experiment_name=f"n2v_{a.video}_{a.modo}",
        data_type="tiff",
        axes=EJES,
        patch_size=PARCHES,
        batch_size=LOTE,
        num_epochs=a.epocas,
        num_steps=(
            a.pasos_por_epoca
            if entrenamiento
            else None
        ),
        n_channels=N_CANALES,
        augmentations=AUMENTOS,
        n_val_patches=8,
        in_memory=False,
        independent_channels=False,
        normalization="mean_std",
        normalization_params={
            "per_channel": True,
        },
        use_n2v2=True,
        roi_size=11,
        masked_pixel_percentage=0.2,
        struct_n2v_axes="none",
        num_workers=trabajadores,
        trainer_params=trainer_params,
        model_params={
            "depth": 3,
            "num_channels_init": 48,
            "residual": False,
            "use_batch_norm": False,
        },
        optimizer="Adam",
        optimizer_params={
            "lr": 1e-4,
        },
        lr_scheduler="ReduceLROnPlateau",
        monitor_metric="val_loss",
        logger="tensorboard",
        seed=SEMILLA,
    )


def preparar(a, r):
    preparar_comun(a, r)


def entrenar(a, r):
    shutil.rmtree(r["work"], ignore_errors=True); r["work"].mkdir(parents=True, exist_ok=True)
    config = crear_config(a, True)
    r["config"].write_text(config.model_dump_json(indent=2), encoding="utf-8")
    careamist = CAREamist(config, work_dir=str(r["work"]), enable_progress_bar=True)
    careamist.train(train_data=str(r["train"]), val_data=str(r["val"]))
    checkpoint_final(r)


def inferir(a, r, inicio):
    """Predice directamente a PNG sin acumular resultados en memoria."""
    if not r["ckpt"].is_file():
        raise FileNotFoundError(
            f"No existe el checkpoint: {r['ckpt']}"
        )

    if not r["predict_input"].is_dir():
        raise FileNotFoundError(
            "No existe el conjunto TIFF de inferencia. "
            "Ejecute primero la etapa preparar."
        )

    cantidad_entrada = sum(
        1
        for archivo in r["predict_input"].iterdir()
        if archivo.is_file()
        and archivo.suffix.lower() in (".tif", ".tiff")
    )

    if cantidad_entrada == 0:
        raise RuntimeError(
            f"No hay TIFF de inferencia en {r['predict_input']}"
        )

    shutil.rmtree(
        r["salida"],
        ignore_errors=True,
    )

    r["salida"].mkdir(
        parents=True,
        exist_ok=True,
    )

    config = crear_config(
        a,
        entrenamiento=False,
    )

    careamist = CAREamist(
        config,
        work_dir=str(r["work"]),
        enable_progress_bar=True,
    )

    careamist.predict_to_disk(
        pred_data=str(r["predict_input"]),
        prediction_dir=r["salida"],
        batch_size=1,
        tile_size=TILE,
        tile_overlap=OVERLAP,
        axes=EJES,
        data_type="tiff",
        num_workers=4,
        in_memory=False,
        checkpoint=r["ckpt"],
        write_type="custom",
        write_extension=".png",
        write_func=guardar_prediccion_png,
        write_func_kwargs={},
    )

    cantidad_salida = sum(
        1
        for archivo in r["salida"].iterdir()
        if archivo.is_file()
        and archivo.name.startswith("frame_")
        and archivo.suffix.lower() == ".png"
    )

    if cantidad_salida != cantidad_entrada:
        raise RuntimeError(
            f"Se esperaban {cantidad_entrada} PNG "
            f"y se encontraron {cantidad_salida}."
        )

    etiqueta = (
        "Noise2Void"
        if a.modo == "original"
        else "Noise2Void sin HUD"
    )

    actualizar_registro(
        a,
        r,
        etiqueta,
        inicio,
        cantidad_salida,
    )

    print(
        f"Inferencia completada: {cantidad_salida} PNG.",
        flush=True,
    )

    del careamist
    del config

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def main():
    a = argumentos(); inicio = time.monotonic(); cfg = cargar_json(); r = rutas_base(a, cfg, "n2v")
    if a.etapa in ("preparar", "todo"): preparar(a, r)
    if a.etapa in ("entrenar", "todo"): entrenar(a, r)
    if a.etapa in ("inferir", "todo"): inferir(a, r, inicio)

if __name__ == "__main__":
    main()