#!/usr/bin/env python3
"""Pipeline Noise2Noise (CAREamics 0.3.2) con filtrado de movimiento y CLI configurable."""
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
from careamics.config.factories import create_advanced_n2n_config
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
RUTA_JSON = RAIZ / "config" / "videos.json"
EJES = "YXC"
N_CANALES = 3
AUMENTOS = ["x_flip", "y_flip", "rotate_90"]
SEMILLA = 42


def argumentos():
    p = argparse.ArgumentParser(description="Pipeline Noise2Noise (N2N) para video FLIR.")
    p.add_argument("--video", required=True, help="ID del video en config/videos.json (ej. video1).")
    p.add_argument("--modo", choices=("original", "sin_hud"), required=True, help="Fuente de frames: original o sin_hud.")
    p.add_argument("--etapa", choices=("preparar", "entrenar", "inferir", "todo"), default="todo")
    p.add_argument("--max-frames", type=int, help="Limite maximo de frames a procesar.")
    p.add_argument("--epocas", type=int, default=30, help="Numero de epocas de entrenamiento.")
    p.add_argument("--pasos-por-epoca", type=int, default=500, help="Pasos por epoca.")
    p.add_argument("--depth", type=int, default=4, help="Profundidad de la UNet (default: 4).")
    p.add_argument("--num-channels-init", type=int, default=48, help="Filtros iniciales de la UNet (default: 48).")
    p.add_argument("--lr", type=float, default=1e-4, help="Tasa de aprendizaje (default: 1e-4).")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8).")
    p.add_argument("--patch-size", type=int, default=128, help="Tamano del parche cuadrado (default: 128).")
    p.add_argument("--filtrar-movimiento", action="store_true", default=True, help="Filtrar parejas con movimiento brusco (default: True).")
    p.add_argument("--sin-filtro-movimiento", dest="filtrar_movimiento", action="store_false", help="Desactivar filtrado de movimiento.")
    p.add_argument("--umbral-movimiento", type=float, default=3.5, help="Umbral maximo de flujo optico en pixeles (default: 3.5).")
    p.add_argument("--umbral-diferencia", type=float, default=60.0, help="Umbral maximo de diferencia absoluta media (default: 60.0).")
    p.add_argument("--id-experimento", help="Nombre personalizado del experimento/carpeta de salida (opcional).")
    p.add_argument("--reiniciar", action="store_true", help="Borrar cache previo antes de ejecutar.")
    return p.parse_args()


def cargar_json():
    with RUTA_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def estimar_movimiento_pareja(img1, img2):
    """Estima diferencia media y magnitud de flujo optico Farneback entre dos frames."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    diff = float(np.mean(np.abs(g1.astype(np.float32) - g2.astype(np.float32))))

    # Downsample para calculo rapido de flujo
    escala = 0.25
    p1 = cv2.resize(g1, (0, 0), fx=escala, fy=escala)
    p2 = cv2.resize(g2, (0, 0), fx=escala, fy=escala)
    flow = cv2.calcOpticalFlowFarneback(p1, p2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))) / escala
    return diff, mag


def rutas_base(a, cfg, sufijo):
    v = cfg[a.video]
    fuente_cfg = v.get("extraccion", {}) if a.modo == "original" else v.get("hud", {}).get("limpieza", {})
    if fuente_cfg.get("estado") != "completada":
        raise RuntimeError(f"La fuente {a.modo} no esta completada")
    fuente = absoluta(fuente_cfg["carpeta_salida"])
    base_video = absoluta(v["ruta"]).parent.parent

    if a.id_experimento:
        nombre = a.id_experimento
    else:
        nombre = sufijo if a.modo == "original" else f"{sufijo}_sin_hud"

    cache = RAIZ / "cache" / "denoising" / a.video / nombre
    return {
        "fuente": fuente, "salida": base_video / nombre, "cache": cache,
        "train": cache / "dataset" / "train", "val": cache / "dataset" / "val",
        "predict_input": cache / "dataset" / "predict",
        "work": cache / "trabajo", "ckpt": cache / "modelo.ckpt",
        "config": cache / "configuracion.json",
        "nombre": nombre,
    }


def checkpoint_final(r):
    candidatos = sorted(r["work"].rglob("*last.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidatos:
        candidatos = sorted(r["work"].rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidatos:
        raise RuntimeError("CAREamics no genero checkpoint")
    shutil.copy2(candidatos[-1], r["ckpt"])


def guardar_prediccion_png(file_path, img, *args, **kwargs):
    destino = Path(file_path)
    if destino.suffix.lower() != ".png":
        destino = destino.with_suffix(".png")

    destino.parent.mkdir(parents=True, exist_ok=True)
    imagen = np.asarray(img)

    while imagen.ndim > 3 and imagen.shape[0] == 1:
        imagen = imagen[0]

    if imagen.ndim == 3 and imagen.shape[0] in (1, 3, 4) and imagen.shape[-1] not in (1, 3, 4):
        imagen = np.moveaxis(imagen, 0, -1)

    if imagen.ndim == 2:
        imagen = np.repeat(imagen[:, :, None], 3, axis=2)
    if imagen.ndim == 3 and imagen.shape[-1] == 1:
        imagen = np.repeat(imagen, 3, axis=2)
    if imagen.ndim == 3 and imagen.shape[-1] == 4:
        imagen = imagen[:, :, :3]

    imagen = np.nan_to_num(imagen, nan=0.0, posinf=255.0, neginf=0.0)
    imagen = np.clip(imagen, 0, 255).round().astype(np.uint8)

    temporal = destino.with_suffix(".png.tmp")
    correcta, codificada = cv2.imencode(".png", cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not correcta:
        raise RuntimeError(f"No se pudo codificar {destino}")

    codificada.tofile(temporal)
    temporal.replace(destino)


def actualizar_registro(a, r, etiqueta, inicio, cantidad):
    ruta_lock = RUTA_JSON.with_suffix(".lock")
    registro = {
        "nombre": etiqueta,
        "estado": "completada",
        "modo": a.modo,
        "carpeta_salida": relativa(r["salida"]),
        "checkpoint": relativa(r["ckpt"]),
        "configuracion": relativa(r["config"]),
        "frames_procesados": cantidad,
        "depth": a.depth,
        "lr": a.lr,
        "filtrado_movimiento": a.filtrar_movimiento,
        "fecha_ejecucion": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tiempo_total_minutos": (time.monotonic() - inicio) / 60.0,
    }

    with ruta_lock.open("w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        configuracion = cargar_json()
        configuracion[a.video].setdefault("modelos", {})
        configuracion[a.video]["modelos"][r["nombre"]] = registro
        temporal = RUTA_JSON.with_suffix(".json.tmp")
        temporal.write_text(json.dumps(configuracion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporal.replace(RUTA_JSON)
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def crear_config(a, entrenamiento):
    trabajadores = max(1, min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))) if entrenamiento else 4
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

    return create_advanced_n2n_config(
        experiment_name=f"n2n_{a.video}_{a.modo}_{r_nombre(a)}",
        data_type="tiff",
        axes=EJES,
        patch_size=(a.patch_size, a.patch_size),
        batch_size=a.batch_size,
        num_epochs=a.epocas,
        num_steps=(a.pasos_por_epoca if entrenamiento else None),
        n_channels_in=N_CANALES,
        n_channels_out=N_CANALES,
        augmentations=AUMENTOS,
        n_val_patches=8,
        in_memory=False,
        independent_channels=False,
        normalization="mean_std",
        normalization_params={"per_channel": True},
        num_workers=trabajadores,
        trainer_params=trainer_params,
        model_params={
            "depth": a.depth,
            "num_channels_init": a.num_channels_init,
            "residual": True,
            "use_batch_norm": False,
        },
        optimizer="Adam",
        optimizer_params={"lr": a.lr},
        lr_scheduler="ReduceLROnPlateau",
        logger="tensorboard",
        seed=SEMILLA,
    )


def r_nombre(a):
    return a.id_experimento or ("n2n" if a.modo == "original" else "n2n_sin_hud")


def preparar(a, r):
    frames = listar(r["fuente"], a.max_frames)
    if len(frames) < 10:
        raise RuntimeError("Se requieren al menos 10 frames")
    if a.reiniciar:
        shutil.rmtree(r["cache"], ignore_errors=True)
        shutil.rmtree(r["salida"], ignore_errors=True)

    shutil.rmtree(r["cache"] / "dataset", ignore_errors=True)
    for c in (r["train"] / "input", r["train"] / "target", r["val"] / "input", r["val"] / "target", r["predict_input"]):
        c.mkdir(parents=True, exist_ok=True)

    for frame in tqdm(frames, desc="Preparando TIFF inferencia N2N", unit="frame"):
        tifffile.imwrite(r["predict_input"] / f"{frame.stem}.tif", leer_rgb(frame), compression="deflate")

    cada = max(2, round(1 / 0.05))
    total_creadas = 0
    total_excluidas = 0

    print(f"Preparando parejas N2N (filtrar_movimiento={a.filtrar_movimiento})...")
    for i in tqdm(range(len(frames) - 1), desc="Parejas N2N", unit="pareja"):
        img_a = leer_rgb(frames[i])
        img_b = leer_rgb(frames[i + 1])

        if a.filtrar_movimiento:
            diff, mag = estimar_movimiento_pareja(img_a, img_b)
            if mag > a.umbral_movimiento or diff > a.umbral_diferencia:
                total_excluidas += 1
                continue

        base = r["val"] if (total_creadas + 1) % cada == 0 else r["train"]
        nombre = f"pareja_{total_creadas:06d}.tif"
        tifffile.imwrite(base / "input" / nombre, img_a, compression="deflate")
        tifffile.imwrite(base / "target" / nombre, img_b, compression="deflate")
        total_creadas += 1

    meta = {
        "frames_totales": len(frames),
        "parejas_creadas": total_creadas,
        "parejas_excluidas": total_excluidas,
        "filtrado_activo": a.filtrar_movimiento,
        "umbral_movimiento": a.umbral_movimiento,
        "train": len(list((r["train"] / "input").glob("*.tif"))),
        "val": len(list((r["val"] / "input").glob("*.tif"))),
    }
    (r["cache"] / "preparacion.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)

    if total_creadas < 5:
        raise RuntimeError("No se generaron suficientes parejas validas. Ajuste los umbrales de movimiento.")


def entrenar(a, r):
    shutil.rmtree(r["work"], ignore_errors=True)
    r["work"].mkdir(parents=True, exist_ok=True)
    config = crear_config(a, True)
    r["config"].write_text(config.model_dump_json(indent=2), encoding="utf-8")
    careamist = CAREamist(config, work_dir=str(r["work"]), enable_progress_bar=True)
    careamist.train(
        train_data=str(r["train"] / "input"), train_data_target=str(r["train"] / "target"),
        val_data=str(r["val"] / "input"), val_data_target=str(r["val"] / "target")
    )
    checkpoint_final(r)


def inferir(a, r, inicio):
    if not r["ckpt"].is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {r['ckpt']}")
    if not r["predict_input"].is_dir():
        raise FileNotFoundError("No existe predict_input. Ejecute primero preparar.")

    cantidad_entrada = sum(1 for f in r["predict_input"].iterdir() if f.is_file() and f.suffix.lower() in (".tif", ".tiff"))
    if cantidad_entrada == 0:
        raise RuntimeError("No hay TIFF de inferencia.")

    shutil.rmtree(r["salida"], ignore_errors=True)
    r["salida"].mkdir(parents=True, exist_ok=True)

    config = crear_config(a, entrenamiento=False)
    careamist = CAREamist(config, work_dir=str(r["work"]), enable_progress_bar=True)

    careamist.predict_to_disk(
        pred_data=str(r["predict_input"]),
        prediction_dir=r["salida"],
        batch_size=4,
        tile_size=(512, 512),
        tile_overlap=(48, 48),
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

    cantidad_salida = sum(1 for f in r["salida"].iterdir() if f.is_file() and f.suffix.lower() == ".png")
    if cantidad_salida != cantidad_entrada:
        raise RuntimeError(f"Se esperaban {cantidad_entrada} PNG y se obtuvieron {cantidad_salida}.")

    etiqueta = a.id_experimento or ("Noise2Noise" if a.modo == "original" else "Noise2Noise sin HUD")
    actualizar_registro(a, r, etiqueta, inicio, cantidad_salida)
    print(f"Inferencia N2N completada: {cantidad_salida} frames guardados en {r['salida']}.")

    del careamist, config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    a = argumentos()
    inicio = time.monotonic()
    cfg = cargar_json()
    r = rutas_base(a, cfg, "n2n")

    if a.etapa in ("preparar", "todo"):
        preparar(a, r)
    if a.etapa in ("entrenar", "todo"):
        entrenar(a, r)
    if a.etapa in ("inferir", "todo"):
        inferir(a, r, inicio)


if __name__ == "__main__":
    main()
