#!/usr/bin/env python3
"""
Experimentos de denoising con UNA sola interfaz para todos los metodos.

  metodos    lista los metodos y sus valores por defecto
  entrenar   entrena un metodo con la lista de frames de una familia
  inferir    limpia una lista de imagenes del dataset (o todas) con un checkpoint
             (o con un metodo clasico, que no se entrena)

Metodos: `python scripts/denoising.py metodos` los lista todos (ronda 3: ~25 espaciales y temporales).
Ver scripts/fac_denoise/metodos.py para referencias y que hace cada uno.

Datos
  * Entrenamiento: --lista resultados/denoising/listas/<FAMILIA>_entrenamiento.csv
    (video, frame). Se leen los frames ORIGINALES a resolucion completa y los parches se
    sortean fuera del HUD (mascaras_hud). Ninguno de esos frames esta a menos de 90 frames de
    una imagen del dataset (lo garantiza scripts/listas_denoising.py).
  * Inferencia: --lista <FAMILIA>_evaluacion.csv (imagen, video, frame) y --entrada
    datasets/dataset_sin_hud/images. Se limpia la imagen del dataset tal cual; UDVD recibe
    ademas los vecinos +-2 construidos sin HUD igual que el dataset (Fuente.compuesto).
    Salida: <salida>/<imagen>.png, mismo nombre que en el dataset.

Todo lo que define el experimento va por parametro y queda en configuracion.json y dentro
del checkpoint. Se guardan instantaneas en --instantaneas (p.ej. 2000 4000) ademas del final.

Ejemplos:
  python scripts/denoising.py entrenar --metodo b2u --familia OPTICO \\
      --lista resultados/denoising/listas/OPTICO_entrenamiento.csv \\
      --salida cache/denoising/OPTICO/b2u_r1 --pasos 4000 --amp
  python scripts/denoising.py inferir --checkpoint cache/denoising/OPTICO/b2u_r1/modelo.pt \\
      --lista resultados/denoising/listas/OPTICO_evaluacion.csv \\
      --entrada datasets/dataset_sin_hud/images --salida cache/denoising/OPTICO/b2u_r1/salidas
  python scripts/denoising.py inferir --metodo nlm --nlm-h 6 --lista ... --entrada ... --salida ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fac_denoise.datos import Fuente, Parches, a_canales, guardar, leer, leer_lista  # noqa: E402
from fac_denoise.metodos import METODOS, argumentos_todos  # noqa: E402
import fac_denoise.metodos_tanda2  # noqa: E402,F401  (registra los metodos de la tanda 2)
import fac_denoise.metodos_tanda3  # noqa: E402,F401  (temporales)
import fac_denoise.metodos_tanda4  # noqa: E402,F401  (tanda 4)

import cv2  # noqa: E402


# ---------------------------------------------------------------------------
def mosaicos(fn, x, tile, overlap, multiplo):
    """Aplica fn por mosaicos de tamano fijo (multiplo de `multiplo`) con mezcla suave.
    x: 1 x Cin x H x W. Devuelve 1 x Cout x H x W."""
    _, _, H, W = x.shape
    tile = max(multiplo, tile - tile % multiplo)
    # si la imagen es mas chica que el mosaico, se rellena por reflexion
    ph, pw = max(0, tile - H), max(0, tile - W)
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")
    _, _, Hp, Wp = x.shape
    paso = max(1, tile - overlap)
    ys = list(range(0, Hp - tile + 1, paso))
    xs = list(range(0, Wp - tile + 1, paso))
    if ys[-1] + tile < Hp:
        ys.append(Hp - tile)
    if xs[-1] + tile < Wp:
        xs.append(Wp - tile)
    rampa = torch.ones(tile, device=x.device)
    if overlap > 0:
        r = torch.linspace(0, 1, overlap + 2, device=x.device)[1:-1]
        rampa[:overlap], rampa[-overlap:] = r, r.flip(0)
    w2d = (rampa[:, None] * rampa[None, :])[None, None]
    out = peso = None
    for y in ys:
        for xx in xs:
            o = fn(x[:, :, y:y + tile, xx:xx + tile]).float()
            if out is None:
                out = torch.zeros(1, o.shape[1], Hp, Wp, device=x.device)
                peso = torch.zeros(1, 1, Hp, Wp, device=x.device)
            out[:, :, y:y + tile, xx:xx + tile] += o * w2d
            peso[:, :, y:y + tile, xx:xx + tile] += w2d
    return (out / peso.clamp_min(1e-6))[:, :, :H, :W]


def semillas(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
def cmd_metodos(a):
    for n, m in METODOS.items():
        print(f"{n:13s} frames={m.temporal}  entrenable={m.entrenable}  parche={m.parche_defecto}  "
              f"batch={m.batch_defecto}  lr={m.lr_defecto}")


def cmd_entrenar(a):
    met = METODOS[a.metodo]
    if not met.entrenable:
        sys.exit(f"{a.metodo} no se entrena: usar directamente 'inferir --metodo {a.metodo}'.")
    a.parche = a.parche or met.parche_defecto
    a.batch = a.batch or met.batch_defecto
    a.lr = a.lr or met.lr_defecto
    disp = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    semillas(a.semilla)
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)

    filas = leer_lista(a.lista)
    if a.max_frames and len(filas) > a.max_frames:
        filas = random.Random(a.semilla).sample(filas, a.max_frames)
    videos = sorted({r["video"] for r in filas})
    fuentes = {v: Fuente(v, margen_hud=a.margen_hud) for v in videos}
    for f in fuentes.values():
        print("  " + f.resumen())
    muestras = [(r["video"], int(r["frame"])) for r in filas if int(r["frame"]) in fuentes[r["video"]].idx_orig]
    print(f"{len(muestras)}/{len(filas)} frames centrales disponibles en disco "
          f"({', '.join(f'{v}={sum(1 for m in muestras if m[0] == v)}' for v in videos)})")
    if not muestras:
        sys.exit("No hay frames para entrenar.")

    red = met.crear(a, a.canales).to(disp)
    if disp.type == "cuda":  # si el batch no cabe en la GPU, se reduce a la mitad (queda registrado)
        a.batch_pedido = a.batch
        while True:
            try:
                t = a.parche - a.parche % met.multiplo
                x = torch.rand(a.batch, met.temporal * a.canales, t, t, device=disp) * met.escala
                with torch.autocast(disp.type, enabled=a.amp):
                    met.perdida(red, x, 0.5, a).backward()
                red.zero_grad(set_to_none=True)
                del x
                torch.cuda.empty_cache()
                print(f"memoria GPU usada en la prueba: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
                break
            except torch.cuda.OutOfMemoryError:
                red.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if a.batch < 2:
                    sys.exit("No cabe ni un parche en la GPU: bajar --parche.")
                a.batch //= 2
                print(f"!! sin memoria en GPU: batch -> {a.batch}")
    a.parches_por_carga = math.gcd(a.batch, a.parches_por_carga)  # el batch debe ser multiplo
    ds = Parches(muestras, fuentes, a.canales, a.parche, a.pasos * a.batch // a.parches_por_carga,
                 temporal=met.temporal, por_carga=a.parches_por_carga, multiplo=met.multiplo)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch // a.parches_por_carga, num_workers=a.workers,
                                     pin_memory=disp.type == "cuda", drop_last=True,
                                     persistent_workers=a.workers > 0)
    n_par = sum(p.numel() for p in red.parameters())
    print(f"{a.metodo}: {n_par / 1e6:.2f} M parametros | parche {a.parche} | batch {a.batch} | lr {a.lr} | "
          f"{a.pasos} pasos | {disp}")
    opt = (torch.optim.AdamW if getattr(met, "optimizador", "adam") == "adamw" else torch.optim.Adam)(red.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.pasos, eta_min=a.lr * a.lr_final)
    usar_amp = a.amp and disp.type == "cuda"
    escalador = torch.amp.GradScaler("cuda", enabled=usar_amp)
    log = open(sal / "entrenamiento.csv", "w")
    log.write("paso,loss,lr,segundos\n")

    def guardar_ck(ruta, paso):
        torch.save({"metodo": a.metodo, "modelo": red.state_dict(), "canales": a.canales,
                    "args": vars(a), "paso": paso}, ruta)

    t0, media = time.time(), None
    red.train()
    for paso, x in enumerate(dl, 1):
        x = x.to(disp, non_blocking=True).flatten(0, 1) * met.escala
        with torch.autocast(disp.type, enabled=usar_amp):
            loss = met.perdida(red, x, paso / a.pasos, a)
        opt.zero_grad(set_to_none=True)
        escalador.scale(loss).backward()
        escalador.step(opt)
        escalador.update()
        sched.step()
        v = loss.item() / met.escala
        media = v if media is None else 0.98 * media + 0.02 * v
        if paso % a.log_cada == 0 or paso == a.pasos:
            log.write(f"{paso},{media:.6g},{sched.get_last_lr()[0]:.2e},{time.time() - t0:.0f}\n")
            log.flush()
            print(f"paso {paso}/{a.pasos}  loss {media:.6g}  ({time.time() - t0:.0f} s)", flush=True)
        if paso in a.instantaneas:
            guardar_ck(sal / f"modelo_p{paso}.pt", paso)
        if paso % a.guardar_cada == 0 or paso == a.pasos:
            guardar_ck(sal / "modelo.pt", paso)
        if paso >= a.pasos:
            break
    meta = {"frames_entrenamiento": len(muestras), "segundos": round(time.time() - t0),
            "parametros_M": round(n_par / 1e6, 3), **vars(a)}
    (sal / "configuracion.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Checkpoint: {sal / 'modelo.pt'}  ({time.time() - t0:.0f} s)")


def cmd_inferir(a):
    disp = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location=disp, weights_only=False)
        conf = Namespace(**{**ck["args"], **{k: v for k, v in vars(a).items() if k.startswith("b2u_beta")
                                              and v}})
        met = METODOS[ck["metodo"]]
        canales = int(ck["canales"])
        red = met.crear(conf, canales).to(disp)
        red.load_state_dict(ck["modelo"], strict=True)  # si algo no carga, falla (no en silencio)
        red.eval()
        print(f"{ck['metodo']} paso {ck['paso']} ({a.checkpoint})")
    else:
        if not a.metodo or METODOS[a.metodo].entrenable:
            sys.exit("Sin --checkpoint solo se pueden usar metodos clasicos o zero-shot.")
        met, conf, canales, red = METODOS[a.metodo], a, a.canales, None

    filas = leer_lista(a.lista) if a.lista else [{"imagen": p.name} for p in sorted(Path(a.entrada).glob("*.png"))]
    if a.max_imagenes:
        filas = filas[:a.max_imagenes]
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    if a.residuo:
        (sal / "residuo").mkdir(exist_ok=True)
    fuentes = {}
    t0 = time.time()
    vecinos_faltantes = 0
    zs = getattr(met, "zero_shot", False)
    if red is None and not zs and a.workers > 1:  # clasicos (BM3D, NLM...): una imagen por proceso
        from concurrent.futures import ProcessPoolExecutor
        tareas = [(Path(r["imagen"]).stem + ".png", str(Path(a.entrada)), str(sal), canales, met.nombre, conf,
                   a.residuo, a.sobrescribir) for r in filas]
        with ProcessPoolExecutor(a.workers) as ex:
            for i, _ in enumerate(ex.map(_clasico_uno, tareas, chunksize=1), 1):
                if i % 20 == 0 or i == len(tareas):
                    print(f"  {i}/{len(tareas)}  ({time.time() - t0:.0f} s)", flush=True)
        filas = []
    for i, r in enumerate(filas, 1):
        nombre = Path(r["imagen"]).stem + ".png"
        dst = sal / nombre
        if dst.exists() and not a.sobrescribir:
            continue
        centro = leer(str(Path(a.entrada) / nombre), canales)
        if centro is None:
            print(f"!! no se pudo leer {nombre}")
            continue
        if zs:  # zero-shot: una red nueva entrenada sobre ESTA imagen
            semillas(i)
            x = torch.from_numpy(centro.transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(disp)
            y = met.zero_shot_imagen(x, conf, disp)
            out = np.rint(y[0].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        elif red is None:
            out = met.clasico(centro, conf)
        else:
            ims = [centro]
            if met.temporal > 1:
                v, k = r["video"], int(r["frame"])
                if v not in fuentes:
                    fuentes[v] = Fuente(v)
                ims = []
                for d in range(-(met.temporal // 2), met.temporal // 2 + 1):
                    im = centro if d == 0 else fuentes[v].compuesto(k + d, canales)
                    if im is None or im.shape != centro.shape:
                        im, vecinos_faltantes = centro, vecinos_faltantes + 1
                    ims.append(im)
            x = torch.from_numpy(np.concatenate(ims, 2).transpose(2, 0, 1).astype(np.float32) / 255.0)[None]
            x = x.to(disp) * met.escala

            def fn(t):
                with torch.no_grad(), torch.autocast(disp.type, enabled=a.amp and disp.type == "cuda"):
                    return met.aplicar(red, t, conf)

            y = mosaicos(fn, x, a.tile, a.overlap, getattr(met, "multiplo_mosaico", max(met.multiplo, 32 if met.nombre not in ("apbsn", "mmbsn") else 2)))
            y = (y[0] / met.escala).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0
            out = np.rint(y).astype(np.uint8)
        guardar(dst, out)
        if a.residuo:  # ruidoso - limpio, centrado en 128 y amplificado x4
            res = np.clip((centro.astype(np.float32) - out.astype(np.float32)) * 4 + 128, 0, 255).astype(np.uint8)
            guardar(sal / "residuo" / nombre, res)
        if i % 20 == 0 or i == len(filas):
            print(f"  {i}/{len(filas)}  ({time.time() - t0:.0f} s)", flush=True)
    if vecinos_faltantes:
        print(f"!! {vecinos_faltantes} vecinos no se pudieron construir sin HUD (se uso el frame central)")
    info = {"checkpoint": a.checkpoint, "metodo": met.nombre, "imagenes": len(filas),
            "segundos": round(time.time() - t0), "vecinos_faltantes": vecinos_faltantes,
            "config": {k: v for k, v in vars(conf).items()}}
    (sal / "inferencia.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"Listo: {sal}")


def _clasico_uno(t):
    nombre, entrada, sal, canales, metodo, conf, residuo, sobrescribir = t
    dst = Path(sal) / nombre
    if dst.exists() and not sobrescribir:
        return
    centro = leer(str(Path(entrada) / nombre), canales)
    if centro is None:
        print(f"!! no se pudo leer {nombre}")
        return
    out = METODOS[metodo].clasico(centro, conf)
    guardar(dst, out)
    if residuo:
        res = np.clip((centro.astype(np.float32) - out.astype(np.float32)) * 4 + 128, 0, 255).astype(np.uint8)
        guardar(Path(sal) / "residuo" / nombre, res)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("metodos")

    e = sub.add_parser("entrenar", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    e.add_argument("--metodo", required=True, choices=list(METODOS))
    e.add_argument("--familia", default="", help="solo informativo (queda en la configuracion)")
    e.add_argument("--lista", required=True, help="CSV video,frame (listas_denoising.py)")
    e.add_argument("--salida", required=True)
    e.add_argument("--max-frames", type=int, default=0, help="usar solo N frames de la lista (0 = todos)")
    e.add_argument("--canales", type=int, default=3, choices=[1, 3])
    e.add_argument("--parche", type=int, default=0, help="0 = el del metodo")
    e.add_argument("--batch", type=int, default=0, help="0 = el del metodo")
    e.add_argument("--parches-por-carga", type=int, default=4,
                   help="parches que se sacan de cada frame leido (acelera la lectura)")
    e.add_argument("--margen-hud", type=int, default=8, help="distancia minima (px) del parche al HUD")
    e.add_argument("--pasos", type=int, default=4000)
    e.add_argument("--lr", type=float, default=0.0, help="0 = el del metodo")
    e.add_argument("--lr-final", type=float, default=0.05, help="lr final = lr * esto (coseno)")
    e.add_argument("--instantaneas", type=int, nargs="*", default=[], help="pasos donde guardar copia")
    e.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    e.add_argument("--log-cada", type=int, default=50)
    e.add_argument("--guardar-cada", type=int, default=500)
    e.add_argument("--semilla", type=int, default=0)
    e.add_argument("--amp", action="store_true", help="precision mixta en GPU")
    e.add_argument("--cpu", action="store_true")
    argumentos_todos(e)

    i = sub.add_parser("inferir", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    i.add_argument("--checkpoint", help="modelo.pt (metodos entrenables)")
    i.add_argument("--metodo", choices=list(METODOS), help="solo para metodos clasicos sin checkpoint")
    i.add_argument("--entrada", required=True, help="carpeta con las imagenes a limpiar (dataset_sin_hud/images)")
    i.add_argument("--lista", help="CSV con columna imagen (y video, frame para UDVD); sin lista = todas")
    i.add_argument("--salida", required=True)
    i.add_argument("--canales", type=int, default=3, choices=[1, 3], help="solo metodos clasicos")
    i.add_argument("--max-imagenes", type=int, default=0)
    i.add_argument("--tile", type=int, default=512)
    i.add_argument("--overlap", type=int, default=48)
    i.add_argument("--residuo", action="store_true", help="guardar tambien ruidoso - limpio (x4)")
    i.add_argument("--sobrescribir", action="store_true")
    i.add_argument("--amp", action="store_true")
    i.add_argument("--cpu", action="store_true")
    i.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                   help="procesos para los metodos clasicos (una imagen por proceso)")
    argumentos_todos(i)

    a = ap.parse_args()
    {"metodos": cmd_metodos, "entrenar": cmd_entrenar, "inferir": cmd_inferir}[a.cmd](a)


if __name__ == "__main__":
    main()
