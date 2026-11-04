#!/usr/bin/env python3
"""
Cuenta los frames de video3 en cada etapa y verifica la repeticion del clip.

video_2hour.mp4 son 186.520 frames = 8 vueltas identicas de un clip de 23.315 frames
(12:57 a 30 fps). Solo hace falta la primera vuelta (frames 1 a 23.315).

Que hace:
  1. Cuenta archivos y el rango de numeros de frame en frames_originales, mascaras_hud,
     frames_sin_hud, frames_sin_hud.tmp, udvd_sin_hud (y cualquier otra carpeta de
     videos/video3 que tenga frame_N.*).
  2. Dice si la primera vuelta (1..23.315) esta completa en cada carpeta y cuales faltan.
  3. Verifica con una muestra que la vuelta 2..8 es identica a la 1 en frames_originales.
  4. Con --limpiar, borra (o mueve con --mover-a) los frames > 23.315 de las carpetas
     que se indiquen. SIN --limpiar no borra nada.

Uso:
    python scripts/contar_frames_video3.py
    python scripts/contar_frames_video3.py --limpiar --carpetas frames_originales mascaras_hud frames_sin_hud
    python scripts/contar_frames_video3.py --limpiar --mover-a /ruta/respaldo_video3_vueltas
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CLIP = 23315
PATRON = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")


def indexar(carpeta: Path) -> dict[int, Path]:
    idx = {}
    with os.scandir(carpeta) as it:
        for e in it:
            if e.is_file():
                m = PATRON.search(e.name)
                if m:
                    idx[int(m.group(1))] = Path(e.path)
    return idx


def rangos(nums: list[int]) -> str:
    if not nums:
        return "-"
    nums = sorted(nums)
    out, ini, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n != prev + 1:
            out.append(f"{ini}-{prev}" if ini != prev else str(ini))
            ini = n
        prev = n
    out.append(f"{ini}-{prev}" if ini != prev else str(ini))
    return ", ".join(out[:15]) + (" ..." if len(out) > 15 else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", default=str(RAIZ / "videos" / "video3"))
    ap.add_argument("--clip", type=int, default=CLIP, help="Frames de una vuelta del clip")
    ap.add_argument("--muestras", type=int, default=20, help="Frames a comparar entre vueltas")
    ap.add_argument("--limpiar", action="store_true", help="Borrar/mover frames > clip")
    ap.add_argument("--carpetas", nargs="*", default=None, help="Carpetas a limpiar (defecto: todas)")
    ap.add_argument("--mover-a", default=None, help="En vez de borrar, mover a esta carpeta")
    a = ap.parse_args()

    base = Path(a.video_dir)
    carpetas = sorted(p for p in base.iterdir() if p.is_dir())
    resumen = {}
    print(f"{'carpeta':28s} {'archivos':>9s} {'min':>7s} {'max':>7s}  vuelta1 completa  faltan(1..{a.clip})")
    indices = {}
    for c in carpetas:
        idx = indexar(c)
        if not idx:
            continue
        indices[c.name] = idx
        v1 = set(range(1, a.clip + 1))
        faltan = sorted(v1 - idx.keys())
        mas_alla = sum(1 for k in idx if k > a.clip)
        print(f"{c.name:28s} {len(idx):9d} {min(idx):7d} {max(idx):7d}  {'SI' if not faltan else 'NO':>8s}"
              f"  {len(faltan)} [{rangos(faltan)}]   (frames > {a.clip}: {mas_alla})")
        resumen[c.name] = {"archivos": len(idx), "min": min(idx), "max": max(idx),
                           "vuelta1_completa": not faltan, "faltan_vuelta1": len(faltan),
                           "frames_despues_de_vuelta1": mas_alla}

    # verificacion de repeticion
    if "frames_originales" in indices:
        try:
            import cv2
            import numpy as np
            idx = indices["frames_originales"]
            paso = max(1, a.clip // a.muestras)
            iguales = total = 0
            for k in range(1, a.clip + 1, paso):
                a0 = cv2.imread(str(idx[k])) if k in idx else None
                for v in range(1, 8):
                    j = k + v * a.clip
                    if a0 is None or j not in idx:
                        continue
                    b = cv2.imread(str(idx[j]))
                    total += 1
                    iguales += int(b is not None and b.shape == a0.shape and int(np.abs(a0.astype(int) - b.astype(int)).max()) == 0)
            print(f"\nRepeticion en frames_originales: {iguales}/{total} pares (vuelta 1 vs vueltas 2-8) identicos pixel a pixel")
            resumen["repeticion_identica"] = f"{iguales}/{total}"
        except ImportError:
            print("(sin opencv: no se verifica la repeticion)")

    (base / "conteo_frames_video3.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(f"\nResumen guardado en {base / 'conteo_frames_video3.json'}")

    if a.limpiar:
        objetivo = a.carpetas or list(indices)
        destino = Path(a.mover_a) if a.mover_a else None
        for nombre in objetivo:
            idx = indices.get(nombre, {})
            sobrantes = [p for k, p in idx.items() if k > a.clip]
            if not sobrantes:
                continue
            if destino:
                (destino / nombre).mkdir(parents=True, exist_ok=True)
            for p in sobrantes:
                if destino:
                    shutil.move(str(p), str(destino / nombre / p.name))
                else:
                    p.unlink()
            print(f"{nombre}: {'movidos' if destino else 'borrados'} {len(sobrantes)} frames > {a.clip}")


if __name__ == "__main__":
    main()
