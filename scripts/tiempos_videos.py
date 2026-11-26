#!/usr/bin/env python3
"""Imprime todos los tiempos guardados en config/videos.json (extraccion, mascara, inpainting...)."""
import json
from pathlib import Path


def recorrer(d, ruta=""):
    if isinstance(d, dict):
        if any("tiempo" in k for k in d):
            extra = {k: v for k, v in d.items() if "tiempo" in k or k in ("frames_procesados", "frames_extraidos", "estado", "modelo")}
            print(f"{ruta:45s} {extra}")
        for k, v in d.items():
            recorrer(v, f"{ruta}/{k}" if ruta else k)


recorrer(json.loads((Path(__file__).resolve().parent.parent / "config/videos.json").read_text(encoding="utf-8")))
