"""
Redes de terceros, copiadas de los repositorios OFICIALES de cada articulo.

Cada carpeta conserva la estructura de imports del repo original (p.ej. `from src.model...`,
`from network...`), asi que no se importan como paquete normal: se cargan con
`cargar(clave, modulo)`, que pone la carpeta al frente de sys.path y limpia de sys.modules los
nombres genericos que se repiten entre repos (src, network, model, util). Cada experimento corre
en su propio proceso, asi que en un proceso solo se usa un repo a la vez.

Cambios al codigo original (marcados con "[FAC]" en el archivo):
  * src/util/util.py de puca, lgbpn y ubsn: los imports de lectura .mat y de metricas
    (mat73, h5py, scipy.io, skimage) van dentro de try, porque la red no los usa.

  clave  articulo                                          repositorio oficial               licencia
  puca   Jang et al., NeurIPS 2023 (PUCA)                   github.com/HyemiEsme/PUCA          MIT
  lgbpn  Wang et al., CVPR 2023 (LG-BPN)                    github.com/Wang-XIaoDingdd/LGBPN   sin licencia (uso academico)
  ubsn   Yu et al., WACV 2025 (U-BSN, multi-scale J-inv.)   github.com/NICALab/UBSN            MIT
  tbsn   Li et al., AAAI 2025 (TBSN)                        github.com/nagejacob/TBSN          GPL-3.0
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
_GENERICOS = ("src", "network", "model", "models", "util", "utils", "dataset", "option")


def cargar(clave: str, modulo: str):
    """Importa `modulo` (p.ej. 'src.model') desde la carpeta del repo `clave`."""
    raiz = str(RAIZ / clave)
    for n in list(sys.modules):
        if n.split(".")[0] in _GENERICOS:
            del sys.modules[n]
    if raiz in sys.path:
        sys.path.remove(raiz)
    sys.path.insert(0, raiz)
    return importlib.import_module(modulo)
