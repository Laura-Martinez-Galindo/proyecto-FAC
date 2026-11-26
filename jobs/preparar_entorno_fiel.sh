#!/usr/bin/env bash
# UNA sola vez antes de la ronda 3 fiel, en el nodo de login (solo instala y descarga pesos; no calcula):
#   * scikit-image: flujo optico TV-L1 de f2f_fiel
#   * DnCNN-S-25 preentrenada (SaoYan/DnCNN-PyTorch, la que usa el repo oficial de Frame2Frame)
#   bash jobs/preparar_entorno_fiel.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source fac_env311/bin/activate
pip install -q scikit-image scipy
mkdir -p pesos/dncnn
if [ ! -s pesos/dncnn/DnCNN-S-25.pth ]; then
  wget -q -O pesos/dncnn/DnCNN-S-25.pth \
    https://raw.githubusercontent.com/SaoYan/DnCNN-PyTorch/master/logs/DnCNN-S-25/net.pth
fi
python - <<'PY'
import torch
from skimage.registration import optical_flow_tvl1  # noqa: F401
sd = torch.load("pesos/dncnn/DnCNN-S-25.pth", map_location="cpu")
assert sd["module.dncnn.0.weight"].shape == (64, 1, 3, 3), "pesos inesperados"
print("DnCNN-S-25 lista:", len(sd), "tensores | scikit-image TV-L1 lista")
PY
echo "Entorno listo para la ronda 3 fiel."
