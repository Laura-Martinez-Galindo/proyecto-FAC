#!/usr/bin/env bash
# UNA sola vez antes de la ronda 3, en el nodo de login (solo instala paquetes y descarga pesos; no
# calcula nada). Los nodos de GPU no tienen internet, por eso los pesos de las metricas se bajan aqui.
#   bash jobs/preparar_entorno_ronda3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source fac_env311/bin/activate
pip install -q einops bm3d pyiqa scikit-image
python - <<'PY'
import pyiqa, torch
for m in ["niqe", "brisque", "musiq", "clipiqa", "paq2piq"]:  # paq2piq: perdida de Self2Self+
    pyiqa.create_metric(m, device=torch.device("cpu"))
    print("pesos listos:", m)
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
raft_small(weights=Raft_Small_Weights.DEFAULT)  # flujo optico de los metodos temporales
print("pesos listos: raft_small")
import einops, bm3d
print("einops", einops.__version__, "| bm3d ok")
PY
echo "Entorno listo para la ronda 3."
