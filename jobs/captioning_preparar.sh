#!/usr/bin/env bash
# UNA vez por modelo, en el nodo de login (solo descarga; no calcula nada). Los nodos de GPU no tienen
# internet, por eso los pesos del modelo de lenguaje visual se bajan aqui a ~/.cache/huggingface.
#   bash jobs/captioning_preparar.sh                                  # Qwen2.5-VL-7B (~17 GB)
#   MODELOS="Qwen/Qwen3-VL-8B-Instruct OpenGVLab/InternVL3-8B-hf" bash jobs/captioning_preparar.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source fac_env311/bin/activate
MODELOS="${MODELOS:-Qwen/Qwen2.5-VL-7B-Instruct}"
# El login mata procesos que usan mucha RAM: se descarga de a un archivo y sin el modo rapido de xet.
# Si se corta, volver a correr: continua donde iba.
export HF_XET_HIGH_PERFORMANCE=0 HF_HUB_ENABLE_HF_TRANSFER=0
for M in ${MODELOS}; do
  echo "Descargando ${M} ..."
  python - "$M" <<'PY'
import sys
from huggingface_hub import snapshot_download
ruta = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
                         max_workers=1)
print("pesos en:", ruta)
from transformers import AutoProcessor
AutoProcessor.from_pretrained(sys.argv[1])
print("procesador listo:", sys.argv[1])
PY
done
python scripts/captioning_ventanas.py --video video2 --simular --fin-s 60 --salida cache/captioning_simulado >/dev/null
echo "Entorno listo para captioning."
