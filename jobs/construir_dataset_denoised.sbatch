#!/usr/bin/env bash
#SBATCH --job-name=dataset_denoised
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=jobs/logs/dataset_denoised_%j.out
#SBATCH --error=jobs/logs/dataset_denoised_%j.err

# Arma datasets/dataset_denoised con el ganador de cada regimen de la fase 2
# (config/denoising/ganadores.json). No entrena ni infiere: copia las salidas que la
# fase 2 ya genero en cache/denoising/<FAMILIA>/<EXPERIMENTO>/salidas para TODAS las
# imagenes de cada regimen. Las 34 imagenes de regimenes minoritarios se copian de
# dataset_sin_hud.
# Deja para traer al PC:
#   datasets/dataset_denoised.tar              el dataset completo
#   resultados/denoising/fase2_visual.tar      6 frames por regimen en original, sin HUD y
#                                              denoised (resolucion completa) + curvas de
#                                              entrenamiento/validacion de la fase 2
# Uso:  sbatch jobs/construir_dataset_denoised.sbatch      (N_VISUAL=6 por defecto)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p jobs/logs
source fac_env311/bin/activate

python scripts/construir_dataset_denoised.py --forzar

tar -cf datasets/dataset_denoised.tar -C datasets dataset_denoised
md5sum datasets/dataset_denoised.tar | tee datasets/dataset_denoised.tar.md5

# frames para revisar a ojo: N_VISUAL por familia, repartidos a lo largo del video
python - <<'PY'
import csv, os
n = int(os.environ.get("N_VISUAL", "6"))
filas = list(csv.DictReader(open("datasets/dataset_denoised/correspondencia.csv", encoding="utf-8")))
sel = []
for fam in sorted({r["familia_denoising"] for r in filas}):
    g = sorted([r["imagen"] for r in filas if r["familia_denoising"] == fam])
    k = n if fam != "(ninguna)" else 3
    paso = max(1, len(g) // k)
    sel += g[paso // 2::paso][:k]
with open("resultados/denoising/fase2_visual.txt", "w") as f:
    for im in sel:
        for d in ("dataset_original", "dataset_sin_hud", "dataset_denoised"):
            f.write(f"datasets/{d}/images/{im}\n")
    f.write("datasets/dataset_denoised/correspondencia.csv\ndatasets/dataset_denoised/resumen.json\n")
print(len(sel), "imagenes elegidas")
PY
ls cache/denoising/*/*_f2/entrenamiento.csv cache/denoising/*/*_f2/validacion.csv \
   cache/denoising/*/*_f2/configuracion.json 2>/dev/null >> resultados/denoising/fase2_visual.txt || true
tar -cf resultados/denoising/fase2_visual.tar -T resultados/denoising/fase2_visual.txt
echo "Traer al PC: datasets/dataset_denoised.tar  y  resultados/denoising/fase2_visual.tar"
