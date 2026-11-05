#!/usr/bin/env bash
#SBATCH --job-name=subir_drive
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=jobs/logs/subir_drive_%j.out
#SBATCH --error=jobs/logs/subir_drive_%j.err

# Sube los datasets a Google Drive como UN archivo .tar por dataset (sin comprimir: los PNG ya
# estan comprimidos). Subir 1 archivo grande es mucho mas rapido que miles de archivos chicos,
# porque Drive limita cuantos archivos por segundo se pueden crear. No se pierde calidad: el tar
# guarda los bytes exactos y al final se compara el md5 de lo subido contra el local.
#
# Uso:  bash jobs/enviar.sh -c 2 --mem=4G -t 03:00:00 jobs/subir_drive.sbatch
# Variables: DESTINO (gdrive:proyecto-FAC/datasets)  DATASETS ("dataset_original dataset_sin_hud")
# Para abrirlo despues: descargar el .tar y  tar -xf dataset_original.tar  (en Windows: 7-Zip)
#
# Si el job termina con codigo 3 (el nodo de computo no tiene internet), correr lo mismo en el
# nodo de login dentro de tmux, que sigue aunque se cierre MobaXterm:
#   tmux new -s drive                 (para volver despues:  tmux attach -t drive)
#   bash jobs/subir_drive.sbatch      (corre igual, sin Slurm)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
DESTINO="${DESTINO:-gdrive:proyecto-FAC/datasets}"
DATASETS="${DATASETS:-dataset_original dataset_sin_hud}"
TMP="${TMPDIR_SUBIDA:-datasets/_para_drive}"

echo "== Prueba de conexion a Drive desde $(hostname)"
if ! timeout 60 rclone lsd "${DESTINO%%:*}:" --max-depth 1 >/dev/null 2>&1; then
  echo "!! Este nodo no llega a Google Drive (los nodos de computo pueden no tener internet)."
  echo "   Correrlo en el nodo de login dentro de tmux (ver el encabezado de jobs/subir_drive.sbatch)."
  exit 3
fi

mkdir -p "$TMP"
for D in $DATASETS; do
  echo "== $D: empaquetando"
  tar -cf "$TMP/$D.tar" -C datasets "$D"
  ls -lh "$TMP/$D.tar"
  ( cd "$TMP" && md5sum "$D.tar" > "$D.tar.md5" )
done

echo "== Subiendo a $DESTINO"
rclone copy "$TMP" "$DESTINO" --include "*.tar" --include "*.tar.md5" \
    --transfers 2 --drive-chunk-size 128M --stats 30s --stats-one-line -v

echo "== Verificando (md5 de Drive contra el local)"
rclone check "$TMP" "$DESTINO" --include "*.tar" --one-way
echo "OK: si arriba dice '0 differences found', lo de Drive es identico byte a byte."
rclone ls "$DESTINO"
rm -f "$TMP"/*.tar          # los .tar locales ya no hacen falta (los datasets siguen intactos)
