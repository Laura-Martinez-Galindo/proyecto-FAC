#!/usr/bin/env python3
"""
Inspecciona exhaustivamente los datasets recién descargados (11.7 GB y 7.07 GB):
- Estructura de carpetas y archivos dataset.yaml
- Conteo de imágenes por split (train / val / test)
- Prefijos y patrones de nombres de archivos
- Detección de origen de video y clases presentes
"""

from collections import Counter
from pathlib import Path
import yaml

RAIZ = Path(__file__).resolve().parent.parent


def inspeccionar_directorio_dataset(nombre, ruta_dir):
    print("=" * 80)
    print(f"INSPECCIÓN: {nombre}")
    print(f"Ruta: {ruta_dir}")
    print("=" * 80)

    if not ruta_dir.is_dir():
        print(f"[-] El directorio no existe todavía: {ruta_dir}\n")
        return

    # 1. Buscar archivos yaml
    yamls = list(ruta_dir.glob("**/*.yaml")) + list(ruta_dir.glob("**/*.yml"))
    print(f"Archivos de configuración encontrados: {len(yamls)}")
    for y in yamls:
        print(f"  [YAML] {y.relative_to(ruta_dir)}")
        try:
            with open(y, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    print(f"         Clases: {data.get('names', 'No especificadas')} (nc={data.get('nc')})")
        except Exception as e:
            print(f"         Error leyendo YAML: {e}")

    # 2. Conteo de imágenes y labels
    extensiones = [".jpg", ".png", ".jpeg", ".bmp"]
    imgs = [p for p in ruta_dir.rglob("*") if p.suffix.lower() in extensiones]
    lbls = [p for p in ruta_dir.rglob("*.txt")]

    print(f"\nTotal imágenes encontradas: {len(imgs)}")
    print(f"Total etiquetas .txt:       {len(lbls)}")

    if not imgs:
        print("[-] No se encontraron imágenes en el directorio.\n")
        return

    # 3. Distribución por carpetas (splits)
    carpetas_splits = Counter()
    prefijos = Counter()
    
    for img in imgs:
        # Detectar split en la ruta
        partes_rel = img.relative_to(ruta_dir).parts
        if len(partes_rel) > 1:
            carpetas_splits[partes_rel[0] if len(partes_rel) == 2 else f"{partes_rel[0]}/{partes_rel[1]}"] += 1
        else:
            carpetas_splits["raiz"] += 1
            
        # Detectar prefijo del nombre
        stem = img.stem
        if "_" in stem:
            pref = stem.split("_")[0] + "_" + stem.split("_")[1] if len(stem.split("_")) > 2 else stem.split("_")[0]
        else:
            pref = stem[:6]
        prefijos[pref] += 1

    print("\nDistribución por carpetas principales:")
    for c, cnt in carpetas_splits.most_common(10):
        print(f"  - {c:<30}: {cnt} imágenes")

    print("\nPatrones y orígenes de nombres de imágenes:")
    for p, cnt in prefijos.most_common(15):
        print(f"  - {p:<30}: {cnt} imágenes")

    # 4. Muestra de primeros 5 nombres
    print("\nPrimeros 5 nombres de archivo de ejemplo:")
    for img in imgs[:5]:
        print(f"  * {img.name} (en {img.parent.relative_to(ruta_dir)})")

    print("\n")


def main():
    dir_datasets = RAIZ / "datasets"
    
    # 1. Dataset 11.7 GB
    inspeccionar_directorio_dataset(
        "Dataset 1: Preprocesado 11.7 GB",
        dir_datasets / "dataset_preprocesado_11gb"
    )

    # 2. Dataset 7.07 GB
    inspeccionar_directorio_dataset(
        "Dataset 2: YOLO Dataset 7.07 GB",
        dir_datasets / "dataset_yolo_7gb"
    )


if __name__ == "__main__":
    main()
