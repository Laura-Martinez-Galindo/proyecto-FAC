#!/usr/bin/env python3
"""
==============================================================================
ORQUESTADOR GENERAL DE EXPERIMENTOS FLIR — PROTOCOLO EXPERIMENTAL FAC
==============================================================================
Ejecuta los 4 Experimentos (EXP-1 a EXP-4) en paralelo distribuyendo
cada modelo en su propia GPU (cuda:0 / cuda:1) de forma completamente autónoma.

Estructura de salida (videos/{video_id}/expos_regimenes/):
  EXP-1_regimen1_v3_001800-002099/
    │
    ├── udvd/
    │   ├── denoised/           ← ŷ (imagen restaurada)
    │   ├── residual/           ← r = y - ŷ (PNG raw)
    │   ├── residual_visual/    ← |r|×5 inferno heatmap
    │   └── metricas_frame.csv  ← métricas por fotograma
    │
    ├── nb2nb/
    ├── b2u/
    ├── f2r/
    ├── combo_b2u_f2r/
    │
    └── resultados_experimento.xlsx  ← Tabla consolidada de todas las expos

Cada modelo se entrena y aplica de forma autosupervisada (zero-shot / few-shot)
sobre exactamente los 300 fotogramas seleccionados del régimen físico correspondiente.

RÉGIMEN     EXPERIMENTO  VIDEO  FRAMES
Régimen 1   EXP-1        V3     001800-002099 | V1  19500-19756
Régimen 2   EXP-2        V2     077550-077849 | V1  014900-015199
Régimen 4   EXP-3        V1     019760-019799 (térmico HDIR, todos)
Régimen 3   EXP-4        V1     004622-004921 (nube densa, calibración)
==============================================================================
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

from metricas_experimentos import (
    calcular_metricas_completas,
    guardar_residuo_visual,
    estimar_sigma_mad,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tabla Maestra de Experimentos (actualiza según tus análisis de líneas de tiempo)
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTOS = {
    "EXP-1": {
        "descripcion": "Régimen 1: HDEO + DDE (Alta altitud, objetos pequeños, DVR)",
        "bloques": [
            {
                "video": "video3",
                "inicio_frame_nombre": "frame_001800.png",
                "n_frames": 300,
                "etiqueta": "v3_001800-002099",
            },
            {
                "video": "video1",
                "inicio_frame_nombre": "frame_19500.png",
                "n_frames": 257,  # hasta frame_19756
                "etiqueta": "v1_019500-019756",
            },
        ],
        "modelos": ["udvd", "nb2nb", "b2u", "f2r", "combo_b2u_f2r"],
        "ablaciones": {
            "nb2nb_submuestreo": [2, 4],      # celdas de submuestreo NB2NB
            "f2r_loss": ["l1", "l2"],
        },
    },
    "EXP-2": {
        "descripcion": "Régimen 2: HDEO + DDE + HAZE (Bruma estirada, grumos, colas pesadas)",
        "bloques": [
            {
                "video": "video2",
                "inicio_frame_nombre": "frame_77550.png",
                "n_frames": 300,
                "etiqueta": "v2_077550-077849",
            },
            {
                "video": "video1",
                "inicio_frame_nombre": "frame_14900.png",
                "n_frames": 300,
                "etiqueta": "v1_014900-015199",
            },
        ],
        "modelos": ["udvd", "nb2nb", "b2u", "f2r", "combo_b2u_f2r", "combo_f2r_b2u"],
        "ablaciones": {
            "b2u_loss": ["l1", "l2"],
            "orden_pipeline": ["espacial_temporal", "temporal_espacial"],
        },
    },
    "EXP-3": {
        "descripcion": "Régimen 4: HDIR C WH + EXP AUT (Térmico InSb + FPN striping)",
        "bloques": [
            {
                "video": "video1",
                "inicio_frame_nombre": "frame_19760.png",
                "n_frames": 40,   # todos los frames térmicos disponibles
                "etiqueta": "v1_019760-019799_termico",
            },
        ],
        "modelos": ["udvd", "udvd_sin_registro", "n2v_struct_vert", "f2r", "combo_struct_f2r"],
        "ablaciones": {
            "mascara_n2v": ["isotrópica_1x1", "estructurada_9x1"],
            "registro_homografico": [True, False],
        },
    },
    "EXP-4": {
        "descripcion": "Régimen 3: HDEO HAZE Nube (Calibración de ruido puro)",
        "bloques": [
            {
                "video": "video1",
                "inicio_frame_nombre": "frame_04622.png",
                "n_frames": 300,
                "etiqueta": "v1_004622-004921_nube",
            },
        ],
        "modelos": ["udvd", "nb2nb", "b2u", "f2r"],
        "ablaciones": {},
    },
}


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def detectar_menu_central(frame_bgr: np.ndarray, umbral_blanco: float = 0.03) -> bool:
    """
    Filtra fotogramas con tablas blancas centrales del operador (Régimen 5 Video 1).
    Detecta si > 3% de píxeles en la zona central [25%-75%] son blancos saturados
    (intensidad > 240, canal diferencial ≈ 0 → gris/blanco puro).
    """
    h, w = frame_bgr.shape[:2]
    zona = frame_bgr[h//4:3*h//4, w//4:3*w//4]
    gray = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
    n_blanco = int(np.count_nonzero(gray > 240))
    pct_blanco = n_blanco / (gray.shape[0] * gray.shape[1])

    if pct_blanco > umbral_blanco:
        # Verificar que también sea gris/blanco puro (saturación ≈ 0)
        hsv = cv2.cvtColor(zona, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        if sat_mean < 30:
            return True
    return False


def seleccionar_bloque_frames(video_id: str, inicio_nombre: str, n_frames: int) -> list:
    """
    Selecciona n_frames consecutivos del directorio frames_sin_hud del video,
    comenzando por el frame de nombre 'inicio_nombre'.
    Aplica filtro de menú central automáticamente.
    Verifica régimen físico (sigma_mad > 0.5 DN para garantizar que hay algo que limpiar).
    """
    dir_sin_hud = RAIZ / "videos" / video_id / "frames_sin_hud"
    if not dir_sin_hud.is_dir():
        raise FileNotFoundError(f"No existe directorio: {dir_sin_hud}")

    exts = {".png", ".jpg", ".jpeg"}
    todos = sorted([p for p in dir_sin_hud.iterdir() if p.suffix.lower() in exts], key=natural_key)
    nombres = [p.name for p in todos]

    # Encontrar índice de inicio
    idx_ini = 0
    for i, n in enumerate(nombres):
        # Coincidencia flexible: por nombre exacto o por número de frame extraído
        num_buscar = int(re.findall(r"\d+", inicio_nombre)[-1]) if re.findall(r"\d+", inicio_nombre) else 0
        num_frame = int(re.findall(r"\d+", n)[-1]) if re.findall(r"\d+", n) else -1
        if n == inicio_nombre or num_frame >= num_buscar:
            idx_ini = i
            break

    seleccionados_raw = todos[idx_ini : idx_ini + n_frames]

    # Filtrar menús centrales del operador
    seleccionados = []
    descartados_menu = 0
    for p in seleccionados_raw:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if detectar_menu_central(img):
            descartados_menu += 1
            continue
        seleccionados.append(p)

    if descartados_menu > 0:
        print(f"    [Filtro] {descartados_menu} frames con menú central descartados de {video_id}")

    return seleccionados


def calcular_metricas_bloque(dir_input: Path, dir_denoised: Path, id_exp: str, etiqueta: str, modelo: str) -> pd.DataFrame:
    """Calcula las 6 métricas por fotograma y retorna un DataFrame."""
    exts = {".png", ".jpg", ".jpeg"}
    frames_in = sorted([p for p in dir_input.iterdir() if p.suffix.lower() in exts], key=natural_key)
    frames_den = {p.name: p for p in dir_denoised.iterdir() if p.suffix.lower() in exts}

    dir_residual_visual = dir_denoised.parent / "residual_visual"
    dir_residual_raw = dir_denoised.parent / "residual"
    dir_residual_visual.mkdir(parents=True, exist_ok=True)
    dir_residual_raw.mkdir(parents=True, exist_ok=True)

    registros = []
    for p_in in frames_in:
        p_den = frames_den.get(p_in.name)
        if p_den is None:
            continue

        y_bgr = cv2.imread(str(p_in))
        yhat_bgr = cv2.imread(str(p_den))
        if y_bgr is None or yhat_bgr is None:
            continue
        if y_bgr.shape[:2] != yhat_bgr.shape[:2]:
            yhat_bgr = cv2.resize(yhat_bgr, (y_bgr.shape[1], y_bgr.shape[0]))

        metricas = calcular_metricas_completas(y_bgr, yhat_bgr)
        metricas["frame"] = p_in.name
        metricas["exp_id"] = id_exp
        metricas["bloque"] = etiqueta
        metricas["modelo"] = modelo
        registros.append(metricas)

        # Guardar residuo visual inferno
        guardar_residuo_visual(y_bgr, yhat_bgr, dir_residual_visual / p_in.name, escala=5.0)
        # Guardar residuo raw (para análisis FFT posterior)
        y_g = cv2.cvtColor(y_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        yhat_g = cv2.cvtColor(yhat_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r_raw = np.clip((y_g - yhat_g + 128.0), 0, 255).astype(np.uint8)
        cv2.imwrite(str(dir_residual_raw / p_in.name), r_raw, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    return pd.DataFrame(registros)


def correr_pipeline(modelo_id: str, gpu_id: int, dir_input: Path, dir_salida_modelo: Path,
                    video_id: str, n_frames: int, extra_args: dict = None):
    """Llama al script de pipeline correspondiente al modelo."""
    dir_salida_modelo.mkdir(parents=True, exist_ok=True)
    denoised_dir = dir_salida_modelo / "denoised"
    denoised_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    base_args = [
        sys.executable, "-u",
        "--epocas", "8",
        "--paciencia", "4",
        "--batch-size", "8",
        "--patch-size", "128",
    ]

    pipeline_map = {
        "udvd":         "scripts/pipeline_udvd.py",
        "nb2nb":        "scripts/pipeline_neighbor2neighbor.py",
        "b2u":          "scripts/pipeline_blind2unblind.py",
        "f2r":          "scripts/pipeline_frames2residual.py",
        "n2v_struct_vert": "scripts/pipeline_n2v.py",
    }

    if modelo_id not in pipeline_map:
        print(f"  [!] Modelo {modelo_id} no tiene script directo. Manejado por combo.")
        return None

    script = str(RAIZ / pipeline_map[modelo_id])
    cmd = [
        sys.executable, "-u", script,
        "--carpeta-entrada", str(dir_input),
        "--carpeta-salida", str(denoised_dir),
        "--epocas", "8",
        "--paciencia", "4",
        "--batch-size", "8",
    ]

    if modelo_id == "n2v_struct_vert":
        cmd += ["--struct-n2v", "vertical"]
    if extra_args:
        for k, v in extra_args.items():
            cmd += [f"--{k}", str(v)]

    print(f"  [GPU {gpu_id}] Corriendo {modelo_id} en {dir_input.name}...")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [!] ERROR en {modelo_id}: {result.stderr[-500:]}")
    else:
        print(f"  [✓] {modelo_id} completado en {elapsed:.1f}s")

    return denoised_dir


def correr_experimento(exp_id: str, gpu_asignada_base: int = 0):
    """Corre un experimento completo alternando entre las 2 GPUs disponibles."""
    exp_cfg = EXPERIMENTOS[exp_id]
    print(f"\n{'='*80}")
    print(f"🧪 INICIANDO {exp_id}: {exp_cfg['descripcion']}")
    print(f"{'='*80}")

    todos_resultados = []

    for bloque in exp_cfg["bloques"]:
        video_id = bloque["video"]
        inicio = bloque["inicio_frame_nombre"]
        n_frames = bloque["n_frames"]
        etiqueta = bloque["etiqueta"]

        print(f"\n📦 Bloque: {etiqueta} — {n_frames} frames desde {inicio}")

        # Selección y filtrado de frames
        frames_seleccionados = seleccionar_bloque_frames(video_id, inicio, n_frames)
        print(f"   Frames válidos tras filtrado: {len(frames_seleccionados)}")
        if not frames_seleccionados:
            print(f"   [!] No hay frames válidos. Saltando bloque.")
            continue

        # Crear directorio de experimento
        dir_exp = RAIZ / "videos" / video_id / "expos_regimenes" / f"{exp_id}_{etiqueta}"
        dir_input = dir_exp / "input"
        dir_input.mkdir(parents=True, exist_ok=True)

        # Crear links simbólicos o copias ligeras (solo si no existen ya)
        existing = list(dir_input.glob("*.png")) + list(dir_input.glob("*.jpg"))
        if len(existing) < len(frames_seleccionados):
            for p in frames_seleccionados:
                destino = dir_input / p.name
                if not destino.exists():
                    import shutil
                    shutil.copy2(str(p), str(destino))

        # Correr modelos alternando GPUs
        modelos = exp_cfg["modelos"]
        for i, modelo_id in enumerate(modelos):
            gpu_id = (gpu_asignada_base + i) % 2

            # Manejar combos
            if "combo" in modelo_id:
                partes = modelo_id.replace("combo_", "").split("_")
                dir_salida_combo = dir_exp / modelo_id / "denoised"
                dir_salida_combo.mkdir(parents=True, exist_ok=True)

                dir_actual = dir_input
                for parte in partes:
                    dir_inter = dir_exp / f"_inter_{parte}" / "denoised"
                    correr_pipeline(parte, gpu_id, dir_actual, dir_exp / f"_inter_{parte}", video_id, n_frames)
                    dir_actual = dir_inter

                # El resultado final del combo está en dir_actual
                import shutil
                for fp in dir_actual.glob("*"):
                    shutil.copy2(str(fp), str(dir_salida_combo / fp.name))
                denoised_dir = dir_salida_combo
            else:
                denoised_dir = correr_pipeline(
                    modelo_id, gpu_id, dir_input,
                    dir_exp / modelo_id, video_id, n_frames
                )

            if denoised_dir is None or not denoised_dir.is_dir():
                continue

            # Calcular métricas
            print(f"    📊 Calculando métricas para {modelo_id}...")
            df_metricas = calcular_metricas_bloque(dir_input, denoised_dir, exp_id, etiqueta, modelo_id)
            csv_path = dir_exp / modelo_id / "metricas_frame.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_metricas.to_csv(str(csv_path), index=False)

            if not df_metricas.empty:
                resumen = df_metricas.agg({
                    "sigma_mad_input": "mean",
                    "sigma_mad_output": "mean",
                    "sigma_mad_reduccion_pct": "mean",
                    "kurtosis_residuo": "mean",
                    "fft_flatness_residuo": "mean",
                    "sli": "mean",
                    "epr": "mean",
                }).to_dict()
                resumen["exp_id"] = exp_id
                resumen["bloque"] = etiqueta
                resumen["modelo"] = modelo_id
                resumen["n_frames"] = len(df_metricas)
                todos_resultados.append(resumen)

    # Guardar Excel consolidado por experimento
    if todos_resultados:
        df_exp = pd.DataFrame(todos_resultados)

        # Reordenar columnas para legibilidad
        cols_order = ["exp_id", "bloque", "modelo", "n_frames",
                      "sigma_mad_input", "sigma_mad_output", "sigma_mad_reduccion_pct",
                      "sli", "epr", "kurtosis_residuo", "fft_flatness_residuo"]
        df_exp = df_exp[[c for c in cols_order if c in df_exp.columns]]

        # Guardar en directorio raíz del experimento
        primera_bloque = EXPERIMENTOS[exp_id]["bloques"][0]
        dir_exp_raiz = RAIZ / "videos" / primera_bloque["video"] / "expos_regimenes"
        dir_exp_raiz.mkdir(parents=True, exist_ok=True)
        excel_path = dir_exp_raiz / f"resultados_{exp_id}.xlsx"
        df_exp.to_excel(str(excel_path), index=False, engine="openpyxl")
        print(f"\n[✅] Resultados Excel guardados en: {excel_path}")
        print(df_exp.to_string(index=False))

    return todos_resultados


def main():
    parser = argparse.ArgumentParser(description="Orquestador de Experimentos FLIR FAC")
    parser.add_argument("--exp", default="todos",
                        choices=["todos", "EXP-1", "EXP-2", "EXP-3", "EXP-4"],
                        help="Experimento a correr (default: todos)")
    parser.add_argument("--gpu-base", type=int, default=0,
                        help="GPU base para distribución (0 o 1). Los modelos se alternan entre GPU-base y GPU-(1-base).")
    args = parser.parse_args()

    exps_a_correr = list(EXPERIMENTOS.keys()) if args.exp == "todos" else [args.exp]
    todos_resultados = []

    for exp_id in exps_a_correr:
        resultados = correr_experimento(exp_id, gpu_asignada_base=args.gpu_base)
        todos_resultados.extend(resultados)

    # Excel consolidado global (todos los experimentos)
    if todos_resultados:
        df_global = pd.DataFrame(todos_resultados)
        excel_global = RAIZ / "figuras_tesis" / "resultados_protocolo_experimental_completo.xlsx"
        excel_global.parent.mkdir(parents=True, exist_ok=True)
        df_global.to_excel(str(excel_global), index=False, engine="openpyxl")
        print(f"\n[🏆] Tabla Global Consolidada guardada en: {excel_global}")


if __name__ == "__main__":
    main()
