#!/usr/bin/env python3
"""
==============================================================================
CONSOLIDADOR DE RESULTADOS Y ELECCIÓN DEL MODELO GANADOR POR RÉGIMEN
==============================================================================
Lee los resultados de todos los bloques ejecutados para un experimento,
consolida las tablas de métricas, corona al modelo ganador mediante un
Score Científico Compuesto:

  Score = Reduccion_sigma(%) * EPR / (1.0 + 5.0 * SLI)

  - Premia alta supresión de ruido (Reducción σ_MAD).
  - Premia preservación de picos calientes (EPR ≈ 1.0).
  - Penaliza severamente la fuga estructural al residuo (SLI > 0.15).

Genera un libro Excel con pestañas formateadas para el reporte y la tesis.
==============================================================================
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description="Consolidador de resultados de experimentos")
    p.add_argument("--exp-id", required=True, help="Identificador del experimento (e.g. EXP-1)")
    p.add_argument("--dirs", nargs="+", required=True, type=Path,
                   help="Directorios de los bloques del experimento")
    p.add_argument("--salida", required=True, type=Path,
                   help="Ruta del archivo Excel final de salida")

    a = p.parse_args()

    dfs_resumen = []
    dfs_frames = []

    for d in a.dirs:
        d = d.resolve()
        resumen_csv = d / "resumen_metricas.csv"
        if resumen_csv.exists():
            df_r = pd.read_csv(resumen_csv)
            dfs_resumen.append(df_r)

        # Buscar métricas por frame de cada modelo
        for p_modelo in d.iterdir():
            if p_modelo.is_dir():
                csv_frame = p_modelo / "metricas_frame.csv"
                if csv_frame.exists():
                    dfs_frames.append(pd.read_csv(csv_frame))

    if not dfs_resumen:
        print(f"[ERROR] No se encontraron archivos resumen_metricas.csv en los directorios indicados.")
        sys.exit(1)

    df_consolidado = pd.concat(dfs_resumen, ignore_index=True)

    # Calcular Score Científico Compuesto para coronar al ganador
    # Score = (Reduccion_sigma_mad_pct * epr_preservacion_picos) / (1.0 + 5.0 * sli_fuga_estructural)
    df_consolidado["score_cientifico"] = (
        df_consolidado["reduccion_sigma_mad_pct"] *
        df_consolidado["epr_preservacion_picos"] /
        (1.0 + 5.0 * df_consolidado["sli_fuga_estructural"])
    ).round(2)

    df_consolidado = df_consolidado.sort_values(
        by=["bloque", "score_cientifico"], ascending=[True, False]
    )

    # Identificar ganador por bloque
    ganadores = df_consolidado.groupby("bloque").first().reset_index()

    salida_path = a.salida.resolve()
    salida_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(salida_path), engine="openpyxl") as writer:
        df_consolidado.to_excel(writer, sheet_name="Resumen_Comparativo", index=False)
        ganadores.to_excel(writer, sheet_name="Modelos_Ganadores", index=False)
        if dfs_frames:
            df_todos_frames = pd.concat(dfs_frames, ignore_index=True)
            # Limitar a primeras 50,000 filas por seguridad de Excel
            df_todos_frames.iloc[:50000].to_excel(writer, sheet_name="Detalle_Frames", index=False)

    print("\n" + "=" * 80)
    print(f"🏆 RESULTADOS CONSOLIDADOS — {a.exp_id}")
    print("=" * 80)
    print(df_consolidado[["bloque", "modelo", "reduccion_sigma_mad_pct", "sli_fuga_estructural", "epr_preservacion_picos", "score_cientifico"]].to_string(index=False))

    print("\n" + "-" * 80)
    print("👑 MODELOS GANADORES CORONADOS:")
    print("-" * 80)
    for _, g in ganadores.iterrows():
        print(f"  • Bloque [{g['bloque']}]: {g['modelo'].upper()} (Score: {g['score_cientifico']}, Reducción: {g['reduccion_sigma_mad_pct']}%, SLI: {g['sli_fuga_estructural']})")

    print(f"\n[✓] Libro Excel consolidado guardado en: {salida_path}")


if __name__ == "__main__":
    main()
