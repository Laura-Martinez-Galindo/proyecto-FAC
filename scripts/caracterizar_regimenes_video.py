#!/usr/bin/env python3
"""
==============================================================================
CARACTERIZACIÓN FÍSICA Y CONTEO DE REGÍMENES PARA VIDEOS FLIR (FAC)
==============================================================================
Analiza fotograma a fotograma la física de la escena y la firma del HUD
(esquina inferior izquierda) sin depender ciegamente de OCR.

Detecta y clasifica automáticamente los 4 regímenes operativos del sensor:
  1. HDIR_Termico: Sensor infrarrojo térmico (InSb / MWIR) -> delta_rgb ~ 0
  2. HDEO_Optico_HAZE: Sensor óptico visible con bruma/nube -> bajo contraste IQR
  3. HDEO_Optico_DDE_Activo: Sensor óptico con realce digital (DDE) -> alta energía laplaciana
  4. HDEO_Optico_Estandar: Sensor óptico en condiciones nominales

Genera automáticamente:
  - CSV con métricas por fotograma y asignación de régimen.
  - Gráfica de línea de tiempo temporal de los regímenes (timeline).
  - Gráfica de distribución porcentual y conteo (barras / torta).
  - Resumen textual con los intervalos continuos de frames de cada régimen.
==============================================================================
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Sin interfaz gráfica (headless para cluster/servidor)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm


def extraer_firma_esquina_inferior_izquierda(img_bgr):
    """
    Recorta la esquina inferior izquierda donde se encuentran HDEO/HDIR,
    HAZE, DDE, FOC y EXP, y extrae descriptores visuales de su contenido.
    """
    h, w, _ = img_bgr.shape
    # Coordenadas exactas de la esquina inferior izquierda del HUD (FLIR Star SAFIRE)
    y1, y2 = int(h * 0.75), int(h * 0.98)
    x1, x2 = 0, int(w * 0.25)
    roi = img_bgr[y1:y2, x1:x2]

    # Binarizar para aislar el texto blanco del HUD (típicamente píxeles >= 210)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    # Analizar cuántas líneas de texto aparecen encendidas y su proyección vertical
    proyeccion_y = np.sum(binaria > 0, axis=1)
    lineas_activas = int(np.count_nonzero(proyeccion_y > 5))  # umbral de ancho de línea

    # Densidad de píxeles activos de texto en el HUD
    densidad_texto = float(np.mean(binaria > 0))

    return binaria, lineas_activas, densidad_texto


def extraer_numero_frame(nombre_archivo):
    """Extrae el número entero de frame de nombres como frame_00123.png o v1_frame_123.png."""
    match = re.search(r"(\d+)", str(nombre_archivo))
    return int(match.group(1)) if match else 0


def caracterizar_y_contar_regimenes(carpeta_frames, salida_csv, generar_graficas=True):
    carpeta_p = Path(carpeta_frames)
    if not carpeta_p.exists():
        raise FileNotFoundError(f"La carpeta especificada no existe: {carpeta_frames}")

    # Buscar imágenes soportadas (.png, .jpg, .jpeg)
    patrones = ["frame_*.png", "frame_*.jpg", "*.png", "*.jpg"]
    rutas = []
    for pat in patrones:
        rutas = list(carpeta_p.glob(pat))
        if rutas:
            break

    if not rutas:
        raise ValueError(f"No se encontraron imágenes en: {carpeta_frames}")

    # Ordenar numéricamente por número de frame
    rutas = sorted(rutas, key=lambda p: extraer_numero_frame(p.name))
    print(f"\n[INFO] Analizando {len(rutas)} fotogramas en: {carpeta_frames}")

    datos_frames = []

    for ruta in tqdm(rutas, desc="Analizando fotogramas"):
        img = cv2.imread(str(ruta))
        if img is None:
            continue

        h, w, _ = img.shape
        # Zona central de la escena (lejos del HUD perimetral)
        escena = img[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]
        b, g, r = cv2.split(escena.astype(np.float32))
        gray_escena = cv2.cvtColor(escena, cv2.COLOR_BGR2GRAY)

        # 1. Métricas físicas de la escena
        # delta_rgb: si es térmico puro (monocromático codificado en 3 canales), delta_rgb ~ 0
        delta_rgb = float(np.median(np.abs(r - g) + np.abs(g - b)))

        # contraste_iqr: rango intercuartílico de intensidad (bruma/HAZE comprime el IQR)
        p75 = np.percentile(gray_escena, 75)
        p25 = np.percentile(gray_escena, 25)
        contraste_iqr = float(p75 - p25)

        # Gradiente de alta frecuencia (detecta DDE / ruido de bordes estirado)
        lap = np.abs(cv2.Laplacian(gray_escena, cv2.CV_32F))
        energia_bordes = float(np.median(lap))

        # Brillo mediano de la escena
        brillo_mediano = float(np.median(gray_escena))

        # 2. Análisis del bloque inferior izquierdo (HUD)
        _, lineas_hud, densidad_hud = extraer_firma_esquina_inferior_izquierda(img)

        # 3. Lógica heurística robusta combinada
        if delta_rgb < 0.6:
            sensor_modo = "HDIR_Termico"
        elif contraste_iqr < 25.0:
            sensor_modo = "HDEO_Optico_HAZE"
        elif energia_bordes > 12.0:
            sensor_modo = "HDEO_Optico_DDE_Activo"
        else:
            sensor_modo = "HDEO_Optico_Estandar"

        frame_num = extraer_numero_frame(ruta.name)

        datos_frames.append({
            "frame_idx": frame_num,
            "archivo": ruta.name,
            "sensor_modo_estimado": sensor_modo,
            "delta_rgb": round(delta_rgb, 3),
            "contraste_iqr": round(contraste_iqr, 2),
            "energia_bordes": round(energia_bordes, 3),
            "brillo_mediano": round(brillo_mediano, 2),
            "lineas_hud_activas": lineas_hud,
            "densidad_hud": round(densidad_hud, 4),
        })

    df = pd.DataFrame(datos_frames)
    if df.empty:
        raise ValueError("No se pudieron procesar fotogramas válidos.")

    # 4. Agrupamiento no supervisado (K-Means) para validar clústeres matemáticos
    n_muestras = len(df)
    n_clusters = min(4, max(2, n_muestras // 50))
    X = df[["delta_rgb", "contraste_iqr", "energia_bordes"]].values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_norm)
    df["cluster_id"] = kmeans.labels_

    # Asegurar que el directorio de salida exista
    salida_csv_p = Path(salida_csv)
    salida_csv_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida_csv_p, index=False)

    # 5. Calcular intervalos continuos de cada régimen
    intervalos = calcular_intervalos_regimen(df)
    ruta_resumen = salida_csv_p.with_suffix(".txt")
    guardar_resumen_textual(df, intervalos, ruta_resumen)

    # 6. Generar gráficas visuales si se solicita
    if generar_graficas:
        ruta_timeline = salida_csv_p.parent / f"{salida_csv_p.stem}_linea_tiempo.png"
        ruta_distribucion = salida_csv_p.parent / f"{salida_csv_p.stem}_distribucion.png"
        generar_graficas_visualizacion(df, ruta_timeline, ruta_distribucion)

    print("\n" + "=" * 65)
    print("      CONTEO DE FOTOGRAMAS POR RÉGIMEN FÍSICO ESTIMADO")
    print("=" * 65)
    conteo_modos = df["sensor_modo_estimado"].value_counts()
    for modo, cant in conteo_modos.items():
        pct = (cant / len(df)) * 100
        print(f"  • {modo:<26}: {cant:>6} frames ({pct:>5.1f}%)")

    print("\n" + "=" * 65)
    print("      CONTEO POR CLÚSTER NO SUPERVISADO (K-MEANS)")
    print("=" * 65)
    conteo_clusters = df["cluster_id"].value_counts()
    for cid, cant in conteo_clusters.items():
        pct = (cant / len(df)) * 100
        print(f"  • Clúster {cid}: {cant:>6} frames ({pct:>5.1f}%)")

    print("\n" + "=" * 65)
    print(f" [OK] CSV guardado en       : {salida_csv_p}")
    print(f" [OK] Resumen guardado en   : {ruta_resumen}")
    if generar_graficas:
        print(f" [OK] Gráfica timeline en   : {ruta_timeline}")
        print(f" [OK] Gráfica conteo en     : {ruta_distribucion}")
    print("=" * 65 + "\n")

    return df


def calcular_intervalos_regimen(df):
    """Agrupa fotogramas consecutivos que pertenecen al mismo régimen."""
    intervalos = []
    if df.empty:
        return intervalos

    regimen_actual = df.iloc[0]["sensor_modo_estimado"]
    inicio_frame = df.iloc[0]["frame_idx"]
    inicio_archivo = df.iloc[0]["archivo"]
    cuenta = 1

    for i in range(1, len(df)):
        fila = df.iloc[i]
        reg = fila["sensor_modo_estimado"]
        if reg == regimen_actual:
            cuenta += 1
        else:
            fin_frame = df.iloc[i - 1]["frame_idx"]
            fin_archivo = df.iloc[i - 1]["archivo"]
            intervalos.append({
                "regimen": regimen_actual,
                "inicio_idx": inicio_frame,
                "fin_idx": fin_frame,
                "inicio_archivo": inicio_archivo,
                "fin_archivo": fin_archivo,
                "duracion_frames": cuenta,
            })
            regimen_actual = reg
            inicio_frame = fila["frame_idx"]
            inicio_archivo = fila["archivo"]
            cuenta = 1

    # Último bloque
    fin_frame = df.iloc[-1]["frame_idx"]
    fin_archivo = df.iloc[-1]["archivo"]
    intervalos.append({
        "regimen": regimen_actual,
        "inicio_idx": inicio_frame,
        "fin_idx": fin_frame,
        "inicio_archivo": inicio_archivo,
        "fin_archivo": fin_archivo,
        "duracion_frames": cuenta,
    })

    return intervalos


def guardar_resumen_textual(df, intervalos, ruta_txt):
    """Guarda un reporte estructurado y legible con conteos e intervalos."""
    total = len(df)
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("  REPORTE DE CARACTERIZACIÓN DE REGÍMENES - VIDEOS FLIR (FAC)\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Total fotogramas analizados: {total}\n\n")

        f.write("--- DISTRIBUCIÓN TOTAL POR RÉGIMEN ---\n")
        for reg, cant in df["sensor_modo_estimado"].value_counts().items():
            pct = (cant / total) * 100
            f.write(f"  • {reg:<26}: {cant:>6} frames ({pct:>5.1f}%)\n")

        f.write("\n--- INTERVALOS TEMPORALES DETECTADOS (SECUENCIAS CONTINUAS) ---\n")
        f.write(f"{'Régimen':<25} | {'Rango Frames':<22} | {'Duración':<10} | {'Archivos'}\n")
        f.write("-" * 85 + "\n")
        for it in intervalos:
            rango = f"{it['inicio_idx']} - {it['fin_idx']}"
            archivos = f"{it['inicio_archivo']} -> {it['fin_archivo']}"
            f.write(f"{it['regimen']:<25} | {rango:<22} | {it['duracion_frames']:>5} frms | {archivos}\n")


def generar_graficas_visualizacion(df, ruta_timeline, ruta_distribucion):
    """Genera las gráficas de línea de tiempo y de distribución."""
    # Colores consistentes por régimen
    paleta = {
        "HDEO_Optico_Estandar": "#2ca02c",     # Verde
        "HDEO_Optico_DDE_Activo": "#ff7f0e",   # Naranja
        "HDEO_Optico_HAZE": "#1f77b4",         # Azul
        "HDIR_Termico": "#d62728",             # Rojo
    }

    # 1. Gráfica de Línea de Tiempo (4 Paneles)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    x = df["frame_idx"]

    # Panel A: Régimen asignado (discreto)
    modos_unicos = sorted(df["sensor_modo_estimado"].unique())
    mapa_y = {m: i for i, m in enumerate(modos_unicos)}
    y_regimen = [mapa_y[m] for m in df["sensor_modo_estimado"]]

    colores_frames = [paleta.get(m, "#7f7f7f") for m in df["sensor_modo_estimado"]]
    axes[0].scatter(x, y_regimen, c=colores_frames, s=6, alpha=0.8)
    axes[0].set_yticks(range(len(modos_unicos)))
    axes[0].set_yticklabels(modos_unicos, fontsize=9)
    axes[0].set_ylabel("Régimen Detectado", fontweight="bold")
    axes[0].set_title("Línea de Tiempo de Regímenes Físicos del Sensor FLIR (FAC)", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel B: Delta RGB (Térmico vs Óptico)
    axes[1].plot(x, df["delta_rgb"], color="#9467bd", lw=1.2, label="Delta RGB (|R-G| + |G-B|)")
    axes[1].axhline(y=0.6, color="red", linestyle=":", label="Umbral Térmico (< 0.6)")
    axes[1].set_ylabel("Delta RGB", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel C: Contraste IQR (Bruma / HAZE)
    axes[2].plot(x, df["contraste_iqr"], color="#1f77b4", lw=1.2, label="Contraste IQR (p75 - p25)")
    axes[2].axhline(y=25.0, color="blue", linestyle=":", label="Umbral HAZE (< 25.0)")
    axes[2].set_ylabel("Contraste IQR", fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Panel D: Energía de Bordes (Laplacian / DDE)
    axes[3].plot(x, df["energia_bordes"], color="#ff7f0e", lw=1.2, label="Energía Bordes (Mediana Laplaciano)")
    axes[3].axhline(y=12.0, color="orange", linestyle=":", label="Umbral DDE Activo (> 12.0)")
    axes[3].set_ylabel("Energía Bordes", fontweight="bold")
    axes[3].set_xlabel("Índice de Fotograma (Frame)", fontweight="bold")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(ruta_timeline, dpi=200)
    plt.close()

    # 2. Gráfica de Distribución (Barras + Torta)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    conteo = df["sensor_modo_estimado"].value_counts()
    colores_pie = [paleta.get(m, "#7f7f7f") for m in conteo.index]

    # Barras
    bars = ax1.bar(conteo.index, conteo.values, color=colores_pie, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Número de Fotogramas", fontweight="bold")
    ax1.set_title("Conteo de Fotogramas por Régimen", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", rotation=25)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Torta (Pie chart)
    ax2.pie(conteo.values, labels=conteo.index, autopct="%1.1f%%",
            colors=colores_pie, startangle=140, explode=[0.05] * len(conteo),
            wedgeprops={"edgecolor": "black", "linewidth": 1})
    ax2.set_title("Proporción Porcentual de Regímenes", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(ruta_distribucion, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Caracterización física y conteo de regímenes en videos FLIR FAC."
    )
    parser.add_argument(
        "--carpeta",
        type=str,
        required=True,
        help="Ruta a la carpeta con los fotogramas (ej. videos/video1/frames_originales).",
    )
    parser.add_argument(
        "--salida",
        type=str,
        default="conteo_regimenes_detectados.csv",
        help="Ruta del CSV de salida (por defecto: conteo_regimenes_detectados.csv).",
    )
    parser.add_argument(
        "--sin-graficas",
        action="store_true",
        help="Desactiva la generación de gráficas (útil si no hay matplotlib o se quiere máxima velocidad).",
    )

    args = parser.parse_args()
    caracterizar_y_contar_regimenes(
        carpeta_frames=args.carpeta,
        salida_csv=args.salida,
        generar_graficas=not args.sin_graficas,
    )


if __name__ == "__main__":
    main()
