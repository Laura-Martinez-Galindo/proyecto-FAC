#!/usr/bin/env python3
"""
Generador de Diapositivas 16:9 de Reducción de Ruido (Denoising) para Sustentación de Tesis.
Genera imágenes estilo PowerPoint (1920x1080 px) comparando:
[Izquierda: Sin HUD] vs [Derecha: UDVD T=5 Global]
para los Top 5 frames con mayor reducción de ruido y Top 5 con menor reducción (casos difíciles).
"""

import argparse
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent


def natural_key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def estimar_sigma_mad(gray):
    """Estima la desviación estándar de ruido usando la desviación absoluta mediana (MAD) del Laplaciano."""
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    return float(mad / 0.6745)


def formatear_tiempo(frame_idx, fps=30.0):
    segundos = frame_idx / fps
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"


def extraer_numero_frame(nombre_stem):
    nums = re.findall(r"\d+", nombre_stem)
    return int(nums[-1]) if nums else 0


def crear_diapositiva_16_9(
    img_sinhud,
    img_udvd,
    id_video,
    num_frame,
    minuto_str,
    sigma_sinhud,
    sigma_udvd,
    pct_reduccion,
    ruta_salida
):
    """
    Renderiza un lienzo 1920x1080 (16:9) limpio, minimalista y elegante:
    - Título centrado arriba: Video X - Frame Y (Minuto Z)
    - Título encima de cada imagen: 'Sin HUD' y 'UDVD (T=5) - Reducción XX.X%'
    - Dos paneles grandes bien centrados ocupando el espacio visual.
    """
    ANCHO_SLIDE = 1920
    ALTO_SLIDE = 1080

    # Fondo oscuro elegante (#141414 BGR: 20, 20, 20)
    canvas = np.zeros((ALTO_SLIDE, ANCHO_SLIDE, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    # 1. TÍTULO PRINCIPAL CENTRADO
    nombre_vid = f"Video {id_video.replace('video', '')}"
    titulo_texto = f"{nombre_vid} - Frame {num_frame:05d} (Minuto {minuto_str})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(titulo_texto, font, 1.25, 2)
    tx = (ANCHO_SLIDE - tw) // 2
    cv2.putText(canvas, titulo_texto, (tx, 85), font, 1.25, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. CÁLCULO DE DIMENSIONES Y POSICIONAMIENTO DE LOS PANELES
    h_orig, w_orig, _ = img_sinhud.shape
    aspecto = w_orig / max(1, h_orig)

    gap = 50
    margen_lat = 70
    ancho_panel = (ANCHO_SLIDE - (2 * margen_lat) - gap) // 2
    alto_panel = int(ancho_panel / aspecto)

    alto_max = 840
    if alto_panel > alto_max:
        alto_panel = alto_max
        ancho_panel = int(alto_panel * aspecto)

    p_izq_res = cv2.resize(img_sinhud, (ancho_panel, alto_panel), interpolation=cv2.INTER_AREA)
    p_der_res = cv2.resize(img_udvd, (ancho_panel, alto_panel), interpolation=cv2.INTER_AREA)

    ancho_total_bloque = (2 * ancho_panel) + gap
    x_izq = (ANCHO_SLIDE - ancho_total_bloque) // 2
    x_der = x_izq + ancho_panel + gap
    y_panel = 175

    # 3. TÍTULOS ENCIMA DE CADA PANEL (AFUERA DE LA IMAGEN)
    # Título Izquierdo: "Sin HUD"
    txt_izq = "Sin HUD"
    (ti_w, ti_h), _ = cv2.getTextSize(txt_izq, font, 0.95, 2)
    t_izq_x = x_izq + (ancho_panel - ti_w) // 2
    cv2.putText(canvas, txt_izq, (t_izq_x, y_panel - 20), font, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    # Título Derecho: "UDVD (T=5) - Reducción XX.X%"
    txt_der = f"UDVD (T=5) - Reducción {pct_reduccion:.1f}%"
    (td_w, td_h), _ = cv2.getTextSize(txt_der, font, 0.95, 2)
    t_der_x = x_der + (ancho_panel - td_w) // 2
    cv2.putText(canvas, txt_der, (t_der_x, y_panel - 20), font, 0.95, (0, 240, 180), 2, cv2.LINE_AA)

    # 4. DIBUJAR PANELES DE IMAGEN CON BORDE FINO
    canvas[y_panel : y_panel + alto_panel, x_izq : x_izq + ancho_panel] = p_izq_res
    cv2.rectangle(canvas, (x_izq - 2, y_panel - 2), (x_izq + ancho_panel + 1, y_panel + alto_panel + 1), (0, 140, 255), 2)

    canvas[y_panel : y_panel + alto_panel, x_der : x_der + ancho_panel] = p_der_res
    cv2.rectangle(canvas, (x_der - 2, y_panel - 2), (x_der + ancho_panel + 1, y_panel + alto_panel + 1), (0, 220, 150), 2)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_salida), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def calcular_riqueza_escena(gray):
    """
    Calcula la riqueza visual y textura de la escena (vegetación, estructuras, ríos, minas).
    Descarta cielos planos o fondos vacíos de calibración térmica.
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energia_grad = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))
    std_intensidad = float(np.std(gray))
    return energia_grad, std_intensidad


def seleccionar_top_diversos(lista_ordenada, top_k=10, min_dist_frames=2000):
    """
    Selecciona los top K frames garantizando:
    1. Separación temporal amplia (min_dist_frames = 2000 frames ~ 1+ min de vuelo).
    2. Priorización de frames con contenido visual real (bosques, ríos, estructuras).
    """
    seleccionados = []
    for item in lista_ordenada:
        idx_act = item["idx"]
        if all(abs(idx_act - s["idx"]) >= min_dist_frames for s in seleccionados):
            seleccionados.append(item)
            if len(seleccionados) == top_k:
                break

    # Si con 2000 frames no llena 10, relajar paso a paso
    if len(seleccionados) < top_k:
        for dist_relax in [1200, 600, 300]:
            for item in lista_ordenada:
                if item not in seleccionados:
                    if all(abs(item["idx"] - s["idx"]) >= dist_relax for s in seleccionados):
                        seleccionados.append(item)
                        if len(seleccionados) == top_k:
                            break
            if len(seleccionados) == top_k:
                break

    if len(seleccionados) < top_k:
        for item in lista_ordenada:
            if item not in seleccionados:
                seleccionados.append(item)
                if len(seleccionados) == top_k:
                    break

    return seleccionados


def es_frame_valido(img_sinhud, img_udvd):
    """
    Filtro estricto de integridad y riqueza visual:
    Descarta frames:
    1. Blancos / quemados / sobreexpuestos (media > 190 o > 5% de píxeles saturados en blanco > 235).
    2. Negros / apagados / corruptos (media < 45 o > 5% de píxeles negros < 18).
    3. Planos / sin información térmica (desv estándar < 20.0).
    4. Sin gradientes / sin objetos (energía de gradientes Sobel < 4.5).
    """
    for img in [img_sinhud, img_udvd]:
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        total_px = gray.shape[0] * gray.shape[1]

        media = float(np.mean(gray))
        desv = float(np.std(gray))

        # 1. Rango de intensidad media (debe ser imagen térmica con buen balance)
        if media < 50.0 or media > 190.0:
            return False

        # 2. Desviación estándar (debe tener contraste térmico de vegetación/objetos)
        if desv < 20.0 or desv > 80.0:
            return False

        # 3. Saturación de blancos (como el frame 02336 o 06581)
        pct_blanco = float(np.count_nonzero(gray > 235)) / total_px
        if pct_blanco > 0.05:
            return False

        # 4. Fondo negro / bloques muertos (como el frame 44356)
        pct_negro = float(np.count_nonzero(gray < 18)) / total_px
        if pct_negro > 0.05:
            return False

        # 5. Riqueza de bordes (árboles, casas, caminos, dragas, ríos)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energia_grad = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))
        if energia_grad < 4.5:
            return False

    return True


def procesar_video(id_video, dir_sinhud, dir_udvd, dir_salida, top_n=10, paso_muestreo=5, min_frame=18000):
    print(f"\n=======================================================")
    print(f"PROCESANDO {id_video.upper()} PARA DIAPOSITIVAS DE REDUCCIÓN DE RUIDO")
    print(f"Sin HUD: {dir_sinhud}")
    print(f"UDVD:    {dir_udvd}")
    print(f"Rango de vuelo: Frame >= {min_frame}")
    print(f"=======================================================")

    if not dir_sinhud.is_dir() or not dir_udvd.is_dir():
        print(f"[Error] Directorios no encontrados para {id_video}. Saltando.")
        return

    exts = {".png", ".jpg", ".jpeg"}
    frames_sinhud = sorted([p for p in dir_sinhud.iterdir() if p.suffix.lower() in exts], key=natural_key)
    frames_udvd_dict = {p.name: p for p in dir_udvd.iterdir() if p.suffix.lower() in exts}

    comunes = [p for p in frames_sinhud if p.name in frames_udvd_dict]
    n_total = len(comunes)
    print(f"Total frames comunes sincronizados: {n_total}")

    if n_total == 0:
        print(f"[Error] No hay frames comunes para {id_video}.")
        return

    # Restringir a sobrevuelo operativo (frames >= 18000)
    indices_candidatos = [
        i for i in range(0, n_total, paso_muestreo)
        if extraer_numero_frame(comunes[i].stem) >= min_frame
    ]
    if not indices_candidatos:
        indices_candidatos = list(range(0, n_total, paso_muestreo))

    print(f"Escaneando reducción de ruido y textura en {len(indices_candidatos)} frames de sobrevuelo operativo...")

    def evaluar_frame(idx):
        p_sin = comunes[idx]
        p_udvd = frames_udvd_dict[p_sin.name]

        img_sin = cv2.imread(str(p_sin))
        img_u = cv2.imread(str(p_udvd))
        if img_sin is None or img_u is None:
            return None

        if img_u.shape[:2] != img_sin.shape[:2]:
            img_u = cv2.resize(img_u, (img_sin.shape[1], img_sin.shape[0]), interpolation=cv2.INTER_AREA)

        if not es_frame_valido(img_sin, img_u):
            return None

        g_sin = cv2.cvtColor(img_sin, cv2.COLOR_BGR2GRAY)
        g_udvd = cv2.cvtColor(img_u, cv2.COLOR_BGR2GRAY)

        energia_grad, std_int = calcular_riqueza_escena(g_sin)

        s_sin = estimar_sigma_mad(g_sin)
        s_udvd = estimar_sigma_mad(g_udvd)

        if s_sin < 4.0:
            return None

        # Reducción porcentual de ruido matemáticamente válida y acotada
        caida_sigma = s_sin - s_udvd
        pct_reduccion = max(0.0, min(80.0, (caida_sigma / s_sin) * 100.0))

        return {
            "idx": idx,
            "p_sinhud": p_sin,
            "p_udvd": p_udvd,
            "s_sin": s_sin,
            "s_udvd": s_udvd,
            "pct_reduccion": pct_reduccion,
            "energia_grad": energia_grad,
            "std_int": std_int,
        }

    resultados = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for res in tqdm(pool.map(evaluar_frame, indices_candidatos), total=len(indices_candidatos), desc="Analizando denoising y textura", dynamic_ncols=True):
            if res is not None:
                resultados.append(res)

    if not resultados:
        return

    # Filtrar frames con textura real (bosques, ríos, campamentos, maquinaria)
    candidatos_con_textura = [
        r for r in resultados
        if r["energia_grad"] >= 4.0 and r["std_int"] >= 20.0
    ]
    if len(candidatos_con_textura) < (2 * top_n):
        candidatos_con_textura = [
            r for r in resultados
            if r["energia_grad"] >= 2.5 and r["std_int"] >= 15.0
        ] or resultados

    print(f"Frames válidos con contenido visual rico: {len(candidatos_con_textura)}/{len(resultados)}")

    # 1. Top N Mayor Reducción de Ruido con separación temporal amplia (2000 frames)
    orden_mayor_reduccion = sorted(candidatos_con_textura, key=lambda x: (x["pct_reduccion"], x["energia_grad"]), reverse=True)
    top_mayor = seleccionar_top_diversos(orden_mayor_reduccion, top_k=top_n, min_dist_frames=2000)

    # 2. Top N Menor Reducción de Ruido / Casos Retadores con separación temporal amplia (2000 frames)
    orden_menor_reduccion = sorted(candidatos_con_textura, key=lambda x: (x["pct_reduccion"], -x["energia_grad"]))
    top_menor = seleccionar_top_diversos(orden_menor_reduccion, top_k=top_n, min_dist_frames=2000)

    # Renderizar diapositivas de Mayor Reducción
    print(f"\nGenerando {len(top_mayor)} diapositivas de Mayor Reducción ({id_video})...")
    for rank, item in enumerate(top_mayor, start=1):
        num_frame = extraer_numero_frame(item["p_sinhud"].stem)
        t_str = formatear_tiempo(num_frame)

        img_sinhud = cv2.imread(str(item["p_sinhud"]))
        img_udvd = cv2.imread(str(item["p_udvd"]))
        if img_sinhud.shape != img_udvd.shape:
            img_udvd = cv2.resize(img_udvd, (img_sinhud.shape[1], img_sinhud.shape[0]), interpolation=cv2.INTER_AREA)

        out_path = dir_salida / f"{id_video}_top{rank:02d}_mayor_reduccion_frame_{num_frame:05d}.png"
        crear_diapositiva_16_9(
            img_sinhud,
            img_udvd,
            id_video,
            num_frame,
            t_str,
            item["s_sin"],
            item["s_udvd"],
            item["pct_reduccion"],
            out_path
        )
        print(f"  -> Guardada: {out_path.name} (Frame {num_frame:05d} | Reducción: {item['pct_reduccion']:.1f}% | Gradiente: {item['energia_grad']:.1f})")

    # Renderizar diapositivas de Menor Reducción
    print(f"\nGenerando {len(top_menor)} diapositivas de Menor Reducción / Casos Difíciles ({id_video})...")
    for rank, item in enumerate(top_menor, start=1):
        num_frame = extraer_numero_frame(item["p_sinhud"].stem)
        t_str = formatear_tiempo(num_frame)

        img_sinhud = cv2.imread(str(item["p_sinhud"]))
        img_udvd = cv2.imread(str(item["p_udvd"]))
        if img_sinhud.shape != img_udvd.shape:
            img_udvd = cv2.resize(img_udvd, (img_sinhud.shape[1], img_sinhud.shape[0]), interpolation=cv2.INTER_AREA)

        out_path = dir_salida / f"{id_video}_top{rank:02d}_menor_reduccion_frame_{num_frame:05d}.png"
        crear_diapositiva_16_9(
            img_sinhud,
            img_udvd,
            id_video,
            num_frame,
            t_str,
            item["s_sin"],
            item["s_udvd"],
            item["pct_reduccion"],
            out_path
        )
        print(f"  -> Guardada: {out_path.name} (Frame {num_frame:05d} | Reducción: {item['pct_reduccion']:.1f}% | Gradiente: {item['energia_grad']:.1f})")


def main():
    p = argparse.ArgumentParser(description="Generar Diapositivas 16:9 de Reducción de Ruido (Sin HUD vs UDVD T=5 Global)")
    p.add_argument("--videos", nargs="+", default=["video2"], help="Lista de videos a procesar (ej. video2)")
    p.add_argument("--top-n", type=int, default=10, help="Cantidad de frames por categoría (top mayor y menor reducción)")
    p.add_argument("--paso-muestreo", type=int, default=5, help="Paso de escaneo de frames")
    p.add_argument("--min-frame", type=int, default=18000, help="Frame mínimo de inicio (ignorar despegue/calibración inicial)")
    p.add_argument("--carpeta-salida", default="figuras_tesis/presentacion_denoising", help="Carpeta destino de las diapositivas")
    p.add_argument("--carpeta-udvd", default="udvd_sin_hud", help="Nombre de la carpeta del modelo UDVD dentro de videos/{video}/expos/")
    args = p.parse_args()

    dir_salida = RAIZ / args.carpeta_salida
    dir_salida.mkdir(parents=True, exist_ok=True)

    for vid in args.videos:
        dir_sinhud = RAIZ / "videos" / vid / "frames_sin_hud"
        dir_udvd = RAIZ / "videos" / vid / "expos" / args.carpeta_udvd

        # Si no está en expos/udvd_sin_hud, verificar rutas alternas
        if not dir_udvd.is_dir():
            alt = RAIZ / "videos" / vid / "expos" / "udvd_t5_ult10min_sin_hud"
            if alt.is_dir():
                dir_udvd = alt

        procesar_video(vid, dir_sinhud, dir_udvd, dir_salida, top_n=args.top_n, paso_muestreo=args.paso_muestreo, min_frame=args.min_frame)

    print("\n=======================================================")
    print("¡TODAS LAS DIAPOSITIVAS DE DENOISING FUERON GENERADAS!")
    print(f"Carpeta de salida: {dir_salida}")
    print("=======================================================")


if __name__ == "__main__":
    main()
