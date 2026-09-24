#!/usr/bin/env python3
"""
Reconstrucción y Rediseño de Mosaicos en Formato Horizontal 1x4 (De Lado a Lado):
1. Layout Horizontal:
   [1. Original con HUD]  |  [2. Sin HUD (Inpainting)]  |  [3. Restaurado UDVD]  |  [4. Ruido Térmico Extraído + Colorbar]
2. Encabezados limpios ARRIBA de cada imagen (en franja superior dedicada, sin tapar píxeles).
3. Ruido Térmico corregido: Se normaliza la escala de alta frecuencia (evitando saturación amarilla)
   e incluye BARRA DE ESCALA / COLORBAR térmica científica.
4. Ejecuta 100% local en 2 segundos sin re-procesar en Hypatia.
"""

import os
from pathlib import Path
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DIR_MOSAICOS_IN = RAIZ / "figuras_tesis/mosaicos_ruido_video2"


def renderizar_mosaico_1x4(img_ori, img_sin, img_den, nombre_salida, info_texto=""):
    """
    Renderiza usando Matplotlib para tener tipografía nítida, encabezados superiores
    espaciados y una barra de escala (Colorbar) matemática profesional.
    """
    h, w = img_sin.shape[:2]
    img_ori = cv2.resize(img_ori, (w, h))
    img_den = cv2.resize(img_den, (w, h))
    
    # Calcular residuo de ruido térmico en escala de grises
    gr_sin = cv2.cvtColor(img_sin, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gr_den = cv2.cvtColor(img_den, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Residuo térmico físico
    ruido_term = gr_sin - gr_den
    
    # Para visualizar claramente el patrón de ruido de alta frecuencia (destriping + estocástico):
    # Restamos el componente de iluminación global de baja frecuencia
    blur_offset = cv2.GaussianBlur(ruido_term, (61, 61), 20.0)
    ruido_alta_frec = ruido_term - blur_offset

    # Escala térmica en niveles digitales (DN)
    v_max = max(15.0, np.percentile(np.abs(ruido_alta_frec), 99.0))
    v_min = -v_max

    # Configurar figura 1x4 horizontal
    plt.close('all')
    fig, axs = plt.subplots(1, 4, figsize=(28, 7.5), gridspec_kw={'width_ratios': [1, 1, 1, 1.18]})
    
    # Colores RGB para matplotlib
    rgb_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    rgb_sin = cv2.cvtColor(img_sin, cv2.COLOR_BGR2RGB)
    rgb_den = cv2.cvtColor(img_den, cv2.COLOR_BGR2RGB)

    # Panel 1: Original
    axs[0].imshow(rgb_ori)
    axs[0].set_title("(a) 1. Original (Con HUD + Ruido)", fontsize=13, fontweight="bold", pad=12, color="#c0392b")
    axs[0].axis("off")

    # Panel 2: Sin HUD
    axs[1].imshow(rgb_sin)
    axs[1].set_title("(b) 2. Sin HUD (Inpainting ProPainter)", fontsize=13, fontweight="bold", pad=12, color="#2980b9")
    axs[1].axis("off")

    # Panel 3: Restaurado UDVD
    axs[2].imshow(rgb_den)
    axs[2].set_title("(c) 3. Restaurado (Sin HUD + UDVD)", fontsize=13, fontweight="bold", pad=12, color="#27ae60")
    axs[2].axis("off")

    # Panel 4: Ruido Térmico Extraído con Colorbar
    im_ruido = axs[3].imshow(ruido_alta_frec, cmap="inferno", vmin=v_min, vmax=v_max)
    axs[3].set_title("(d) 4. Ruido Térmico Extraído (Sensor FLIR)", fontsize=13, fontweight="bold", pad=12, color="#8e44ad")
    axs[3].axis("off")

    # Añadir Colorbar al panel 4
    cbar = fig.colorbar(im_ruido, ax=axs[3], fraction=0.046, pad=0.04)
    cbar.set_label("Desviación de Ruido Térmico (Niveles Digitales DN)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # Pie de figura con métricas
    if info_texto:
        fig.text(0.5, 0.03, info_texto, ha="center", fontsize=12, fontweight="bold", 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#bdc3c7", lw=1.5))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    salida_p = Path(nombre_salida)
    salida_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida_p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Mosaico 1x4 regenerado: {salida_p.name}")


def main():
    archivos = sorted(list(DIR_MOSAICOS_IN.glob("*.png")))
    print(f"Regenerando {len(archivos)} mosaicos a formato Horizontal 1x4...")

    for f_p in archivos:
        img_full = cv2.imread(str(f_p))
        if img_full is None:
            continue

        h_full, w_full = img_full.shape[:2]
        h_half = h_full // 2
        w_half = w_full // 2

        # Extraer los 3 paneles visuales originales (cortando encabezados y pie de pagina antiguos)
        margen_arriba = 60
        margen_abajo = 60
        img_ori = img_full[margen_arriba:h_half-margen_abajo, 0:w_half]
        img_sin = img_full[margen_arriba:h_half-margen_abajo, w_half:w_full]
        img_den = img_full[h_half+margen_arriba:h_full-margen_abajo, 0:w_half]

        # Extraer info de frame del nombre
        match_frame = re.search(r"frame_(\d+)", f_p.name)
        num_frame = match_frame.group(1) if match_frame else "N/A"
        tipo = "Zona de Mayor Ruido Térmico (High Noise Area)" if "MAX" in f_p.name else "Zona de Menor Ruido Térmico (Low Noise Area)"
        
        info_str = f"Misión Santander (FLIR Systems) | Frame #{num_frame} | {tipo} | Modelo: UDVD Blind-Spot"

        renderizar_mosaico_1x4(img_ori, img_sin, img_den, f_p, info_str)

    print("\n[+] Todos los mosaicos fueron convertidos exitosamente al formato Horizontal 1x4 con encabezados limpios y barra de escala.")


if __name__ == "__main__":
    main()
