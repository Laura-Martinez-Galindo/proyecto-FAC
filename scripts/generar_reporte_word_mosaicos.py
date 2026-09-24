#!/usr/bin/env python3
"""
Generador de Reporte en Word (.docx) Horizontal para Mosaicos de Ruido Térmico:
- Formato de página Horizontal (Landscape) A4 / Letter.
- Inserta los 10 mosaicos 1x4 de alta resolución ocupando el ancho óptimo de la página.
- Incluye introducción técnica, metodología y tablas explicativas.
"""

from pathlib import Path
import re
import docx
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor

RAIZ = Path(__file__).resolve().parent.parent
DIR_MOSAICOS = RAIZ / "figuras_tesis/mosaicos_ruido_video2"
DOC_SALIDA = RAIZ / "figuras_tesis/Reporte_Mosaicos_Ruido_Termico_Video2.docx"


def configurar_estilos(doc):
    # Configurar estilo Normal
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)


def crear_reporte_word():
    print(f"Generando documento Word horizontal: {DOC_SALIDA.name}...")
    doc = docx.Document()
    configurar_estilos(doc)

    # Configurar todas las secciones en orientación Horizontal (Landscape)
    for section in doc.sections:
        section.orientation = WD_ORIENT.LANDSCAPE
        # Invertir ancho y alto para Landscape (Letter: 11 x 8.5 in)
        section.page_width = Inches(11.0)
        section.page_height = Inches(8.5)
        section.top_margin = Inches(0.6)
        section.bottom_margin = Inches(0.6)
        section.left_margin = Inches(0.6)
        section.right_margin = Inches(0.6)

    # Título Principal
    p_tit = doc.add_paragraph()
    p_tit.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_tit = p_tit.add_run("ANÁLISIS COMPARATIVO DE EXTRACCIÓN DE RUIDO TÉRMICO FLIR Y RESTAURACIÓN CON UDVD")
    run_tit.bold = True
    run_tit.font.size = Pt(16)
    run_tit.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    # Subtítulo institucional
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Proyecto Tesis FAC - Universidad de los Andes | Video de Misión 2 (Santander)")
    run_sub.font.size = Pt(11)
    run_sub.italic = True
    run_sub.font.color.rgb = RGBColor(0x7F, 0x8C, 0x8D)

    doc.add_paragraph()

    # Resumen Metodológico
    p_intro = doc.add_paragraph()
    run_intro_tit = p_intro.add_run("1. Estructura de Comparación (Layout 1x4 Horizontal):\n")
    run_intro_tit.bold = True
    run_intro_tit.font.size = Pt(12)
    run_intro_tit.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p_intro.add_run(
        "Para cada escena seleccionada se presenta una comparación visual de cuatro etapas de izquierda a derecha:\n"
        " • (a) Original: Cuadro capturado por el sensor FLIR aerotransportado con simbología HUD de navegación.\n"
        " • (b) Sin HUD: Cuadro tras la remoción de telemetría e inpainting espaciotemporal con ProPainter.\n"
        " • (c) Restaurado (UDVD): Cuadro procesado mediante el modelo UDVD (Universal Denoising for Video),\n"
        "       removiendo el ruido térmico sensor sin degradar bordes ni elementos de interés.\n"
        " • (d) Ruido Térmico Extraído: Mapa residual de alta frecuencia (|Sin HUD - UDVD|) con barra de escala en\n"
        "       Niveles Digitales (DN) de 0 a 25. Permite aislar y cuantificar el bandeado térmico y grano del sensor."
    )

    doc.add_paragraph()

    # Obtener mosaicos disponibles
    mosaicos_max = sorted(list(DIR_MOSAICOS.glob("mosaico_MAX_RUIDO_*.png")))
    mosaicos_min = sorted(list(DIR_MOSAICOS.glob("mosaico_MIN_RUIDO_*.png")))

    # Sección: Zonas con Mayor Ruido Térmico
    doc.add_heading("2. Escenas con Mayor Nivel de Ruido Térmico (High Noise Areas)", level=1)
    p_max_desc = doc.add_paragraph()
    p_max_desc.add_run(
        "Corresponden a los cuadros con mayor energía de ruido residual térmico donde el modelo UDVD eliminó "
        "intensas líneas de no-uniformidad del sensor (striping) y perturbaciones estocásticas:"
    )

    for idx, f_img in enumerate(mosaicos_max, 1):
        match_frame = re.search(r"frame_(\d+)", f_img.name)
        num_frame = match_frame.group(1) if match_frame else f"{idx}"
        
        p_fig_tit = doc.add_paragraph()
        p_fig_tit.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r_f = p_fig_tit.add_run(f"Figura 2.{idx}: Caso #{idx} - Mayor Ruido Térmico (Frame #{num_frame})")
        r_f.bold = True
        r_f.font.size = Pt(11)
        r_f.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

        # Insertar imagen ocupando el ancho útil horizontal (9.8 pulgadas)
        try:
            doc.add_picture(str(f_img), width=Inches(9.8))
            last_p = doc.paragraphs[-1]
            last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception as e:
            doc.add_paragraph(f"[Error cargando imagen {f_img.name}: {e}]")

        doc.add_paragraph()  # Espaciador

    # Sección: Zonas con Menor Ruido Térmico
    doc.add_heading("3. Escenas con Menor Nivel de Ruido Térmico (Low Noise Areas)", level=1)
    p_min_desc = doc.add_paragraph()
    p_min_desc.add_run(
        "Corresponden a cuadros térmicamente homogéneos o estables donde el residuo extraído es bajo, "
        "demostrando la capacidad de UDVD de no sobre-suavizar texturas naturales cuando el ruido es mínimo:"
    )

    for idx, f_img in enumerate(mosaicos_min, 1):
        match_frame = re.search(r"frame_(\d+)", f_img.name)
        num_frame = match_frame.group(1) if match_frame else f"{idx}"
        
        p_fig_tit = doc.add_paragraph()
        p_fig_tit.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r_f = p_fig_tit.add_run(f"Figura 3.{idx}: Caso #{idx} - Menor Ruido Térmico (Frame #{num_frame})")
        r_f.bold = True
        r_f.font.size = Pt(11)
        r_f.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

        try:
            doc.add_picture(str(f_img), width=Inches(9.8))
            last_p = doc.paragraphs[-1]
            last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception as e:
            doc.add_paragraph(f"[Error cargando imagen {f_img.name}: {e}]")

        doc.add_paragraph()

    # Guardar documento
    DOC_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(DOC_SALIDA))
    print(f"\n[+] Documento Word generado exitosamente en:\n    {DOC_SALIDA}")


if __name__ == "__main__":
    crear_reporte_word()
