#!/usr/bin/env python3
"""
Ranking final de la seleccion de denoising usando SOLO metodos fieles a su articulo.

Junta, por regimen, los ranking.csv de la ronda 3 (resultados/denoising/ronda3/<regimen>/) y de la
ronda 3 fiel (resultados/denoising/ronda3_fiel/<regimen>/), marca la fidelidad de cada experimento
segun la auditoria (proyecto "Tesis": fidelidad_metodos_vs_articulos.md) y escribe:

  resultados/denoising/ranking_fiel/<regimen>.csv   todos los experimentos, con su columna fidelidad
  resultados/denoising/ranking_fiel/ganadores.csv   por regimen: mejor por grano y mejor por NIQE
                                                    (y por BRISQUE), solo entre los fieles

Criterios (los mismos de antes):
  grano  : menor grano_p50 entre los que cumplen retencion_bordes_p25 >= 0,8 y delta_e_suave_p50 < 2,5
  NIQE   : mayor niqe_cambio_pct (mejora frente a la imagen sin HUD), con las mismas dos condiciones
  BRISQUE: mayor brisque_cambio_pct, mismas condiciones. Ojo: BRISQUE cerca de 0 infla el porcentaje.

Uso:  python scripts/ranking_fiel.py
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# Fieles tal como se corrieron en la ronda 3 (sufijo _r3). Ver la auditoria.
FIELES_R3 = {
    "bm3d", "guiado", "mediana_adaptativa", "nlm", "mediana_nlm",            # sin entrenamiento
    "apbsn", "mmbsn", "puca", "lgbpn", "ubsn", "tbsn", "cvfsid", "sass", "nsp", "tmbsn",
    "zsn2n_zs", "zskan", "dip", "filter2noise", "noise2detail",
    "gr2r", "saved", "n2s",                                                   # casi fieles
}
# Adaptados que se corrigieron: su version valida es la _r3f (la _r3 queda fuera)
CORREGIDOS = {"nb2nb", "b2u", "n2v", "structn2v", "n2v2", "noise2same", "noise2score", "udvd", "r2r",
              "s2splus", "f2f", "zsn2n"}
REIMPLEMENTADOS = {"mash", "swinia", "selfformer", "stbn", "rdrf", "p2n"}
SIN_ARREGLO = {"f2r", "tap", "pcst", "self2self", "cut2self", "mf2f"}
# Zero-shot (una red por imagen): sirven para las imagenes del dataset, pero no para limpiar videos
# completos; la fase 2 por grano los dejo fuera por eso. Se reporta el ranking con y sin ellos.
ZERO_SHOT = {"zsn2n_zs", "zskan", "dip", "filter2noise", "noise2detail", "s2splus", "mash"}


def metodo_de(exp: str) -> tuple[str, str]:
    """('n2v', 'r3f') a partir de 'n2v_fiel_r3f_ycolor'."""
    e = re.sub(r"_ycolor$", "", exp)
    m = re.match(r"^(.*?)_(r3f|r3)$", e)
    if not m:
        return e, ""
    base = re.sub(r"_fiel$", "", m.group(1))
    return base, m.group(2)


def fidelidad(exp: str) -> str:
    met, ronda = metodo_de(exp)
    if exp.startswith(("ens3", "stack")):
        return "combinacion (no es un metodo)"
    if ronda == "r3f":
        return "fiel (corregido)"
    if met in FIELES_R3:
        return "fiel"
    if met in CORREGIDOS:
        return "adaptado (reemplazado por su version _r3f)"
    if met in REIMPLEMENTADOS:
        return "reimplementacion sin codigo oficial"
    if met in SIN_ARREGLO:
        return "adaptado sin arreglo posible"
    return "sin clasificar"


def num(r, k):
    try:
        return float(r.get(k, ""))
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rondas", nargs="+", default=["resultados/denoising/ronda3", "resultados/denoising/ronda3_fiel"])
    ap.add_argument("--salida", default="resultados/denoising/ranking_fiel")
    ap.add_argument("--min-bordes", type=float, default=0.8)
    ap.add_argument("--max-delta-e", type=float, default=2.5)
    a = ap.parse_args()
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    regimenes = sorted({p.parent.name for r in a.rondas for p in Path(r).glob("*/ranking.csv")})
    if not regimenes:
        raise SystemExit("No hay ranking.csv en " + ", ".join(a.rondas))
    ganadores = []
    for reg in regimenes:
        filas = []
        for r in a.rondas:
            p = Path(r) / reg / "ranking.csv"
            if not p.exists():
                print(f"!! falta {p}")
                continue
            with open(p, encoding="utf-8", newline="") as f:
                for fila in csv.DictReader(f):
                    if any(g["experimento"] == fila["experimento"] for g in filas):
                        continue  # el ranking de la ronda 3 trae algunas filas repetidas
                    fila["ronda"] = Path(r).name
                    fila["zero_shot"] = "si" if metodo_de(fila["experimento"])[0] in ZERO_SHOT else "no"
                    fila["fidelidad"] = fidelidad(fila["experimento"])
                    filas.append(fila)
        cols = ["experimento", "ronda", "fidelidad", "zero_shot", "n", "grano_p50", "retencion_bordes_p25", "delta_e_suave_p50",
                "niqe_cambio_pct", "brisque_cambio_pct", "musiq_cambio_pct", "clipiqa_cambio_pct", "estado"]
        filas.sort(key=lambda f: num(f, "grano_p50"))
        with open(sal / f"{reg}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(filas)
        validos = [f for f in filas if f["fidelidad"].startswith("fiel")
                   and num(f, "retencion_bordes_p25") >= a.min_bordes and num(f, "delta_e_suave_p50") < a.max_delta_e]
        sin_clas = sorted({f["experimento"] for f in filas if f["fidelidad"] == "sin clasificar"})
        if sin_clas:
            print(f"!! {reg}: sin clasificar (no entran al ranking fiel): {', '.join(sin_clas)}")
        print(f"\n===== {reg}: {len(validos)} experimentos fieles que cumplen bordes >= {a.min_bordes} y dE < {a.max_delta_e}")
        for crit, clave, mejor_es_menor, conj in (
                ("grano", "grano_p50", True, validos),
                ("grano (sin zero-shot)", "grano_p50", True, [f for f in validos if f["zero_shot"] == "no"]),
                ("NIQE", "niqe_cambio_pct", False, validos),
                ("BRISQUE", "brisque_cambio_pct", False, validos)):
            orden = sorted(conj, key=lambda f: num(f, clave), reverse=not mejor_es_menor)
            print(f"  top 3 por {crit}:")
            for f in orden[:3]:
                print(f"    {f['experimento']:30s} grano {num(f, 'grano_p50'):6.2f} | bordes {num(f, 'retencion_bordes_p25'):.2f} | "
                      f"dE {num(f, 'delta_e_suave_p50'):.2f} | NIQE {num(f, 'niqe_cambio_pct'):+.1f} % | "
                      f"BRISQUE {num(f, 'brisque_cambio_pct'):+.1f} % | n {f['n']}")
            if orden:
                g = orden[0]
                ganadores.append({"regimen": reg, "criterio": crit, "experimento": g["experimento"],
                                  **{k: g.get(k, "") for k in cols[4:12]}})
    with open(sal / "ganadores.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["regimen", "criterio", "experimento", "n", "grano_p50", "retencion_bordes_p25",
                                          "delta_e_suave_p50", "niqe_cambio_pct", "brisque_cambio_pct",
                                          "musiq_cambio_pct", "clipiqa_cambio_pct"])
        w.writeheader()
        w.writerows(ganadores)
    print(f"\nListo: {sal}/<regimen>.csv y {sal}/ganadores.csv")


if __name__ == "__main__":
    main()
