#!/usr/bin/env python3
"""Analisis del experimento 2 de dense video captioning (orientado a mineria ilegal).

Cruza cada ventana que respondio el modelo con las imagenes anotadas del dataset que caen dentro de
ella (por video y segundo). Una ventana es SDZI+ si alguna de sus imagenes tiene una caja SDZI, SDZI-
si tiene imagenes anotadas y ninguna tiene SDZI, y "sin etiqueta" si no cae ninguna imagen anotada.

Salidas en --salida:
  ventanas_eval.csv     una fila por ventana: respuesta del modelo + etiquetas que caen en ella
  metricas.csv          por configuracion (version x prompt x etapa): AUROC del puntaje contra SDZI,
                        sensibilidad / falsas alarmas con umbral 5 y 7, falsas alarmas en el control (video2)
  indicadores.csv       acuerdo de cada indicador del modelo con la etiqueta correspondiente
  hud.csv               con HUD vs sin HUD, ventana por ventana
  errores.csv           falsos positivos y falsos negativos (con su descripcion) para revisarlos a ojo
  figuras/              violines del puntaje por clase, linea de tiempo por video, efecto del HUD
  resumen.txt
Uso:
  python scripts/analizar_captioning_mineria.py --resultados resultados/captioning_mineria \\
      --correspondencia datasets/dataset_original/correspondencia.csv --salida resultados/captioning_mineria_analisis
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETIQ = {"SDZI": "SDZI", "river": "rio", "road": "via", "building": "edificacion", "vehicle": "vehiculo"}
# indicador del modelo -> etiqueta del dataset con la que se compara
PAREJAS = {"rio": "river", "via_o_trocha": "road", "maquinaria": "vehicle", "campamento": "building",
           "zona_urbana": "building", "suelo_removido": "SDZI", "piscinas": "SDZI", "draga_o_balsa": "SDZI",
           "agua_turbia": "SDZI", "deforestacion": "SDZI"}
MINEROS = ["draga_o_balsa", "agua_turbia", "piscinas", "suelo_removido", "maquinaria", "campamento", "deforestacion"]


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    p, n = s[y], s[~y]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def si(v):
    return str(v).strip().lower() in ("si", "sí", "true", "yes")


def cargar(resultados: Path) -> pd.DataFrame:
    filas = []
    for f in sorted(resultados.rglob("ventanas.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            j = d.get("json") or {}
            ind = j.get("indicadores") or {}
            v = d.get("verificacion") or {}
            vj = v.get("json") or {}
            r = dict(video=d["video"], version=d["version"], prompt=d["prompt"], modelo=d["modelo"],
                     inicio_s=d["inicio_s"], fin_s=d["fin_s"], json_valido=bool(j),
                     escena=j.get("escena", ""), descripcion=j.get("descripcion", ""),
                     justificacion=j.get("justificacion", ""), calidad=j.get("calidad_visual", ""),
                     motivo_calidad=j.get("motivo_calidad", ""),
                     riesgo_inicial=d.get("riesgo_inicial"), riesgo_final=d.get("riesgo_final"),
                     verificada=bool(v), confirmados="|".join(map(str, vj.get("confirmados") or [])),
                     descartados="|".join(map(str, vj.get("descartados") or [])),
                     n_evidencia=len(j.get("evidencia") or []), segundos=d.get("segundos"),
                     ruta_mosaico=f"{f.parent.name}/mosaicos/{d['video']}_{int(round(d['inicio_s'] * 1000)):08d}.jpg")
            for k in PAREJAS:
                r[f"m_{k}"] = ind.get(k, "")
            filas.append(r)
    return pd.DataFrame(filas)


def etiquetar(df, corr):
    c = pd.read_csv(corr, encoding="utf-8-sig")
    out = []
    for _, r in df.iterrows():
        g = c[(c.video == r.video) & (c.segundo >= r.inicio_s) & (c.segundo < r.fin_s)]
        e = {"n_imagenes": len(g)}
        for k in ETIQ:
            e[f"l_{k}"] = int((g[k] > 0).sum()) if len(g) else 0
        e["clase"] = "sin etiqueta" if len(g) == 0 else ("SDZI+" if e["l_SDZI"] > 0 else "SDZI-")
        out.append(e)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def metricas(df):
    filas = []
    for (ver, pr), g in df.groupby(["version", "prompt"]):
        for etapa in ("riesgo_inicial", "riesgo_final"):
            if etapa == "riesgo_inicial" and not g.verificada.any():
                continue
            e = g[g.clase != "sin etiqueta"].dropna(subset=[etapa])
            y, s = e.clase == "SDZI+", e[etapa]
            ctrl = g[(g.video == "video2")].dropna(subset=[etapa])
            fila = dict(version=ver, prompt=pr, etapa=etapa.replace("riesgo_", ""), ventanas=len(g),
                        json_valido=f"{g.json_valido.mean():.0%}", ventanas_etiquetadas=len(e),
                        sdzi_pos=int(y.sum()), sdzi_neg=int((~y).sum()), auroc=round(auroc(y, s), 3),
                        mediana_riesgo_sdzi_pos=s[y].median(), mediana_riesgo_sdzi_neg=s[~y].median())
            for u in (5, 7):
                tp, fp = int(((s >= u) & y).sum()), int(((s >= u) & ~y).sum())
                fila[f"sensibilidad@{u}"] = round(tp / max(1, y.sum()), 3)
                fila[f"falsas_alarmas@{u}"] = round(fp / max(1, (~y).sum()), 3)
                fila[f"precision@{u}"] = round(tp / max(1, tp + fp), 3) if tp + fp else np.nan
                fila[f"alarmas_control_video2@{u}"] = round(float((ctrl[etapa] >= u).mean()), 3) if len(ctrl) else np.nan
            filas.append(fila)
    return pd.DataFrame(filas)


def indicadores(df):
    filas = []
    e = df[(df.clase != "sin etiqueta") & (df.prompt == "mineria")]
    for (ver,), g in e.groupby(["version"]):
        for k, lab in PAREJAS.items():
            m = g[f"m_{k}"].map(si)
            inc = (g[f"m_{k}"].astype(str).str.lower() == "incierto").mean()
            y = g[f"l_{lab}"] > 0
            filas.append(dict(version=ver, indicador=k, etiqueta=lab, ventanas=len(g), con_etiqueta=int(y.sum()),
                              modelo_dice_si=int(m.sum()), incierto=f"{inc:.0%}",
                              sensibilidad=round((m & y).sum() / max(1, y.sum()), 3),
                              especificidad=round((~m & ~y).sum() / max(1, (~y).sum()), 3)))
    return pd.DataFrame(filas)


def hud(df):
    k = ["video", "prompt", "inicio_s"]
    a = df[df.version == "sin_hud"].set_index(k)
    b = df[df.version == "original"].set_index(k)
    j = a.join(b, lsuffix="_sin_hud", rsuffix="_con_hud", how="inner")
    if j.empty:
        return j
    j["dif_riesgo"] = j.riesgo_final_con_hud - j.riesgo_final_sin_hud
    return j.reset_index()[k + ["clase_sin_hud", "riesgo_final_sin_hud", "riesgo_final_con_hud", "dif_riesgo",
                                "descripcion_sin_hud", "descripcion_con_hud"]]


def figuras(df, met, h, sal):
    fd = sal / "figuras"
    fd.mkdir(parents=True, exist_ok=True)
    confs = sorted(df.groupby(["version", "prompt"]).groups)
    clases = ["SDZI+", "SDZI-", "sin etiqueta"]
    col = {"SDZI+": "#b7472a", "SDZI-": "#4a7c59", "sin etiqueta": "#9a9a9a"}
    fig, ax = plt.subplots(1, len(confs), figsize=(4.2 * len(confs), 4), sharey=True, squeeze=False)
    for a, (ver, pr) in zip(ax[0], confs):
        g = df[(df.version == ver) & (df.prompt == pr)]
        datos = [g[g.clase == c].riesgo_final.dropna().values for c in clases]
        pos = [i + 1 for i, d in enumerate(datos) if len(d) > 1]
        if pos:
            p = a.violinplot([datos[i - 1] for i in pos], positions=pos, showmedians=True, showextrema=False)
            for b, i in zip(p["bodies"], pos):
                b.set_facecolor(col[clases[i - 1]]); b.set_alpha(.6)
            p["cmedians"].set_color("k")
        for i, d in enumerate(datos):
            a.scatter(np.full(len(d), i + 1) + np.random.uniform(-.08, .08, len(d)), d, s=6, c="k", alpha=.35)
        a.set_xticks([1, 2, 3]); a.set_xticklabels([f"{c}\n(n={len(d)})" for c, d in zip(clases, datos)], fontsize=8)
        a.set_title(f"{'sin HUD' if ver == 'sin_hud' else 'con HUD'} · {pr}", fontsize=10)
        a.axhline(5, color="gray", ls=":", lw=.8)
        a.spines[["top", "right"]].set_visible(False)
    ax[0][0].set_ylabel("riesgo de mineria (0-10)")
    plt.tight_layout(); plt.savefig(fd / "riesgo_por_clase.png", dpi=150); plt.close()

    for vid, g0 in df.groupby("video"):
        g = g0[g0.prompt == "mineria"]
        if g.empty:
            continue
        fig, a = plt.subplots(figsize=(13, 3.4))
        for ver, c in (("sin_hud", "#1f4e79"), ("original", "#c98a2b")):
            x = g[g.version == ver].sort_values("inicio_s")
            if len(x):
                a.plot((x.inicio_s + x.fin_s) / 2 / 60, x.riesgo_final, "-o", ms=3, lw=1.2, color=c,
                       label="sin HUD" if ver == "sin_hud" else "con HUD")
        x = g[g.version == g.version.iloc[0]]
        for _, r in x[x.clase == "SDZI+"].iterrows():
            a.axvspan(r.inicio_s / 60, r.fin_s / 60, color="#b7472a", alpha=.12, lw=0)
        a.set_ylim(-.3, 10.3); a.set_xlabel("minuto del video"); a.set_ylabel("riesgo")
        a.set_title(f"{vid}: puntaje del modelo (prompt 'mineria'); en rojo las ventanas con imagenes SDZI", fontsize=10)
        a.legend(frameon=False, fontsize=8); a.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); plt.savefig(fd / f"linea_tiempo_{vid}.png", dpi=150); plt.close()

    if len(h):
        fig, a = plt.subplots(figsize=(5, 4))
        prs = sorted(h.prompt.unique())
        d = [h[h.prompt == p].dif_riesgo.dropna().values for p in prs]
        p = a.violinplot(d, showmedians=True, showextrema=False)
        p["cmedians"].set_color("k")
        a.axhline(0, color="gray", ls="--", lw=.8)
        a.set_xticks(range(1, len(prs) + 1)); a.set_xticklabels(prs)
        a.set_ylabel("riesgo con HUD - riesgo sin HUD"); a.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); plt.savefig(fd / "efecto_hud.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", required=True)
    ap.add_argument("--correspondencia", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    sal = Path(a.salida); sal.mkdir(parents=True, exist_ok=True)
    df = cargar(Path(a.resultados))
    if df.empty:
        raise SystemExit("no hay ventanas.jsonl en " + a.resultados)
    df = etiquetar(df, a.correspondencia)
    df.to_csv(sal / "ventanas_eval.csv", index=False)
    met = metricas(df); met.to_csv(sal / "metricas.csv", index=False)
    ind = indicadores(df); ind.to_csv(sal / "indicadores.csv", index=False)
    h = hud(df); h.to_csv(sal / "hud.csv", index=False)
    e = df[df.clase != "sin etiqueta"].copy()
    e["error"] = np.where((e.riesgo_final >= 5) & (e.clase == "SDZI-"), "falso positivo",
                          np.where((e.riesgo_final < 5) & (e.clase == "SDZI+"), "falso negativo", ""))
    e[e.error != ""].sort_values(["error", "video", "inicio_s"])[
        ["error", "video", "version", "prompt", "inicio_s", "fin_s", "riesgo_inicial", "riesgo_final", "escena",
         "descripcion", "justificacion", "descartados", "calidad", "motivo_calidad", "ruta_mosaico"]
    ].to_csv(sal / "errores.csv", index=False)
    figuras(df, met, h, sal)
    with open(sal / "resumen.txt", "w", encoding="utf-8") as f:
        f.write(met.to_string(index=False) + "\n\n" + ind.to_string(index=False) + "\n")
        if len(h):
            f.write("\nHUD: diferencia de riesgo (con - sin), percentiles 25/50/75 por prompt\n")
            f.write(h.groupby("prompt").dif_riesgo.quantile([.1, .25, .5, .75, .9]).unstack().round(2).to_string() + "\n")
            f.write("ventanas con el mismo nivel (bajo/medio/alto) con y sin HUD: " + h.assign(ig=(h.riesgo_final_sin_hud >= 4).astype(int) + (h.riesgo_final_sin_hud >= 7) == (h.riesgo_final_con_hud >= 4).astype(int) + (h.riesgo_final_con_hud >= 7)).groupby("prompt").ig.mean().round(3).to_string() + "\n")
    print(open(sal / "resumen.txt", encoding="utf-8").read())


if __name__ == "__main__":
    main()
