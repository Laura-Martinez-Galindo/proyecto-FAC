"""Dense video captioning por ventanas con un modelo de lenguaje visual (VLM) que corre en Hypatia.

Idea: el video se recorre en ventanas de tiempo. De cada ventana se toman unos frames (con su tiempo)
y se le pide al modelo que describa lo que pasa y diga que tan relevante es para mineria ilegal.
Despues, las ventanas seguidas con el mismo nivel de interes se unen en SEGMENTOS: cada segmento
(inicio, fin, descripcion) es una entrada del dense captioning.

Salidas en --salida:
  ventanas.jsonl   una linea por ventana: tiempos, respuesta cruda del modelo, JSON leido, segundos
  segmentos.csv    segmentos de interes (ventanas seguidas unidas)
  configuracion.json

El modelo se elige con --modelo (id de Hugging Face); sirve cualquiera que cargue con
AutoModelForImageTextToText (Qwen2.5-VL, Qwen3-VL, InternVL3, LLaVA-OneVision, VideoLLaMA3).
--simular no carga modelo ni frames: prueba el flujo completo con respuestas de mentira.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NIVELES = {"bajo": 0, "medio": 1, "alto": 2}


def mmss(t: float) -> str:
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def leer_json(texto: str) -> dict:
    """El modelo a veces envuelve el JSON en texto o en ```json; se toma el primer objeto {...} valido
    (admite objetos anidados). Si el texto se corto, se intenta cerrar las llaves que falten."""
    dec = json.JSONDecoder()
    for i, c in enumerate(texto):
        if c != "{":
            continue
        try:
            d, _ = dec.raw_decode(texto[i:])
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            continue
    i = texto.find("{")
    if i >= 0:  # respuesta cortada por max_tokens
        t = re.sub(r",\s*$", "", texto[i:].rstrip())
        for cierre in ("}", "]}", "\"}", "\"]}", "\"}]}", "}]}", "}}"):
            try:
                return json.loads(t + cierre)
            except json.JSONDecodeError:
                pass
    return {}


def riesgo(d: dict):
    try:
        return max(0.0, min(10.0, float(d.get("riesgo_mineria"))))
    except (TypeError, ValueError):
        return None


def nivel(d: dict) -> str:
    r = riesgo(d)
    if r is not None:  # prompts con puntaje 0-10
        return "alto" if r >= 7 else "medio" if r >= 4 else "bajo"
    v = str(d.get("interes", "")).strip().lower()
    return v if v in NIVELES else "sin_dato"


class Limpiador:
    """Aplica a un frame del video el modelo de denoising ganador de SU regimen (fase 2), igual que
    en dataset_denoised: config/denoising/ganadores.json dice que experimento usar por familia y
    videos/<video>/regimenes_<video>.csv el regimen de cada frame (lectura del HUD mas cercana).
    Los frames de regimenes sin modelo (termico, baja luz, SWIR) se devuelven sin HUD, sin limpiar."""

    def __init__(self, video: str, fuente, tile: int = 384, overlap: int = 48):
        import numpy as np
        import torch
        import denoising as dn
        from color_original import recolorear
        self.np, self.torch, self.dn, self.recolorear = np, torch, dn, recolorear
        self.f, self.tile, self.overlap = fuente, tile, overlap
        self.disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fams = json.loads((RAIZ / "config/denoising/familias.json").read_text(encoding="utf-8"))
        self.fam_de = {r: f for f, rs in fams.items() for r in rs}
        self.gan = json.loads((RAIZ / "config/denoising/ganadores.json").read_text(encoding="utf-8"))
        lect = []
        with (RAIZ / f"videos/{video}/regimenes_{video}.csv").open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("regimen") and r["regimen"] != "DESCONOCIDO":
                    lect.append((int(r["frame"]), r["regimen"]))
        lect.sort()
        self.k_lect = np.array([k for k, _ in lect])
        self.r_lect = [r for _, r in lect]
        self.redes, self.conteo = {}, {}

    def regimen(self, k: int) -> str:
        i = int(self.np.searchsorted(self.k_lect, k))
        cand = [j for j in (i - 1, i) if 0 <= j < len(self.r_lect)]
        return self.r_lect[min(cand, key=lambda j: abs(self.k_lect[j] - k))] if cand else "DESCONOCIDO"

    def _red(self, fam: str):
        if fam not in self.redes:
            from argparse import Namespace
            g = self.gan[fam]
            ruta = RAIZ / "cache/denoising" / fam / g["entrenamiento"] / "modelo.pt"
            ck = self.torch.load(ruta, map_location=self.disp, weights_only=False)
            conf = Namespace(**ck["args"])
            met = self.dn.METODOS[ck["metodo"]]
            red = met.crear(conf, int(ck["canales"])).to(self.disp)
            red.load_state_dict(ck["modelo"], strict=True)
            red.eval()
            self.redes[fam] = (met, conf, int(ck["canales"]), red, g["experimento"].endswith("_ycolor"),
                               bool(ck["args"].get("amp")))
            print(f"denoising {fam}: {ck['metodo']} paso {ck['paso']} ({ruta})", flush=True)
        return self.redes[fam]

    def frame(self, k: int):
        """BGR del frame k sin HUD y limpio con el modelo de su regimen; (imagen, etiqueta)."""
        reg = self.regimen(k)
        fam = self.fam_de.get(reg)
        if fam is None or fam not in self.gan:
            self.conteo[reg] = self.conteo.get(reg, 0) + 1
            return self.f.compuesto(k, 3), f"{reg}:sin_modelo"
        met, conf, canales, red, ycolor, amp = self._red(fam)
        np, torch = self.np, self.torch
        centro = self.f.compuesto(k, canales)
        if centro is None:
            return None, f"{reg}:sin_frame"
        ims = [centro]
        if met.temporal > 1:
            ims = []
            for d in range(-(met.temporal // 2), met.temporal // 2 + 1):
                im = centro if d == 0 else self.f.compuesto(k + d, canales)
                ims.append(centro if im is None or im.shape != centro.shape else im)
        x = torch.from_numpy(np.concatenate(ims, 2).transpose(2, 0, 1).astype(np.float32) / 255.0)[None]
        x = x.to(self.disp) * met.escala

        def fn(t):
            with torch.no_grad(), torch.autocast(self.disp.type, enabled=amp and self.disp.type == "cuda"):
                return met.aplicar(red, t, conf)

        mult = getattr(met, "multiplo_mosaico", max(met.multiplo, 32 if met.nombre not in ("apbsn", "mmbsn") else 2))
        tile = self.tile
        while True:
            try:
                y = self.dn.mosaicos(fn, x, tile, self.overlap, mult)
                break
            except RuntimeError as e:
                if not self.dn.sin_memoria(e) or tile // 2 < max(mult, 64):
                    raise
                torch.cuda.empty_cache()
                tile //= 2
        out = np.rint((y[0] / met.escala).clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        if out.ndim == 2 or out.shape[2] == 1:
            import cv2
            out = cv2.cvtColor(out.reshape(out.shape[0], out.shape[1]), cv2.COLOR_GRAY2BGR)
        if ycolor:
            base = self.f.compuesto(k, 3)
            if base is not None:
                out = self.recolorear(base, out, 5, 1.5)
        del x, y
        if self.disp.type == "cuda":
            torch.cuda.empty_cache()  # la GPU la comparte con el modelo de lenguaje visual
        self.conteo[fam] = self.conteo.get(fam, 0) + 1
        return out, f"{reg}:{self.gan[fam]['metodo']}"


class Frames:
    """Lee frames del video por numero. version: original (con HUD), sin_hud (compuesto) o
    denoised (sin HUD + modelo de denoising de su regimen, ver Limpiador)."""

    def __init__(self, video: str, version: str, lado_max: int):
        sys.path.insert(0, str(RAIZ / "scripts"))
        from fac_denoise.datos import Fuente, cargar_config
        self.f = Fuente(video)
        self.version, self.lado_max = version, lado_max
        self.limpiador = Limpiador(video, self.f) if version == "denoised" else None
        self.etiquetas = []
        cfg = cargar_config().get(video, {})
        self.fps = float(cfg.get("extraccion", {}).get("fps") or cfg.get("metadatos", {}).get("fps_original") or 30.0)
        self.n = max(self.f.idx_orig) if self.f.idx_orig else 0
        print(self.f.resumen(), "| fps", self.fps, flush=True)

    def imagen(self, t: float):
        import cv2
        from PIL import Image
        k = int(round(t * self.fps)) + 1
        if self.limpiador is not None:
            im, etq = self.limpiador.frame(k)
            self.etiquetas.append(etq)
        else:
            im = self.f.compuesto(k, 3) if self.version == "sin_hud" else self.f.original(k, 3)
        if im is None and self.version in ("sin_hud", "denoised"):
            im = self.f.original(k, 3)  # si falta la mascara o el relleno, se usa el original
        if im is None:
            return None
        h, w = im.shape[:2]
        e = self.lado_max / max(h, w)
        if e < 1:
            im = cv2.resize(im, (int(round(w * e)), int(round(h * e))), interpolation=cv2.INTER_AREA)
        return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


class Modelo:
    def __init__(self, nombre: str, cuatro_bits: bool):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(nombre)
        extra = {}
        if cuatro_bits:
            from transformers import BitsAndBytesConfig
            extra["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.m = AutoModelForImageTextToText.from_pretrained(nombre, dtype=torch.float16, device_map="auto", **extra)
        self.m.eval()
        print("modelo cargado:", nombre, "| GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no",
              flush=True)

    def responder(self, imagenes, tiempos, prompt: str, max_tokens: int) -> str:
        contenido = []
        for j, (im, t) in enumerate(zip(imagenes, tiempos), 1):
            contenido.append({"type": "text", "text": f"Frame {j} (t={mmss(t)}):"})
            contenido.append({"type": "image", "image": im})
        contenido.append({"type": "text", "text": prompt})
        ent = self.proc.apply_chat_template([{"role": "user", "content": contenido}], add_generation_prompt=True,
                                            tokenize=True, return_dict=True, return_tensors="pt").to(self.m.device)
        with self.torch.inference_mode():
            sal = self.m.generate(**ent, max_new_tokens=max_tokens, do_sample=False)
        return self.proc.decode(sal[0][ent["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def hoja_contacto(ims, tiempos, lado=320):
    """Hoja de contacto (frames en 2 filas, con su tiempo) para revisar despues que vio el modelo."""
    from PIL import Image, ImageDraw
    peq = [im.copy() for im in ims]
    for im in peq:
        im.thumbnail((lado, lado))
    w, h = max(i.width for i in peq), max(i.height for i in peq)
    cols = (len(peq) + 1) // 2
    hoja = Image.new("RGB", (cols * w, 2 * h), (255, 255, 255))
    for k, (im, t) in enumerate(zip(peq, tiempos)):
        x, y = (k % cols) * w, (k // cols) * h
        hoja.paste(im, (x, y))
        ImageDraw.Draw(hoja).text((x + 4, y + 4), f"{k + 1} {mmss(t)}", fill=(255, 255, 0))
    return hoja


def verificar(modelo, frames, d, usados, plantilla, a):
    """Segunda mirada: los frames citados como evidencia, a mas resolucion, con las afirmaciones del modelo."""
    citados = []
    for e in d.get("evidencia") or []:
        for k in (e.get("frames") or []) if isinstance(e, dict) else []:
            try:
                k = int(k)
            except (TypeError, ValueError):
                continue
            if 1 <= k <= len(usados) and k not in citados:
                citados.append(k)
    citados = citados[:2] or [len(usados) // 2 + 1]
    afirm = [f"- {e.get('indicador')}: {e.get('detalle', '')} (frame {e.get('frames')}, {e.get('ubicacion', '')})"
             for e in d.get("evidencia") or [] if isinstance(e, dict)]
    if not afirm:
        afirm = [f"- {k}: si" for k, v in (d.get("indicadores") or {}).items() if str(v).lower() == "si"]
    prompt = plantilla.replace("{afirmaciones}", "\n".join(afirm) or "- (ninguna)").replace(
        "{riesgo}", str(d.get("riesgo_mineria")))
    lado_ant = frames.lado_max
    frames.lado_max = a.lado_verificacion
    ts = [usados[k - 1] for k in citados]
    ims = [frames.imagen(t) for t in ts]
    frames.lado_max = lado_ant
    ini = time.time()
    try:
        cruda = modelo.responder(ims, ts, prompt, a.max_tokens)
    except modelo.torch.OutOfMemoryError:
        modelo.torch.cuda.empty_cache()
        return {"error": "sin memoria", "frames": citados}
    return {"frames": citados, "respuesta": cruda, "json": leer_json(cruda), "segundos": round(time.time() - ini, 2)}


def unir_segmentos(filas: list[dict], minimo: str) -> list[dict]:
    """Ventanas seguidas con el mismo nivel de interes (>= minimo) y sin cambio de escena -> un segmento."""
    segs, act = [], None
    for f in filas:
        nv = f["interes"]
        if NIVELES.get(nv, -1) < NIVELES[minimo]:
            act = None
            continue
        corta = bool(f["json"].get("cambio_de_escena"))
        if act and act["interes"] == nv and not corta and f["inicio_s"] <= act["fin_s"] + 1e-6:
            act["fin_s"] = f["fin_s"]
            act["descripciones"].append(f["json"].get("descripcion", ""))
            act["riesgos"].append(f.get("riesgo_final"))
        else:
            act = {"inicio_s": f["inicio_s"], "fin_s": f["fin_s"], "interes": nv,
                   "descripciones": [f["json"].get("descripcion", "")], "riesgos": [f.get("riesgo_final")]}
            segs.append(act)
    return segs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--version", choices=["original", "sin_hud", "denoised"], default="sin_hud")
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--prompt", default="lista", help="nombre en config/captioning/prompts.json")
    ap.add_argument("--inicio-s", type=float, default=0.0)
    ap.add_argument("--fin-s", type=float, default=None, help="por defecto, hasta el final del video")
    ap.add_argument("--ventana-s", type=float, default=24.0)
    ap.add_argument("--frames-por-ventana", type=int, default=8)
    ap.add_argument("--lado-max", type=int, default=640, help="lado mayor de cada frame que ve el modelo")
    ap.add_argument("--paso-s", type=float, default=0.0,
                    help="cada cuanto empieza una ventana; 0 = igual a --ventana-s (sin solape)")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--mosaicos", action="store_true",
                    help="guarda en mosaicos/ una hoja de contacto (jpg pequeno) con los frames que vio el modelo")
    ap.add_argument("--verificar", type=float, default=0.0,
                    help="si > 0: cuando riesgo_mineria >= este valor, se re-pregunta con los frames de la evidencia "
                         "a --lado-verificacion px (prompt 'verificacion') y se guarda riesgo_final")
    ap.add_argument("--lado-verificacion", type=int, default=1280)
    ap.add_argument("--max-ventanas", type=int, default=0, help="0 = todas (para la prueba de humo: 3)")
    ap.add_argument("--interes-minimo", choices=["medio", "alto"], default="medio")
    ap.add_argument("--cuatro-bits", action="store_true", help="cargar el modelo en 4 bits si no cabe en la GPU")
    ap.add_argument("--simular", action="store_true")
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()

    prompts = json.loads((RAIZ / "config" / "captioning" / "prompts.json").read_text(encoding="utf-8"))
    prompt = prompts[a.prompt]
    sal = Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    if a.mosaicos:
        (sal / "mosaicos").mkdir(exist_ok=True)

    if a.simular:
        frames, fps, dur = None, 30.0, a.fin_s or 120.0
    else:
        frames = Frames(a.video, a.version, a.lado_max)
        fps, dur = frames.fps, (frames.n - 1) / frames.fps
    fin = min(a.fin_s, dur) if a.fin_s else dur
    ventanas, t = [], a.inicio_s
    while t + 1e-6 < fin:
        ventanas.append((t, min(t + a.ventana_s, fin)))
        t += a.paso_s or a.ventana_s
    if a.max_ventanas:
        ventanas = ventanas[: a.max_ventanas]
    (sal / "configuracion.json").write_text(json.dumps({**vars(a), "fps": fps, "n_ventanas": len(ventanas),
                                                        "texto_prompt": prompt}, indent=2, ensure_ascii=False),
                                            encoding="utf-8")

    # Reanudable: las ventanas que ya estan en ventanas.jsonl no se repiten.
    hechas, ruta = {}, sal / "ventanas.jsonl"
    if ruta.exists():
        for ln in ruta.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(ln)
                hechas[round(d["inicio_s"], 3)] = d
            except (json.JSONDecodeError, KeyError):
                pass
    modelo = None if a.simular else Modelo(a.modelo, a.cuatro_bits)
    lado = a.lado_max
    with ruta.open("a", encoding="utf-8") as fh:
        for i, (t0, t1) in enumerate(ventanas):
            if round(t0, 3) in hechas:
                continue
            n = a.frames_por_ventana
            tiempos = [t0 + (t1 - t0) * (j + 0.5) / n for j in range(n)]
            ini = time.time()
            if a.simular:
                cruda = json.dumps({"descripcion": f"ventana simulada {i}", "rio": i % 2 == 0, "via": False,
                                    "edificacion": i % 3 == 0, "vehiculo_o_maquinaria": False, "suelo_alterado": False,
                                    "interes": ["bajo", "medio", "medio", "alto"][i % 4], "cambio_de_escena": False,
                                    "indicadores": {"suelo_removido": "si" if i % 3 == 0 else "no", "rio": "si"},
                                    "evidencia": [{"indicador": "suelo_removido", "frames": [2], "ubicacion": "centro"}],
                                    "riesgo_mineria": i % 11})
                usados = tiempos
            else:
                ims, usados = [], []
                if frames.limpiador is not None:
                    frames.etiquetas = []
                for tt in tiempos:
                    im = frames.imagen(tt)
                    if im is not None:
                        ims.append(im)
                        usados.append(tt)
                if not ims:
                    print(f"[{i + 1}/{len(ventanas)}] {mmss(t0)}: sin frames, se salta", flush=True)
                    continue
                while True:  # si no cabe en la GPU, se achica el frame y se reintenta
                    try:
                        cruda = modelo.responder(ims, usados, prompt, a.max_tokens)
                        break
                    except modelo.torch.OutOfMemoryError:
                        modelo.torch.cuda.empty_cache()
                        lado = int(lado * 0.75)
                        if lado < 224:
                            raise
                        print(f"  sin memoria: frames a {lado} px de lado", flush=True)
                        frames.lado_max = lado
                        ims = [frames.imagen(tt) for tt in usados]
            d = leer_json(cruda)
            r0, verif = riesgo(d), None
            if a.mosaicos and not a.simular:
                hoja_contacto(ims, usados).save(sal / "mosaicos" / f"{a.video}_{int(round(t0 * 1000)):08d}.jpg",
                                                quality=80)
            if a.verificar and r0 is not None and r0 >= a.verificar and not a.simular:
                verif = verificar(modelo, frames, d, usados, prompts.get("verificacion", ""), a)
            dv = (verif or {}).get("json", {})
            rf = riesgo(dv) if verif else r0
            fila = {"video": a.video, "version": a.version, "modelo": a.modelo, "prompt": a.prompt,
                    "inicio_s": t0, "fin_s": t1, "n_frames": len(usados), "lado_max": lado,
                    "tiempos_frames": [round(x, 3) for x in usados],
                    "denoising": list(frames.etiquetas[:len(usados)]) if (frames is not None and frames.limpiador is not None) else None,
                    "respuesta": cruda, "json": d, "riesgo_inicial": r0, "verificacion": verif, "riesgo_final": rf,
                    "interes": nivel({"riesgo_mineria": rf}) if rf is not None else nivel(d),
                    "segundos": round(time.time() - ini, 2)}
            fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
            fh.flush()
            hechas[round(t0, 3)] = fila
            print(f"[{i + 1}/{len(ventanas)}] {mmss(t0)}-{mmss(t1)} interes={fila['interes']} "
                  f"({fila['segundos']} s): {str(d.get('descripcion', cruda))[:110]}", flush=True)

    filas = [hechas[k] for k in sorted(hechas)]
    segs = unir_segmentos(filas, a.interes_minimo)
    with (sal / "segmentos.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "inicio_s", "fin_s", "inicio", "fin", "interes", "riesgo_max", "n_ventanas", "descripcion"])
        for s in segs:
            w.writerow([a.video, s["inicio_s"], s["fin_s"], mmss(s["inicio_s"]), mmss(s["fin_s"]), s["interes"],
                        max([x for x in s["riesgos"] if x is not None], default=""), len(s["descripciones"]), " | ".join(x for x in s["descripciones"] if x)])
    sin = sum(1 for f in filas if f["interes"] == "sin_dato")
    print(f"ventanas: {len(filas)} | sin JSON valido: {sin} | segmentos de interes: {len(segs)} -> {sal}", flush=True)


if __name__ == "__main__":
    main()
