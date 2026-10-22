# -*- coding: utf-8 -*-
"""
MATCHING EXACTO: Dataset_Balanceado  <->  frames_originales (video1, video2, video3)
==================================================================================
Para cada imagen del Dataset_Balanceado encuentra el frame exacto del video del que
salio, comparando pixeles contra los PNG de frames_originales en Google Drive.

Etapa 1 (busqueda gruesa, todos los frames):
    Decodifica videoN_original.mp4 (videos_renderizados) con OpenCV y guarda una
    miniatura gris 64x36 de CADA frame. Cada imagen del dataset se compara contra
    todos los frames de los 3 videos -> candidatos.
Etapa 2 (verificacion exacta):
    Para los mejores candidatos se lee el PNG real de frames_originales
    (frame_N.png = indice N-1 del video, igual que en extraer_frames.py) y se
    compara a resolucion completa (MAE y PSNR en BGR). Se revisan los vecinos
    para asegurar que el elegido es el minimo exacto.

Salida (en <carpeta videos>/matching_dataset_balanceado/):
    matching_dataset_balanceado.csv   -> una fila por imagen del dataset
    verificacion/*.jpg                -> trípticos [dataset | frame | diferencia x5]
    cache/*.npz                       -> miniaturas de la etapa 1 (para re-ejecutar rapido)

Es reanudable: si Colab se desconecta, vuelva a correr la celda y continua.
"""
import os, sys, re, glob, json, time, math, zipfile, shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import pandas as pd

# ----------------------------------------------------------------------------
# Configuracion (se puede dejar en None: se detecta solo)
# ----------------------------------------------------------------------------
DRIVE = os.environ.get("MATCH_DRIVE", "/content/drive")
RAIZ_VIDEOS = os.environ.get("MATCH_RAIZ")          # carpeta que contiene video1/ video2/ video3/
ZIP_DATASET = os.environ.get("MATCH_ZIP")           # ruta a Dataset_Balanceado.zip
DIR_TRABAJO = os.environ.get("MATCH_WORK", "/content/matching_work")
VIDEOS = ["video1", "video2", "video3"]
TW, TH = 64, 36            # miniatura etapa 1
TOPK_POR_BLOQUE = 40
N_HILOS_PNG = int(os.environ.get("MATCH_HILOS", "12"))
MAX_TRIPTICOS_OK = 60
DIR_CACHE = os.path.join(DIR_TRABAJO, "cache")  # se redefine en main() hacia Drive      # muestras aleatorias de matches exactos para revisar a ojo

cv2.setNumThreads(max(1, os.cpu_count() or 1))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------
# Localizar carpetas
# ----------------------------------------------------------------------------
CARPETAS_PESADAS = {"frames_originales", "frames_sin_hud", "mascaras_hud", "udvd_sin_hud",
                    "comprimidos", "videos_renderizados", "images", "labels"}


def _bases_drive():
    bases = []
    for b in ("MyDrive", "Shareddrives", ".shortcut-targets-by-id", "Othercomputers"):
        p = os.path.join(DRIVE, b)
        if os.path.isdir(p):
            bases.append(p)
    return bases


def buscar(predicado, max_prof=5):
    for base in _bases_drive():
        for dp, dn, fn in os.walk(base):
            prof = dp[len(base):].count(os.sep)
            r = predicado(dp, dn, fn)
            if r:
                return r
            if prof >= max_prof:
                dn[:] = []
            else:
                dn[:] = [d for d in dn if d not in CARPETAS_PESADAS and not d.startswith(".")]
    return None


def encontrar_raiz_videos():
    if RAIZ_VIDEOS:
        return RAIZ_VIDEOS
    log("Buscando la carpeta 'videos' (con video1/ video2/ video3/) en Drive...")
    r = buscar(lambda dp, dn, fn: dp if set(VIDEOS) <= set(dn)
               and os.path.isdir(os.path.join(dp, "video1", "frames_originales")) else None)
    if not r:
        raise SystemExit("No encontre la carpeta con video1/video2/video3. Si esta en 'Compartido conmigo', "
                         "agreguele un acceso directo en Mi unidad, o defina MATCH_RAIZ con la ruta.")
    return r


def encontrar_zip():
    if ZIP_DATASET and os.path.isfile(ZIP_DATASET):
        return ZIP_DATASET
    for p in ("/content/Dataset_Balanceado.zip",):
        if os.path.isfile(p):
            return p
    log("Buscando Dataset_Balanceado.zip en Drive...")
    r = buscar(lambda dp, dn, fn: os.path.join(dp, "Dataset_Balanceado.zip") if "Dataset_Balanceado.zip" in fn else None)
    if r:
        return r
    try:
        from google.colab import files  # type: ignore
        log("No esta en Drive: suba Dataset_Balanceado.zip desde su computador.")
        up = files.upload()
        for k in up:
            if k.lower().endswith(".zip"):
                shutil.move(k, "/content/Dataset_Balanceado.zip")
                return "/content/Dataset_Balanceado.zip"
    except ImportError:
        pass
    raise SystemExit("No encontre Dataset_Balanceado.zip")


# ----------------------------------------------------------------------------
# Utilidades de imagen
# ----------------------------------------------------------------------------
def miniatura(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (TW, TH), interpolation=cv2.INTER_AREA)


def comparar(ref_bgr, cand_bgr):
    if cand_bgr.shape != ref_bgr.shape:
        cand_bgr = cv2.resize(cand_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    d = cv2.absdiff(ref_bgr, cand_bgr)
    mae = float(d.mean())
    mse = float(np.mean(d.astype(np.float32) ** 2))
    psnr = 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
    return mae, psnr


def leer(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:  # reintento (FUSE de Drive a veces falla la primera vez)
        time.sleep(1.0)
        im = cv2.imread(p, cv2.IMREAD_COLOR)
    return im


# ----------------------------------------------------------------------------
# Etapa 0: dataset
# ----------------------------------------------------------------------------
def cargar_dataset(zip_path):
    destino = os.path.join(DIR_TRABAJO, "dataset")
    if not os.path.isdir(destino) or not glob.glob(os.path.join(destino, "**", "*.jpg"), recursive=True):
        log("Descomprimiendo", zip_path)
        os.makedirs(destino, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(destino)
    rutas = sorted(glob.glob(os.path.join(destino, "**", "*.jpg"), recursive=True)
                   + glob.glob(os.path.join(destino, "**", "*.png"), recursive=True))
    filas = []
    for r in rutas:
        partes = r.replace("\\", "/").split("/")
        split = next((s for s in ("train", "val", "test") if s in partes), "")
        filas.append({"archivo_dataset": os.path.basename(r), "split": split, "ruta": r})
    df = pd.DataFrame(filas)
    log(f"Dataset: {len(df)} imagenes", df.split.value_counts().to_dict())
    return df


# ----------------------------------------------------------------------------
# Etapa 1: miniaturas de todos los frames de cada video
# ----------------------------------------------------------------------------
def info_video(raiz, v):
    carpeta_frames = os.path.join(raiz, v, "frames_originales")
    mp4 = os.path.join(raiz, v, "videos_renderizados", f"{v}_original.mp4")
    pad = None
    for n in (4, 5, 6, 7):
        if os.path.isfile(os.path.join(carpeta_frames, f"frame_{1:0{n}d}.png")):
            pad = n
            break
    return {"video": v, "carpeta_frames": carpeta_frames, "mp4": mp4, "pad": pad}


def miniaturas_video(info):
    v = info["video"]
    cache = os.path.join(DIR_CACHE, f"{v}_mini_{TW}x{TH}.npz")
    if os.path.isfile(cache):
        z = np.load(cache)
        log(f"{v}: miniaturas en cache ({z['mini'].shape[0]} frames)")
        return z["mini"], float(z["fps"]), (int(z["w"]), int(z["h"]))
    if not os.path.isfile(info["mp4"]):
        raise SystemExit(f"No existe {info['mp4']}")
    cap = cv2.VideoCapture(info["mp4"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"{v}: decodificando {os.path.basename(info['mp4'])} {w}x{h} @ {fps:.3f} fps, ~{n_est} frames")
    minis = []
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        minis.append(miniatura(fr))
        if len(minis) % 5000 == 0:
            vel = len(minis) / (time.time() - t0)
            log(f"  {v}: {len(minis)}/{n_est} frames ({vel:.0f} fps, faltan ~{(n_est-len(minis))/max(vel,1)/60:.1f} min)")
    cap.release()
    mini = np.stack(minis)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, mini=mini, fps=fps, w=w, h=h)
    log(f"{v}: {mini.shape[0]} frames decodificados en {(time.time()-t0)/60:.1f} min")
    return mini, fps, (w, h)


def candidatos_gruesos(Q, minis_por_video):
    """Q: (M, D) float32. Devuelve por imagen lista de (video, idx, rmse) ordenada."""
    M = Q.shape[0]
    qn = (Q ** 2).sum(1)
    res = [[] for _ in range(M)]
    for v, mini in minis_por_video.items():
        N = mini.shape[0]
        for a in range(0, N, 20000):
            B = mini[a:a + 20000].reshape(-1, TW * TH).astype(np.float32)
            d = qn[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * Q @ B.T
            k = min(TOPK_POR_BLOQUE, B.shape[0])
            idx = np.argpartition(d, k - 1, axis=1)[:, :k]
            for i in range(M):
                for j in idx[i]:
                    res[i].append((v, a + int(j), math.sqrt(max(float(d[i, j]), 0.0) / (TW * TH))))
    salida = []
    for lst in res:
        lst.sort(key=lambda t: t[2])
        # minimos locales separados (>15 frames) para no repetir el mismo tramo
        elegidos = []
        for c in lst:
            if all(not (c[0] == e[0] and abs(c[1] - e[1]) <= 15) for e in elegidos):
                elegidos.append(c)
            if len(elegidos) == 4:
                break
        salida.append(elegidos)
    return salida


# ----------------------------------------------------------------------------
# Etapa 2: verificacion contra los PNG reales
# ----------------------------------------------------------------------------
def ruta_png(info, idx0):
    return os.path.join(info["carpeta_frames"], f"frame_{idx0 + 1:0{info['pad']}d}.png")


def verificar(fila, cands, infos, n_frames):
    ref = cv2.imread(fila["ruta"], cv2.IMREAD_COLOR)
    evaluados = {}  # (v, idx) -> (mae, psnr)

    def evaluar(v, idx):
        if (v, idx) in evaluados or idx < 0 or idx >= n_frames[v]:
            return
        im = leer(ruta_png(infos[v], idx))
        evaluados[(v, idx)] = (float("inf"), 0.0) if im is None else comparar(ref, im)

    # el mejor candidato siempre; los demas si estan cerca en la etapa 1
    r0 = cands[0][2]
    revisar = [c for c in cands if c[2] <= max(1.5 * r0, r0 + 2.0)]
    for v, idx, _ in revisar:
        for k in range(idx - 2, idx + 3):
            evaluar(v, k)
        # extender si el minimo queda en el borde de la ventana
        for _ in range(10):
            locales = {k: evaluados[(v, k)][0] for (vv, k) in evaluados if vv == v and abs(k - idx) <= 40}
            kmin = min(locales, key=locales.get)
            nuevos = [k for k in (kmin - 1, kmin + 1, kmin - 2, kmin + 2) if (v, k) not in evaluados
                      and 0 <= k < n_frames[v]]
            if not nuevos:
                break
            for k in nuevos:
                evaluar(v, k)

    orden = sorted(evaluados.items(), key=lambda kv: kv[1][0])
    (vb, ib), (mae, psnr) = orden[0]
    ant = evaluados.get((vb, ib - 1), (float("nan"),))[0]
    sig = evaluados.get((vb, ib + 1), (float("nan"),))[0]
    lejanos = [(k, m) for k, m in orden[1:] if not (k[0] == vb and abs(k[1] - ib) <= 2)]
    (v2, i2), (mae2, _) = lejanos[0] if lejanos else ((None, None), (float("nan"), 0))
    vecinos = [x for x in (ant, sig) if not math.isnan(x)]
    mae_vec = min(vecinos) if vecinos else float("nan")
    return {
        "video": vb, "indice_frame_0based": ib,
        "archivo_frame": os.path.basename(ruta_png(infos[vb], ib)),
        "mae": round(mae, 3), "psnr_db": round(psnr, 2),
        "mae_frame_anterior": round(ant, 3), "mae_frame_siguiente": round(sig, 3),
        "mae_mejor_vecino": round(mae_vec, 3),
        "mae_otro_candidato": round(mae2, 3) if not math.isnan(mae2) else None,
        "otro_candidato": f"{v2}:{os.path.basename(ruta_png(infos[v2], i2))}" if v2 else None,
        "rmse_etapa1": round(cands[0][2], 3),
        "pngs_leidos": len(evaluados),
    }


def estado(r):
    if r["psnr_db"] >= 34 and (math.isnan(r["mae_mejor_vecino"]) or r["mae_mejor_vecino"] > r["mae"] * 1.15 + 0.05):
        return "EXACTO"
    if r["psnr_db"] >= 34:
        return "EXACTO_CON_VECINO_CASI_IGUAL"
    if r["psnr_db"] >= 27:
        return "PROBABLE_REVISAR"
    return "SIN_MATCH"


def triptico(ref, cand, titulo, ruta):
    if cand.shape != ref.shape:
        cand = cv2.resize(cand, (ref.shape[1], ref.shape[0]))
    d = cv2.applyColorMap(np.clip(cv2.absdiff(ref, cand).mean(2) * 5, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    t = np.hstack([ref, cand, d])
    esc = 1600 / t.shape[1]
    t = cv2.resize(t, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    barra = np.full((40, t.shape[1], 3), 25, np.uint8)
    cv2.putText(barra, titulo, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(ruta, np.vstack([barra, t]), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----------------------------------------------------------------------------
def main():
    raiz = encontrar_raiz_videos()
    log("Carpeta de videos:", raiz)
    salida = os.path.join(raiz, "matching_dataset_balanceado")
    os.makedirs(os.path.join(salida, "verificacion"), exist_ok=True)
    csv_final = os.path.join(salida, "matching_dataset_balanceado.csv")
    global DIR_CACHE
    DIR_CACHE = os.path.join(salida, "cache")
    csv_parcial = os.path.join(salida, "cache", "parcial.csv")
    os.makedirs(DIR_CACHE, exist_ok=True)
    os.makedirs(DIR_TRABAJO, exist_ok=True)

    df = cargar_dataset(encontrar_zip())

    infos = {v: info_video(raiz, v) for v in VIDEOS}
    for v, i in infos.items():
        log(f"{v}: frames_originales pad={i['pad']}  mp4={'OK' if os.path.isfile(i['mp4']) else 'NO EXISTE'}")
        if i["pad"] is None:
            raise SystemExit(f"No encontre frame_0..1.png en {i['carpeta_frames']}")

    minis, fpss, res, n_frames = {}, {}, {}, {}
    for v in VIDEOS:
        minis[v], fpss[v], res[v] = miniaturas_video(infos[v])
        n_frames[v] = minis[v].shape[0]
        ultimo = ruta_png(infos[v], n_frames[v] - 1)
        log(f"{v}: {n_frames[v]} frames; ultimo PNG esperado {os.path.basename(ultimo)} existe={os.path.isfile(ultimo)}")

    log("Etapa 1: candidatos para", len(df), "imagenes")
    Q = np.stack([miniatura(cv2.imread(r)).reshape(-1) for r in df.ruta]).astype(np.float32)
    cands = candidatos_gruesos(Q, minis)

    hechos = {}
    if os.path.isfile(csv_parcial):
        for r in pd.read_csv(csv_parcial).to_dict("records"):
            hechos[r["archivo_dataset"] + "|" + r["split"]] = r
        log(f"Reanudando: {len(hechos)} ya verificadas")

    filas = df.to_dict("records")
    pendientes = [(i, f) for i, f in enumerate(filas) if f["archivo_dataset"] + "|" + f["split"] not in hechos]
    log(f"Etapa 2: verificando {len(pendientes)} imagenes contra los PNG originales ({N_HILOS_PNG} hilos)")

    def tarea(par):
        i, f = par
        r = verificar(f, cands[i], infos, n_frames)
        r.update({"archivo_dataset": f["archivo_dataset"], "split": f["split"]})
        return r

    nuevos = []
    t0 = time.time()
    with ThreadPoolExecutor(N_HILOS_PNG) as ex:
        for n, r in enumerate(ex.map(tarea, pendientes), 1):
            nuevos.append(r)
            if n % 50 == 0 or n == len(pendientes):
                pd.DataFrame(list(hechos.values()) + nuevos).to_csv(csv_parcial, index=False)
                vel = n / (time.time() - t0)
                log(f"  {n}/{len(pendientes)} verificadas ({vel:.2f} img/s, faltan ~{(len(pendientes)-n)/max(vel,1e-6)/60:.1f} min)")

    out = pd.DataFrame(list(hechos.values()) + nuevos)
    out = df[["archivo_dataset", "split", "ruta"]].merge(out, on=["archivo_dataset", "split"], how="left")
    dims = {r: cv2.imread(r).shape[:2] for r in out.ruta}
    out["resolucion_dataset"] = [f"{dims[r][1]}x{dims[r][0]}" for r in out.ruta]
    out["resolucion_video"] = [f"{res[v][0]}x{res[v][1]}" for v in out.video]
    out["fps_video"] = [round(fpss[v], 4) for v in out.video]
    out["numero_frame_archivo"] = out.indice_frame_0based + 1
    out["tiempo_s"] = (out.indice_frame_0based / out.fps_video).round(3)
    out["tiempo_hh_mm_ss_ff"] = [
        f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}.{int(i % round(fp)):02d}"
        for t, i, fp in zip(out.tiempo_s, out.indice_frame_0based, out.fps_video)]
    out["estado"] = [estado(r) for r in out.to_dict("records")]
    cols = ["archivo_dataset", "split", "resolucion_dataset", "video", "archivo_frame", "numero_frame_archivo",
            "indice_frame_0based", "fps_video", "tiempo_s", "tiempo_hh_mm_ss_ff", "resolucion_video",
            "mae", "psnr_db", "mae_frame_anterior", "mae_frame_siguiente", "mae_mejor_vecino",
            "mae_otro_candidato", "otro_candidato", "rmse_etapa1", "pngs_leidos", "estado"]
    out[cols].to_csv(csv_final, index=False, encoding="utf-8-sig")
    log("CSV final:", csv_final)
    log("Estados:", out.estado.value_counts().to_dict())
    log("Por video:", out.video.value_counts().to_dict())
    log("PSNR (percentiles 0/1/5/50):", np.percentile(out.psnr_db, [0, 1, 5, 50]).round(2).tolist())

    # tripticos: todos los que no son EXACTO + una muestra de los exactos
    rev = out[out.estado != "EXACTO"]
    ok = out[out.estado == "EXACTO"].sample(min(MAX_TRIPTICOS_OK, (out.estado == "EXACTO").sum()), random_state=0)
    for r in pd.concat([rev, ok]).to_dict("records"):
        cand = leer(ruta_png(infos[r["video"]], r["indice_frame_0based"]))
        if cand is None:
            continue
        tit = f"{r['estado']} | {r['archivo_dataset']} ({r['split']}) -> {r['video']}/{r['archivo_frame']} | MAE {r['mae']} PSNR {r['psnr_db']}"
        triptico(cv2.imread(r["ruta"]), cand, tit,
                 os.path.join(salida, "verificacion", f"{r['estado']}__{os.path.splitext(r['archivo_dataset'])[0]}__{r['split']}.jpg"))
    log("Listo. Tripticos en", os.path.join(salida, "verificacion"))
    return out


if __name__ == "__main__":
    main()
