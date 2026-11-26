"""
RONDA 3 FIEL: versiones fieles a su articulo / repositorio oficial de los metodos que la auditoria de
fidelidad (proyecto "Tesis": fidelidad_metodos_vs_articulos.md) marco como adaptados pero corregibles.
Son metodos NUEVOS (sufijo _fiel): los experimentos anteriores siguen reproducibles con su nombre.

Se mantiene igual para todos (a proposito): presupuesto de 4.000 pasos, los mismos frames por regimen,
parches/recortes fuera del HUD y la inferencia sobre la imagen sin HUD completa.

  n2v_fiel, structn2v_fiel   juglab/n2v, ejemplo oficial de imagenes RGB (examples/2D/denoising2D_RGB):
        U-Net de CSBDeep (profundidad 3, 64 filtros, kernel 3, BatchNorm, salida lineal), datos normalizados
        con la media y desviacion de cada canal, 0,198 % de pixeles ocultos POR CANAL con muestreo
        estratificado, reemplazo "uniform_withCP" en una ventana de radio 5, MSE solo en los ocultos,
        parche 64, batch 128, lr 4e-4. StructN2V: ademas tapa con valores U(-2, 2) una linea de 8 vecinos
        (mascara [0,1,1,1,1,1,1,1,1,1,0] del ejemplo oficial) a lo largo de --struct-eje.
  n2v2_fiel   igual que n2v_fiel con los tres cambios de N2V2 del mismo repositorio: MaxBlurPool,
        sin el salto de mayor resolucion (skip_skipone) y reemplazo por la MEDIANA de la ventana de
        radio 5 sin el centro.
  noise2same_fiel   divelab/Noise2Same (models.py, network.py, network_configure.py, ejemplo RGB):
        U-Net residual (profundidad 3, 96 filtros, bloques de codificacion [1,1,0] y decodificacion [1,1],
        bajada con conv 2x2/2, subida con conv. transpuesta 2x2/2, union por concatenacion), BN+ReLU+conv
        1x1 a la salida; mascara GAUSSIANA N(0; 0,2) en el 0,5 % de los pixeles (estratificado, por canal),
        cada imagen normalizada con su media y desviacion por canal,
        L = MSE(f(x), x) + 2 * sqrt(MSE_J(f(x), f(x_J))), parche 64, batch 64, lr 4e-4.
  f2f_fiel   tehret/blind-denoising (Ehret et al., CVPR 2019): DnCNN-S-25 PREENTRENADO (gris, 17 capas,
        pesos de SaoYan/DnCNN-PyTorch, el mismo submodulo del repo oficial) aplicado a cada canal por
        separado; el frame t se ajusta para parecerse al frame t-1 alineado con flujo optico TV-L1
        (scikit-image) de t a t-1, con la mascara de oclusion del codigo oficial (divergencia del flujo
        > 0,75, dilatada, bordes fuera, pixeles mal interpolados fuera), perdida L1 sumada, Adam lr 5e-5,
        weight decay 1e-5, BN en modo entrenamiento y 20 iteraciones por par de frames (esquema del
        repo). Frames COMPLETOS sin HUD; para que quepa en la GPU se usa un recorte de 540x960 (el tamano
        de los frames del articulo). La zona del HUD (dilatada) queda FUERA de la perdida, igual que
        las zonas ocluidas: alli el frame sin HUD es relleno de ProPainter y no ruido real del sensor.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fac_denoise.metodos import METODOS, Metodo


# ===========================================================================
# Utilidades comunes: muestreo estratificado de N2V / Noise2Same
def _mascara_estratificada(b, c, h, w, perc, dev):
    """get_stratified_coords2D de juglab/n2v (y Noise2Same), independiente por imagen y por canal:
    una caja de lado round(sqrt(100/perc)) por cada celda de la rejilla y un pixel al azar en cada
    caja (los que caen fuera del parche se descartan). Devuelve bool B x C x H x W."""
    caja = int(np.round(np.sqrt(100.0 / perc)))
    ny, nx = math.ceil(h / caja), math.ceil(w / caja)
    oy = (torch.rand(b, c, ny, nx, device=dev) * caja).long()
    ox = (torch.rand(b, c, ny, nx, device=dev) * caja).long()
    ys = (torch.arange(ny, device=dev) * caja).view(1, 1, ny, 1) + oy
    xs = (torch.arange(nx, device=dev) * caja).view(1, 1, 1, nx) + ox
    ok = (ys < h) & (xs < w)
    m = torch.zeros(b, c, h, w, dtype=torch.bool, device=dev)
    bi = torch.arange(b, device=dev).view(b, 1, 1, 1).expand_as(ys)
    ci = torch.arange(c, device=dev).view(1, c, 1, 1).expand_as(ys)
    m[bi[ok], ci[ok], ys[ok], xs[ok]] = True
    return m


class _Normalizador(nn.Module):
    """Media y desviacion por canal de los datos de entrenamiento (N2V las calcula sobre todo el
    conjunto antes de entrenar): aqui se fijan con el primer lote y quedan guardadas en el modelo."""

    def __init__(self, c):
        super().__init__()
        self.register_buffer("media", torch.zeros(1, c, 1, 1))
        self.register_buffer("desv", torch.ones(1, c, 1, 1))
        self.register_buffer("listo", torch.zeros(()))

    def fijar(self, x):
        if self.listo.item() == 0:
            self.media.copy_(x.mean(dim=(0, 2, 3), keepdim=True).float())
            self.desv.copy_(x.std(dim=(0, 2, 3), keepdim=True).float().clamp_min(1e-6))
            self.listo.fill_(1)

    def ida(self, x):
        return (x - self.media) / self.desv

    def vuelta(self, y):
        return y * self.desv + self.media


# ===========================================================================
# U-Net de CSBDeep / N2V (n2v/nets/unet_blocks.py + csbdeep conv_block2), en PyTorch
class _MaxBlurPool(nn.Module):
    """n2v/nets/blurpool.py: max-pool 2x2 con paso 1 ('SAME') y luego filtro [1 2 1]^2/16 con paso 2."""

    def __init__(self, c):
        super().__init__()
        k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
        self.register_buffer("k", k.view(1, 1, 3, 3).repeat(c, 1, 1, 1))
        self.c = c

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode="replicate"), 2, stride=1)  # SAME (relleno abajo/derecha)
        return F.conv2d(F.pad(x, (0, 1, 0, 1)), self.k, stride=2, groups=self.c)  # SAME con paso 2 (lado par)


class _ConvBN(nn.Sequential):
    """conv_block2 de CSBDeep con batch_norm=True: Conv2D (con sesgo) -> BatchNorm -> ReLU."""

    def __init__(self, ci, co, k=3, bn=True):
        conv = nn.Conv2d(ci, co, k, padding=k // 2)
        nn.init.xavier_uniform_(conv.weight)  # glorot_uniform de Keras
        nn.init.zeros_(conv.bias)
        capas = [conv] + ([nn.BatchNorm2d(co, momentum=0.01, eps=1e-3)] if bn else []) + [nn.ReLU(inplace=True)]
        super().__init__(*capas)


class _UNetN2V(nn.Module):
    def __init__(self, c, profundidad=3, base=64, k=3, bn=True, blurpool=False, skip_skipone=False):
        super().__init__()
        self.prof, self.skipone = profundidad, skip_skipone
        self.norm = _Normalizador(c)
        self.bajada, self.pools = nn.ModuleList(), nn.ModuleList()
        ch, saltos = c, []
        for n in range(profundidad):
            f = base * 2 ** n
            self.bajada.append(nn.Sequential(_ConvBN(ch, f, k, bn), _ConvBN(f, f, k, bn)))
            ch = f
            if not skip_skipone or n > 0:
                saltos.append(f)
            self.pools.append(_MaxBlurPool(f) if blurpool else nn.MaxPool2d(2))
        self.medio = nn.Sequential(_ConvBN(ch, base * 2 ** profundidad, k, bn),
                                   _ConvBN(base * 2 ** profundidad, base * 2 ** max(0, profundidad - 1), k, bn))
        ch = base * 2 ** max(0, profundidad - 1)
        self.subida = nn.ModuleList()
        for n in reversed(range(profundidad)):
            if skip_skipone:
                ci = ch + (saltos[n - 1] if n > 0 else 0)
            else:
                ci = ch + saltos[n]
            f1 = base * 2 ** n if (skip_skipone and n > 0) else base  # asi esta en unet_blocks.py de n2v
            f2 = base * 2 ** max(0, n - 1)
            self.subida.append(nn.Sequential(_ConvBN(ci, f1, k, bn), _ConvBN(f1, f2, k, bn)))
            ch = f2
        self.fin = nn.Conv2d(ch, c, 1)
        nn.init.xavier_uniform_(self.fin.weight)
        nn.init.zeros_(self.fin.bias)

    def red(self, x):
        saltos = []
        for n in range(self.prof):
            x = self.bajada[n](x)
            if not self.skipone or n > 0:
                saltos.append(x)
            x = self.pools[n](x)
        x = self.medio(x)
        for i, n in enumerate(reversed(range(self.prof))):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if self.skipone:
                if n > 0:
                    x = torch.cat([x, saltos[n - 1]], 1)
            else:
                x = torch.cat([x, saltos[n]], 1)
            x = self.subida[i](x)
        return self.fin(x)

    def forward(self, x):  # x en [0,1]; sale en [0,1]
        return self.norm.vuelta(self.red(self.norm.ida(x)))


# ===========================================================================
class N2VFiel(Metodo):
    nombre, multiplo, parche_defecto, batch_defecto, lr_defecto = "n2v_fiel", 8, 64, 128, 4e-4
    multiplo_mosaico = 32
    perc, radio, mediana, blurpool, skipone, struct = 0.198, 5, False, False, False, False

    @staticmethod
    def argumentos(p):
        p.add_argument("--n2v-fiel-perc", type=float, default=0.0,
                       help="%% de pixeles ocultos por canal (0 = el del metodo: 0,198, ejemplo RGB oficial)")

    def crear(self, a, c):
        return _UNetN2V(c, 3, 64, 3, True, self.blurpool, self.skipone)

    def _reemplazo(self, xn, m):
        """Valores que reemplazan a los pixeles ocultos (xn normalizado)."""
        b, c, h, w = xn.shape
        dev = xn.device
        bi, ci, ys, xs = m.nonzero(as_tuple=True)
        r = self.radio
        if not self.mediana:  # pm_uniform_withCP: un pixel al azar de la ventana (puede ser el centro)
            dy = torch.randint(-r, r + 1, ys.shape, device=dev)
            dx = torch.randint(-r, r + 1, xs.shape, device=dev)
            return xn[bi, ci, (ys + dy).clamp(0, h - 1), (xs + dx).clamp(0, w - 1)], (bi, ci, ys, xs)
        d = torch.arange(-r, r + 1, device=dev)
        dy, dx = torch.meshgrid(d, d, indexing="ij")
        sin_centro = ~((dy == 0) & (dx == 0))
        dy, dx = dy[sin_centro], dx[sin_centro]  # pm_median: mediana de la ventana SIN el centro
        yy, xx = ys[:, None] + dy[None], xs[:, None] + dx[None]
        dentro = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)  # la ventana se recorta en el borde
        v = xn[bi[:, None], ci[:, None], yy.clamp(0, h - 1), xx.clamp(0, w - 1)]
        v = torch.where(dentro, v, torch.full_like(v, float("nan")))
        return v.nanmedian(dim=1).values, (bi, ci, ys, xs)

    def perdida(self, red, x, progreso, a):
        red.norm.fijar(x)
        xn = red.norm.ida(x)
        b, c, h, w = xn.shape
        perc = getattr(a, "n2v_fiel_perc", 0.0) or self.perc
        m = _mascara_estratificada(b, c, h, w, perc, x.device)
        val, (bi, ci, ys, xs) = self._reemplazo(xn, m)
        ent = xn.clone()
        if self.struct:  # apply_structN2Vmask: linea de 8 vecinos con valores U(-2, 2)
            mitad = 4
            for d in range(-mitad, mitad + 1):
                if d == 0:
                    continue
                if getattr(a, "struct_eje", "vertical") == "vertical":
                    yy, xx = (ys + d).clamp(0, h - 1), xs
                else:
                    yy, xx = ys, (xs + d).clamp(0, w - 1)
                ent[bi, ci, yy, xx] = torch.rand(ys.shape, device=x.device, dtype=ent.dtype) * 4 - 2
        ent[bi, ci, ys, xs] = val.to(ent.dtype)
        out = red.red(ent).float()
        return ((out[bi, ci, ys, xs] - xn[bi, ci, ys, xs].float()) ** 2).sum() / max(1, ys.numel())

    def aplicar(self, red, x, a):
        return red(x)


class StructN2VFiel(N2VFiel):
    nombre, struct = "structn2v_fiel", True


class N2V2Fiel(N2VFiel):
    nombre, mediana, blurpool, skipone = "n2v2_fiel", True, True, True


# ===========================================================================
# Noise2Same: U-Net residual de divelab/Noise2Same (network.py + resnet_module.py)
def _conv(ci, co, k, s=1, sesgo=False):
    c = nn.Conv2d(ci, co, k, s, padding=(k // 2 if s == 1 else 0), bias=sesgo)
    nn.init.xavier_uniform_(c.weight)  # glorot_uniform (conf_basic_ops)
    if sesgo:
        nn.init.zeros_(c.bias)
    return c


def _bn(c):
    return nn.BatchNorm2d(c, momentum=0.003, eps=1e-5)  # momentum TF 0,997


class _ResBloque(nn.Module):
    """res_block con pre-activacion: (BN, ReLU) -> conv3 -> BN, ReLU -> conv3, + atajo."""

    def __init__(self, ci, co):
        super().__init__()
        self.bn1, self.bn2 = _bn(ci), _bn(co)
        self.c1, self.c2 = _conv(ci, co, 3), _conv(co, co, 3)
        self.proy = None if ci == co else _conv(ci, co, 1)

    def forward(self, x):
        if self.proy is None:
            atajo = x
            x = F.relu(self.bn1(x))
        else:
            x = F.relu(self.bn1(x))
            atajo = self.proy(x)
        x = self.c1(x)
        x = self.c2(F.relu(self.bn2(x)))
        return atajo + x


class _UNetN2Same(nn.Module):
    """conf_unet: depth 3, first_output_filters 96, encoding [1,1,0], decoding [1,1], downsampling
    'convolution' (conv 2x2 paso 2 con sesgo), subida con conv. transpuesta 2x2 paso 2 con sesgo,
    skip 'concat'; al final (models.py) BN + ReLU + conv 1x1 sin sesgo."""

    def __init__(self, c, f=96):
        super().__init__()
        self.primera = _conv(c, f, 3)
        self.enc1 = _ResBloque(f, f)
        self.baja2, self.enc2 = _conv(f, 2 * f, 2, 2, True), _ResBloque(2 * f, 2 * f)
        self.baja3 = _conv(2 * f, 4 * f, 2, 2, True)
        self.fondo = _ResBloque(4 * f, 4 * f)
        self.sube2 = nn.ConvTranspose2d(4 * f, 2 * f, 2, 2)
        self.dec2 = _ResBloque(4 * f, 2 * f)
        self.sube1 = nn.ConvTranspose2d(2 * f, f, 2, 2)
        self.dec1 = _ResBloque(2 * f, f)
        for t in (self.sube2, self.sube1):
            nn.init.xavier_uniform_(t.weight)
            nn.init.zeros_(t.bias)
        self.bn_sal = _bn(f)
        self.sal = _conv(f, c, 1)

    def forward(self, x):
        s1 = self.enc1(self.primera(x))
        s2 = self.enc2(self.baja2(s1))
        y = self.fondo(self.baja3(s2))
        y = self.dec2(torch.cat([self.sube2(y), s2], 1))
        y = self.dec1(torch.cat([self.sube1(y), s1], 1))
        return self.sal(F.relu(self.bn_sal(y)))


def _norm_por_imagen(x):
    m = x.mean(dim=(2, 3), keepdim=True)
    s = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (x - m) / s, m, s


class Noise2SameFiel(Metodo):
    nombre, multiplo, parche_defecto, batch_defecto, lr_defecto = "noise2same_fiel", 4, 64, 64, 4e-4
    multiplo_mosaico = 32

    def crear(self, a, c):
        return _UNetN2Same(c)

    def perdida(self, red, x, progreso, a):
        xn, _, _ = _norm_por_imagen(x)
        b, c, h, w = xn.shape
        m = _mascara_estratificada(b, c, h, w, 0.5, x.device).to(xn.dtype)
        ruido = torch.randn_like(xn) * 0.2
        enmascarada = (1 - m) * xn + m * ruido
        pred = red(enmascarada).float()
        cruda = red(xn).float()
        m = m.float()
        rec = F.mse_loss(cruda, xn.float())
        inv = ((cruda - pred) ** 2 * m).sum() / m.sum().clamp_min(1)
        return rec + 2.0 * torch.sqrt(inv + 1e-12)

    def aplicar(self, red, x, a):
        xn, mu, s = _norm_por_imagen(x.float())
        return red(xn).float() * s + mu


# ===========================================================================
# Frame2Frame fiel
class _DnCNN(nn.Module):
    """SaoYan/DnCNN-PyTorch models.py (17 capas, 64 filtros, BN), el mismo del repo de Frame2Frame."""

    def __init__(self, canales=1, capas=17):
        super().__init__()
        L = [nn.Conv2d(canales, 64, 3, padding=1, bias=False), nn.ReLU(inplace=True)]
        for _ in range(capas - 2):
            L += [nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        L.append(nn.Conv2d(64, canales, 3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*L)

    def forward(self, x):
        return self.dncnn(x)


def _gris(rgb):
    """skimage.color.rgb2gray (as_gray=True en el repo oficial)."""
    return rgb[..., 0] * 0.2125 + rgb[..., 1] * 0.7154 + rgb[..., 2] * 0.0721


def _oclusion_ehret(flujo):
    """WarpedLoss.occlusion_mask de tehret/blind-denoising, tal cual (incluida su formula de 'divergencia').
    flujo: 2 x H x W (u, v). Devuelve float H x W con 1 = pixel bueno."""
    from scipy.ndimage import binary_dilation
    u, v = flujo
    a = np.zeros(u.shape, np.float32)
    b = np.zeros(u.shape, np.float32)
    a[:-1, :] = u[1:, :] - u[:-1, :]
    b[:, :-1] = v[:, 1:] - v[:, :-1]
    m = np.abs(a + b) > 0.75
    bola = np.zeros((3, 3), bool)
    bola[1, 0] = bola[0, 1] = bola[1, 1] = bola[2, 1] = bola[1, 2] = True
    m = binary_dilation(m, bola)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return (~m).astype(np.float32)


class _ParesF2F(torch.utils.data.Dataset):
    """Cada item: un par (t, t-1) de frames completos SIN HUD (como el dataset), el flujo TV-L1 de t a
    t-1 sobre el gris de los frames completos, el frame t-1 alineado a t, la mascara buena (oclusion de
    Ehret x pixeles bien interpolados x fuera del HUD) y un recorte de alto x ancho."""

    def __init__(self, pares, fuentes, alto, ancho, semilla):
        self.pares, self.fuentes, self.alto, self.ancho, self.semilla = pares, fuentes, alto, ancho, semilla

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, i):
        from skimage.registration import optical_flow_tvl1
        video, k = self.pares[i]
        f = self.fuentes[video]
        t = f.compuesto(k, 3)
        p = f.compuesto(k - 1, 3)
        if t is None or p is None:  # sin frame sin HUD: original (el HUD igual queda fuera de la perdida)
            t, p = f.original(k, 3), f.original(k - 1, 3)
        H, W = t.shape[:2]
        tf, pf = t.astype(np.float32) / 255.0, p.astype(np.float32) / 255.0
        fl = optical_flow_tvl1(_gris(tf), _gris(pf))  # (v, u): p(x+u, y+v) ~ t(x, y)
        u, v = fl[1].astype(np.float32), fl[0].astype(np.float32)
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        mx, my = gx + u, gy + v
        alin = cv2.remap(pf, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bien = ((mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)).astype(np.float32)  # grid_sample de unos
        bien *= _oclusion_ehret(np.stack([u, v]))
        hud = f.mascara(k, (H, W), margen=f.margen_hud)
        hud1 = f.mascara(k - 1, (H, W), margen=f.margen_hud)
        for mm in (hud, hud1):
            if mm is not None:
                bien *= (~mm).astype(np.float32)
        rng = random.Random(self.semilla * 100003 + i)
        a, b = min(self.alto, H), min(self.ancho, W)
        y0, x0 = rng.randint(0, H - a), rng.randint(0, W - b)
        sl = (slice(y0, y0 + a), slice(x0, x0 + b))
        aT = lambda z: torch.from_numpy(np.ascontiguousarray(z[sl].transpose(2, 0, 1)))
        return aT(tf), aT(alin), torch.from_numpy(np.ascontiguousarray(bien[sl]))[None]


class F2FFiel(Metodo):
    nombre, multiplo, lr_defecto = "f2f_fiel", 1, 5e-5
    multiplo_mosaico = 8

    @staticmethod
    def argumentos(p):
        p.add_argument("--f2f-pesos", default="pesos/dncnn/DnCNN-S-25.pth",
                       help="DnCNN-S-25 de SaoYan/DnCNN-PyTorch (logs/DnCNN-S-25/net.pth)")
        p.add_argument("--f2f-iter", type=int, default=20, help="iteraciones por par de frames (repo oficial: 20)")
        p.add_argument("--f2f-recorte", type=int, nargs=2, default=[540, 960], help="alto ancho del recorte")

    def crear(self, a, c):
        return _DnCNN(1)

    def aplicar(self, red, x, a):  # red gris aplicada a cada canal: y = x - DnCNN(x)
        b, c, h, w = x.shape
        xc = x.reshape(b * c, 1, h, w)
        return (xc - red(xc)).reshape(b, c, h, w)

    def entrenar_propio(self, a, fuentes, muestras, disp):
        """Bucle de entrenamiento del repo oficial (blind_denoising.py), en modo fuera de linea sobre
        los frames de entrenamiento del regimen: a.pasos / f2f_iter pares (t, t-1) al azar."""
        sal = Path(a.salida)
        red = self.crear(a, 1).to(disp)
        sd = torch.load(a.f2f_pesos, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        faltan = red.load_state_dict(sd, strict=False)
        malos = [k for k in faltan.missing_keys if not k.endswith("num_batches_tracked")] + list(faltan.unexpected_keys)
        if malos:
            raise SystemExit(f"Los pesos {a.f2f_pesos} no corresponden a la DnCNN: {malos[:5]}")
        print(f"f2f_fiel: DnCNN-S-25 preentrenada de {a.f2f_pesos}")
        pares = [(v, k) for v, k in muestras if (k - 1) in fuentes[v].idx_orig]
        n_pares = max(1, a.pasos // a.f2f_iter)
        rng = random.Random(a.semilla)
        pares = [rng.choice(pares) for _ in range(n_pares)]
        ds = _ParesF2F(pares, fuentes, a.f2f_recorte[0], a.f2f_recorte[1], a.semilla)
        dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers,
                                         prefetch_factor=4 if a.workers > 0 else None)
        opt = torch.optim.Adam(red.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        log = open(sal / "entrenamiento.csv", "w")
        log.write("paso,loss,lr,segundos\n")
        print(f"f2f_fiel: {n_pares} pares x {a.f2f_iter} iteraciones | recorte {a.f2f_recorte} | lr {a.lr} | {disp}")
        t0, paso, media = time.time(), 0, None
        red.train()

        def guardar(ruta):
            torch.save({"metodo": self.nombre, "modelo": red.state_dict(), "canales": 3, "args": vars(a),
                        "paso": paso}, ruta)

        for j, (cur, alin, bien) in enumerate(dl, 1):
            cur, alin, bien = cur.to(disp), alin.to(disp), bien.to(disp)
            for it in range(a.f2f_iter):
                ch = it % 3  # la red es gris: cada iteracion ajusta con un canal (batch 1, como el repo)
                x = cur[ch:ch + 1][None]
                out = x - red(x)
                loss = torch.abs(bien * out - bien * alin[ch:ch + 1][None]).sum()  # L1Loss(size_average=False)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                paso += 1
                v = loss.item() / max(1.0, bien.sum().item())
                media = v if media is None else 0.98 * media + 0.02 * v
                if paso % a.log_cada == 0 or paso == n_pares * a.f2f_iter:
                    log.write(f"{paso},{media:.6g},{a.lr:.2e},{time.time() - t0:.0f}\n")
                    log.flush()
                    print(f"paso {paso}/{n_pares * a.f2f_iter}  L1 por pixel {media:.6g}  ({time.time() - t0:.0f} s)",
                          flush=True)
                if paso in a.instantaneas:
                    guardar(sal / f"modelo_p{paso}.pt")
            if j % 10 == 0:
                guardar(sal / "modelo.pt")
        guardar(sal / "modelo.pt")
        log.close()
        meta = {"pares": n_pares, "pasos": paso, "segundos": round(time.time() - t0),
                "frames_entrenamiento": len(muestras), **vars(a)}
        (sal / "configuracion.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"Checkpoint: {sal / 'modelo.pt'}  ({time.time() - t0:.0f} s)")


FIELES = [N2VFiel(), StructN2VFiel(), N2V2Fiel(), Noise2SameFiel(), F2FFiel()]
METODOS.update({m.nombre: m for m in FIELES})
