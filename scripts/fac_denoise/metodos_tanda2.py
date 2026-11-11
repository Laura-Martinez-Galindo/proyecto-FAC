"""
RONDA 3, tanda 2: mas metodos autosupervisados espaciales.

Entrenados con el conjunto de frames (igual que la tanda 1):
  cvfsid       Neshatavar et al., CVPR 2022. Red y perdida del repo oficial (Reyhanehne/CVF-SID_PyTorch,
               src/model/model.py, loss.py y trainer.py), reescritas aqui sin el BaseModel del repo.
  noise2score  Kim & Ye, NeurIPS 2021 (cubeyoung/Noise2Score, modelo Gaussian_blind): AR-DAE con sigma
               recocido de 0,05 a 0,001; al inferir, Tweedie y = x + s^2 * score(x) con la s que
               minimiza la variacion total (busqueda "blind" del repo; rango adaptado a nuestro ruido).
  n2v2         Hoeck et al., ECCVW 2022: N2V sin la conexion de salto superior, con BlurPool y
               reemplazo de los pixeles ocultos por la MEDIANA de su vecindario.
  self2self    Quan et al., CVPR 2020. Red con convoluciones parciales y dropout de la implementacion
               PyTorch de referencia (JinYize/self2self_pytorch, BSD-3; terceros/self2self). El
               articulo entrena una red por imagen (500.000 iteraciones): aqui se entrena con el
               conjunto de frames (condiciones iguales); inferencia = promedio de T mascaras.
  cut2self     Variante de Self2Self con huecos cuadrados ("cutout", jubyrea/Cut2Self): misma red.
  gr2r         Monroy et al., CVPR 2025 (Generalized R2R, bemc22/GeneralizedR2R, pytorch_demo.py),
               corruptor gaussiano alpha 0,2; sigma estimada por parche (Immerkaer).
  p2n          Li et al., CVPR 2025 (Positive2Negative). SIN codigo publico: reimplementado del
               articulo (ec. 3-11): y_p = x^ + s_n n^, y_n = x^ - s_p n^, s ~ N(1, 0,75), perdida
               ||F(y_p) - F(y_n)||^gamma con gamma de 2 a 1,5, AdamW lr 1e-4. El articulo parte de una
               red YA entrenada (si no, la identidad es solucion trivial): aqui parte de la UNet de
               Neighbor2Neighbor de la ronda 3 del mismo regimen (--p2n-base) y se ajusta con los frames.

Zero-shot (una red NUEVA por imagen, como en su articulo; no usan el conjunto de frames):
  zsn2n_zs     Mansour & Heckel, CVPR 2023: 2.000 iteraciones por imagen, lr 1e-3 (paso 1.000, x0,5).
  noise2detail Chobola et al., MICCAI 2025 (ctom2/noise2detail): ZS-N2N + pseudo-limpias con
               pixel-unshuffle 2 y 4 + segunda red entrenada sobre la pseudo-limpia.
  dip          Ulyanov et al., CVPR 2018: red "skip" de 5 niveles (128 canales), entrada de ruido de 32
               canales, lr 0,01, reg_noise_std 1/30, promedio exponencial 0,99, 1.800 iteraciones
               (config. de denoising del repo). Lento: se evalua en --max-imagenes del regimen.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fac_denoise.metodos import Metodo, N2V, _unet, _sigma_lote, _par_zs, _ZSNet


# ---------------------------------------------------------------------------
# CVF-SID
class _GenClean(nn.Module):
    def __init__(self, channels=3, num_of_layers=17, features=64):
        super().__init__()
        capas = [nn.Conv2d(channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            capas += [nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True)]
        capas.append(nn.Conv2d(features, channels, 1))
        self.genclean = nn.Sequential(*capas)
        for m in self.genclean:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.genclean(x)


class _GenNoise(nn.Module):
    def __init__(self, c=3, NLayer=10, FSize=64):
        super().__init__()
        m = [nn.Conv2d(c, FSize, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(NLayer - 1):
            m += [nn.Conv2d(FSize, FSize, 3, padding=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*m)

        def rama():
            r = []
            for _ in range(4):
                r += [nn.Conv2d(FSize, FSize, 3, padding=1), nn.ReLU(inplace=True)]
            r.append(nn.Conv2d(FSize, c, 1))
            return nn.Sequential(*r)

        self.gen_noise_w, self.gen_noise_b = rama(), rama()
        for seq in (self.body, self.gen_noise_w, self.gen_noise_b):
            for mm in seq:
                if isinstance(mm, nn.Conv2d):
                    nn.init.xavier_uniform_(mm.weight)
                    nn.init.constant_(mm.bias, 0)

    def forward(self, x):
        f = self.body(x)
        w, b = self.gen_noise_w(f), self.gen_noise_b(f)
        return w - w.mean(dim=(-1, -2), keepdim=True), b - b.mean(dim=(-1, -2), keepdim=True)


class _CVF(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.gen_noise = _GenNoise(c)
        self.genclean = _GenClean(c)

    def forward(self, x):
        clean = self.genclean(x)
        w, b = self.gen_noise(x - clean)
        return w, b, clean


def _padr(img, pad=20):
    return F.pad(img, (pad, pad, pad, pad), mode="reflect")


def _padr_crop(img, pad=20):
    return F.pad(img[:, :, pad:-pad, pad:-pad], (pad, pad, pad, pad), mode="reflect")


def _std_local(x, k=6):
    """data.py::process_image del repo: media de la desviacion estandar en ventanas 6x6, por canal."""
    m = F.avg_pool2d(x, k, stride=1)
    m2 = F.avg_pool2d(x * x, k, stride=1)
    return torch.sqrt((m2 - m * m).clamp_min(0)).mean(dim=(-1, -2))  # B x C


class CVFSID(Metodo):
    """config_SIDD_Val.json oficial: parche 40, batch 64, Adam lr 1e-4, gamma 1."""
    nombre, multiplo, parche_defecto, batch_defecto = "cvfsid", 1, 40, 64
    multiplo_mosaico = 8

    def crear(self, a, c):
        return _CVF(c)

    def perdida(self, red, x, progreso, a):
        g = 1.0
        mse = F.mse_loss
        std = _std_local(x)
        x = _padr(x)
        w, b, clean = red(x)
        w1, b1, c1 = red(_padr_crop(clean))
        w2, b2, c2 = red(_padr_crop(clean + clean.clamp_min(0) ** g * w))
        w3, b3, c3 = red(_padr_crop(b))
        w4, b4, c4 = red(_padr_crop(clean + clean.clamp_min(0) ** g * w - b))
        w5, b5, c5 = red(_padr_crop(clean - clean.clamp_min(0) ** g * w + b))
        w6, b6, c6 = red(_padr_crop(clean - clean.clamp_min(0) ** g * w - b))
        w10, b10, c10 = red(_padr_crop(clean + clean.clamp_min(0) ** g * w + b))
        w7, b7, c7 = red(_padr_crop(clean + b))
        w8, b8, c8 = red(_padr_crop(clean - b))
        w9, b9, c9 = red(_padr_crop(clean - clean.clamp_min(0) ** g * w))
        pred = clean + clean.clamp_min(0) ** g * w + b
        # loss_main
        sb = b.flatten(2).std(-1)
        sw = w.flatten(2).std(-1)
        blur = F.avg_pool2d(clean, 6, stride=1, padding=3)
        cm = (blur.clamp_min(0) ** g).flatten(2).square().mean(-1)
        swb = torch.sqrt(cm * sw.square() + sb.square() + 1e-12)
        z = torch.zeros_like
        loss = (mse(pred, x) + mse(c1, clean) + mse(b3, b) + mse(w2, w) + mse(c2, clean) + mse(c3, z(c3))
                + mse(w1, z(w1)) + mse(b1, z(b1)) + mse(b2, z(b2)) + mse(swb, std))

        def aug(cl, cl1, nw, nw1, nb, nb1):
            return mse(cl1, cl) + mse(nw1, nw) + mse(nb1, nb)

        neg = (aug(clean, c4, w, w4, b, -b4) + aug(clean, c5, w, -w5, b, b5) + aug(clean, c6, w, -w6, b, -b6)
               + aug(clean, c7, z(w), w7, b, b7) + aug(clean, c8, z(w), w8, b, -b8)
               + aug(clean, c9, -w, w9, z(b), b9) + aug(clean, c10, w, w10, b, b10))
        return loss + 0.1 * neg

    def aplicar(self, red, x, a):
        return red(_padr(x))[2][:, :, 20:-20, 20:-20]


# ---------------------------------------------------------------------------
# Noise2Score (Gaussian_blind)
class Noise2Score(Metodo):
    nombre, multiplo = "noise2score", 32

    @staticmethod
    def argumentos(p):
        p.add_argument("--n2score-smax", type=float, default=0.05, help="sigma AR-DAE inicial")
        p.add_argument("--n2score-smin", type=float, default=0.001, help="sigma AR-DAE final")
        p.add_argument("--n2score-busqueda", type=float, nargs=2, default=[0.5, 20.0],
                       help="rango (DN) de la sigma de Tweedie buscada con variacion total")

    def crear(self, a, c):
        return _unet(a, c)

    def perdida(self, red, x, progreso, a):
        s = a.n2score_smax * (1 - progreso) + a.n2score_smin * progreso
        mu = torch.randn_like(x)
        return F.mse_loss(s * red(x + s * mu).float(), -mu)

    def aplicar(self, red, x, a):
        score = red(x).float()
        mejor, mejor_tv = None, float("inf")
        lo, hi = a.n2score_busqueda
        for i in range(50):
            sig = (lo + (hi - lo) * (i + 1) / 50) / 255.0
            rec = x + sig ** 2 * score
            tv = (torch.sqrt((rec[:, :, 1:] - rec[:, :, :-1]) ** 2 + 1e-12).mean()
                  + torch.sqrt((rec[..., 1:] - rec[..., :-1]) ** 2 + 1e-12).mean())
            if tv < mejor_tv:
                mejor_tv, mejor = tv, rec
        return mejor


# ---------------------------------------------------------------------------
# N2V2
class _BlurPool(nn.Module):
    def __init__(self, c):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = (k[:, None] * k[None, :])
        self.register_buffer("k", (k / k.sum())[None, None].repeat(c, 1, 1, 1))
        self.c = c

    def forward(self, x):
        x = F.max_pool2d(x, 2, stride=1, padding=0)
        x = F.pad(x, (0, 1, 0, 1), mode="replicate")
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), self.k, stride=2, groups=self.c)


class _UNetN2V2(nn.Module):
    """UNet de N2V (profundidad 3, 32 filtros) con BlurPool y SIN el salto de mayor resolucion."""

    def __init__(self, c=3, f=32):
        super().__init__()

        def bloque(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))
        self.e0, self.p0 = bloque(c, f), _BlurPool(f)
        self.e1, self.p1 = bloque(f, 2 * f), _BlurPool(2 * f)
        self.e2, self.p2 = bloque(2 * f, 4 * f), _BlurPool(4 * f)
        self.m = bloque(4 * f, 8 * f)
        self.d2 = bloque(8 * f + 4 * f, 4 * f)
        self.d1 = bloque(4 * f + 2 * f, 2 * f)
        self.d0 = bloque(2 * f, f)  # sin salto desde e0 (N2V2)
        self.fin = nn.Conv2d(f, c, 1)

    def forward(self, x):
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
        e0 = self.e0(x)
        e1 = self.e1(self.p0(e0))
        e2 = self.e2(self.p1(e1))
        m = self.m(self.p2(e2))
        d2 = self.d2(torch.cat([up(m), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        d0 = self.d0(up(d1))
        return self.fin(d0)


class N2V2(N2V):
    nombre, multiplo = "n2v2", 8
    lr_defecto = 4e-4

    def crear(self, a, c):
        a.struct_largo = 0
        return _UNetN2V2(c)

    def perdida(self, red, x, progreso, a):
        b, c, h, w = x.shape
        n = max(1, int(round(a.n2v_perc / 100.0 * h * w)))
        dev = x.device
        ys, xs = torch.randint(1, h - 1, (b, n), device=dev), torch.randint(1, w - 1, (b, n), device=dev)
        bi = torch.arange(b, device=dev)[:, None].expand(b, n)
        vecinos = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    vecinos.append(x[bi, :, ys + dy, xs + dx])
        med = torch.stack(vecinos, 0).median(0).values  # mediana del vecindario SIN el centro
        entrada = x.clone()
        entrada[bi, :, ys, xs] = med
        out = red(entrada)
        return F.mse_loss(out[bi, :, ys, xs].float(), x[bi, :, ys, xs])


# ---------------------------------------------------------------------------
# Self2Self / Cut2Self (red de terceros/self2self)
class Self2Self(Metodo):
    nombre, multiplo, parche_defecto, batch_defecto = "self2self", 64, 256, 4
    multiplo_mosaico = 64
    corte = False

    @staticmethod
    def argumentos(p):
        p.add_argument("--s2s-p", type=float, default=0.3, help="fraccion de pixeles ocultos")
        p.add_argument("--s2s-t", type=int, default=20, help="mascaras promediadas al inferir (articulo: 100)")

    def crear(self, a, c):
        from fac_denoise.terceros import cargar
        return cargar("self2self", "model").self2self(c, a.s2s_p)

    def _mascara(self, x, p):
        if not self.corte:
            return (torch.rand_like(x) > p).float()
        b, c, h, w = x.shape  # Cut2Self: huecos de 4x4 que cubren ~p del area
        n = max(1, int(p * h * w / 16))
        m = torch.ones(b, 1, h, w, device=x.device)
        ys = torch.randint(0, h, (b, n), device=x.device)
        xs = torch.randint(0, w, (b, n), device=x.device)
        for dy in range(-2, 2):
            for dx in range(-2, 2):
                m[torch.arange(b, device=x.device)[:, None], 0, (ys + dy).clamp(0, h - 1), (xs + dx).clamp(0, w - 1)] = 0
        return m.expand(b, c, h, w).contiguous()

    def perdida(self, red, x, progreso, a):
        m = self._mascara(x, a.s2s_p) * 0.7  # self2self.py de referencia: mascara * 0.7
        out = red(x * m, m).float()
        return torch.sum((out - x) ** 2 * (1 - m)) / torch.sum(1 - m)

    def aplicar(self, red, x, a):
        acc = 0
        for _ in range(a.s2s_t):
            m = self._mascara(x, a.s2s_p)
            acc = acc + red(x * m, m).float()
        return acc / a.s2s_t


class Cut2Self(Self2Self):
    nombre, corte = "cut2self", True


# ---------------------------------------------------------------------------
class GR2R(Metodo):
    """pytorch_demo.py oficial, corruptor gaussiano: y1 = y + alpha*sigma*z, y2 = (y - y1(1-alpha))/alpha,
    MSE(f(y1), y2); inferencia = promedio de f(y1) en T muestras (mc_samples)."""
    nombre, multiplo = "gr2r", 32

    @staticmethod
    def argumentos(p):
        p.add_argument("--gr2r-alpha", type=float, default=0.2)
        p.add_argument("--gr2r-t", type=int, default=5)

    def crear(self, a, c):
        return _unet(a, c)

    def perdida(self, red, x, progreso, a):
        al = a.gr2r_alpha
        y1 = x + al * _sigma_lote(x) * torch.randn_like(x)
        y2 = (x - y1 * (1 - al)) / al
        return F.mse_loss(red(y1).float(), y2)

    def aplicar(self, red, x, a):
        s = _sigma_lote(x)
        return torch.stack([red(x + a.gr2r_alpha * s * torch.randn_like(x)).float()
                            for _ in range(a.gr2r_t)]).mean(0)


# ---------------------------------------------------------------------------
class P2N(Metodo):
    nombre, multiplo, lr_defecto = "p2n", 32, 1e-4
    optimizador = "adamw"

    @staticmethod
    def argumentos(p):
        p.add_argument("--p2n-sigma", type=float, default=0.75)
        p.add_argument("--p2n-base", default="", help="checkpoint de la UNet de partida (p.ej. nb2nb_r3)")

    def crear(self, a, c):
        red = _unet(a, c)
        if getattr(a, "p2n_base", ""):
            ck = torch.load(a.p2n_base, map_location="cpu", weights_only=False)
            red.load_state_dict(ck["modelo"], strict=True)
            print(f"p2n: parte de {a.p2n_base} ({ck['metodo']}, paso {ck['paso']})")
        return red

    def perdida(self, red, y, progreso, a):
        with torch.no_grad():
            xh = red(y).float()
        nh = y - xh
        b = y.shape[0]
        sn = torch.randn(b, 1, 1, 1, device=y.device) * a.p2n_sigma + 1
        sp = torch.randn(b, 1, 1, 1, device=y.device) * a.p2n_sigma + 1
        yp, yn = xh + sn * nh, xh - sp * nh
        gamma = 2.0 - 0.5 * progreso
        d = (red(yp).float() - red(yn).float()).abs()
        return torch.mean((d + 1e-8) ** gamma)

    def aplicar(self, red, x, a):
        return red(x)


# ---------------------------------------------------------------------------
# Zero-shot: una red por imagen (sin el conjunto de frames)
class ZeroShot(Metodo):
    entrenable, zero_shot = False, True

    def zero_shot_imagen(self, x, a, disp):
        raise NotImplementedError


def _entrenar_zs(red, x, it, lr, perdida, paso=1000, gamma=0.5):
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, paso, gamma)
    for _ in range(it):
        opt.zero_grad(set_to_none=True)
        perdida().backward()
        opt.step()
        sch.step()


def _zs_res(model, y):
    n1, n2 = _par_zs(y)
    p1, p2 = n1 - model(n1), n2 - model(n2)
    res = 0.5 * (F.mse_loss(n1, p2) + F.mse_loss(n2, p1))
    d1, d2 = _par_zs(y - model(y))
    return res + 0.5 * (F.mse_loss(p1, d1) + F.mse_loss(p2, d2))


def _pad_par(x):
    h, w = x.shape[-2:]
    return F.pad(x, (0, w % 2, 0, h % 2), mode="replicate"), h, w


class ZSN2NZS(ZeroShot):
    nombre = "zsn2n_zs"

    @staticmethod
    def argumentos(p):
        p.add_argument("--zs-iter", type=int, default=2000)

    def zero_shot_imagen(self, x, a, disp):
        xp, h, w = _pad_par(x)
        red = _ZSNet(x.shape[1]).to(disp)
        _entrenar_zs(red, xp, a.zs_iter, 1e-3, lambda: _zs_res(red, xp))
        with torch.no_grad():
            return (xp - red(xp))[..., :h, :w].clamp(0, 1)


def _unshuffle_lote(x, f):
    b, c, h, w = x.shape
    return F.pixel_unshuffle(x, f).view(b, c, f * f, h // f, w // f).permute(0, 2, 1, 3, 4).reshape(b * f * f, c, h // f, w // f)


def _shuffle_lote(x, f):
    bf, c, h, w = x.shape
    b = bf // (f * f)
    return F.pixel_shuffle(x.view(b, f * f, c, h, w).permute(0, 2, 1, 3, 4).reshape(b, c * f * f, h, w), f)


class Noise2Detail(ZeroShot):
    nombre = "noise2detail"

    def zero_shot_imagen(self, x, a, disp):
        h, w = x.shape[-2:]
        xp = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
        c = x.shape[1]
        m1 = _ZSNet(c).to(disp)
        _entrenar_zs(m1, xp, a.zs_iter, 1e-3, lambda: _zs_res(m1, xp))
        with torch.no_grad():
            d1 = (xp - m1(xp)).clamp(0, 1)
            pseudo = [d1]
            for f in (2, 4):
                t = _unshuffle_lote(xp, f)
                t = _shuffle_lote((t - m1(t)).clamp(0, 1), f)
                pseudo.append((t - m1(t)).clamp(0, 1))
            semi = sum(pseudo) / 3
        m2 = _ZSNet(c).to(disp)

        def perdida2():
            n1, n2 = _par_zs(xp)
            s1, s2 = _par_zs(semi)
            p1, p2 = m2(s1), m2(s2)
            res = 0.5 * (F.mse_loss(n1, p2) + F.mse_loss(n2, p1))
            e1, e2 = _par_zs(m2(semi))
            return res + 0.5 * (F.mse_loss(p1, e1) + F.mse_loss(p2, e2))

        _entrenar_zs(m2, xp, a.zs_iter, 1e-3, perdida2)
        with torch.no_grad():
            return m2(semi).clamp(0, 1)[..., :h, :w]


# ---- Deep Image Prior: red "skip" (models/skip.py del repo oficial, config de denoising) ----
class _Skip(nn.Module):
    def __init__(self, cin=32, cout=3, d=128, s=4, niveles=5):
        super().__init__()
        act = lambda: nn.LeakyReLU(0.2, inplace=True)

        def cbr(i, o, k, st=1):
            return [nn.ReflectionPad2d(k // 2), nn.Conv2d(i, o, k, st), nn.BatchNorm2d(o), act()]
        self.down = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.up = nn.ModuleList()
        c = cin
        for _ in range(niveles):
            self.skip.append(nn.Sequential(*cbr(c, s, 1)))
            self.down.append(nn.Sequential(*cbr(c, d, 3, 2), *cbr(d, d, 3)))
            c = d
        for i in range(niveles):
            self.up.append(nn.Sequential(nn.BatchNorm2d(s + d), *cbr(s + d, d, 3), *cbr(d, d, 1)))
        self.fin = nn.Sequential(nn.Conv2d(d, cout, 1), nn.Sigmoid())

    def forward(self, z):
        skips, x = [], z
        for sk, dn in zip(self.skip, self.down):
            skips.append(sk(x))
            x = dn(x)
        for up, sk in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=sk.shape[-2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([sk, x], 1))
        return self.fin(x)


class DIP(ZeroShot):
    nombre = "dip"

    @staticmethod
    def argumentos(p):
        p.add_argument("--dip-iter", type=int, default=1800)

    def zero_shot_imagen(self, x, a, disp):
        h, w = x.shape[-2:]
        xp = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
        red = _Skip(32, x.shape[1]).to(disp)
        z0 = torch.rand(1, 32, *xp.shape[-2:], device=disp) * 0.1
        opt = torch.optim.Adam(red.parameters(), lr=0.01)
        prom = None
        for _ in range(a.dip_iter):
            opt.zero_grad(set_to_none=True)
            out = red(z0 + torch.randn_like(z0) / 30.0)
            F.mse_loss(out, xp).backward()
            opt.step()
            o = out.detach()
            prom = o if prom is None else prom * 0.99 + o * 0.01
        return prom[..., :h, :w]


TANDA2 = [CVFSID(), Noise2Score(), N2V2(), Self2Self(), Cut2Self(), GR2R(), P2N(), ZSN2NZS(), Noise2Detail(), DIP()]

from fac_denoise.metodos import METODOS as _M  # noqa: E402
_M.update({m.nombre: m for m in TANDA2})
