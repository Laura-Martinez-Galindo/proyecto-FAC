"""
Metodos de denoising auto-supervisado / zero-shot con una sola interfaz.

Cada metodo define:
  temporal   frames que recibe (5 = video: UDVD; 1 = imagen)
  multiplo   el lado del parche/mosaico debe ser multiplo de esto
  escala     1 -> la red trabaja en [0,1]; 255 -> en [0,255] (AP-BSN, como el original)
  crear(a, c)                  red
  perdida(red, x, progreso, a) perdida de un lote (x: B x T*C x t x t en [0,1])
  aplicar(red, x, a)           salida limpia de un mosaico (1 x T*C x h x w -> 1 x C x h x w)

Los hiperparametros propios de cada metodo se declaran en `argumentos()` y quedan en el
checkpoint y en configuracion.json.

Referencias (perdidas reimplementadas siguiendo el codigo oficial):
  udvd       Sheth et al. ICCV 2021  - red oficial (scripts/udvd_oficial/modelo.py, MIT)
  n2v        Krull et al. CVPR 2019  - Noise2Void (mascara de pixeles, UNet)
  structn2v  Broaddus et al. ISBI 2020 - Noise2Void con mascara en linea (ruido en rayas)
  nb2nb      Huang et al. CVPR 2021  - Neighbor2Neighbor (train.py oficial)
  b2u        Wang et al. CVPR 2022   - Blind2Unblind (train_b2u.py / test_b2u.py oficiales)
  apbsn      Lee et al. CVPR 2022    - AP-BSN (ruido real espacialmente correlacionado)
  nlm        Buades et al. 2005      - Non-Local Means de OpenCV (sin entrenamiento)
  destripe_nlm  correccion de offset por columna (FPN del termico) + NLM
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fac_denoise.redes import APBSN, UNet  # noqa: E402
from udvd_oficial.modelo import BlindVideoNet  # noqa: E402


# ---------------------------------------------------------------------------
class Metodo:
    nombre = ""
    temporal = 1
    multiplo = 1
    escala = 1.0
    entrenable = True
    parche_defecto = 128
    batch_defecto = 16
    lr_defecto = 1e-4

    @staticmethod
    def argumentos(p):
        pass

    def crear(self, a, c):
        raise NotImplementedError

    def perdida(self, red, x, progreso, a):
        raise NotImplementedError

    def aplicar(self, red, x, a):
        raise NotImplementedError


# ---------------------------------------------------------------------------
class UDVD(Metodo):
    nombre, temporal, multiplo = "udvd", 5, 8

    def crear(self, a, c):
        return BlindVideoNet(channels_per_frame=c, out_channels=c, bias=False, blind=True, sigma_known=True)

    def perdida(self, red, x, progreso, a):
        c = x.shape[1] // 5
        out, _ = red(x)
        return F.mse_loss(out[:, :c].float(), x[:, 2 * c:3 * c])

    def aplicar(self, red, x, a):
        c = x.shape[1] // 5
        return red(x)[0][:, :c]


# ---------------------------------------------------------------------------
def _unet(a, c):
    return UNet(in_nc=c, out_nc=c, n_feature=a.unet_canales)


class N2V(Metodo):
    """Noise2Void / StructN2V. Se ocultan `perc` de los pixeles (reemplazados por un vecino
    al azar) y la perdida solo se mide ahi. StructN2V ademas tapa con valores aleatorios
    los pixeles de una linea a lo largo de `struct_eje` (la direccion del ruido en rayas)."""
    nombre, multiplo = "n2v", 32
    @staticmethod
    def argumentos(p):
        p.add_argument("--n2v-perc", type=float, default=0.5, help="%% de pixeles enmascarados por parche")
        p.add_argument("--n2v-radio", type=int, default=2, help="radio del vecino que reemplaza al pixel")
        p.add_argument("--struct-largo", type=int, default=None,
                       help="StructN2V: pixeles de la linea tapada (por defecto 9 en structn2v, 0 en n2v)")
        p.add_argument("--struct-eje", choices=["vertical", "horizontal"], default="vertical",
                       help="vertical = rayas de columna (FPN tipico del termico)")

    struct_defecto = 0

    def crear(self, a, c):
        if a.struct_largo is None:
            a.struct_largo = self.struct_defecto
        return _unet(a, c)

    def perdida(self, red, x, progreso, a):
        b, c, h, w = x.shape
        n = max(1, int(round(a.n2v_perc / 100.0 * h * w)))
        dev = x.device
        ys = torch.randint(0, h, (b, n), device=dev)
        xs = torch.randint(0, w, (b, n), device=dev)
        r = a.n2v_radio
        dy = torch.randint(-r, r + 1, (b, n), device=dev)
        dx = torch.randint(-r, r + 1, (b, n), device=dev)
        cero = (dy == 0) & (dx == 0)
        dx = torch.where(cero, torch.ones_like(dx), dx)
        yv, xv = (ys + dy).clamp(0, h - 1), (xs + dx).clamp(0, w - 1)
        bi = torch.arange(b, device=dev)[:, None].expand(b, n)
        entrada = x.clone()
        if a.struct_largo > 1:
            lo, hi = x.amin(dim=(1, 2, 3)), x.amax(dim=(1, 2, 3))
            mitad = a.struct_largo // 2
            for d in range(-mitad, mitad + 1):
                if d == 0:
                    continue
                if a.struct_eje == "vertical":
                    yy, xx = (ys + d).clamp(0, h - 1), xs
                else:
                    yy, xx = ys, (xs + d).clamp(0, w - 1)
                rnd = lo[:, None, None] + (hi - lo)[:, None, None] * torch.rand(b, n, c, device=dev)
                entrada[bi, :, yy, xx] = rnd
        entrada[bi, :, ys, xs] = x[bi, :, yv, xv]
        out = red(entrada)
        return F.mse_loss(out[bi, :, ys, xs].float(), x[bi, :, ys, xs])

    def aplicar(self, red, x, a):
        return red(x)


class StructN2V(N2V):
    nombre = "structn2v"
    struct_defecto = 9


def _norma(t, a):
    """Media de t^2 (l2, la del articulo) o de |t| (l1, robusta a impulsos)."""
    return torch.mean(t.abs()) if getattr(a, "perdida", "l2") == "l1" else torch.mean(t ** 2)


# ---------------------------------------------------------------------------
# Neighbor2Neighbor (train.py oficial)
def _space_to_depth(x, bs):
    n, c, h, w = x.size()
    return F.unfold(x, bs, stride=bs).view(n, c * bs ** 2, h // bs, w // bs)


def _mascaras_par(img):
    n, c, h, w = img.shape
    m1 = torch.zeros(n * h // 2 * w // 2 * 4, dtype=torch.bool, device=img.device)
    m2 = torch.zeros_like(m1)
    pares = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]], device=img.device)
    rd = torch.randint(0, 8, (n * h // 2 * w // 2,), device=img.device)
    idx = pares[rd] + torch.arange(0, n * h // 2 * w // 2 * 4, 4, device=img.device).reshape(-1, 1)
    m1[idx[:, 0]] = 1
    m2[idx[:, 1]] = 1
    return m1, m2


def _subimagen(img, mask):
    n, c, h, w = img.shape
    sub = torch.zeros(n, c, h // 2, w // 2, dtype=img.dtype, device=img.device)
    for i in range(c):
        t = _space_to_depth(img[:, i:i + 1], 2).permute(0, 2, 3, 1).reshape(-1)
        sub[:, i:i + 1] = t[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return sub


class NB2NB(Metodo):
    nombre, multiplo = "nb2nb", 64
    lr_defecto = 3e-4

    @staticmethod
    def argumentos(p):
        p.add_argument("--nb2nb-lambda1", type=float, default=1.0)
        p.add_argument("--nb2nb-lambda2", type=float, default=1.0)
        p.add_argument("--nb2nb-increase-ratio", type=float, default=2.0,
                       help="peso de la regularizacion al final (crece lineal desde 0)")

    def crear(self, a, c):
        return _unet(a, c)

    def perdida(self, red, x, progreso, a):
        m1, m2 = _mascaras_par(x)
        s1, s2 = _subimagen(x, m1), _subimagen(x, m2)
        # OJO: primero la pasada CON gradiente. Con AMP (autocast) la primera pasada guarda en cache
        # los pesos convertidos a fp16; si esa primera pasada es bajo no_grad, la cache queda sin
        # gradiente y la segunda pasada no entrena ("element 0 of tensors does not require grad").
        out = red(s1)
        with torch.no_grad():
            den = red(x)
        d1, d2 = _subimagen(den, m1), _subimagen(den, m2)
        lam = progreso * a.nb2nb_increase_ratio
        diff = out.float() - s2
        exp_diff = (d1 - d2).float()
        return (a.nb2nb_lambda1 * _norma(diff, a) + a.nb2nb_lambda2 * lam * _norma(diff - exp_diff, a))

    def aplicar(self, red, x, a):
        return red(x)


# ---------------------------------------------------------------------------
# Blind2Unblind (train_b2u.py / test_b2u.py oficiales). Masker width=4, mode interpolate.
def _depth_to_space(x, bs):
    return F.pixel_shuffle(x, bs)


def _mascara_fija(img, width, i):
    n, c, h, w = img.shape
    m = torch.zeros(n * h // width * w // width * width ** 2, dtype=torch.float32, device=img.device)
    idx = torch.full((n * h // width * w // width,), i, dtype=torch.int64, device=img.device)
    idx = idx + torch.arange(0, n * h // width * w // width * width ** 2, width ** 2, device=img.device)
    m[idx] = 1
    return _depth_to_space(m.view(n, h // width, w // width, width ** 2).permute(0, 3, 1, 2), width)


def _interpolar(t, mask):
    n, c, h, w = t.shape
    k = torch.tensor([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]], device=t.device, dtype=t.dtype)
    k = (k / k.sum())[None, None]
    f = F.conv2d(t.reshape(n * c, 1, h, w), k, padding=1).view_as(t)
    return f * mask + t * (1 - mask)


def _masker_train(img, width=4):
    n, c, h, w = img.shape
    ts, ms = [], []
    for i in range(width ** 2):
        m = _mascara_fija(img, width, i)
        ts.append(_interpolar(img, m))
        ms.append(m)
    return torch.stack(ts, 1).reshape(-1, c, h, w), torch.stack(ms, 1).reshape(-1, 1, h, w)


class B2U(Metodo):
    nombre, multiplo = "b2u", 32
    lr_defecto = 3e-4
    batch_defecto = 4

    @staticmethod
    def argumentos(p):
        p.add_argument("--b2u-lambda1", type=float, default=1.0)
        p.add_argument("--b2u-lambda2", type=float, default=2.0, help="beta inicial")
        p.add_argument("--b2u-increase-ratio", type=float, default=20.0, help="beta final")
        p.add_argument("--b2u-beta-inferencia", type=float, default=0.0,
                       help="beta al inferir (0 = usar increase-ratio, como test_b2u.py)")

    def crear(self, a, c):
        return _unet(a, c)

    def perdida(self, red, x, progreso, a):
        n, c, h, w = x.shape
        entrada, mask = _masker_train(x)
        out = (red(entrada).float() * mask).view(n, -1, c, h, w).sum(1)
        diff = out - x
        with torch.no_grad():
            exp_diff = red(x).float() - x
        t1, t2 = 0.4, 1.0
        if progreso <= t1:
            beta = a.b2u_lambda2
        elif progreso <= t2:
            beta = a.b2u_lambda2 + (progreso - t1) * (a.b2u_increase_ratio - a.b2u_lambda2) / (t2 - t1)
        else:
            beta = a.b2u_increase_ratio
        return a.b2u_lambda1 * _norma(diff, a) + _norma(diff + beta * exp_diff, a)

    def aplicar(self, red, x, a):
        n, c, h, w = x.shape
        entrada, mask = _masker_train(x)
        dn = (red(entrada).float() * mask).view(n, -1, c, h, w).sum(1)
        exp = red(x).float()
        beta = a.b2u_beta_inferencia or a.b2u_increase_ratio
        return (dn + beta * exp) / (1 + beta)


# ---------------------------------------------------------------------------
class APBSNm(Metodo):
    """Configuracion oficial de SIDD: pd_a=5, pd_b=2, pd_pad=2, R3 (T=8, p=0.16), DBSNl 128x9,
    perdida L1 contra la entrada ruidosa, escala 0-255, parche 120, batch 8, lr 1e-4."""
    nombre, multiplo, escala = "apbsn", 10, 255.0
    parche_defecto = 120
    batch_defecto = 8

    @staticmethod
    def argumentos(p):
        p.add_argument("--pd-a", type=int, default=5, help="factor PD al entrenar")
        p.add_argument("--pd-b", type=int, default=2, help="factor PD al inferir")
        p.add_argument("--pd-pad", type=int, default=2)
        p.add_argument("--r3-t", type=int, default=8, help="R3: repeticiones (0 = sin R3)")
        p.add_argument("--r3-p", type=float, default=0.16)
        p.add_argument("--bsn-canales", type=int, default=128)
        p.add_argument("--bsn-modulos", type=int, default=9)

    def crear(self, a, c):
        return APBSN(pd_a=a.pd_a, pd_b=a.pd_b, pd_pad=a.pd_pad, R3=a.r3_t > 0, R3_T=max(a.r3_t, 1),
                     R3_p=a.r3_p, in_ch=c, bsn_base_ch=a.bsn_canales, bsn_num_module=a.bsn_modulos)

    def perdida(self, red, x, progreso, a):
        return F.l1_loss(red(x).float(), x)

    def aplicar(self, red, x, a):
        return red.denoise(x)


# ---------------------------------------------------------------------------
class MMBSNm(APBSNm):
    """MM-BSN (CVPRW 2023): como AP-BSN pero con varias mascaras de punto ciego (p.ej. 'o' + diagonal
    'a45') para ruido correlacionado en varias direcciones. Configuracion oficial de SIDD:
    mask_type o_a45, DCL1=2, DCL2=7, base 128, PD 5/2, R3 (8, 0.16), L1, parche 120, batch 8, lr 1e-4.
    Usa los mismos --pd-a/--pd-b/--pd-pad/--r3-t/--r3-p/--bsn-canales que apbsn."""
    nombre = "mmbsn"

    @staticmethod
    def argumentos(p):
        p.add_argument("--mm-mascaras", default="o_a45", help="mascaras de MM-BSN: o, c, r, sz, fsz, a45, a135, cha, fcha (unir con _)")
        p.add_argument("--mm-dcl1", type=int, default=2)
        p.add_argument("--mm-dcl2", type=int, default=7)

    def crear(self, a, c):
        from fac_denoise.mmbsn import BSN_MM
        return BSN_MM(pd_a=a.pd_a, pd_b=a.pd_b, pd_pad=a.pd_pad, R3=a.r3_t > 0, R3_T=max(a.r3_t, 1), R3_p=a.r3_p,
                      in_ch=c, bsn_base_ch=a.bsn_canales, DCL1_num=a.mm_dcl1, DCL2_num=a.mm_dcl2,
                      mask_type=a.mm_mascaras)


# ---------------------------------------------------------------------------
class NLM(Metodo):
    nombre, entrenable = "nlm", False

    @staticmethod
    def argumentos(p):
        p.add_argument("--nlm-h", type=float, default=5.0, help="fuerza en luminancia")
        p.add_argument("--nlm-h-color", type=float, default=5.0, help="fuerza en color")
        p.add_argument("--nlm-plantilla", type=int, default=7)
        p.add_argument("--nlm-busqueda", type=int, default=21)

    def clasico(self, im, a):
        """im HxWxC uint8 (RGB o gris)."""
        if im.shape[2] == 1:
            return cv2.fastNlMeansDenoising(im[..., 0], None, a.nlm_h, a.nlm_plantilla, a.nlm_busqueda)[..., None]
        bgr = np.ascontiguousarray(im[..., ::-1])
        out = cv2.fastNlMeansDenoisingColored(bgr, None, a.nlm_h, a.nlm_h_color, a.nlm_plantilla, a.nlm_busqueda)
        return np.ascontiguousarray(out[..., ::-1])


class DestripeNLM(NLM):
    nombre = "destripe_nlm"

    @staticmethod
    def argumentos(p):
        p.add_argument("--destripe-sigma", type=float, default=3.0,
                       help="suavizado perpendicular a las rayas para separar la raya de la escena")
        p.add_argument("--destripe-eje", choices=["vertical", "horizontal"], default="vertical")

    def clasico(self, im, a):
        f = im.astype(np.float32)
        g = f.mean(axis=2)
        if a.destripe_eje == "vertical":  # offset constante por columna
            alto = g - cv2.GaussianBlur(g, (0, 0), sigmaX=a.destripe_sigma, sigmaY=0.01)
            off = np.median(alto, axis=0)[None, :, None]
        else:
            alto = g - cv2.GaussianBlur(g, (0, 0), sigmaX=0.01, sigmaY=a.destripe_sigma)
            off = np.median(alto, axis=1)[:, None, None]
        im2 = np.clip(np.rint(f - off), 0, 255).astype(np.uint8)
        return super().clasico(im2, a)


class MedianaNLM(NLM):
    """Linea base para sal y pimienta: mediana 3x3 SOLO en los pixeles atipicos (difieren mas de
    --mediana-umbral de su mediana) y luego NLM."""
    nombre = "mediana_nlm"

    @staticmethod
    def argumentos(p):
        p.add_argument("--mediana-umbral", type=float, default=30.0)

    def clasico(self, im, a):
        med = cv2.medianBlur(np.ascontiguousarray(im), 3)
        if med.ndim == 2:
            med = med[..., None]
        atip = np.abs(im.astype(np.int16) - med.astype(np.int16)).max(axis=2, keepdims=True) > a.mediana_umbral
        im2 = np.where(atip, med, im).astype(np.uint8)
        return super().clasico(im2, a)



# ===========================================================================
# RONDA 3: metodos nuevos
# ===========================================================================
# ---- Redes de punto ciego de terceros (codigo OFICIAL en fac_denoise/terceros) -----------
# Todas siguen el esquema de AP-BSN de su repo: entrenan con L1 contra la entrada ruidosa
# (self_L1), escala 0-255, e infieren con su propio denoise() (PD asimetrico + R3).
class TerceroAPBSN(Metodo):
    escala = 255.0
    clave = modulo = clase = ""
    kwargs: dict = {}

    def crear(self, a, c):
        from fac_denoise.terceros import cargar
        mod = cargar(self.clave, self.modulo)
        return mod.get_model_class(self.clase)(**self.kwargs)

    def perdida(self, red, x, progreso, a):
        return F.l1_loss(red(x).float(), x)

    def aplicar(self, red, x, a):
        return red.denoise(x)


class PUCAm(TerceroAPBSN):
    """PUCA (Jang et al., NeurIPS 2023). conf/PUCA_SIDD.yaml oficial: pd [4,2,1], dilation 2,
    width 128, enc [3,4], middle 4, dec [4,3], R3 (8, 0.16), parche 256, batch 2, lr 1e-4."""
    nombre, multiplo, parche_defecto, batch_defecto = "puca", 64, 256, 2
    multiplo_mosaico = 64
    clave, modulo, clase = "puca", "src.model", "apbsn"
    kwargs = dict(bsn="PUCA", pd=[4, 2, 1], dilation=2, width=128, enc_blk_nums=[3, 4], middle_blk_nums=4,
                  dec_blk_nums=[4, 3], R3=True, R3_T=8, R3_p=0.16)


class LGBPNm(TerceroAPBSN):
    """LG-BPN (Wang et al., CVPR 2023). conf/APBSN_SIDD/BSN_SIDD.yaml oficial: BSN My_BSN, base 128,
    9 modulos, PD interno (entrena 4, infiere 3, refina 2), R3 (8, 0.16), parche 120, batch 8, lr 1e-4."""
    nombre, multiplo, parche_defecto, batch_defecto = "lgbpn", 24, 120, 8
    multiplo_mosaico = 48
    clave, modulo, clase = "lgbpn", "src.model", "bsn"
    kwargs = dict(pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16, SIDD=True, bsn="My_BSN",
                  in_ch=3, bsn_base_ch=128, bsn_num_module=9)


class UBSNm(TerceroAPBSN):
    """U-BSN (Yu et al., WACV 2025). conf/UBSN_SIDD_val.yaml oficial: pd [4,2,1] (al entrenar sortea
    3-5), dilation 2, R3 (8, 0.16), parche 240, batch 4, lr 1e-4."""
    nombre, multiplo, parche_defecto, batch_defecto = "ubsn", 120, 240, 4
    multiplo_mosaico = 120
    clave, modulo, clase = "ubsn", "src.model", "apbsn"
    kwargs = dict(bsn="UBSN", pd=[4, 2, 1], dilation=2, width=128, enc_blk_nums=[3, 4], middle_blk_nums=4,
                  dec_blk_nums=[4, 3], R3=True, R3_T=8, R3_p=0.16)


def _pd_tbsn(x, f, train=False):
    """pixel_shuffle_down_sampling de model/apbsn.py (TBSN oficial), sin cambios."""
    b, c, w, h = x.shape
    u = F.pixel_unshuffle(x, f).view(b, c, f * f, w // f, h // f).permute(0, 2, 1, 3, 4)
    if train:
        return u[:, 0].reshape(b, c, w // f, h // f)
    return u.reshape(b * f * f, c, w // f, h // f)


def _pu_tbsn(x, f):
    b, c, h, w = x.shape
    t = x.view(b // (f * f), f * f, c, h, w).permute(0, 2, 1, 3, 4).reshape(b // (f * f), c * f * f, h, w)
    return F.pixel_shuffle(t, f)


class _TBSNEnvoltura(torch.nn.Module):
    """model/apbsn.py::APBSNModel de TBSN (train_step / validation_step) como modulo."""

    def __init__(self, bsn, pd_a=5, pd_b=2, R3=True, R3_T=8, R3_p=0.16):
        super().__init__()
        self.bsn, self.pd_a, self.pd_b, self.R3, self.R3_T, self.R3_p = bsn, pd_a, pd_b, R3, R3_T, R3_p

    def denoise(self, x):
        b, c, h, w = x.shape
        out = _pu_tbsn(self.bsn(_pd_tbsn(x, self.pd_b)), self.pd_b)
        if not self.R3:
            return out[:, :, :h, :w]
        den = torch.empty(*x.shape, self.R3_T, device=x.device)
        for t in range(self.R3_T):
            m = torch.rand_like(x) < self.R3_p
            tmp = out.clone().detach()
            tmp[m] = x[m]
            den[..., t] = self.bsn(tmp)
        return den.mean(-1)


class TBSNm(Metodo):
    """TBSN (Li et al., AAAI 2025). option/tbsn_sidd.json oficial: pd_a 5 (entrena con la primera
    sub-imagen PD), pd_b 2, R3 (8, 0.16), parche 640, batch 4, lr 3e-4, L1."""
    nombre, multiplo, escala, parche_defecto, batch_defecto = "tbsn", 40, 255.0, 640, 4
    multiplo_mosaico = 64
    lr_defecto = 3e-4

    def crear(self, a, c):
        from fac_denoise.terceros import cargar
        net = cargar("tbsn", "network")
        return _TBSNEnvoltura(net.TBSN(in_ch=c, out_ch=c))

    def perdida(self, red, x, progreso, a):
        inp = _pd_tbsn(x, red.pd_a, train=True)
        return F.l1_loss(red.bsn(inp).float(), inp)

    def aplicar(self, red, x, a):
        return red.denoise(x)


# ---- Reimplementados desde el articulo / codigo oficial pequeno, sobre la UNet comun ------
class Noise2Self(Metodo):
    """Noise2Self (Batson & Royer, ICML 2019). Masker oficial (czbiohub/noise2self, mode
    'interpolate', width 4): en cada paso se oculta una rejilla de 1 de cada 16 pixeles, se rellena
    con el promedio de sus vecinos y la perdida (MSE) se mide solo ahi. Inferencia: imagen completa."""
    nombre, multiplo = "n2s", 32

    def crear(self, a, c):
        return _unet(a, c)

    def perdida(self, red, x, progreso, a):
        n, c, h, w = x.shape
        fase = random.randrange(16)
        m = torch.zeros(1, 1, h, w, device=x.device)
        m[..., (fase // 4)::4, (fase % 4)::4] = 1
        entrada = _interpolar(x, m)
        out = red(entrada).float()
        return torch.sum(((out - x) * m) ** 2) / (m.sum() * n * c)

    def aplicar(self, red, x, a):
        return red(x)


class Noise2Same(N2V):
    """Noise2Same (Xie et al., NeurIPS 2020): L = MSE(f(x), x) + lambda * sqrt(MSE_J(f(x), f(x_J)))
    donde x_J tiene ocultos los pixeles J (reemplazo por vecino, como N2V). lambda = 2 (articulo)."""
    nombre = "noise2same"

    @staticmethod
    def argumentos(p):
        p.add_argument("--n2same-lambda", type=float, default=2.0)

    def perdida(self, red, x, progreso, a):
        b, c, h, w = x.shape
        n = max(1, int(round(a.n2v_perc / 100.0 * h * w)))
        dev = x.device
        ys, xs = torch.randint(0, h, (b, n), device=dev), torch.randint(0, w, (b, n), device=dev)
        dy = torch.randint(-2, 3, (b, n), device=dev)
        dx = torch.randint(-2, 3, (b, n), device=dev)
        dx = torch.where((dy == 0) & (dx == 0), torch.ones_like(dx), dx)
        bi = torch.arange(b, device=dev)[:, None].expand(b, n)
        mascarada = x.clone()
        mascarada[bi, :, ys, xs] = x[bi, :, (ys + dy).clamp(0, h - 1), (xs + dx).clamp(0, w - 1)]
        f_x = red(x).float()
        f_m = red(mascarada).float()
        rec = F.mse_loss(f_x, x)
        inv = torch.mean((f_x[bi, :, ys, xs] - f_m[bi, :, ys, xs]) ** 2)
        return rec + a.n2same_lambda * torch.sqrt(inv + 1e-12)

    def aplicar(self, red, x, a):
        return red(x)


def _sigma_lote(x):
    """Immerkaer (1996) por imagen del lote, en la escala de x. Devuelve B x 1 x 1 x 1."""
    g = x.mean(1, keepdim=True)
    k = torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=x.dtype, device=x.device)[None, None]
    r = F.conv2d(g, k)
    return (math.sqrt(math.pi / 2) * r.abs().mean(dim=(1, 2, 3)) / 6.0).view(-1, 1, 1, 1)


class R2R(Metodo):
    """Recorrupted-to-Recorrupted (Pang et al., CVPR 2021). y^ = y + alpha*sigma*z,
    y~ = y - sigma*z/alpha, perdida MSE(f(y^), y~); inferencia = promedio de f(y + alpha*sigma*z) en T
    muestras. alpha 0.5 (oficial). El articulo asume sigma conocida: aqui se estima por parche con
    Immerkaer (adaptacion necesaria porque no la conocemos)."""
    nombre, multiplo = "r2r", 32

    @staticmethod
    def argumentos(p):
        p.add_argument("--r2r-alpha", type=float, default=0.5)
        p.add_argument("--r2r-t", type=int, default=8, help="muestras promediadas al inferir (oficial 50)")

    def crear(self, a, c):
        return _unet(a, c)

    def perdida(self, red, x, progreso, a):
        s = _sigma_lote(x)
        z = torch.randn_like(x)
        return F.mse_loss(red(x + a.r2r_alpha * s * z).float(), x - s * z / a.r2r_alpha)

    def aplicar(self, red, x, a):
        s = _sigma_lote(x)
        return torch.stack([red(x + a.r2r_alpha * s * torch.randn_like(x)).float() for _ in range(a.r2r_t)]).mean(0)


class _ZSNet(torch.nn.Module):
    """Red del articulo ZS-N2N: 2 conv 3x3 (48 canales) + conv 1x1, LeakyReLU 0.2; predice el ruido."""

    def __init__(self, c, ch=48):
        super().__init__()
        self.c1 = torch.nn.Conv2d(c, ch, 3, padding=1)
        self.c2 = torch.nn.Conv2d(ch, ch, 3, padding=1)
        self.c3 = torch.nn.Conv2d(ch, c, 1)

    def forward(self, x):
        x = F.leaky_relu(self.c1(x), 0.2)
        x = F.leaky_relu(self.c2(x), 0.2)
        return self.c3(x)


def _par_zs(img):
    c = img.shape[1]
    f1 = torch.tensor([[[[0, 0.5], [0.5, 0]]]], dtype=img.dtype, device=img.device).repeat(c, 1, 1, 1)
    f2 = torch.tensor([[[[0.5, 0], [0, 0.5]]]], dtype=img.dtype, device=img.device).repeat(c, 1, 1, 1)
    return F.conv2d(img, f1, stride=2, groups=c), F.conv2d(img, f2, stride=2, groups=c)


class ZSN2N(Metodo):
    """Zero-Shot Noise2Noise (Mansour & Heckel, CVPR 2023), perdida residual + consistencia del
    articulo (pair_downsampler). Aqui se entrena con los mismos parches que los demas metodos
    (condiciones iguales con los demas metodos)."""
    nombre, multiplo, lr_defecto, parche_defecto = "zsn2n", 2, 1e-3, 128

    def crear(self, a, c):
        return _ZSNet(c)

    def perdida(self, red, y, progreso, a):
        n1, n2 = _par_zs(y)
        p1, p2 = n1 - red(n1), n2 - red(n2)
        res = 0.5 * (F.mse_loss(n1, p2) + F.mse_loss(n2, p1))
        d1, d2 = _par_zs(y - red(y))
        cons = 0.5 * (F.mse_loss(p1, d1) + F.mse_loss(p2, d2))
        return res + cons

    def aplicar(self, red, x, a):
        return x - red(x)


# ---- Clasicos (sin entrenamiento) ------------------------------------------------------
def _sigma_img(im):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) if im.shape[2] == 3 else im[..., 0].astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    return float(math.sqrt(math.pi / 2) * np.abs(cv2.filter2D(g, -1, k))[1:-1, 1:-1].mean() / 6.0)


class BM3Dm(NLM):
    """BM3D (Dabov et al. 2007; Makinen et al. 2020), paquete oficial `bm3d` de Tampere. Color con
    bm3d_rgb (YCbCr). sigma = Immerkaer de la imagen x --bm3d-factor."""
    nombre = "bm3d"

    @staticmethod
    def argumentos(p):
        p.add_argument("--bm3d-factor", type=float, default=1.0)

    def clasico(self, im, a):
        import bm3d
        s = max(_sigma_img(im) * a.bm3d_factor, 0.5) / 255.0
        f = im.astype(np.float32) / 255.0
        out = bm3d.bm3d_rgb(f, s) if im.shape[2] == 3 else bm3d.bm3d(f[..., 0], s)[..., None]
        return np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)


def _guiado(I, p, r, eps):
    """Filtro guiado (He et al., ECCV 2010 / TPAMI 2013), version con box filter."""
    caja = lambda t: cv2.boxFilter(t, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = caja(I), caja(p)
    a = (caja(I * p) - mI * mp) / (caja(I * I) - mI * mI + eps)
    b = mp - a * mI
    return caja(a) * I + caja(b)


class GuiadoM(NLM):
    """Filtro guiado auto-guiado por canal (recomendacion de la investigacion para DDE): r = 4,
    eps = (k * sigma)^2 con sigma de Immerkaer (k = --guiado-k)."""
    nombre = "guiado"

    @staticmethod
    def argumentos(p):
        p.add_argument("--guiado-r", type=int, default=4)
        p.add_argument("--guiado-k", type=float, default=2.0)

    def clasico(self, im, a):
        eps = (a.guiado_k * max(_sigma_img(im), 0.5) / 255.0) ** 2
        f = im.astype(np.float32) / 255.0
        out = np.stack([_guiado(f[..., i], f[..., i], a.guiado_r, eps) for i in range(f.shape[2])], 2)
        return np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)


class MedianaAdaptativa(NLM):
    """Filtro de mediana adaptativo (Hwang & Haddad, 1995): por pixel crece la ventana 3 -> 7
    hasta que la mediana no es un extremo; solo reemplaza el pixel si el mismo es un extremo
    (impulso). Deja intactos los pixeles que no son sal y pimienta."""
    nombre = "mediana_adaptativa"

    def clasico(self, im, a):
        out = im.copy()
        for ch in range(im.shape[2]):
            x = np.ascontiguousarray(im[..., ch])
            decidido = np.zeros(x.shape, bool)
            res = x.copy()
            for k in (3, 5, 7):
                med = cv2.medianBlur(x, k)
                ker = np.ones((k, k), np.uint8)
                mn, mx = cv2.erode(x, ker), cv2.dilate(x, ker)
                nivel_a = (med > mn) & (med < mx) & ~decidido
                impulso = (x <= mn) | (x >= mx)
                res = np.where(nivel_a & impulso, med, res)
                decidido |= nivel_a
            out[..., ch] = res
        return out

METODOS = {m.nombre: m for m in [UDVD(), N2V(), StructN2V(), NB2NB(), B2U(), APBSNm(), MMBSNm(), NLM(), DestripeNLM(),
                                 MedianaNLM(),
                                 # ronda 3
                                 PUCAm(), LGBPNm(), UBSNm(), TBSNm(), Noise2Self(), Noise2Same(), R2R(), ZSN2N(),
                                 BM3Dm(), GuiadoM(), MedianaAdaptativa()]}


def argumentos_todos(p):
    p.add_argument("--unet-canales", type=int, default=48, help="ancho de la UNet (n2v, nb2nb, b2u)")
    p.add_argument("--perdida", choices=["l2", "l1"], default="l2",
                   help="nb2nb y b2u: l2 = la del articulo; l1 = robusta a sal y pimienta (converge a la mediana)")
    vistos = set()
    for m in METODOS.values():
        f = type(m).__dict__.get("argumentos")
        if f is None or f in vistos:
            continue
        vistos.add(f)
        f.__func__(p)
