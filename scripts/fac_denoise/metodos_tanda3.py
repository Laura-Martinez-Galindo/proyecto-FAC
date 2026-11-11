"""
RONDA 3, tanda 3: metodos TEMPORALES (usan frames vecinos). UDVD ya estaba en la tanda 1.

El flujo optico es RAFT-small de torchvision (pesos publicos, se bajan con
jobs/preparar_entorno_ronda3.sh). Los articulos usan TV-L1 (F2F/MF2F) o PWC-Net (F2R, que necesita
cupy): RAFT cumple el mismo papel y corre en la GPU sin dependencias extra.

  f2f    Ehret et al., CVPR 2019 (Frame2Frame). Entrada: el frame t; objetivo: el frame vecino t+-1
         alineado a t con flujo optico, con mascara de oclusion (consistencia ida-vuelta del flujo),
         perdida L1 solo donde no hay oclusion. El articulo ajusta una red ya entrenada; aqui se
         entrena desde cero con nuestros frames. Flujo calculado sobre los frames suavizados.
  mf2f   Dewil et al., CVPR 2021 (Multi-Frame2Frame, centreborelli/mf2f): la red recibe VARIOS frames
         (t-2, t, t+2) y el objetivo es t+-1 alineado (que NO esta en la entrada, asi el ruido del
         objetivo es independiente). Misma perdida.
  saved  Stathatos et al., WACV 2026 (suzanne-stathatos/SAVeD, Apache-2.0). Red DenoisingCNN_512_with_skips
         oficial (bottleneck 512, fine_layers), entrada (t, t-1, t-2), objetivo "gap1_and_curr_frame"
         = t + relu(t+1 - t) + relu(t-1 - t), MSE. El repo trabaja en gris (sonar): aqui se aplica al
         brillo (Y) y se conserva el color original.
  f2r    Ji et al., ECCV 2026 (m1NGGi/F2R). Redes oficiales Fusion de la etapa 1 (estimador temporal
         ciego) y etapa 2 (refinador espacial), T = 9 frames, L1, lr 3e-4. Adaptaciones necesarias
         para que sea autosupervisado con ruido real: (1) la "limpieza gruesa" la hace nuestro modelo
         autosupervisado del mismo regimen (--f2r-base, p.ej. nb2nb_r3) en vez de la NAFNet supervisada;
         (2) el ruido de los pseudo-pares de la etapa 2 no se sintetiza con un modelo conocido: se toma el
         residuo real del frame (ruidoso - grueso) desplazado y volteado al azar; (3) flujo RAFT.
         La primera mitad de los pasos entrena la etapa 1 y la segunda la etapa 2 (etapa 1 congelada).
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from fac_denoise.metodos import METODOS, Metodo, _unet
from fac_denoise.redes import UNet

_RAFT = {}


def _raft(dev):
    if dev not in _RAFT:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        m = raft_small(weights=Raft_Small_Weights.DEFAULT).to(dev).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _RAFT[dev] = m
    return _RAFT[dev]


@torch.no_grad()
def flujo(a, b, suavizar=1.5):
    """Flujo de a hacia b (en la grilla de a), Bx2xHxW en pixeles. a, b en [0,1], H y W multiplos de 8."""
    if suavizar > 0:
        k = int(2 * round(3 * suavizar) + 1)
        a = _gauss(a, suavizar, k)
        b = _gauss(b, suavizar, k)
    if a.shape[1] == 1:
        a, b = a.repeat(1, 3, 1, 1), b.repeat(1, 3, 1, 1)
    h, w = a.shape[-2:]  # RAFT pide lados multiplos de 8 y de al menos 128
    ph, pw = max(128, -(-h // 8) * 8) - h, max(128, -(-w // 8) * 8) - w
    if ph or pw:
        a, b = F.pad(a, (0, pw, 0, ph), mode="replicate"), F.pad(b, (0, pw, 0, ph), mode="replicate")
    with torch.autocast(a.device.type, enabled=False):
        return _raft(a.device)(a.float() * 2 - 1, b.float() * 2 - 1, num_flow_updates=12)[-1][..., :h, :w]


def _gauss(x, s, k):
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-t ** 2 / (2 * s * s))
    g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)


def alinear(x, f):
    """Muestrea x en grilla + f (backward warping). Devuelve (alineado, validos)."""
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    gx = (xx[None] + f[:, 0]) / max(w - 1, 1) * 2 - 1
    gy = (yy[None] + f[:, 1]) / max(h - 1, 1) * 2 - 1
    g = torch.stack([gx, gy], -1).to(x.dtype)
    out = F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)
    dentro = ((gx.abs() <= 1) & (gy.abs() <= 1))[:, None]
    return out, dentro


@torch.no_grad()
def mascara_oclusion(ft, fv):
    """Consistencia ida-vuelta (Sundaram et al. 2010): ft = flujo t->v, fv = flujo v->t."""
    fv_en_t, dentro = alinear(fv, ft)
    s = (ft + fv_en_t).square().sum(1, keepdim=True)
    lim = 0.01 * (ft.square().sum(1, keepdim=True) + fv_en_t.square().sum(1, keepdim=True)) + 0.5
    return ((s < lim) & dentro).float()


def _objetivo_vecino(centro, vecino):
    ft, fv = flujo(centro, vecino), flujo(vecino, centro)
    obj, _ = alinear(vecino, ft)
    return obj, mascara_oclusion(ft, fv)


def _frame(x, i, c):
    return x[:, i * c:(i + 1) * c]


# ---------------------------------------------------------------------------
class F2F(Metodo):
    nombre, temporal, multiplo, lr_defecto = "f2f", 3, 32, 1e-4

    def crear(self, a, c):
        return _unet(a, c)

    def perdida(self, red, x, progreso, a):
        c = x.shape[1] // 3
        centro = _frame(x, 1, c)
        vec = _frame(x, random.choice((0, 2)), c)
        obj, m = _objetivo_vecino(centro, vec)
        out = red(centro).float()
        return (torch.abs(out - obj) * m).sum() / (m.sum() * c + 1e-6)

    def aplicar(self, red, x, a):
        c = x.shape[1] // 3
        return red(_frame(x, 1, c))


class MF2F(Metodo):
    nombre, temporal, multiplo, lr_defecto = "mf2f", 5, 32, 1e-4

    def crear(self, a, c):
        return UNet(in_nc=3 * c, out_nc=c, n_feature=a.unet_canales)

    def _entrada(self, x, c):
        return torch.cat([_frame(x, 0, c), _frame(x, 2, c), _frame(x, 4, c)], 1)

    def perdida(self, red, x, progreso, a):
        c = x.shape[1] // 5
        centro = _frame(x, 2, c)
        obj, m = _objetivo_vecino(centro, _frame(x, random.choice((1, 3)), c))
        out = red(self._entrada(x, c)).float()
        return (torch.abs(out - obj) * m).sum() / (m.sum() * c + 1e-6)

    def aplicar(self, red, x, a):
        return red(self._entrada(x, x.shape[1] // 5))


# ---------------------------------------------------------------------------
def _y(rgb):
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def _con_y(rgb, y_nueva):
    """Reemplaza el brillo de rgb por y_nueva conservando Cb y Cr (BT.601)."""
    y = _y(rgb)
    cb, cr = (rgb[:, 2:3] - y) * 0.564, (rgb[:, 0:1] - y) * 0.713
    r = y_nueva + 1.403 * cr
    b = y_nueva + 1.773 * cb
    g = (y_nueva - 0.299 * r - 0.114 * b) / 0.587
    return torch.cat([r, g, b], 1)


class SAVeD(Metodo):
    nombre, temporal, multiplo, lr_defecto = "saved", 5, 16, 1e-4
    parche_defecto, batch_defecto = 256, 16
    optimizador = "adamw"

    def crear(self, a, c):
        from fac_denoise.terceros import cargar
        return cargar("saved", "DenoisingCNN").DenoisingCNN_512_with_skips(fine_layers=True, bottleneck_size=512)

    def _y_frames(self, x):
        c = x.shape[1] // 5
        return [_y(_frame(x, i, c)) if c == 3 else _frame(x, i, c) for i in range(5)]

    def perdida(self, red, x, progreso, a):
        tm2, tm1, t, tp1, _ = self._y_frames(x)
        obj = t + F.relu(tp1 - t) + F.relu(tm1 - t)
        return F.mse_loss(red(torch.cat([t, tm1, tm2], 1)).float(), obj)

    def aplicar(self, red, x, a):
        tm2, tm1, t, _, _ = self._y_frames(x)
        y = red(torch.cat([t, tm1, tm2], 1)).float()
        c = x.shape[1] // 5
        centro = _frame(x, 2, c)
        return _con_y(centro, y) if c == 3 else y


# ---------------------------------------------------------------------------
class _F2RRed(nn.Module):
    def __init__(self, base, base_met, base_conf, T, dim):
        super().__init__()
        from fac_denoise.terceros import cargar
        s1 = cargar("f2r", "fusion_s1").Fusion(t=T - 1, dim=dim)
        s2 = cargar("f2r", "fusion_s2").Fusion(t=T - 1, dim=dim)
        self.s1, self.s2, self.base = s1, s2, base
        self.base_met, self.base_conf, self.T = base_met, base_conf, T
        for p in self.base.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def grueso(self, frames):  # frames: (B*T) x C x H x W en [0,1]
        self.base.eval()
        e = self.base_met.escala
        return (self.base_met.aplicar(self.base, frames * e, self.base_conf).float() / e).clamp(0, 1)

    @torch.no_grad()
    def flujos_a_ref(self, g, ref):
        b, t = g.shape[:2]
        fl = [flujo(g[:, ref], g[:, i]) for i in range(t) if i != ref]
        return torch.stack(fl, 1).flatten(0, 1)


class F2R(Metodo):
    nombre, temporal, multiplo, lr_defecto = "f2r", 9, 8, 3e-4
    parche_defecto, batch_defecto = 128, 4

    @staticmethod
    def argumentos(p):
        p.add_argument("--f2r-base", default="", help="checkpoint autosupervisado para la limpieza gruesa")
        p.add_argument("--f2r-dim", type=int, default=32)

    def crear(self, a, c):
        from argparse import Namespace
        ck = torch.load(a.f2r_base, map_location="cpu", weights_only=False)
        bm = METODOS[ck["metodo"]]
        bc = Namespace(**ck["args"])
        base = bm.crear(bc, int(ck["canales"]))
        base.load_state_dict(ck["modelo"], strict=True)
        print(f"f2r: limpieza gruesa con {a.f2r_base} ({ck['metodo']})")
        return _F2RRed(base, bm, bc, self.temporal, a.f2r_dim)

    def _partes(self, red, x):
        b, tc, h, w = x.shape
        T = red.T
        c = tc // T
        ruid = x.view(b, T, c, h, w)
        g = red.grueso(ruid.flatten(0, 1)).view(b, T, c, h, w)
        ref = T // 2
        return ruid, g, ref, red.flujos_a_ref(g, ref), c

    def perdida(self, red, x, progreso, a):
        ruid, g, ref, fl, c = self._partes(red, x)
        b, T, _, h, w = ruid.shape
        no_ref = [i for i in range(T) if i != ref]
        ent = torch.cat([g, ruid - g], 2)
        if progreso < 0.5:  # etapa 1: estimador temporal ciego (el frame t no entra)
            out = red.s1(ent[:, no_ref].reshape(-1, 2 * c, h, w), g[:, ref], fl)
            return torch.mean(torch.abs(out.float() - ruid[:, ref]))
        with torch.no_grad():  # etapa 2: pseudo-pares sobre el ancla de la etapa 1
            ancla = red.s1(ent[:, no_ref].reshape(-1, 2 * c, h, w), g[:, ref], fl).float()
            res = ruid[:, ref] - g[:, ref]
            res = torch.roll(res, shifts=(random.randrange(h), random.randrange(w)), dims=(-2, -1))
            if random.random() < 0.5:
                res = res.flip(-1)
            if random.random() < 0.5:
                res = res.flip(-2)
            ps = ruid.clone()
            ps[:, ref] = (ancla + res).clamp(0, 1)
            pg = red.grueso(ps.flatten(0, 1)).view_as(ps)
            pfl = red.flujos_a_ref(pg, ref)
            pent = torch.cat([pg, ps - pg], 2)
        out = red.s2(pent.reshape(-1, 2 * c, h, w), pg[:, ref], pfl)
        return torch.mean(torch.abs(out.float() - ancla))

    def aplicar(self, red, x, a):
        ruid, g, ref, fl, c = self._partes(red, x)
        b, T, _, h, w = ruid.shape
        ent = torch.cat([g, ruid - g], 2)
        return red.s2(ent.reshape(-1, 2 * c, h, w), g[:, ref], fl)


TANDA3 = [F2F(), MF2F(), SAVeD(), F2R()]
METODOS.update({m.nombre: m for m in TANDA3})
