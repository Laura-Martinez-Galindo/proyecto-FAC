"""
RONDA 3, tanda 4: los metodos que faltaban de las investigaciones.

Codigo oficial (fac_denoise/terceros):
  sass         Li et al., CVPR 2023 (nagejacob/SpatiallyAdaptiveSSID, GPL-3.0). Tres etapas del
               model/three_stage.py oficial: BNN (punto ciego 9), LAN (punto ciego 3, imita a BNN) y UNet
               que mezcla ambos segun la desviacion local (generate_alpha). Escala 0-255, L1, lr 3e-4,
               parche 256, batch 8. Los pasos se reparten en tercios.
  nsp          Shan et al., CVPR 2026 (XLearning-SCU/2026-CVPR-NSP). Red TBSN_NSP oficial; entrenamiento
               "next-scale": de cada bloque 5x5 se toma 1 pixel como entrada y un grupo 2x2 contiguo como
               objetivo a doble resolucion (generate_mask_pair / generate_imgs_pair, vectorizado aqui);
               L1 (criterion del repo), escala 0-255, parche 640, batch 4, lr 1e-4. Inferencia:
               pixel-unshuffle 2, red, promedio (pixel_shuffle con sr_rec=False).
  tmbsn        Park et al., CVPR 2026 (parkjun210/TM-BSN). Red TMBSN oficial (48 canales, 5 bloques),
               L1 con hueco 5 al entrenar y 2 al inferir, parche 128, batch 4, lr 1e-4. (Sin la
               destilacion a la UNet alumna.)
  filter2noise Sun et al., 2026 (sypsyp97/Filter2Noise). Zero-shot por imagen: DenoisingPipeline de 2
               etapas AGBF + LossFunction oficial (lambda 400), AdamW 1e-3, OneCycle, 500 iteraciones.
               Es para imagenes en gris (TC): se aplica al brillo (Y) y se conserva el color.
  s2splus      Ko et al., 2023 (JK-the-Ko/Self2SelfPlus). Zero-shot por imagen: red DAE con conv.
               compuertas, perdida L1 enmascarada + IQA (paq2piq de pyiqa, lambda 2e-8), Adam 4e-4,
               4.000 iteraciones; inferencia con --s2sp-muestras mascaras (el repo usa 500).
  tap          Fu et al., ECCV 2024 (zfu006/TAP). Red Baseline_Video oficial (NAFNet + alineacion PCD con
               convoluciones deformables), 5 frames. El repo ajusta con pseudo-limpias generadas por una
               red supervisada y ruido sintetico conocido; aqui (adaptacion): pseudo-limpia = nuestro
               modelo autosupervisado del regimen (--tap-base) y pseudo-ruido = residuo real de cada frame
               desplazado al azar. L1, parche 128, batch 8.
  pcst         Vaksman & Elad, CVPR 2023 (grishavak/pcst). Funciones oficiales find_nn / create_im: el
               objetivo de cada frame es un "patch-craft frame" armado con los parches mas parecidos de
               los OTROS frames (sin registro). Red UNet, MSE, lr 5e-5. Busqueda reducida a +-8 px y
               parche 7 (el repo usa 500 px y 15 sobre secuencias completas; aqui son recortes de 128).

Reimplementados del articulo (sin codigo publico; marcados como reimplementacion):
  mash         Chihaoui & Favaro, CVPR 2024. Zero-shot: dos redes base con tau 0,8 y 0,2 -> eps; tau optima
               (ec. 10) y, si eps > 2,5, Local Pixel Shuffling en zonas planas (s = 4). N = 800, K = 10.
               lr y lambda no los da el articulo: 1e-3 y la mediana de la desviacion local.
  zskan        Cheng et al., 2024 ("You KAN do it in a single shot"). Prior KAN (B-splines, grid 5, capas
               ocultas 128-32-16) entrenado 100 iteraciones sobre la propia imagen con ruido gaussiano
               anadido (0,1), sobre parches 5x5; luego PnP-ADMM con A = I (5 iteraciones).
  swinia       Papkov et al., WACV 2025. Autocodificador solo-atencion: K,V fijos de un solo parche
               (pixel-shuffle 1, 2, 4), consultas = embeddings posicionales, ventanas 8x8 desplazadas,
               mascara diagonal al entrenar y sin ella al inferir, dim 144, 16 cabezas, 3 encoders + 2
               decoders de 4 bloques, MSE sobre la imagen completa. (AdamW en vez de Lion.)
  selfformer   Quan et al., NeurIPS 2024 (Pseudo-Siamese Blind-spot Transformer). Atencion direccional en
               semiplano (4 ramas) sobre una rejilla dilatada (pixel-unshuffle 2) = rama ciega D; la rama F
               comparte pesos con atencion completa. L1(D, y) + L1(F, sg(D)). El articulo no detalla todo:
               reimplementacion aproximada.
  stbn         Chen et al., AAAI 2025. Vecinos alineados con flujo (RAFT) e interpolacion al vecino mas
               cercano (no correlaciona el ruido), propagacion bidireccional de caracteristicas de los
               vecinos + rama ciega espacial (convoluciones desplazadas, Laine et al. 2019) para el frame t.
               T = 7, MSE. Reimplementacion aproximada.
  rdrf         Wang et al., ACM MM 2023. Propagacion recurrente bidireccional SOLO con los vecinos (sin
               flujo explicito) + rama local ciega en (t-1, t, t+1) con el centro tapado. T = 9, MSE.
               Reimplementacion aproximada.
"""
from __future__ import annotations

import math
import random
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from fac_denoise.metodos import METODOS, Metodo, _unet, _sigma_lote, _interpolar
from fac_denoise.metodos_tanda2 import ZeroShot, _entrenar_zs
from fac_denoise.metodos_tanda3 import _con_y, _y, alinear, flujo


def _cargar_base(ruta):
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    bm = METODOS[ck["metodo"]]
    bc = Namespace(**ck["args"])
    red = bm.crear(bc, int(ck["canales"]))
    red.load_state_dict(ck["modelo"], strict=True)
    for p in red.parameters():
        p.requires_grad_(False)
    return red.eval(), bm, bc


@torch.no_grad()
def _aplicar_base(red, bm, bc, x):
    red.eval()
    return (bm.aplicar(red, x * bm.escala, bc).float() / bm.escala).clamp(0, 1)


def _residuo_barajado(r):
    h, w = r.shape[-2:]
    r = torch.roll(r, shifts=(random.randrange(h), random.randrange(w)), dims=(-2, -1))
    if random.random() < 0.5:
        r = r.flip(-1)
    if random.random() < 0.5:
        r = r.flip(-2)
    return r


# ===========================================================================
class SASS(Metodo):
    nombre, multiplo, escala, parche_defecto, batch_defecto, lr_defecto = "sass", 8, 255.0, 256, 8, 3e-4

    def crear(self, a, c):
        from fac_denoise.terceros import cargar
        n = cargar("sass", "network")
        r = nn.Module()
        r.bnn, r.lan, r.unet = n.BNN(blindspot=9, in_ch=c, out_ch=c), n.LAN(blindspot=3), n.UNet()
        return r

    @staticmethod
    def _std(img, k=7):
        img = img.mean(1, keepdim=True)
        p = k // 2
        u = F.unfold(F.pad(img, [p] * 4, mode="reflect"), k).view(img.shape[0], 1, k * k, *img.shape[-2:])
        return torch.sqrt(((u - u.mean(2, keepdim=True)) ** 2).mean(2))

    def _alpha(self, x, lo=1, hi=5):
        r = x.new_ones((x.shape[0], 1, *x.shape[-2:])) * 0.5
        s = self._std(x)
        r[s < lo] = torch.sigmoid(s - lo)[s < lo]
        r[s > hi] = torch.sigmoid(s - hi)[s > hi]
        return r.detach()

    def perdida(self, red, x, progreso, a):
        l1 = F.l1_loss
        if progreso < 1 / 3:
            return l1(red.bnn(x).float(), x)
        if progreso < 2 / 3:
            with torch.no_grad():
                b = red.bnn(x)
            return l1(b.float(), red.lan(x).float())
        with torch.no_grad():
            b, l = red.bnn(x), red.lan(x)
        u = red.unet(x).float()
        al = self._alpha(b)
        return l1(b * (1 - al), u * (1 - al)) + l1(l * al, u * al)

    def aplicar(self, red, x, a):
        return red.unet(x)


# ===========================================================================
class NSP(Metodo):
    nombre, multiplo, escala, parche_defecto, batch_defecto, lr_defecto = "nsp", 40, 255.0, 640, 4, 1e-4
    multiplo_mosaico = 64

    def crear(self, a, c):
        from fac_denoise.terceros import cargar
        return cargar("nsp", "nsp_tbsn").TBSN_NSP(in_ch=c, out_ch=c, ids=2)

    def perdida(self, red, x, progreso, a):
        b, c, h, w = x.shape
        p, s = 5, 2
        bl = F.unfold(x, p, stride=p).view(b, c, p * p, h // p, w // p)  # bloques p x p
        nb = (h // p) * (w // p)
        dev = x.device
        # grupo s x s contiguo al azar (topleft_pos del repo) y 1 pixel al azar fuera del grupo
        r0 = torch.randint(0, p - s + 1, (b, nb), device=dev)
        c0 = torch.randint(0, p - s + 1, (b, nb), device=dev)
        grupo = torch.stack([(r0 + i) * p + (c0 + j) for i in range(s) for j in range(s)], -1)  # b,nb,s*s
        fuera = torch.ones(b, nb, p * p, dtype=torch.bool, device=dev)
        fuera.scatter_(-1, grupo, False)
        pesos = fuera.float() + 1e-9
        sel = torch.multinomial(pesos.view(-1, p * p), 1).view(b, nb)
        blf = bl.view(b, c, p * p, nb)
        ent = torch.gather(blf, 2, sel[:, None, None, :].expand(b, c, 1, nb)).view(b, c, h // p, w // p)
        obj = torch.gather(blf, 2, grupo.permute(0, 2, 1)[:, None].expand(b, c, s * s, nb))  # b,c,s*s,nb
        obj = obj.view(b, c, s, s, h // p, w // p).permute(0, 1, 4, 2, 5, 3).reshape(b, c, h // p * s, w // p * s)
        return F.l1_loss(red(ent).float(), obj)

    def aplicar(self, red, x, a):
        f = 2
        b, c, h, w = x.shape
        sub = x.view(b, c, h // f, f, w // f, f).permute(0, 3, 5, 1, 2, 4).reshape(b * f * f, c, h // f, w // f)
        out = red(sub).float()  # b*f*f, c, h, w
        return out.view(b, f * f, c, h, w).mean(1)


# ===========================================================================
class TMBSNm(Metodo):
    nombre, multiplo, parche_defecto, batch_defecto, lr_defecto = "tmbsn", 8, 128, 4, 1e-4
    multiplo_mosaico = 32

    def crear(self, a, c):
        from fac_denoise.terceros import cargar
        return cargar("tmbsn", "TMBSN").TMBSN(in_channels=c, feat_channels=48, block_nums=5)

    def perdida(self, red, x, progreso, a):
        return F.l1_loss(red(x, 5).float(), x)

    def aplicar(self, red, x, a):
        return red(x, 2)


# ===========================================================================
class Filter2NoiseM(ZeroShot):
    nombre = "filter2noise"

    @staticmethod
    def argumentos(p):
        p.add_argument("--f2n-iter", type=int, default=500)
        p.add_argument("--f2n-lambda", type=float, default=400.0)

    def zero_shot_imagen(self, x, a, disp):
        from fac_denoise.terceros import cargar
        mod = cargar("filter2noise", "filter2noise")
        y = _y(x)
        h, w = y.shape[-2:]
        yp = F.pad(y, (0, (-w) % 16, 0, (-h) % 16), mode="reflect")
        red = mod.DenoisingPipeline(num_stages=2, patch_size=8).to(disp)
        opt = torch.optim.AdamW(red.parameters(), lr=1e-3, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=a.f2n_iter, steps_per_epoch=1)
        perd = mod.LossFunction(disp, lambda_=a.f2n_lambda)
        for _ in range(a.f2n_iter):
            opt.zero_grad(set_to_none=True)
            perd(yp, red).backward()
            opt.step()
            sch.step()
        with torch.no_grad():
            yd = red(yp)[..., :h, :w]
        return _con_y(x, yd).clamp(0, 1)


class S2SPlus(ZeroShot):
    nombre = "s2splus"

    @staticmethod
    def argumentos(p):
        p.add_argument("--s2sp-iter", type=int, default=4000)
        p.add_argument("--s2sp-muestras", type=int, default=50)

    def zero_shot_imagen(self, x, a, disp):
        from fac_denoise.terceros import cargar
        mod = cargar("s2splus", "models.models")
        h, w = x.shape[-2:]
        xp = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
        o = Namespace(seed=42, p=0.4, inputDim=x.shape[1], channels=48, initType="normal",
                      numSample=a.s2sp_muestras, lossType="L1", lambdaIQA=2e-8, numIters=a.s2sp_iter)
        red = mod.Self2SelfPlus(o).to(disp)
        opt = torch.optim.Adam(red.parameters(), lr=4e-4, betas=(0.9, 0.999))
        for _ in range(a.s2sp_iter):
            opt.zero_grad(set_to_none=True)
            sum(red(xp, "train").values()).backward()
            opt.step()
        return red(xp, "inference")[..., :h, :w].clamp(0, 1)


# ===========================================================================
class TAPm(Metodo):
    nombre, temporal, multiplo, parche_defecto, batch_defecto, lr_defecto = "tap", 5, 8, 128, 8, 1e-4

    @staticmethod
    def argumentos(p):
        p.add_argument("--tap-base", default="", help="checkpoint autosupervisado para las pseudo-limpias")

    def crear(self, a, c):
        from fac_denoise.terceros import cargar
        net = cargar("tap", "models.network_nafnet").Baseline_Video(
            n_frames=5, img_channel=c, width=64, middle_blk_num=6, enc_blk_nums=[2, 2, 4],
            dec_blk_nums=[2, 2, 2], dw_expand=2, ffn_expand=2, current_stage=3, total_stages=3)
        r = nn.Module()
        r.net = net
        r.base, self._bm, self._bc = _cargar_base(a.tap_base)
        return r

    def _sec(self, x):
        b, tc, h, w = x.shape
        return x.view(b, 5, tc // 5, h, w)

    def perdida(self, red, x, progreso, a):
        s = self._sec(x)
        b, t, c, h, w = s.shape
        with torch.no_grad():
            limpio = _aplicar_base(red.base, self._bm, self._bc, s.flatten(0, 1)).view_as(s)
            ps = torch.stack([(limpio[:, i] + _residuo_barajado(s[:, i] - limpio[:, i])).clamp(0, 1)
                              for i in range(t)], 1)
        return F.l1_loss(red.net(ps).float(), limpio[:, t // 2])

    def aplicar(self, red, x, a):
        return red.net(self._sec(x))


# ===========================================================================
class PCST(Metodo):
    nombre, temporal, multiplo, parche_defecto, batch_defecto, lr_defecto = "pcst", 5, 32, 128, 16, 5e-5

    @staticmethod
    def argumentos(p):
        p.add_argument("--pcst-parche", type=int, default=7)
        p.add_argument("--pcst-busqueda", type=int, default=8)
        p.add_argument("--pcst-salto", type=int, default=2)

    def crear(self, a, c):
        return _unet(a, c)

    @torch.no_grad()
    def _craft(self, seq, a):
        """seq: C x T x H x W (frame t al centro). data/preprocess_unknown_noise.py del repo."""
        from fac_denoise.terceros import cargar
        pu = cargar("pcst", "patchcraft_utils")
        ps, so = a.pcst_parche, a.pcst_busqueda
        o = Namespace(patch_size=ps, search_offs=so, search_skip=a.pcst_salto, search_offs2=None)
        pad = F.pad(seq[None], (ps - 1, ps - 1, ps - 1, ps - 1, 0, 0), mode="reflect")
        pad = F.pad(pad, (so, so, so, so, 0, 0), mode="constant", value=-1)[0]
        min_i = pu.find_nn(o, pad)
        return pu.create_im(o, pad, min_i, random.randrange(ps * ps))

    def perdida(self, red, x, progreso, a):
        b, tc, h, w = x.shape
        t = 5
        c = tc // t
        s = x.view(b, t, c, h, w)
        obj = torch.stack([self._craft(s[i].permute(1, 0, 2, 3), a) for i in range(b)])
        m = torch.isfinite(obj).float()
        obj = torch.nan_to_num(obj)
        out = red(s[:, t // 2]).float()
        return ((out - obj) ** 2 * m).sum() / (m.sum() + 1e-6)

    def aplicar(self, red, x, a):
        c = x.shape[1] // 5
        return red(x[:, 2 * c:3 * c])


# ===========================================================================
class MASH(ZeroShot):
    nombre = "mash"

    @staticmethod
    def argumentos(p):
        p.add_argument("--mash-iter", type=int, default=800)

    def zero_shot_imagen(self, x, a, disp):
        h, w = x.shape[-2:]
        y = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
        a2 = Namespace(unet_canales=48)

        def entrenar(tau, it, objetivo=None, red=None):
            red = red or _unet(a2, x.shape[1]).to(disp)
            opt = torch.optim.Adam(red.parameters(), lr=1e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, it)
            obj = y if objetivo is None else objetivo
            for _ in range(it):
                m = (torch.rand_like(y) > tau).float()
                opt.zero_grad(set_to_none=True)
                (((1 - m) * (red(y * m) - obj)) ** 2).mean().backward()
                opt.step()
                sch.step()
            return red

        def sigma(red, tau):
            with torch.no_grad():
                m = (torch.rand_like(y) > tau).float()
                return float(torch.sqrt(((red(m * y) - y) ** 2).mean()) * 255)

        eps = abs(sigma(entrenar(0.8, a.mash_iter), 0.8) - sigma(entrenar(0.2, a.mash_iter), 0.2))
        tau = 0.2 if eps <= 1.5 else (0.5 if eps < 2.5 else 0.8)
        if eps <= 2.5:
            red = entrenar(tau, a.mash_iter)
        else:  # Local Pixel Shuffling en zonas planas
            n1 = a.mash_iter // 2
            red = entrenar(tau, n1)
            with torch.no_grad():
                m = (torch.rand_like(y) > tau).float()
                pseudo = red(y * m)
                s = 4
                b, c, H, W = y.shape
                std = F.unfold(pseudo, s, stride=s).view(b, c, s * s, -1).std(2).mean(1)  # b x n
                plano = std < std.median()
                t = F.unfold(y, s, stride=s).view(b, c, s * s, -1)
                perm = torch.argsort(torch.rand(b, 1, s * s, t.shape[-1], device=disp), dim=2).expand_as(t)
                t2 = torch.where(plano[:, None, None, :], torch.gather(t, 2, perm), t)
                y_sh = F.fold(t2.view(b, c * s * s, -1), (H, W), s, stride=s)
            red = entrenar(tau, a.mash_iter - n1, objetivo=y_sh, red=red)
        with torch.no_grad():
            out = sum(red(y * (torch.rand_like(y) > tau).float()) for _ in range(10)) / 10
        return out[..., :h, :w].clamp(0, 1)


# ===========================================================================
class _KANCapa(nn.Module):
    """KAN con B-splines (version eficiente: SiLU base + splines cubicos, grid 5)."""

    def __init__(self, din, dout, grid=5, k=3, rango=(-1, 1)):
        super().__init__()
        hgrid = (rango[1] - rango[0]) / grid
        g = torch.arange(-k, grid + k + 1).float() * hgrid + rango[0]
        self.register_buffer("g", g.expand(din, -1).contiguous())
        self.k = k
        self.base = nn.Linear(din, dout)
        self.spline = nn.Parameter(torch.randn(dout, din, grid + k) * 0.1)

    def _bsplines(self, x):
        g = self.g
        x = x[..., None]
        B = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for p in range(1, self.k + 1):
            B = ((x - g[:, :-(p + 1)]) / (g[:, p:-1] - g[:, :-(p + 1)]) * B[..., :-1]
                 + (g[:, p + 1:] - x) / (g[:, p + 1:] - g[:, 1:(-p)]) * B[..., 1:])
        return B

    def forward(self, x):
        return self.base(F.silu(x)) + torch.einsum("nik,oik->no", self._bsplines(x), self.spline)


class ZSKAN(ZeroShot):
    nombre = "zskan"

    def zero_shot_imagen(self, x, a, disp):
        c = x.shape[1]
        k = 5
        red = nn.Sequential(_KANCapa(c * k * k, 128), _KANCapa(128, 32), _KANCapa(32, 16), nn.Linear(16, c)).to(disp)
        h, w = x.shape[-2:]

        def D(img):
            v = F.unfold(F.pad(img, (2, 2, 2, 2), mode="reflect"), k).transpose(1, 2).reshape(-1, c * k * k)
            return torch.cat([red(v[i:i + 65536] * 2 - 1) for i in range(0, v.shape[0], 65536)]).view(1, h, w, c).permute(0, 3, 1, 2)

        opt = torch.optim.Adam(red.parameters(), lr=1e-3)
        v = F.unfold(F.pad(x, (2, 2, 2, 2), mode="reflect"), k).transpose(1, 2).reshape(-1, c * k * k)
        cen = x.permute(0, 2, 3, 1).reshape(-1, c)
        for _ in range(100):
            idx = torch.randint(0, v.shape[0], (16384,), device=disp)
            opt.zero_grad(set_to_none=True)
            F.mse_loss(red((v[idx] + 0.1 * torch.randn_like(v[idx])) * 2 - 1), cen[idx]).backward()
            opt.step()
        with torch.no_grad():  # PnP-ADMM con A = I, mu en descenso logaritmico
            z, u = x.clone(), torch.zeros_like(x)
            for mu in torch.logspace(0, -1, 5).tolist():
                xk = D(z - u)
                z = (x + mu * (xk + u)) / (1 + mu)
                u = u + xk - z
        return xk.clamp(0, 1)


# ===========================================================================
# SwinIA (reimplementacion)
def _ventanas(t, ws):
    b, d, h, w = t.shape
    return t.view(b, d, h // ws, ws, w // ws, ws).permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, d)


def _desventanas(t, b, h, w, ws):
    d = t.shape[-1]
    return t.view(b, h // ws, w // ws, ws, ws, d).permute(0, 5, 1, 3, 2, 4).reshape(b, d, h, w)


class _BloqueQ(nn.Module):
    def __init__(self, d, heads, ws, desplazar):
        super().__init__()
        self.nq, self.nm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d) for _ in range(4))
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h, self.ws, self.s = heads, ws, ws // 2 if desplazar else 0

    def forward(self, q, kv, ciego):
        b, d, H, W = q.shape
        if self.s:
            q, kv = (torch.roll(t, (-self.s, -self.s), (2, 3)) for t in (q, kv))
        qw, kw = _ventanas(q, self.ws), _ventanas(kv, self.ws)
        n = qw.shape[1]
        sp = lambda t: t.view(t.shape[0], n, self.h, -1).transpose(1, 2)
        Q, K, V = sp(self.wq(self.nq(qw))), sp(self.wk(kw)), sp(self.wv(kw))
        att = (Q @ K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
        if ciego:
            att = att - torch.eye(n, device=q.device, dtype=att.dtype) * 1e9
        o = (att.softmax(-1) @ V).transpose(1, 2).reshape(-1, n, d)
        qw = qw + self.wo(o)
        qw = qw + self.mlp(self.nm(qw))
        q = _desventanas(qw, b, H, W, self.ws)
        return torch.roll(q, (self.s, self.s), (2, 3)) if self.s else q


class _SwinIA(nn.Module):
    def __init__(self, c=3, d=144, heads=16, ws=8, bloques=4):
        super().__init__()
        self.ws, self.d = ws, d
        self.kv = nn.ModuleDict({str(p): nn.Linear(c * p * p, d) for p in (1, 2, 4)})
        self.pos = nn.ParameterDict({str(p): nn.Parameter(torch.zeros(1, d, ws, ws)) for p in (1, 2, 4)})
        self.qpos = nn.ParameterDict({str(p): nn.Parameter(torch.randn(1, d, ws, ws) * 0.02) for p in (1, 2, 4)})
        self.nkv = nn.LayerNorm(d)
        mk = lambda: nn.ModuleList([_BloqueQ(d, heads, ws, i % 2 == 1) for i in range(bloques)])
        self.enc = nn.ModuleDict({str(p): mk() for p in (1, 2, 4)})
        self.dec = nn.ModuleDict({"2": mk(), "1": mk()})
        self.up = nn.ModuleDict({"2": nn.Linear(d, 4 * d), "1": nn.Linear(d, 4 * d)})
        self.fus = nn.ModuleDict({"2": nn.Linear(2 * d, d), "1": nn.Linear(2 * d, d)})
        self.out = nn.Linear(d, c)

    def _kv(self, x, p):
        t = F.pixel_unshuffle(x, p) if p > 1 else x
        b, _, h, w = t.shape
        e = self.kv[str(p)](t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + self.pos[str(p)].repeat(1, 1, h // self.ws, w // self.ws)
        return self.nkv(e.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _lin(self, m, t):
        return m(t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x, ciego=True):
        xs = (x - 0.5) / 0.25
        b, _, H, W = x.shape
        kv, q = {}, {}
        for p in (1, 2, 4):
            kv[p] = self._kv(xs, p)
            h, w = H // p, W // p
            qq = self.qpos[str(p)].repeat(b, 1, h // self.ws, w // self.ws)
            for blk in self.enc[str(p)]:
                qq = blk(qq, kv[p], ciego)
            q[p] = qq
        cur = q[4]
        for p in (2, 1):
            up = F.pixel_shuffle(self._lin(self.up[str(p)], cur), 2)
            cur = self._lin(self.fus[str(p)], torch.cat([up, q[p]], 1))
            for blk in self.dec[str(p)]:
                cur = blk(cur, kv[p], ciego)
        return self._lin(self.out, cur) * 0.25 + 0.5


class SwinIA(Metodo):
    nombre, multiplo, parche_defecto, batch_defecto, lr_defecto = "swinia", 32, 64, 64, 3e-4
    multiplo_mosaico = 32
    optimizador = "adamw"

    def crear(self, a, c):
        return _SwinIA(c)

    def perdida(self, red, x, progreso, a):
        return F.mse_loss(red(x, ciego=True).float(), x)

    def aplicar(self, red, x, a):
        return red(x, ciego=False)  # "unblinding" al inferir


# ===========================================================================
# SelfFormer (reimplementacion aproximada)
class _BloqueDir(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.h = heads

    def forward(self, t, mascara, residuo=True, qfuente=None):
        bn, n, d = t.shape
        q, k, v = self.qkv(self.n1(t)).view(bn, n, 3, self.h, -1).permute(2, 0, 3, 1, 4)
        if qfuente is not None:  # primera capa ciega: la consulta NO puede depender del propio pixel
            q = self.qkv(self.n1(qfuente.expand(bn, -1, -1))).view(bn, n, 3, self.h, -1).permute(2, 0, 3, 1, 4)[0]
        att = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if mascara is not None:
            att = att.masked_fill(~mascara, -1e9)
        o = self.o((att.softmax(-1) @ v).transpose(1, 2).reshape(bn, n, d))
        if mascara is not None:  # una consulta sin claves permitidas (fila de arriba) no recibe nada
            o = o * mascara.any(-1).to(o.dtype)[None, :, None]
        t = t + o if residuo else o
        return t + self.mlp(self.n2(t))


class _SelfFormer(nn.Module):
    def __init__(self, c=3, d=64, heads=4, ws=8, capas=4):
        super().__init__()
        self.ws = ws
        self.emb = nn.Linear(c, d)
        self.qpos = nn.Parameter(torch.randn(1, ws * ws, d) * 0.02)
        self.bloques = nn.ModuleList([_BloqueDir(d, heads) for _ in range(capas)])
        self.fus_d = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, c))  # 1x1 de la rama ciega
        self.fus_f = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, c))      # 1x1 de la rama completa
        fila = torch.arange(ws).repeat_interleave(ws)
        self.register_buffer("estricta", fila[None, :] < fila[:, None])    # j estrictamente arriba de i
        self.register_buffer("inclusiva", fila[None, :] <= fila[:, None])

    def _rama(self, x, ciega):
        b, c, h, w = x.shape
        t = _ventanas(self.emb(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), self.ws)
        for i, blk in enumerate(self.bloques):
            if ciega:
                t = blk(t, self.estricta if i == 0 else self.inclusiva, residuo=i > 0,
                        qfuente=self.qpos if i == 0 else None)
            else:
                t = blk(t, None)
        return _desventanas(t, b, h, w, self.ws)

    def forward(self, x, ciega=True):
        b, c, H, W = x.shape
        g = F.pixel_unshuffle(x, 2).view(b, c, 4, H // 2, W // 2).permute(0, 2, 1, 3, 4).reshape(b * 4, c, H // 2, W // 2)
        if ciega:
            fs = [torch.rot90(self._rama(torch.rot90(g, r, (2, 3)), True), -r, (2, 3)) for r in range(4)]
            o = self.fus_d(torch.cat(fs, 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            o = self.fus_f(self._rama(g, False).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        o = o.view(b, 4, c, H // 2, W // 2).permute(0, 2, 1, 3, 4).reshape(b, c * 4, H // 2, W // 2)
        return F.pixel_shuffle(o, 2)


class SelfFormer(Metodo):
    nombre, multiplo, parche_defecto, batch_defecto, lr_defecto = "selfformer", 32, 128, 8, 2e-4
    multiplo_mosaico = 32

    def crear(self, a, c):
        return _SelfFormer(c)

    def perdida(self, red, x, progreso, a):
        d = red(x, ciega=True).float()
        f = red(x, ciega=False).float()
        return F.l1_loss(d, x) + F.l1_loss(f, d.detach())

    def aplicar(self, red, x, a):
        return red(x, ciega=False)


# ===========================================================================
# STBN / RDRF (reimplementaciones aproximadas). Rama ciega espacial de Laine et al. (2019):
# convoluciones con desplazamiento vertical (el campo receptivo queda estrictamente arriba del
# pixel), en 4 rotaciones con pesos compartidos.
class _ConvArriba(nn.Module):
    def __init__(self, ci, co, dil=1):
        super().__init__()
        self.c, self.d = nn.Conv2d(ci, co, 3, dilation=dil), dil

    def forward(self, x):
        d = self.d
        return self.c(F.pad(x, (d, d, 2 * d, 0), mode="replicate"))  # solo filas de arriba y la propia


class _RamaCiega(nn.Module):
    def __init__(self, c, f=48):
        super().__init__()
        self.capas = nn.ModuleList([_ConvArriba(c, f)] + [_ConvArriba(f, f, d) for d in (1, 2, 4, 1)])

    def _arriba(self, x):
        for l in self.capas:
            x = F.leaky_relu(l(x), 0.1)
        return F.pad(x, (0, 0, 1, 0))[:, :, :-1]  # desplazar 1 fila: ya no ve su propia fila

    def forward(self, x):
        return torch.cat([torch.rot90(self._arriba(torch.rot90(x, r, (2, 3))), -r, (2, 3)) for r in range(4)], 1)


class _Res(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.a, self.b = nn.Conv2d(f, f, 3, padding=1), nn.Conv2d(f, f, 3, padding=1)

    def forward(self, x):
        return x + self.b(F.leaky_relu(self.a(x), 0.1))


class _Temporal(nn.Module):
    def __init__(self, c, T, f=48, con_flujo=True, local=False):
        super().__init__()
        self.T, self.con_flujo, self.local = T, con_flujo, local
        self.ciega = _RamaCiega(c, f)
        self.prop_f = nn.Sequential(nn.Conv2d(c + f, f, 3, padding=1), nn.LeakyReLU(0.1), _Res(f), _Res(f))
        self.prop_b = nn.Sequential(nn.Conv2d(c + f, f, 3, padding=1), nn.LeakyReLU(0.1), _Res(f), _Res(f))
        nloc = 2 * c if local else 0
        self.fus = nn.Sequential(nn.Conv2d(4 * f + 2 * f + nloc, 2 * f, 1), nn.LeakyReLU(0.1),
                                 nn.Conv2d(2 * f, f, 1), nn.LeakyReLU(0.1), nn.Conv2d(f, c, 1))
        self.f = f

    def _al(self, ref, img):
        if not self.con_flujo:
            return img
        fl = flujo(ref, img)
        b, _, h, w = img.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing="ij")
        gx = ((xx[None] + fl[:, 0]).round() / max(w - 1, 1)) * 2 - 1
        gy = ((yy[None] + fl[:, 1]).round() / max(h - 1, 1)) * 2 - 1
        return F.grid_sample(img, torch.stack([gx, gy], -1), mode="nearest", padding_mode="border", align_corners=True)

    def forward(self, x):
        b, tc, h, w = x.shape
        T, c = self.T, tc // self.T
        fr = x.view(b, T, c, h, w)
        r = T // 2
        cen = fr[:, r]
        hf = cen.new_zeros(b, self.f, h, w)
        for i in range(0, r):  # hacia adelante con t-r..t-1 (el frame t NO entra)
            hf = self.prop_f(torch.cat([self._al(cen, fr[:, i]), hf], 1))
        hb = cen.new_zeros(b, self.f, h, w)
        for i in range(T - 1, r, -1):
            hb = self.prop_b(torch.cat([self._al(cen, fr[:, i]), hb], 1))
        partes = [self.ciega(cen), hf, hb]
        if self.local:  # rama local de RDRF: t-1 y t+1 (el centro esta tapado)
            partes += [fr[:, r - 1], fr[:, r + 1]]
        return self.fus(torch.cat(partes, 1))


class STBN(Metodo):
    nombre, temporal, multiplo, parche_defecto, batch_defecto, lr_defecto = "stbn", 7, 8, 96, 8, 1e-4

    def crear(self, a, c):
        return _Temporal(c, 7, con_flujo=True)

    def perdida(self, red, x, progreso, a):
        c = x.shape[1] // 7
        return F.mse_loss(red(x).float(), x[:, 3 * c:4 * c])

    def aplicar(self, red, x, a):
        return red(x)


class RDRF(Metodo):
    nombre, temporal, multiplo, parche_defecto, batch_defecto, lr_defecto = "rdrf", 9, 8, 96, 8, 1e-4

    def crear(self, a, c):
        return _Temporal(c, 9, con_flujo=False, local=True)

    def perdida(self, red, x, progreso, a):
        c = x.shape[1] // 9
        return F.mse_loss(red(x).float(), x[:, 4 * c:5 * c])

    def aplicar(self, red, x, a):
        return red(x)


TANDA4 = [SASS(), NSP(), TMBSNm(), Filter2NoiseM(), S2SPlus(), TAPm(), PCST(), MASH(), ZSKAN(), SwinIA(),
          SelfFormer(), STBN(), RDRF()]
METODOS.update({m.nombre: m for m in TANDA4})
