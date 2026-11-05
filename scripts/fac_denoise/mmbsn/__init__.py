"""
MM-BSN (Zhang et al., CVPRW 2023, "MM-BSN: Self-Supervised Image Denoising for Real-World with
Multi-Mask based on Blind-Spot Network"). https://github.com/dannie125/MM-BSN  (licencia MIT,
ver ../licencias/LICENSE_MM-BSN_MIT.txt).
MMBSN.py y masks.py copiados SIN cambios de model/ del repositorio oficial. La envoltura con
pixel-shuffle (PD) y R3 es la de model/get_model.py (clase BSN), identica a la de AP-BSN; lo unico agregado es recortar la salida de denoise() al tamano de entrada.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .MMBSN import MMBSN
from ..redes import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling


class BSN_MM(nn.Module):
    """Copia de model/get_model.py::BSN del repo oficial, solo con la rama type='MMBSN'."""

    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16,
                 in_ch=3, bsn_base_ch=128, DCL1_num=2, DCL2_num=7, mask_type='o_a45'):
        super().__init__()
        self.pd_a, self.pd_b, self.pd_pad = pd_a, pd_b, pd_pad
        self.R3, self.R3_T, self.R3_p = R3, R3_T, R3_p
        self.bsn = MMBSN(in_ch, in_ch, bsn_base_ch, DCL1_num, DCL2_num, mask_type)

    def forward(self, img, pd=None):
        if pd is None:
            pd = self.pd_a
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p, p, p, p))
        pd_img_denoised = self.bsn(pd_img)
        if pd > 1:
            return pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        p = self.pd_pad
        return pd_img_denoised[:, :, p:-p, p:-p]

    def denoise(self, x):
        b, c, h, w = x.shape
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
        img_pd_bsn = self.forward(img=x, pd=self.pd_b)
        if not self.R3:
            return img_pd_bsn[:, :, :h, :w]
        denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
        for t in range(self.R3_T):
            mask = torch.rand_like(x) < self.R3_p
            tmp_input = torch.clone(img_pd_bsn).detach()
            tmp_input[mask] = x[mask]
            p = self.pd_pad
            tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
            denoised[..., t] = self.bsn(tmp_input) if p == 0 else self.bsn(tmp_input)[:, :, p:-p, p:-p]
        return torch.mean(denoised, dim=-1)[:, :, :h, :w]
