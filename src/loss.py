from typing import List
import math
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim

class metric_ms_ssim(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, x):
        return 1 - ms_ssim(x, x_hat, data_range=1.0)


class metric_mse(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, x_hat, x):
        return self.mse(x, x_hat).mean(dim=[1, 2, 3])

def compute_aux_loss(aux_list: List, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss

        if backward is True and aux_loss.grad_fn is not None:
            aux_loss.backward()

    return aux_loss_sum


class R_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, num_pixels, y_likelihoods, z_likelihoods=None):
        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)
        if z_likelihoods is None:
            bpp_z = 0
        else:
            bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * num_pixels)

        return bpp_y + bpp_z, bpp_y, bpp_z


class D_Loss(nn.Module):
    def __init__(self, metric="mse"):
        super().__init__()
        assert metric in ["mse", "ms-ssim"]

        if metric == "mse":
            self.metric = metric_mse()
        else:
            self.metric = metric_ms_ssim()

    def forward(self, x_hat, x):
        return self.metric(x_hat=x_hat, x=x)
