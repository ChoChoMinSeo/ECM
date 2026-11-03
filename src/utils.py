import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import numpy as np
import pandas as pd

import torch.nn.functional as F
import matplotlib.colors as mcolors

class LogManager:
    def __init__(self, filepath):
        self.logfile = filepath

    def create(self, columns, **kwargs):
        with open(self.logfile, "w") as f:
            if "l_dataloader" in kwargs.keys():
                defaults = f"time\tepoch\tglobal step / {kwargs['l_dataloader']}\tlearning rate\t"
            else:
                defaults = f"time\tepoch\tglobal step\tlearning rate\t"
            f.write(defaults + columns)
            f.flush()

    def logging(self, text):
        with open(self.logfile, "a") as f:
            f.write(f"{text}\n")
            f.flush()

    def resume(self, target_step):
        df = pd.read_csv(self.logfile, sep=f"\t")
        df.drop(df[df[df.columns[2]] > target_step].index, inplace=True)
        df.to_csv(self.logfile, header=True, index=False, sep="\t")

    def target_epoch(self):
        df = pd.read_csv(self.logfile, sep=f"\t")
        return df['epoch'].max() + 1

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.sum = 0
        self.count = 0
        self.data_list = []

    def update(self, val):
        self.sum += val
        self.count += 1
        self.data_list.append(val)

    def avg(self):
        if self.count != 0:
            return self.sum / self.count
        else:
            return 0

    def init(self):
        self.sum = 0
        self.count = 0
        self.data_list = []

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def configure_optimizer(net, args):
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())

    optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer

def mkdir_(path):
    if not os.path.exists(path):
        os.mkdir(path)


def mkfulldir(path):
    if not isinstance(path, str):
        raise Exception('not str')

    if not '/' in path:
        mkdir_(path)
    else:
        subpath_list = path.split("/")
        subpath = subpath_list[0]
        if subpath != "":
            mkdir_(subpath)
        del subpath_list[0]

        for p in subpath_list:
            subpath = "/".join([subpath, p])
            mkdir_(subpath)

    return path

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def cal_psnr(a, b):
    return (-10 * torch.log10(((a - b) ** 2).mean(dim=[1, 2, 3]))).mean().item()

def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")

def make_diff_heatmap(target, test):
    diff = (target[0] - test[0]).detach().cpu().numpy()
    diff = np.array(np.power(np.sum(diff ** 2, axis=0), 1/3) * 255.0, dtype=np.uint8)
    heat_diff = torch.Tensor((cv2.applyColorMap(diff, cv2.COLORMAP_JET) / 255.0).transpose(2, 0, 1))
    heat_diff = torch.stack([heat_diff[2], heat_diff[1], heat_diff[0]], dim=0).unsqueeze(0)
    return heat_diff

task_colors = {
    0: '#4DAF4A',      # natural
    1: '#377EB8',     # document
    2: '#984EA3',    # Residue
    3: '#E41A1C',   # y1y2y3
    4: '#009E73',      # driving
    5: '#FF7F00',    # line drawing
}
def save_colored_segmentation_masks_blend(seg: torch.Tensor, save_dir: str, base_idx = 0, add_idx=1):
    """
    Save segmentation as RGB image using soft blending (K=6 assumed).

    Args:
        seg (Tensor): Tensor of shape (K, H, W), logits or scores per class
        save_dir (str): Path to save the output image (e.g., './output.png')
    """
    # 1. Apply softmax to get per-class probability
    probs = seg[0]

    # H, W = probs.shape[1], probs.shape[2]

    # 2. Convert colors to RGB
    color0 = np.array(mcolors.to_rgb(task_colors[base_idx]))  # class 0
    color1 = np.array(mcolors.to_rgb(task_colors[add_idx]))  # class 1

    # 3. Blend colors according to probabilities
    rgb_img = (probs[base_idx].cpu().numpy()[..., None] * color0[None,None,:] +
               probs[add_idx].cpu().numpy()[..., None] * color1[None,None,:])

    rgb_img = (rgb_img * 255).astype(np.uint8)

    # 4. Save
    img = Image.fromarray(rgb_img)
    img.save(save_dir)
    
def make_diff_RB(orig, rec, clip=False):
    """
    B x C x H x W
    """
    diff = ((orig[0] - rec[0]) ** 2).sum(dim=0, keepdim=True).detach().cpu()
    if clip:
        min_diff = diff.min()
        max_diff = diff.max()
        diff = (diff - min_diff) / max_diff

    diff_rgb = torch.cat([diff, torch.zeros_like(diff), 1 - diff], dim=0)
    return diff_rgb

def fix_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
