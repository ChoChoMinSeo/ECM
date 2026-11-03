# Copyright 2026 MinSeo Cho
# SPDX-License-Identifier: MIT
# See LICENSE.txt in the repository root for the full license text.

import torch
import numpy as np
try:
    from .utils import expand_to_onehot
except ImportError:
    from utils import expand_to_onehot


PATHS = {
    "base": "./data/BMS/",
    "save": "./data/BMS/modes/"
}

for img_idx in range(1,24):
    display_scale = 0.5

    PATHS['img'] = PATHS['base'] + f'pad_png/{img_idx}.png'
    PATHS['mode'] = PATHS['save'] + f'{img_idx}.pt'

    clicked_map = np.array(torch.load(PATHS['mode'])[::16,::16], dtype = np.uint8)

    pixel_map = expand_to_onehot(clicked_map, grid_size=16, num_classes=3)
    torch.save(torch.from_numpy(pixel_map), PATHS['mode'])
    print("clicked_mask.pt 저장 완료!")