# Copyright 2026 MinSeo Cho
# SPDX-License-Identifier: MIT
# See LICENSE.txt in the repository root for the full license text.

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from .utils import onehot_to_clicked_map
except ImportError:
    from utils import onehot_to_clicked_map

parser = argparse.ArgumentParser(description='Inspect a manually annotated BMS mode map.')
parser.add_argument('--image', required=True)
parser.add_argument('--mode_map', required=True)
parser.add_argument('--output', default=None, help='Optional overlay PNG')
parser.add_argument('--no_gui', action='store_true', help='Validate and save without opening a window')
args = parser.parse_args()
PATHS = {'img': args.image, 'mode': args.mode_map}
img = cv2.imread(PATHS['img'])
if img is None:
    raise FileNotFoundError(PATHS['img'])
if img.shape[0] % 256 or img.shape[1] % 256:
    raise ValueError('Prepare the image with preprocess.py before annotation: H and W must be multiples of 256.')
pixel_map = torch.as_tensor(torch.load(PATHS['mode'], map_location='cpu', weights_only=True))
if tuple(pixel_map.shape) != (3, *img.shape[:2]):
    raise ValueError('Mode map must have shape (3, H, W) matching the image.')
if not torch.all((pixel_map == 0) | (pixel_map == 1)) or not torch.all(pixel_map.sum(dim=0) == 1):
    raise ValueError('Mode map must be one-hot at every pixel.')
mask = onehot_to_clicked_map(np.asarray(pixel_map), grid_size=16)
expanded = np.repeat(np.repeat(mask, 16, axis=0), 16, axis=1)
if not np.array_equal(expanded, np.asarray(pixel_map.argmax(dim=0))):
    raise ValueError('Every 16x16 block must use one mode.')
print('Mode map is valid: 0=natural, 1=document, 2=prediction.')

# 오버레이 만들기
overlay = img.copy()

H, W, _ = img.shape
grid_size = 16

rows, cols = H // grid_size, W // grid_size  # 격자 개수

def render_display():
    display = img.copy()
    overlay = display.copy()

    # 각 셀 오버레이 색칠
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * grid_size, r * grid_size
            x2, y2 = x1 + grid_size, y1 + grid_size
            if mask[r, c] == 1:
                color = (255, 255, 255)  
            elif mask[r, c] == 0:
                color = (0, 0, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            elif mask[r, c] == 2:
                color = (0, 255, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # 반투명 합성
    display = cv2.addWeighted(overlay, 0.5, display, 0.5, 0)

    # 격자 선 (녹색)
    for x in range(0, W, grid_size):
        cv2.line(display, (x, 0), (x, H), (0, 255, 0), 1)
    for y in range(0, H, grid_size):
        cv2.line(display, (0, y), (W, y), (0, 255, 0), 1)
    return display
combined = render_display()
if args.output:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(args.output, combined):
        raise OSError(f'Could not write {args.output}')
    print(f'Saved overlay: {args.output}')

if not args.no_gui:
    cv2.imshow('Overlayed Result', combined)
    print('ESC exits; S saves the overlay.')
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('s'):
            output = args.output or 'mode_overlay.png'
            cv2.imwrite(output, combined)
            print(f'Saved overlay: {output}')
    cv2.destroyAllWindows()
