# Copyright 2026 MinSeo Cho
# SPDX-License-Identifier: MIT
# See LICENSE.txt in the repository root for the full license text.

import argparse
from pathlib import Path

import cv2
import numpy as np
import math

def pad_to_multiple_of_256(img, save_path="padded_image.png"):
    H, W = img.shape[:2]

    # 256 배수 크기 계산
    new_H = math.ceil(H / 256) * 256
    new_W = math.ceil(W / 256) * 256

    pad_h = new_H - H
    pad_w = new_W - W

    # 위/아래, 좌/우 padding 계산
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # White padding
    padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255))

    print(f"패딩 완료: {H}x{W} → {new_H}x{new_W}, 저장: {save_path}")
    return padded

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pad an image to multiples of 256 before BMS annotation.')
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True, help='Output PNG image')
    args = parser.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    if Path(args.output).suffix.lower() != '.png':
        parser.error('--output must be a .png file to preserve the annotated image')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    padded_img = pad_to_multiple_of_256(img, args.output)
    if not cv2.imwrite(args.output, padded_img):
        raise OSError(f'Could not write {args.output}')
