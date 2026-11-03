# Copyright 2026 MinSeo Cho
# SPDX-License-Identifier: MIT
# See LICENSE.txt in the repository root for the full license text.

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from .utils import expand_to_onehot, onehot_to_clicked_map
except ImportError:
    from utils import expand_to_onehot, onehot_to_clicked_map

parser = argparse.ArgumentParser(description='Manually annotate 16x16 BMS blocks.')
parser.add_argument('--image', required=True)
parser.add_argument('--output', required=True, help='Output .pt mode map')
parser.add_argument('--scale', type=float, default=1.0)
args = parser.parse_args()
if args.scale <= 0:
    parser.error('--scale must be positive')
PATHS = {'img': args.image, 'mode': args.output}
display_scale = args.scale
grid_size = 16
img = cv2.imread(PATHS['img'])
if img is None:
    raise FileNotFoundError(PATHS['img'])
H, W, _ = img.shape
if H % 256 or W % 256:
    raise ValueError('Prepare the image with preprocess.py before annotation: H and W must be multiples of 256.')
rows, cols = H // grid_size, W // grid_size
Path(PATHS['mode']).parent.mkdir(parents=True, exist_ok=True)
if Path(PATHS['mode']).exists():
    pixel_map = torch.load(PATHS['mode'], map_location='cpu', weights_only=True)
    if tuple(pixel_map.shape) != (3, H, W):
        raise ValueError('Existing selector map must have shape (3, H, W) matching the image.')
    clicked_map = onehot_to_clicked_map(np.asarray(pixel_map), grid_size)
else:
    clicked_map = np.ones((rows, cols), dtype=np.uint8)

# --- 드래그 상태 ---
dragging = False
start_pt = None
end_pt = None

# --- 격자 및 오버레이 그리기 함수 ---
def render_display(temp_rect=None):
    display = img.copy()
    overlay = display.copy()

    # 각 셀 오버레이 색칠
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * grid_size, r * grid_size
            x2, y2 = x1 + grid_size, y1 + grid_size
            if clicked_map[r, c] == 1:
                color = (255, 255, 255)  
            elif clicked_map[r, c] == 0:
                color = (0, 0, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            elif clicked_map[r, c] == 2:
                color = (0, 255, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # 반투명 합성
    display = cv2.addWeighted(overlay, 0.5, display, 0.5, 0)

    # 격자 선 (녹색)
    for x in range(0, W, grid_size):
        cv2.line(display, (x, 0), (x, H), (0, 255, 0), 1)
    for y in range(0, H, grid_size):
        cv2.line(display, (0, y), (W, y), (0, 255, 0), 1)

    # 드래그 직사각형 표시 (빨강 테두리)
    if temp_rect is not None:
        cv2.rectangle(display, temp_rect[0], temp_rect[1], (0, 0, 255), 2)

    # 축소해서 출력
    disp_small = cv2.resize(display, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
    return disp_small

# --- 마우스 콜백 ---
def mouse_callback(event, x, y, flags, param):
    global dragging, start_pt, end_pt, clicked_map

    # 화면 좌표 → 원본 좌표 변환
    orig_x = int(x / display_scale)
    orig_y = int(y / display_scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_pt = (orig_x, orig_y)
        end_pt = (orig_x, orig_y)

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        end_pt = (orig_x, orig_y)
        cv2.imshow("Grid Image", render_display((start_pt, end_pt)))

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        end_pt = (orig_x, orig_y)

        # 드래그 영역 내 격자 토글
        x1, y1 = min(start_pt[0], end_pt[0]), min(start_pt[1], end_pt[1])
        x2, y2 = max(start_pt[0], end_pt[0]), max(start_pt[1], end_pt[1])

        row1, col1 = y1 // grid_size, x1 // grid_size
        row2, col2 = y2 // grid_size, x2 // grid_size

        for r in range(row1, min(row2 + 1, rows)):
            for c in range(col1, min(col2 + 1, cols)):
                clicked_map[r, c] = (clicked_map[r, c]+1)%3
                
        cv2.imshow("Grid Image", render_display())

# --- 실행 ---
cv2.imshow("Grid Image", render_display())
cv2.setMouseCallback("Grid Image", mouse_callback)

print("Click or drag to cycle: 0=natural, 1=document, 2=prediction. S saves; ESC exits.")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키
        break
    elif key == ord("s"):  # 's' 키 누르면 저장
        # 격자 클릭 맵(0/1)을 픽셀 단위로 확장
        pixel_map = expand_to_onehot(clicked_map, grid_size=16, num_classes=3)
        torch.save(torch.from_numpy(pixel_map), PATHS['mode'])
        print(f"Saved: {PATHS['mode']}")

cv2.destroyAllWindows()
