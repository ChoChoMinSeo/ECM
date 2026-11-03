# Copyright 2026 MinSeo Cho
# SPDX-License-Identifier: MIT
# See LICENSE.txt in the repository root for the full license text.

import numpy as np

def onehot_to_clicked_map(pixel_map, grid_size=16):
    """
    one-hot pixel 맵을 다시 grid 단위 clicked_map으로 복원
    
    Args:
        pixel_map (ndarray): (C, H*grid_size, W*grid_size)
        grid_size (int): 셀의 픽셀 크기
    
    Returns:
        clicked_map (ndarray): (H, W), 각 셀의 모드 ID
    """
    # (C, H, W) → (H, W), 픽셀별 클래스 ID
    class_map = np.argmax(pixel_map, axis=0)
    
    # grid_size 간격으로 슬라이싱
    clicked_map = class_map[::grid_size, ::grid_size]
    
    return clicked_map
        
def expand_to_onehot(clicked_map, grid_size=16, num_classes=2):
    """
    클릭 맵을 픽셀 단위 one-hot (C,H,W) 표현으로 확장
    
    Args:
        clicked_map (ndarray): (h, w), 각 셀의 모드 ID
        grid_size (int): 각 셀의 픽셀 크기
        num_classes (int): 클래스 수 (기본 2)
    
    Returns:
        pixel_map (ndarray): (C, H*grid_size, W*grid_size)
    """
    # grid_size 배로 확장 (h*grid, w*grid)
    expanded = np.kron(clicked_map, np.ones((grid_size, grid_size), dtype=np.uint8))
    
    # one-hot 변환 (H*grid, W*grid, C)
    onehot = np.eye(num_classes, dtype=np.uint8)[expanded]
    
    # (H*grid, W*grid, C) → (C, H*grid, W*grid)
    pixel_map = np.transpose(onehot, (2, 0, 1))
    
    return pixel_map