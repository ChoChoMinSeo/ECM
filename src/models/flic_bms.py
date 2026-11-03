from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel
from compressai.layers import (
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
import os
from compressai.ops import LowerBound
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch

from einops import rearrange 
from einops.layers.torch import Rearrange
from .entropy_models import GsnConditionalLocScaleShift
from .tca_bms import TCA_EntropyModel
from timm.layers import  DropPath
from .multi_modules_bms import *
from .layers import ModeEstimator
from tqdm import tqdm
from range_coder import RangeEncoder, RangeDecoder

from build_dataset import onehot_convert
import math

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(B, -1, H_sp * W_sp, C)
    img_perm = img_perm.reshape(-1, H_sp * W_sp, C)
    
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


def ste_round(x: Tensor) -> Tensor:
    return (torch.round(x) - x).detach() + x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, idx, split_size=8, dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias
   
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size*2, self.split_size*2
        elif idx == 1:
            H_sp, W_sp = self.split_size//2, self.split_size//2
        elif idx == 2:
            H_sp, W_sp = self.split_size//2, self.split_size*2
        elif idx == 3:
            H_sp, W_sp = self.split_size*2, self.split_size//2
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        window_size = [H_sp, W_sp]
        self.attn_drop = nn.Dropout(attn_drop)
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(x.shape[0], -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)
      
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)
        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x
    
class FM_FFN(nn.Module):
    def __init__(self, dim, window_size_fm, norm_layer = nn.LayerNorm):
        super().__init__()
        self.fm = WindowFrequencyModulation(dim,window_size_fm)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm2 = norm_layer(dim)
        
    def forward(self, x, H, W):
        return self.fm(self.ffn(self.norm2(x)), H, W)


class Swin_FDWA(nn.Module):
   
    def __init__(self, dim,  num_heads,
                 window_size=8,  window_size_fm=16, shift_size=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,norm_layer=nn.LayerNorm, num_class = 1, multi = False, multi_channels = 10, patch_size = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.multi = multi
        if multi:
            self.qkv = MultiModule_linear_input(dim, dim * 3, num_class, multi_channel_idx = list(range(multi_channels)),bias = qkv_bias, patch_size = patch_size)
            self.proj = MultiModule_linear_input(dim, dim, num_class, multi_channel_idx=list(range(multi_channels)), patch_size = patch_size)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            
        self.norm1 = norm_layer(dim)
            
        self.branch_num = 4
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            WindowAttention(
                dim//self.branch_num ,  idx = i,
                split_size=window_size, num_heads=num_heads//self.branch_num , dim_out=dim//self.branch_num ,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
            for i in range(self.branch_num)])

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fm = WindowFrequencyModulation(dim,window_size_fm)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        if multi:
            self.ffn[0] = MultiModule_linear_input(dim, 4 * dim, num_class, multi_channel_idx=list(range(multi_channels)), patch_size = patch_size)
            self.ffn[2] = MultiModule_linear_input(4 * dim, dim, num_class, multi_channel_idx=list(range(multi_channels)), patch_size = patch_size)

        self.norm2 = norm_layer(dim)

    def calculate_mask(self, H, W,split_size=[8,8]):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for Rwin
        img_mask_0 = torch.zeros((1, H, W, 1)).cpu() # 1 H W 1 idx=0
        shift_size =(split_size[0]//2,split_size[1]//2)

        h_slices_0 = (slice(0, -split_size[0]),
                    slice(-split_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices_0 = (slice(0, -split_size[1]),
                    slice(-split_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))

        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt +=1

        # calculate mask for H-Shift
        img_mask_0 = img_mask_0.view(1, H // split_size[0], split_size[0], W // split_size[1], split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, split_size[0], split_size[1], 1) # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, split_size[0] * split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for V-Shift
    
        return attn_mask_0

    def forward(self, x, x_size, mode_map_one_hot = None, mode_map2 = None, unknown_mask = None, est = True):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
            
        if self.multi:
            qkv = self.qkv(img, H, W, mode_map_one_hot, mode_map2, unknown_mask, est)
        else:
            qkv = self.qkv(img)
        qkv = qkv.reshape(B, -1, 3, C).permute(2, 0, 1, 3).contiguous() # 3, B, HW, C

        if self.shift_size > 0:
            qkv = qkv.view(3, B, H, W, C)
     
            qkv0,qkv1,qkv2,qkv3= qkv.chunk(4,4)
            qkv_0 = torch.roll(qkv0, shifts=(-self.split_size,-self.split_size), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, L, C//4)

            qkv_1 = torch.roll(qkv1, shifts=(-self.split_size//4,-self.split_size//4), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, L, C//4)
         
            qkv_2 = torch.roll(qkv2, shifts=(-self.split_size//4,-self.split_size), dims=(2, 3))
            qkv_2 = qkv_2.view(3, B, L, C//4)
         
            qkv_3 = torch.roll(qkv3, shifts=(-self.split_size,-self.split_size//4), dims=(2, 3))
            qkv_3 = qkv_3.view(3, B, L, C//4)
            
            x1_shift = self.attns[0](qkv_0, H, W)
            x2_shift = self.attns[1](qkv_1, H, W)
            x3_shift = self.attns[2](qkv_2, H, W)
            x4_shift = self.attns[3](qkv_3, H, W)
                
            x1 = torch.roll(x1_shift, shifts=(self.split_size, self.split_size), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.split_size//4, self.split_size//4), dims=(1, 2))
            x3 = torch.roll(x3_shift, shifts=(self.split_size//4, self.split_size), dims=(1, 2))
            x4 = torch.roll(x4_shift, shifts=(self.split_size, self.split_size//4), dims=(1, 2))
        else:
            qkv0, qkv1, qkv2, qkv3 = qkv.chunk(4,3)

            x1 = self.attns[0](qkv0, H, W)
            x2 = self.attns[1](qkv1, H, W)
            x3 = self.attns[2](qkv2, H, W)
            x4 = self.attns[3](qkv3, H, W)

        x1 = x1.view(B, L, C // 4)
        x2 = x2.view(B, L, C // 4)
        x3 = x3.view(B, L, C // 4)
        x4 = x4.view(B, L, C // 4)

        if self.multi:
            x = x + self.drop_path(self.proj(torch.cat([x1, x2, x3, x4], dim=2), H, W, mode_map_one_hot, mode_map2, unknown_mask, est))
        else:
            x = x + self.drop_path(self.proj(torch.cat([x1, x2, x3, x4], dim=2)))
            
        if self.multi:
            x = x + self.fm(self.forward_ffn(self.norm2(x), H, W, mode_map_one_hot, mode_map2, unknown_mask, est),H,W)
        else:
            x = x + self.fm(self.forward_ffn(self.norm2(x)), H,W)
        return x
    
    def forward_ffn(self,x, H = None, W = None, mode_map_one_hot = None, mode_map2 = None, unknown_mask = None, est = True):
        if self.multi:
            x = self.ffn[0](x, H, W, mode_map_one_hot, mode_map2, unknown_mask, est)
            x = self.ffn[1](x)
            x = self.ffn[2](x, H, W, mode_map_one_hot, mode_map2, unknown_mask, est)
        else:
            x = self.ffn[0](x)
            x = self.ffn[1](x)
            x = self.ffn[2](x)
            
        return x
        
    
class WindowFrequencyModulation(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.ratio = 1
        self.complex_weight= nn.Parameter(torch.cat((torch.ones(self.window_size, self.window_size//2+1, self.ratio*dim, 1, dtype=torch.float32),\
        torch.zeros(self.window_size, self.window_size//2+1, self.ratio*dim, 1, dtype=torch.float32)),dim=-1))
        
    def forward(self, x, H,W,spatial_size=None):
        B,L,C = x.shape
   
        x = x.view(B,H,W,self.ratio*C)
        B, H,W, C = x.shape

        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)

        x = x.to(torch.float32)
        
        x= torch.fft.rfft2(x,dim=(3, 4), norm='ortho')
      
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.window_size, self.window_size), dim=(3, 4), norm='ortho')

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 p1) (w2 p2) c ')

        x = x.view(B, -1, C)
        return x

    
class FAT_Block(nn.Module):
    def __init__(self, trans_dim, head_dim, window_size, window_size_fm, drop_path, type='W',
                 num_class = 1, multi = False, multi_channels = 10, patch_size = 8):
        """ SwinTransformer and Conv Block
        """
        super(FAT_Block, self).__init__()
        self.trans_dim = trans_dim
        self.head_dim = head_dim

        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']

        self.trans_block = Swin_FDWA(
            dim=trans_dim,
            num_heads=head_dim,
            window_size=window_size,
            window_size_fm=window_size_fm,
            shift_size=0 if (type=='W') else window_size//2,
            num_class=num_class,
            multi = multi,
            multi_channels = multi_channels,
            patch_size = patch_size,
        )
        # output multi
        self.multi = multi
        if multi:
            self.conv1_1 = MultiModule_conv_input(self.trans_dim, self.trans_dim, 1,1,0, 
                            num_class=num_class, multi_channel_idx=list(range(multi_channels)), patch_size = patch_size)
            self.conv1_2 = MultiModule_conv_input(self.trans_dim, self.trans_dim, 1,1,0, 
                            num_class=num_class, multi_channel_idx=list(range(multi_channels)), patch_size = patch_size)
        else:
            self.conv1_1 = nn.Conv2d(self.trans_dim, self.trans_dim, 1, 1, 0, bias=True)
            self.conv1_2 = nn.Conv2d(self.trans_dim, self.trans_dim, 1, 1, 0, bias=True)

    def forward(self, x, mode_map_one_hot = None, mode_map2 = None, unknown_mask = None, est = True):
        _,_,h,w = x.shape
        if self.multi:            
            res = self.conv1_1(x, mode_map_one_hot, mode_map2, unknown_mask, est)
        else:
            res = self.conv1_1(x)
                
        res = Rearrange('b c h w -> b (h w) c')(res)
        res = self.trans_block(res, (h, w), mode_map_one_hot, mode_map2, unknown_mask, est)
        res = Rearrange('b (h w) c -> b c h w', h=h, w=w)(res)
        if self.multi:
            res = self.conv1_2(res, mode_map_one_hot, mode_map2, unknown_mask, est)
        else:
            res = self.conv1_2(res)
        x = x + res
        return x


class AnalysisTransform(nn.Module):
    def __init__(self, dpr,num_class=6, multi_channels = 90, patch_size = 16):
        super().__init__()

        self.down0 = ResidualBlockWithStride(3, 96, 2)
        self.fat1 = nn.ModuleList(
            FAT_Block(96, 8, 8, 8 * 2, dpr[i], 'W' if not i % 2 else 'SW', num_class = num_class,
                      multi = True, multi_channels=multi_channels, patch_size=patch_size//2) for i in range(2)
        )   
        self.down1 = ResidualBlockWithStride(96, 144, stride=2)
        
        self.fat2 = nn.ModuleList(
            FAT_Block(144, 16, 8, 8 * 2, dpr[i + 2], 'W' if not i % 2 else 'SW', num_class = num_class, 
                      multi = True, multi_channels=multi_channels, patch_size=patch_size//4) for i in range(2)
        )   
        self.down2 = ResidualBlockWithStride(144, 256, stride=2)
        
        self.fat3 = nn.ModuleList(
            FAT_Block(256, 32, 8, 8 * 2, dpr[i + 4], 'W' if not i % 2 else 'SW') for i in range(2)
        )
        self.down3 = conv3x3(256, 320, stride=2)
        self.num_class = num_class
        
        self.patch_size = patch_size
        
        self.patch_mode_estimator1 = ModeEstimator(96 , num_class)
        self.patch_mode_estimator2 = ModeEstimator(144, num_class)
        
    def mode_map_init(self, mode_map, est = True):
        '''
        mode_map : B 1 H/16 W/16  -- 0: natural 1: document 2: unknown
        
        unknown_mask: B H/(16/p) W/(16/p) -- 스케일 별 예측이 필요한 부분 마스킹 1 : unknown
        
        return onehot mode_map, unknown_mask
        ''' 
        _mode_map = mode_map.clone().squeeze(1)
        if not est:
            _mode_map[_mode_map == self.num_class] = 0
        mode_map_one_hot = F.one_hot(_mode_map, num_classes = self.num_class+1)
        mode_map_one_hot = mode_map_one_hot.permute(0,3,1,2).float()[:,:self.num_class]
        
        unknown_mask = (_mode_map == self.num_class)
        
        return mode_map_one_hot, unknown_mask
    
    def forward(self, x, mode_map, est = True):
        mode_map_one_hot, unknown_mask = self.mode_map_init(mode_map, est)
        
        x = self.down0(x)
        
        if est:
            _, _, H, W = x.shape
            _mode_map = self.patch_mode_estimator1(x.clone())
            _mode_map = rearrange(_mode_map, "(B H W) c -> B c H W", H = H, W = W).contiguous()
        else:
            _mode_map = None
            
        for layer in self.fat1:
            x = layer(x, mode_map_one_hot, _mode_map, unknown_mask, est)
        x = self.down1(x)
        
        if est:
            _, _, H, W = x.shape
            _mode_map = self.patch_mode_estimator2(x.clone())
            _mode_map = rearrange(_mode_map, "(B H W) c -> B c H W", H = H, W = W).contiguous()
        else:
            _mode_map = None
        
        for layer in self.fat2:
            x = layer(x, mode_map_one_hot, _mode_map, unknown_mask, est)
        x = self.down2(x)
        for layer in self.fat3:
            x = layer(x)
        x = self.down3(x)
        return x


class SynthesisTransform(nn.Module):
    def __init__(self, dpr,num_class=1, multi_channels = 10, patch_size = 16):
        super().__init__()
        self.up0 = ResidualBlockUpsample(320, 256, 2)
        self.fat1 = nn.ModuleList(
            FAT_Block(256, 32, 8, 8 * 2, dpr[i], 'W' if not i % 2 else 'SW') for i in range(2)
        )
        self.up1 = ResidualBlockUpsample(256, 144, 2)

        self.fat2 = nn.ModuleList(
            FAT_Block(144, 16, 8, 8 * 2, dpr[i + 2], 'W' if not i % 2 else 'SW',num_class=num_class, 
                      multi = True, multi_channels = multi_channels, patch_size=patch_size//4) for i in range(2)
        )
        self.up2 = ResidualBlockUpsample(144, 96, 2)
        
        self.fat3 = nn.ModuleList(
            FAT_Block(96, 8, 8, 8 * 2, dpr[i + 4], 'W' if not i % 2 else 'SW', num_class = num_class, 
                      multi = True, multi_channels = multi_channels, patch_size=patch_size//2) for i in range(2)
        )
        self.up3 = subpel_conv3x3(96, 3, 2)

        self.num_class = num_class
            
        self.patch_mode_estimator1 = ModeEstimator(96, num_class)
        self.patch_mode_estimator2 = ModeEstimator(144, num_class)
        self.patch_size = patch_size
        
    def mode_map_init(self, mode_map, est = True):
        '''
        mode_map : B 1 H/16 W/16  -- 0: natural 1: document 2: unknown
        
        unknown_mask: B H/(16/p) W/(16/p) -- 스케일 별 예측이 필요한 부분 마스킹 1 : unknown
        
        return onehot mode_map, unknown_mask
        ''' 
        _mode_map = mode_map.clone().squeeze(1)
        if not est:
            _mode_map[_mode_map == self.num_class] = 0
        mode_map_one_hot = F.one_hot(_mode_map, num_classes = self.num_class+1)
        mode_map_one_hot = mode_map_one_hot.permute(0,3,1,2).float()[:,:self.num_class]
        
        unknown_mask = (_mode_map == self.num_class)
        
        return mode_map_one_hot, unknown_mask
    
    def forward(self, x, mode_map, est = True):
        mode_map_one_hot, unknown_mask = self.mode_map_init(mode_map, est)
        
        x = self.up0(x)
        for layer in self.fat1:
            x = layer(x)
        x = self.up1(x)
        
        if est:
            _, _, H, W = x.shape
            _mode_map = self.patch_mode_estimator2(x.clone())
            _mode_map = rearrange(_mode_map, "(B H W) c -> B c H W", H = H, W = W)
        else:
            _mode_map = None
        
        for layer in self.fat2:
            x = layer(x, mode_map_one_hot, _mode_map, unknown_mask, est)
        x = self.up2(x)
        
        if est:
            _, _, H, W = x.shape
            _mode_map = self.patch_mode_estimator1(x.clone())
            _mode_map = rearrange(_mode_map, "(B H W) c -> B c H W", H = H, W = W)
        else:
            _mode_map = None
                
        for layer in self.fat3:
            x = layer(x, mode_map_one_hot, _mode_map, unknown_mask, est)
        x = self.up3(x)

        return x
        
class FrequencyAwareTransFormer(CompressionModel):
    def __init__(
            self,
            config=(2, 2, 2, 2, 2, 2),
            head_dim=(8, 16, 32, 32, 16, 8),
            drop_path_rate=0,
            N=128,
            M=320,
            num_slices=5,
            max_support_slices=5,
            num_class = 6,
            multi_channels = 90,
            multi_channels_tca = 30,
            patch_size = 16,
    ):
        super().__init__(192)
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_class = num_class
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.tca_depth = 12
        self.tca_ratio = 4
        self.lower_bound = 0.01
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        self.g_a = AnalysisTransform(dpr,num_class, multi_channels, patch_size=patch_size)
        self.g_s = SynthesisTransform(dpr[::-1],num_class, multi_channels, patch_size=patch_size)
        self.h_a = nn.Sequential(
            *[ResidualBlockWithStride(320, 256, 2)] + \
            [FAT_Block(N * 2, 32, 2, 4, 0, 'W' if not i % 2 else 'SW') for i in range(config[0])] +\
            [conv3x3(2 * N, 192, stride=2)]
        )
        self.h_mean_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2 * N, 2)] + \
            [FAT_Block(N * 2, 32, 2, 4, 0, 'W' if not i % 2 else 'SW') for i in range(config[3])] +\
            [subpel_conv3x3(2 * N, 320, 2)]
        ) 
        
        self.h_scale_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2 * N, 2)] + \
            [FAT_Block(N * 2, 32, 2, 4, 0, 'W' if not i % 2 else 'SW') for i in range(config[3])] +\
            [subpel_conv3x3(2 * N, 320, 2)]
        )
        self.tca= TCA_EntropyModel(dim=M,ratio=self.tca_ratio,depth=self.tca_depth,slices=self.num_slices,
                num_class=num_class, multi_channels = multi_channels_tca, patch_size = patch_size//16)
        self.gaussian_conditional= GsnConditionalLocScaleShift(
            num_scales=256, num_means=100, min_scale=self.lower_bound, tail_mass=(2 ** (-8))
        )
        self.entropy_bottleneck = EntropyBottleneck(192)
               
        self.patch_size = patch_size
        
    def mode_map_init(self, idx_cls):            
        idx_cls16 = rearrange(idx_cls, 'b c (h p1) (w p2) -> b c h w (p1 p2)', p1= self.patch_size, p2 = self.patch_size).contiguous()
        idx_cls16 = torch.sum(idx_cls16, dim=-1)
        idx_cls16 = idx_cls16 / (self.patch_size**2)
        idx_cls16, mode_map = torch.max(idx_cls16, dim=1, keepdim = True)# b 1 h w
        # 비율대로 initialize
        mode_map[idx_cls16 != 1] = self.num_class
        # mode_map = torch.ones_like(mode_map) * self.num_class
        return mode_map
        
    def forward(self, x, idx_cls, est = True):
        B,_,h,w = x.shape
        if len(idx_cls.shape)!=4:
            idx_cls = idx_cls.view(B,self.num_class,1,1).repeat(1,1,h,w)
            
        if idx_cls.shape[-1]==w:
            # pixel-wise mode to block-wise
            mode_map = self.mode_map_init(idx_cls) # B 1 H/16 W/16
        else:
            mode_map = idx_cls
            
        y = self.g_a(x, mode_map, est)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        scales_hyper = self.h_scale_s(z_hat)
        means_hyper  = self.h_mean_s(z_hat)        
 
        y_hat = ste_round(y)
        means,scales,lrp = self.tca(torch.cat((means_hyper,scales_hyper),dim=1), y_hat, mode_map, est)
        _, y_likelihoods =  self.gaussian_conditional(y, scales, means)

        lrp = 0.5 * torch.tanh(lrp)
        y_hat += lrp
        
        # restoration
        x_hat = self.g_s(y_hat, mode_map, est)
        
        return x_hat, y_likelihoods, z_likelihoods
            
    def forward_rd_search_compress(self, x, lmbda = 0.0483, verbose=False):
        # try encoding
        B, _, H, W = x.shape
        num_pixels = H*W

        all_losses = []
        # Encoding
        for i in range(self.num_class+1):
            _mode_map = torch.full(
                (B, 1, H // self.patch_size, W // self.patch_size),
                i,
                device=x.device,
            )

            _y = self.g_a(x, _mode_map)
            _z = self.h_a(_y)

            _, _z_likelihoods = self.entropy_bottleneck(_z)

            _z_offset = self.entropy_bottleneck._get_medians()
            _z_tmp = _z - _z_offset
            _z_hat = ste_round(_z_tmp) + _z_offset

            _scales_hyper = self.h_scale_s(_z_hat)
            _means_hyper  = self.h_mean_s(_z_hat)        
    
            _y_hat = ste_round(_y)
            _means,_scales,_lrp = self.tca(torch.cat((_means_hyper,_scales_hyper),dim=1), _y_hat, _mode_map)
            _, _y_likelihoods =  self.gaussian_conditional(_y, _scales, _means)

            _lrp = 0.5 * torch.tanh(_lrp)
            _y_hat += _lrp
            
            # restoration
            _x_hat = self.g_s(_y_hat, _mode_map)
            distortion = F.avg_pool2d(torch.mean(((x - _x_hat)**2) * (255**2) * lmbda, dim=1, keepdim=True), kernel_size=self.patch_size, stride = self.patch_size)
            y_bitmap = torch.log(torch.sum(_y_likelihoods, dim=1, keepdim=True))/(-math.log(2)*num_pixels)
            z_bitmap = F.interpolate(torch.log(torch.sum(_z_likelihoods, dim=1, keepdim=True))/(-math.log(2)*num_pixels), scale_factor=4, mode = 'nearest')

            loss = distortion + y_bitmap + z_bitmap
            loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, float('inf')))
            all_losses.append(loss)
        # [B, num_class+1, H/16, W/16]
        all_losses = torch.cat(all_losses, dim=1)
        if not torch.isfinite(all_losses).any(dim=1).all():
            raise RuntimeError('BMS RD search found no finite candidate for one or more blocks.')

        # 각 patch에서 가장 loss가 작은 mode index
        best_mode = torch.argmin(all_losses, dim=1, keepdim=True)
        return best_mode
    

    def forward_encoding_oracle(self,x, mode_map):
        y = self.g_a(x, mode_map)
        z = self.h_a(y)
        return y, z, mode_map

    def forward_entropy_coding_oracle(self,y, z, mode_map):
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        scales_hyper = self.h_scale_s(z_hat)
        means_hyper  = self.h_mean_s(z_hat)        
 
        y_hat = ste_round(y)
        means,scales,lrp = self.tca(torch.cat((means_hyper,scales_hyper),dim=1), y_hat, mode_map)
        _, y_likelihoods =  self.gaussian_conditional(y, scales, means)

        lrp = 0.5 * torch.tanh(lrp)
        y_hat += lrp
        
        return y_hat, mode_map
    
    def forward_decoding_oracle(self,y_hat, mode_map):
        x_hat = self.g_s(y_hat, mode_map)
        return x_hat

    def forward_encoding(self,x):
        mode_map = self.forward_rd_search_compress(x)
        y = self.g_a(x, mode_map)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        scales_hyper = self.h_scale_s(z_hat)
        means_hyper  = self.h_mean_s(z_hat)      
        y_hat = ste_round(y)
        means,scales,lrp = self.tca(torch.cat((means_hyper,scales_hyper),dim=1), y_hat, mode_map)
        _, y_likelihoods =  self.gaussian_conditional(y, scales, means)
        lrp = 0.5 * torch.tanh(lrp)
        y_hat += lrp
        return z_hat, y, mode_map
    
    def forward_decoding(self,z_hat, y, mode_map):
        y_hat = ste_round(y)
        scales_hyper = self.h_scale_s(z_hat)
        means_hyper  = self.h_mean_s(z_hat)     
        y_hat = ste_round(y)
        means,scales,lrp = self.tca(torch.cat((means_hyper,scales_hyper),dim=1), y_hat, mode_map)
        _, y_likelihoods =  self.gaussian_conditional(y, scales, means)
        lrp = 0.5 * torch.tanh(lrp)
        y_hat += lrp
        # restoration
        x_hat = self.g_s(y_hat, mode_map)
        return x_hat
    
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net
    
    def compress(self, x, output_path, est = True, lmbda = 0.0483, mode_map = None):
        if mode_map is None:
            mode_map = self.forward_rd_search_compress(x, lmbda=lmbda)
        else:
            expected = (x.shape[0], 1, x.shape[2] // self.patch_size, x.shape[3] // self.patch_size)
            if tuple(mode_map.shape) != expected:
                raise ValueError(f'Expected block mode map shape {expected}, got {tuple(mode_map.shape)}')
            if mode_map.min() < 0 or mode_map.max() > self.num_class:
                raise ValueError('Block mode IDs are outside the supported range.')
            mode_map = mode_map.to(device=x.device, dtype=torch.long).contiguous()
        y = self.g_a(x, mode_map, est)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        hyper = torch.cat((latent_means,latent_scales),dim=1)
        # Use the decoder layout; CDF rounding is sensitive to numerical differences.
        y_hat_coded = torch.zeros_like(y, memory_format=torch.contiguous_format)
        channel_per_slices = self.M//self.num_slices
        minmax = np.maximum(abs(torch.round(y.cpu()).max()), abs(torch.round(y.cpu()).min()))
        minmax = int(np.maximum(minmax, 1))
        encoder = RangeEncoder(output_path)
        samples = np.arange(0, minmax*2+1)
        lower_bound =  LowerBound(1e-9)
        lower_bound.bound = lower_bound.bound.to(x.device)

        
        # encode mode information 1 1 H W
        for mode_element in mode_map.view(-1).tolist():
            encoder.encode([int(mode_element)],list(range(self.num_class+2)))
        print('mode info coded!')
        ####################################
        
        for slice_index in range(self.num_slices):
            # Match the decoder context exactly, including floating-point evaluation.
            means,scales,_ = self.tca(hyper,y_hat_coded, mode_map, est)
            mu = means[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            scale = scales[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            y_slice = y[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            y_hat_slice = torch.round(y_slice)

            for h_idx in (range(y_shape[0])):
                for w_idx in range(y_shape[1]):
                    for c_idx in range(channel_per_slices):

                        pmf = self._likelihood(torch.tensor(samples).to(x.device),scale[0][c_idx][h_idx][w_idx],means=mu[0][c_idx][h_idx][w_idx]+minmax)
                        pmf = lower_bound(pmf)
                        pmf_clip = np.clip(np.array(pmf.cpu()), 1.0/65536, 1.0)   
                        pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                        
                        cdf = list(np.add.accumulate(pmf_clip))
                        cdf = [0] + [int(i) for i in cdf]
                    
                        symbol = int(y_hat_slice[0, c_idx,h_idx, w_idx] + minmax )
                        encoder.encode([symbol], cdf)
     
            y_hat_coded[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices] = y_hat_slice

        encoder.close()
        y_size = os.path.getsize(output_path)
        return {"z_strings":  z_strings, "z_shape": z.size()[-2:],"y_size":y_size,"minmax":minmax}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        lower_bound =  LowerBound(self.lower_bound)
        lower_bound.bound = lower_bound.bound.to(inputs.device)
        scales = lower_bound(scales)
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
      
        return half * torch.erfc(const * inputs)

    def decompress(self, z_strings, minmax, z_shape, output_path, est = True):
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        hyper = torch.cat((latent_means,latent_scales),dim=1)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_hat_coded = torch.zeros((1,self.M,z_hat.shape[2]*4,z_hat.shape[3]*4)).to(z_hat.device)
        lrp_coded = torch.zeros((1,self.M,z_hat.shape[2]*4,z_hat.shape[3]*4)).to(z_hat.device)
        channel_per_slices = self.M//self.num_slices
        decoder = RangeDecoder(output_path)
        samples = np.arange(0, minmax*2+1)

        lower_bound =  LowerBound(1e-9)
        lower_bound.bound = lower_bound.bound.to(z_hat.device)
        
        # decode mode information
        mode_map_coded = torch.zeros((1,1,z_shape[0]*4, z_shape[1]*4),dtype=torch.long, device=z_hat.device)
        for h_idx in range(z_shape[0]*4):
            for w_idx in range(z_shape[1]*4):
                mode_map_coded[0,:,h_idx,w_idx] = torch.tensor(decoder.decode(1, list(range(self.num_class+2))),dtype=torch.long, device=z_hat.device)
        print('mode info decoded!')
        
        for slice_index in range(self.num_slices):
            means,scales,lrps = self.tca(hyper,y_hat_coded, mode_map_coded, est)
            mu = means[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            scale = scales[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            lrp = lrps[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices]
            y_hat_slice = torch.zeros_like(mu).to(z_hat.device)
            for h_idx in (range(y_shape[0])):
                for w_idx in range(y_shape[1]):
                    for c_idx in range(channel_per_slices):

                        pmf = self._likelihood(torch.tensor(samples).to(z_hat.device),scale[0][c_idx][h_idx][w_idx],means=mu[0][c_idx][h_idx][w_idx]+minmax)
                        pmf = lower_bound(pmf)
                        pmf_clip = np.clip(np.array(pmf.cpu()), 1.0/65536, 1.0)   
                        pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                       
                        cdf = list(np.add.accumulate(pmf_clip))
                        cdf = [0] + [int(i) for i in cdf]
                       
                        y_hat_slice[0,  c_idx,h_idx, w_idx] = decoder.decode(1, cdf)[0] - minmax 

            y_hat_coded[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices] = y_hat_slice
            lrp_coded[:,slice_index*channel_per_slices:(slice_index+1)*channel_per_slices] = lrp


        lrp_coded = 0.5 * torch.tanh(lrp_coded)
  
        y_hat= y_hat_coded+lrp_coded
        x_hat = self.g_s(y_hat, mode_map_coded, est).clamp_(0, 1)

        return {"x_hat": x_hat}
