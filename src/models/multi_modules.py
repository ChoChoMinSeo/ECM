# Copyright 2026 MinSeo Cho
# SPDX-License-Identifier: MIT
# See LICENSE.txt in the repository root for the full license text.

import torch.nn as nn
import torch
from einops import rearrange 
import torch.nn.functional as F

class MultiModule_conv_input(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size = 1,
        stride = 1,
        padding = 0,
        groups = 1,
        num_class = 1,
        multi_channel_idx = [1,2,3],
        multi_bias = True
    ):
        super().__init__()
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.out_ch = out_ch
        self.chunk_size = out_ch//groups
        self.topk = len(multi_channel_idx)//groups
        self.mode_adaptive_channel_idx = [i % self.chunk_size for i in multi_channel_idx]
        self.main_weight = nn.Parameter(torch.ones(1, out_ch, in_ch//groups, kernel_size, kernel_size))
        self.multi_weights = nn.Parameter(torch.ones(num_class-1, out_ch//groups, len(self.mode_adaptive_channel_idx), kernel_size, kernel_size))
        self.main_bias = nn.Parameter(torch.ones(1,out_ch))
        if multi_bias:
            self.multi_bias = nn.Parameter(torch.ones(num_class-1, out_ch))
        else:
            self.multi_bias = None
        self.num_class = num_class
        
    def forward(self, x, idx_cls):
        B,C,H,W = x.shape
        # weight
        weight = self.main_weight.repeat(B,1,1,1,1)
        for i in range(self.groups):
            weight[:,self.chunk_size*i:self.chunk_size*(i+1), self.mode_adaptive_channel_idx[self.topk*i:self.topk*(i+1)]] = \
            torch.sum(
                torch.cat([
                    self.main_weight[:,self.chunk_size*i:self.chunk_size*(i+1), self.mode_adaptive_channel_idx[self.topk*i:self.topk*(i+1)]],self.multi_weights[:,:,self.topk*i:self.topk*(i+1)]],dim=0)
                .unsqueeze(0).repeat(B,1,1,1,1,1)*idx_cls.view(B,self.num_class,1,1,1,1), dim=1)
        weight = rearrange(weight,"B Cout Cin k1 k2 -> (B Cout) Cin k1 k2").contiguous()
        
        # bias
        if self.multi_bias is not None:
            bias = torch.sum(torch.cat([self.main_bias,self.multi_bias],dim=0).unsqueeze(0).repeat(B,1,1)*idx_cls.view(B,self.num_class,1),dim=1)
        else:
            bias = self.main_bias.repeat(B,1)
            
        bias = bias.view(-1)
        x = x.view(1,B*C,H,W)
        x = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding, groups=self.groups * B)
        x = x.view(B, self.out_ch, H//self.stride, W//self.stride)
        return x
    
class MultiModule_linear_input(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        num_class = 1,
        multi_channel_idx = [1,2,3],
        multi_bias = True,
        bias = True
    ):
        super().__init__()
        self.out_ch = out_ch
        self.mode_adaptive_channel_idx = multi_channel_idx
        self.main_weight = nn.Parameter(torch.ones(1,out_ch, in_ch))
        self.multi_weights = nn.Parameter(torch.ones(num_class-1, out_ch, len(self.mode_adaptive_channel_idx)))
        if bias:
            self.main_bias = nn.Parameter(torch.ones(1,out_ch))
            if multi_bias:
                self.multi_bias = nn.Parameter(torch.ones(num_class-1, out_ch))
            else:
                self.multi_bias = None
        self.num_class = num_class
        self.bias = bias
        
    def forward(self, x, idx_cls):
        B,_,_ = x.shape
        # weight
        weight = self.main_weight.repeat(B,1,1)
        weight[:,:,self.mode_adaptive_channel_idx] = torch.sum(torch.cat([self.main_weight[:,:,self.mode_adaptive_channel_idx],self.multi_weights ],dim=0).unsqueeze(0).repeat(B,1,1,1)*idx_cls.view(B,self.num_class,1,1),dim=1)
        x = torch.bmm(x,weight.transpose(1,2))
        
        # bias
        if self.bias:
            if self.multi_bias is None:
                bias = self.main_bias.repeat(B,1)
            else:
                bias = torch.sum(torch.cat([self.main_bias,self.multi_bias],dim=0).unsqueeze(0).repeat(B,1,1)*idx_cls.view(B,self.num_class,1),dim=1)
            x = x + bias.unsqueeze(1)
        return x