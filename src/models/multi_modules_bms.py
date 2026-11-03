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
        multi_bias = True,
        patch_size = 8,
    ):
        super().__init__()
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.out_ch = out_ch
        self.chunk_size = out_ch//groups
        self.in_ch_chunk_size = in_ch//groups
        self.mode_adaptive_channel_idx = torch.tensor(multi_channel_idx)
        self.topk = len(self.mode_adaptive_channel_idx)
        self.mode_independent_channel_idx = torch.tensor(list(range(len(multi_channel_idx),self.in_ch_chunk_size)))
        self.mode_independent_channel_idx_ = torch.cat([self.mode_independent_channel_idx + self.in_ch_chunk_size*i for i in range(self.groups)])
        
        self.main_weight = nn.Parameter(torch.ones(1, out_ch, in_ch//groups, kernel_size, kernel_size))
        self.multi_weights = nn.Parameter(torch.ones(num_class-1, out_ch//groups, len(multi_channel_idx)*self.groups, kernel_size, kernel_size))
        self.main_bias = nn.Parameter(torch.ones(1,out_ch))
        if multi_bias:
            self.multi_bias = nn.Parameter(torch.ones(num_class-1, out_ch))
        else:
            self.multi_bias = None
        self.num_class = num_class
        
        self.patch_size = patch_size
        self.out_ch = out_ch
        
    def forward(self, x,  mode_map = None, mode_map2 = None, unknown_mask = None, est=True )   :
        if est:
            return self.forward_BMS(x,  mode_map, mode_map2, unknown_mask)
        else:
            idx_cls = mode_map[:,:,0,0]
            return self.forward_PartiMat(x, idx_cls)
    
    def forward_PartiMat(self, x, idx_cls):  # idx_cls: (B, 6, H//p, W//p)
        # Bound per-pixel weight allocations during evaluation.
        if x.shape[0] > 256:
            return torch.cat([
                self.forward_PartiMat(x[start:start + 256], idx_cls[start:start + 256])
                for start in range(0, x.shape[0], 256)
            ], dim=0)
        B, _, _, _ = x.shape
        mode_idx = torch.argmax(idx_cls, dim=1)  # (B, H//p, W//p)
        
        # mode-independent channel
        out = F.conv2d(x[:,self.mode_independent_channel_idx_], self.main_weight[0,:, self.mode_independent_channel_idx], bias = None, stride=self.stride, padding=self.padding, groups = self.groups)
        for i in range(self.groups):
            # mode specific channel
            weight = torch.cat([self.main_weight[:,self.chunk_size*i:self.chunk_size*(i+1), self.mode_adaptive_channel_idx],self.multi_weights[:,:,self.topk*i:self.topk*(i+1)]], dim=0) # 6 Cout Cin k k 
            # 배치별 weight 선택
            weight = weight[mode_idx]  # (B, Cout, Cin, k1, k2)
            weight = rearrange(weight, "B Cout Cin k1 k2 -> (B Cout) Cin k1 k2").contiguous()
            out[:,self.chunk_size*i:self.chunk_size*(i+1)] = out[:,self.chunk_size*i:self.chunk_size*(i+1)] + rearrange(F.conv2d(rearrange(x[:,self.in_ch_chunk_size*i+ self.mode_adaptive_channel_idx],"B C H W -> 1 (B C) H W"), weight, bias = None, stride=self.stride, padding=self.padding, groups = B), "1 (B Cout) H W -> B Cout H W", B = B)

        # bias
        if self.multi_bias is not None:
            full_bias = torch.cat([self.main_bias, self.multi_bias], dim=0)
            bias = full_bias[mode_idx]  # (B, Cout)
        else:
            bias = self.main_bias.expand(B, -1)
        out = out + bias.view(B,self.out_ch,1,1)
        return out
    
    def forward_dynamic_PartiMat(self, x, mode_map):
        # Bound per-pixel weight allocations during evaluation.
        if x.shape[0] > 256:
            return torch.cat([
                self.forward_dynamic_PartiMat(x[start:start + 256], mode_map[start:start + 256])
                for start in range(0, x.shape[0], 256)
            ], dim=0)
        B,_,_,_ = x.shape
        
        # mode-independent channel
        out = F.conv2d(x[:,self.mode_independent_channel_idx_], self.main_weight[0,:, self.mode_independent_channel_idx], bias = None, stride=self.stride, padding=self.padding, groups = self.groups)
        # mode specific channel
        for i in range(self.groups):
            weight = torch.cat([self.main_weight[:,self.chunk_size*i:self.chunk_size*(i+1), self.mode_adaptive_channel_idx],self.multi_weights[:,:,self.topk*i:self.topk*(i+1)]], dim=0) # 6 Cout Cin k k 
            # 배치별 weight 조합
            weight = torch.sum(mode_map.view(B,self.num_class,1,1,1,1) * weight.unsqueeze(0),dim=1)
            weight = rearrange(weight,"B Cout Cin k1 k2 -> (B Cout) Cin k1 k2").contiguous()
            out[:,self.chunk_size*i:self.chunk_size*(i+1)] = out[:,self.chunk_size*i:self.chunk_size*(i+1)] + rearrange(F.conv2d(rearrange(x[:,self.in_ch_chunk_size*i+ self.mode_adaptive_channel_idx],"B C H W -> 1 (B C) H W"), weight, bias = None, stride=self.stride, padding=self.padding, groups = B), "1 (B Cout) H W -> B Cout H W", B = B)
        
        # bias
        if self.multi_bias is not None:
            bias = torch.sum(mode_map.unsqueeze(-1) * torch.cat([self.main_bias, self.multi_bias],dim=0).unsqueeze(0),dim=1) # B 6 1 *1 6 cout
        else:
            bias = self.main_bias.repeat(B,1)
        out = out + bias.view(B,self.out_ch,1,1)
        return out
    
    def forward_BMS(self, x, mode_map = None, mode_map2 = None, unknown_mask = None):
        '''
        x : B C H/2 W/2
        unknown_mask : B H/16 W/16
        mode_map : B m+1 H/16 W/16
        mode_map2 : 
        '''
        B, C, H, W = x.shape        
        out = torch.zeros((B, self.out_ch, H, W)).to(device = x.device).contiguous()
        with torch.no_grad():
            flag = unknown_mask.sum()>0
        
        ######### run PartiMat ###########
        _H = H // self.patch_size
        _W = W // self.patch_size
        
        _unknown_mask = rearrange(unknown_mask, "b h w -> (b h w)").contiguous().bool()
        if (~_unknown_mask).any():
            out = rearrange(out, "b c (h p1) (w p2) -> (b h w) c p1 p2", p1 = self.patch_size, p2 = self.patch_size)
            out[~_unknown_mask] = self.forward_PartiMat(rearrange(x.contiguous(), 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1= self.patch_size, p2= self.patch_size)[~_unknown_mask], rearrange(mode_map.contiguous(), 'b c h w -> (b h w) c')[~_unknown_mask]) # (B h w) Cout P1 P2
            out = rearrange(out, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h = _H, w = _W)
        
        ######### run Dynamic PartiMat ###########
        if flag:
            _unknown_mask = unknown_mask.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
            _unknown_mask = rearrange(_unknown_mask, "b h w -> (b h w)").contiguous()
            out = rearrange(out, "b c h w -> (b h w) c 1 1")
            out[_unknown_mask] = self.forward_dynamic_PartiMat(rearrange(x, 'b c h w -> (b h w) c 1 1')[_unknown_mask], rearrange(mode_map2, 'b c h w -> (b h w) c')[_unknown_mask]) # (B h w) Cout P1 P2
                
            out = rearrange(out, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h = H, w = W)
            
        return out 
    
   
class MultiModule_linear_input(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        num_class = 1,
        multi_channel_idx = [1,2,3],
        multi_bias = True,
        bias = True,
        patch_size = 8,
    ):
        super().__init__()
        self.out_ch = out_ch
        self.mode_adaptive_channel_idx = multi_channel_idx
        self.mode_independent_channel_idx = list(range(len(self.mode_adaptive_channel_idx),in_ch))
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
        
        self.patch_size = patch_size
        
    def forward(self,x, H, W, mode_map = None, mode_map2 = None, unknown_mask = None, est = True):
        if est:
            return self.forward_BMS(x, H, W, mode_map, mode_map2, unknown_mask)
        else:
            idx_cls = mode_map[:,:,0,0]
            return self.forward_PartiMat(x,idx_cls)
                    
    def forward_BMS(self, x, H, W, mode_map = None, mode_map2 = None, unknown_mask = None):
        '''
        x : B (H/2 W/2) C
        unknown_mask : B H/16 W/16
        mode_map : B m+1 H/16 W/16
        mode_map2 : 
        '''
        B, L, C = x.shape        
        out = torch.zeros((B, L, self.out_ch), device = x.device).contiguous()
        with torch.no_grad():
            flag = unknown_mask.sum()>0
        
        ######### run PartiMat ###########
        _H = H // self.patch_size
        _W = W // self.patch_size
        
        _unknown_mask = rearrange(unknown_mask, "b h w -> (b h w)").contiguous().bool()
        
        out = rearrange(out, 'b (h p1 w p2) c -> (b h w) (p1 p2) c', h = _H, w = _W, p1 = self.patch_size, p2 = self.patch_size)
        out[~_unknown_mask] = self.forward_PartiMat(rearrange(x, 'b (h p1 w p2) c -> (b h w) (p1 p2) c', h = _H, w = _W, p1= self.patch_size, p2= self.patch_size)[~_unknown_mask], rearrange(mode_map, 'b c h w -> (b h w) c')[~_unknown_mask]) # (B h w) Cout P1 P2
        
        out = rearrange(out, '(b h w) (p1 p2) c -> b (h p1 w p2) c', h = _H, w = _W, p1 = self.patch_size, p2 = self.patch_size)     
        
        ######### run Dynamic PartiMat ###########
        if flag:
            _unknown_mask = unknown_mask.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
            _unknown_mask = rearrange(_unknown_mask, "b h w -> (b h w)").contiguous().bool()   
            
            out = rearrange(out, 'b (h w) c -> (b h w) 1 c', h = H, w = W)
     
            out[_unknown_mask] = self.forward_dynamic_PartiMat(rearrange(x, 'b (h w) c -> (b h w) 1 c', h = H, w = W)[_unknown_mask], rearrange(mode_map2, 'b c h w -> (b h w) c')[_unknown_mask]) # (B h w) Cout P1 P2
            
            out = rearrange(out, '(b h w) 1 c -> b (h w) c', h = H, w = W)
        return out 
    
    def forward_PartiMat(self, x, idx_cls): # idx_cls : B 6 H//p W//p
        # Bound per-pixel weight allocations during evaluation.
        if x.shape[0] > 256:
            return torch.cat([
                self.forward_PartiMat(x[start:start + 256], idx_cls[start:start + 256])
                for start in range(0, x.shape[0], 256)
            ], dim=0)
        B,_,_ = x.shape
        # mode independent channel
        out = torch.bmm(x[:,:,self.mode_independent_channel_idx],self.main_weight[:,:,self.mode_independent_channel_idx].repeat(B,1,1).transpose(1,2))
        # mode specific channel
        weight = torch.cat([self.main_weight[:,:,self.mode_adaptive_channel_idx],self.multi_weights],dim=0).unsqueeze(0).repeat(B,1,1,1)[torch.arange(B, device = x.device),torch.argmax(idx_cls,dim=1)] # B 6 Cout k -> B Cout k
        out = out + torch.bmm(x[:,:,self.mode_adaptive_channel_idx],weight.transpose(1,2))
        # bias
        if self.bias:
            if self.multi_bias is None:
                bias = self.main_bias.repeat(B,1)
            else:
                bias = torch.cat([self.main_bias,self.multi_bias],dim=0).unsqueeze(0).repeat(B,1,1)[torch.arange(B, device = x.device),torch.argmax(idx_cls,dim=1)]
            out = out + bias.unsqueeze(1)
        return out
    
    def forward_dynamic_PartiMat(self, x, mode_map): # idx_cls : B 6 H//p W//p
        # Bound per-pixel weight allocations during evaluation.
        if x.shape[0] > 256:
            return torch.cat([
                self.forward_dynamic_PartiMat(x[start:start + 256], mode_map[start:start + 256])
                for start in range(0, x.shape[0], 256)
            ], dim=0)
        B,_,_ = x.shape
        
        # mode independent channel
        out = torch.bmm(x[:,:,self.mode_independent_channel_idx],self.main_weight[:,:,self.mode_independent_channel_idx].repeat(B,1,1).transpose(1,2))
        
        # mode adaptive
        weight = torch.sum(torch.cat([self.main_weight[:,:,self.mode_adaptive_channel_idx],self.multi_weights],dim=0).unsqueeze(0) * mode_map.view(B,self.num_class,1,1), dim=1)          # B 6 Cout k * B 6 -> B Cout k
        out = out + torch.bmm(x[:,:,self.mode_adaptive_channel_idx],weight.transpose(1,2))
        # bias
        if self.bias:
            if self.multi_bias is None:
                bias = self.main_bias.repeat(B,1)
            else:
                bias = torch.sum(torch.cat([self.main_bias, self.multi_bias],dim=0).unsqueeze(0) * mode_map.view(B,self.num_class,1),dim=1)
            out = out + bias.unsqueeze(1)
        return out