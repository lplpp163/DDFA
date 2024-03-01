import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np

from VideoSwin import SwinTransformerBlock3D, get_window_size, compute_mask

class DDFA(nn.Module):
    def __init__(self, W=16, D=4, window_size=(4,4,4)): # W: model width, D: model depth
        super(DDFA, self).__init__()
        
        self.stem = nn.Sequential(
            conv_bn_act(3, W, kernel_size=1),
            ResBlock(W),
        )
        
        self.enc_res = nn.Sequential(
            DownBlock( W, W*2),
            ResBlock(W*2),
        )
        
        self.encs_swin = nn.ModuleList([nn.Sequential(
            DownBlock( W*(2**(i-1)), W*(2**i)),
            Swin3DLayer( dim=W*(2**i), num_heads=4, window_size=window_size),
        ) for i in range(2, D)])
        
        self.decs = nn.ModuleList([UpBlock(W*(2**i), W*(2**(i-1))) for i in range(D-1, 0, -1)])
        
        self.conv_out = nn.Sequential(
            conv_bn(W, 1, kernel_size=1),
            Rearrange('b c n h w -> b (c n) h w'),
            nn.Softmax(1),
        )

    def forward(self, x):
        
        x = x.transpose(1,2) # B C N H W
        
        x_s = []
        x = self.stem(x)
        x_s.append(x)
        
        # Encode Res
        x = self.enc_res(x)
        x_s.append(x)
        
        # Encode Swin
        for enc in (self.encs_swin):
            x = enc(x)
            x_s.append(x)
        x_s.pop(-1)

        
        # Decode
        for i, dec in enumerate(self.decs):
            x = dec(x, x_s.pop(-1))

        # out
        output = self.conv_out(x).squeeze(1) # B H W
        
        return output

class Swin3DLayer(nn.Module):
    def __init__(self, dim, depth=2, num_heads=4, window_size=(4,4,4),
                 mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0.1, attn_drop=0.1, drop_path=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

    def forward(self,x):
        B,C,N,H,W = x.shape
        window_size, shift_size = get_window_size((N,H,W), self.window_size, self.shift_size)
        
        x = rearrange(x, 'b c n h w -> b n h w c')
        Dp = int(np.ceil(N / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, N, H, W, -1)

        x = rearrange(x, 'b n h w c -> b c n h w')
        
        return x

class ResBlock(nn.Module):
    def __init__(self, in_ch, kernel_size=3, stride=1, padding='same'):
        super().__init__()
        
        self.net = nn.Sequential(
            conv_bn_act(in_ch, in_ch, kernel_size, stride, padding),
            conv_bn(in_ch, in_ch, kernel_size, stride, padding),
        )
        self.act = nn.PReLU()
        
    def forward(self,x):
        return self.act( x + self.net(x) )

def conv_bn(in_ch, out_ch, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_ch),
        )

def conv_bn_act(in_ch, out_ch, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(
            conv_bn(in_ch, out_ch, kernel_size, stride, padding),
            nn.PReLU(),
        )
    
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, factor=2):
        super().__init__()
        
        factor3d = (1,factor,factor)
        self.strided = conv_bn(in_ch, out_ch, stride=factor3d, padding=1)
        self.max = nn.Sequential(
            nn.MaxPool3d(kernel_size=factor3d,stride=factor3d),
            conv_bn(in_ch, out_ch, kernel_size=1),
        )
        self.act= nn.PReLU()
    def forward(self,x):
        return self.act( self.strided(x) + self.max(x) )
    
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, factor=2):
        super().__init__()
        
        factor3d = (1,factor,factor)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=factor3d, mode='trilinear'),
            conv_bn(in_ch, out_ch),
        )
        
        self.cca = CSA(out_ch, out_ch)
        
        self.out = conv_bn_act(out_ch*2,out_ch, 1)
    def forward(self, x, skip):
        up = self.net(x)
        
        skip_att = self.cca(up, skip)

        return self.out( torch.cat([up, skip_att], dim=1) )

    
class CSA(nn.Module):
    """
    Cross-Slice Attention Block
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.mlp_x = nn.Linear(in_ch, out_ch)
        self.mlp_g = nn.Linear(in_ch, out_ch)
        
        self.act = nn.PReLU()

    def forward(self, g, x):

        avg_pool_x = F.adaptive_avg_pool2d( x, 1 ).flatten(2).transpose(1,2) # b n c
        att_x = self.mlp_x(avg_pool_x)
        
        avg_pool_g = F.adaptive_avg_pool2d( g, 1 ).flatten(2).transpose(1,2)
        att_g = self.mlp_g(avg_pool_g)
        
        scale = torch.sigmoid(att_x + att_g).transpose(1,2)[...,None,None].expand_as(x)

        return self.act(x * scale)