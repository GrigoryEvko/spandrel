from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

from spandrel.util import store_hyperparameters
from spandrel.util.timm import DropPath, trunc_normal_


# Import vectorized operations
from .vectorized_ops import img2windows, windows2img, calculate_mask_vectorized, generate_position_bias




class SpatialGate(nn.Module):
    """Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
        )  # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, _N, C = x.shape
        x2 = (
            self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W))
            .flatten(2)
            .transpose(-1, -2)
            .contiguous()
        )

        return x1 * x2


class SGFN(nn.Module):
    """Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads),
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Spatial_Attention(nn.Module):
    """Spatial Window Self-Attention.
    It supports rectangle window (containing square window).
    Args:
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """

    def __init__(
        self,
        dim,
        idx,
        split_size=[8, 8],
        dim_out=None,
        num_heads=6,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_scale=None,
        position_bias=True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            raise ValueError(f"ERROR MODE: {idx}")
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # Use JIT-friendly position bias generation
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rpe_biases, relative_position_index = generate_position_bias(self.H_sp, self.W_sp, device)
            self.register_buffer("rpe_biases", rpe_biases)
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, _N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = (
            x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1
            )
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0
            )
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(
            -1, self.H_sp * self.W_sp, C
        )  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x


class Adaptive_Spatial_Attention(nn.Module):
    # The implementation builds on CAT code https://github.com/Zhengchen1999/CAT
    """Adaptive Spatial Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        split_size (tuple(int)): Height and Width of spatial window.
        shift_size (tuple(int)): Shift size for spatial window.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        rg_idx (int): The indentix of Residual Group (RG)
        b_idx (int): The indentix of Block in each RG
    """

    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=[8, 8],
        shift_size=[1, 2],
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        rg_idx=0,
        b_idx=0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert (
            0 <= self.shift_size[0] < self.split_size[0]
        ), "shift_size must in 0-split_size0"
        assert (
            0 <= self.shift_size[1] < self.split_size[1]
        ), "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList(
            [
                Spatial_Attention(
                    dim // 2,
                    idx=i,
                    split_size=split_size,
                    num_heads=num_heads // 2,
                    dim_out=dim // 2,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    position_bias=True,
                )
                for i in range(self.branch_num)
            ]
        )

        # Determine if we actually need the mask based on block indices
        # shift in block: (0, 4, 8, ...), (2, 6, 10, ...), (0, 4, 8, ...), (2, 6, 10, ...), ...
        use_real_mask = (
            (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or 
            (self.rg_idx % 2 != 0 and self.b_idx % 4 == 0)
        )
        
        # Pre-compute shift values as Python tuples in __init__ 
        # Since these are static values, we can compute them once here
        # This avoids ANY runtime operations in the forward pass
        if use_real_mask:
            # Actual shift values
            self.shift_0 = (-self.shift_size[0], -self.shift_size[1])
            self.shift_1 = (-self.shift_size[1], -self.shift_size[0])
            self.unshift_0 = (self.shift_size[0], self.shift_size[1])
            self.unshift_1 = (self.shift_size[1], self.shift_size[0])
        else:
            # Zero shifts (no-op)
            self.shift_0 = (0, 0)
            self.shift_1 = (0, 0)
            self.unshift_0 = (0, 0)
            self.unshift_1 = (0, 0)
        
        # Note: torch._dynamo.mark_static doesn't work for non-tensor attributes
        # The shift values being constants in __init__ should be enough for torch.compile
        
        # Don't pre-compute masks - always calculate dynamically to avoid recompilations
        # This ensures consistent behavior across all input sizes
        self.use_real_mask = use_real_mask

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
        )

    def calculate_mask(self, H, W):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.use_real_mask:
            # Use vectorized implementation for better GPU utilization
            return calculate_mask_vectorized(H, W, self.split_size, self.shift_size, device)
        else:
            # Return zero masks that have no effect
            num_windows_0 = (H // self.split_size[0]) * (W // self.split_size[1])
            num_windows_1 = (H // self.split_size[1]) * (W // self.split_size[0])
            window_size = self.split_size[0] * self.split_size[1]
            
            zero_mask_0 = torch.zeros((num_windows_0, window_size, window_size), device=device)
            zero_mask_1 = torch.zeros((num_windows_1, window_size, window_size), device=device)
            
            return [zero_mask_0, zero_mask_1]
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Handle backward compatibility - ignore attn_mask buffers from old checkpoints."""
        # Remove attn_mask keys from state_dict if present (we don't use them anymore)
        attn_mask_0_key = prefix + 'attn_mask_0'
        attn_mask_1_key = prefix + 'attn_mask_1'
        
        # Pop the keys if they exist in the checkpoint (we calculate masks dynamically now)
        if attn_mask_0_key in state_dict:
            state_dict.pop(attn_mask_0_key)
        if attn_mask_1_key in state_dict:
            state_dict.pop(attn_mask_1_key)
        
        # Initialize shift values if not already set (for backward compatibility)
        if not hasattr(self, 'shift_0'):
            use_real_mask = (
                (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or 
                (self.rg_idx % 2 != 0 and self.b_idx % 4 == 0)
            )
            if use_real_mask:
                self.shift_0 = (-self.shift_size[0], -self.shift_size[1])
                self.shift_1 = (-self.shift_size[1], -self.shift_size[0])
                self.unshift_0 = (self.shift_size[0], self.shift_size[1])
                self.unshift_1 = (self.shift_size[1], self.shift_size[0])
            else:
                self.shift_0 = (0, 0)
                self.shift_1 = (0, 0)
                self.unshift_0 = (0, 0)
                self.unshift_1 = (0, 0)
            
        
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, HW, C
        # V without partition
        v = qkv[2].transpose(-2, -1).contiguous().view(B, C, H, W)

        # image padding
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)  # 3B C H W
        qkv = (
            F.pad(qkv, (pad_l, pad_r, pad_t, pad_b))
            .reshape(3, B, C, -1)
            .transpose(-2, -1)
        )  # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        # Unified forward pass without ANY conditionals or Python operations
        # All shift values are pre-computed as tensor buffers in __init__
        # This completely eliminates runtime conditionals, computation, and type conversions
        
        # Always perform the same operations
        qkv = qkv.view(3, B, _H, _W, C)
        
        # Apply shifts using pre-computed tuples (will be no-op with (0,0) shifts)
        qkv_0 = torch.roll(
            qkv[:, :, :, :, : C // 2],
            shifts=self.shift_0,  # Direct use of pre-computed tuple
            dims=(2, 3),
        )
        qkv_0 = qkv_0.view(3, B, _L, C // 2)
        
        qkv_1 = torch.roll(
            qkv[:, :, :, :, C // 2 :],
            shifts=self.shift_1,  # Direct use of pre-computed tuple
            dims=(2, 3),
        )
        qkv_1 = qkv_1.view(3, B, _L, C // 2)

        # Always calculate masks dynamically to avoid recompilations
        mask_tmp = self.calculate_mask(_H, _W)
        x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device).to(x.dtype))
        x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device).to(x.dtype))

        # Apply unshifts using pre-computed tuples (will be no-op with (0,0) shifts)
        x1 = torch.roll(
            x1_shift, shifts=self.unshift_0, dims=(1, 2)
        )
        x2 = torch.roll(
            x2_shift, shifts=self.unshift_1, dims=(1, 2)
        )
        
        x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
        x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
        
        # attention output
        attened_x = torch.cat([x1, x2], dim=2)

        # convolution output
        conv_x = self.dwconv(v)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        channel_map = (
            self.channel_interaction(conv_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(B, 1, C)
        )
        # S-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)

        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map)
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Adaptive_Channel_Attention(nn.Module):
    # The implementation builds on XCiT code https://github.com/facebookresearch/xcit
    """Adaptive Channel Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1),
        )

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # convolution output
        conv_x = self.dwconv(v_)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = (
            self.spatial_interaction(conv_x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(B, N, 1)
        )

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DATB(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=[2, 4],
        shift_size=[1, 2],
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rg_idx=0,
        b_idx=0,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        if b_idx % 2 == 0:
            # DSTB
            self.attn = Adaptive_Spatial_Attention(
                dim,
                num_heads=num_heads,
                reso=reso,
                split_size=split_size,
                shift_size=shift_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                rg_idx=rg_idx,
                b_idx=b_idx,
            )
        else:
            # DCTB
            self.attn = Adaptive_Channel_Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn = SGFN(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))

        return x


class ResidualGroup(nn.Module):
    """ResidualGroup
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (tuple(int)): Height and Width of spatial window.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float): Dropout rate. Default: 0
        attn_drop(float): Attention dropout rate. Default: 0
        drop_paths (float | None): Stochastic depth rate.
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        depth (int): Number of dual aggregation Transformer blocks in residual group.
        use_chk (bool): Whether to use checkpointing to save memory.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        dim,
        reso,
        num_heads,
        split_size=[2, 4],
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_paths=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        depth=2,
        use_chk=False,
        resi_connection="1conv",
        rg_idx=0,
        regional_compile=False,
    ):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso
        self.regional_compile = regional_compile

        self.blocks = nn.ModuleList()
        for i in range(depth):
            datb = DATB(
                dim=dim,
                num_heads=num_heads,
                reso=reso,
                split_size=split_size,
                shift_size=[split_size[0] // 2, split_size[1] // 2],
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],  # type: ignore
                act_layer=act_layer,
                norm_layer=norm_layer,
                rg_idx=rg_idx,
                b_idx=i,
            )
            
            # Don't apply torch.compile here - it will be done after state dict loading
            
            self.blocks.append(datb)

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        res = x
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = res + x

        return x


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. " "Supported scales: 2^n and 3."
            )
        super().__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super().__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@store_hyperparameters()
class DAT(nn.Module):
    """Dual Aggregation Transformer
    Args:
        img_size (int): Input image size. Default: 64
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 180
        depths (tuple(int)): Depth of each residual group (number of DATB in each RG).
        split_size (tuple(int)): Height and Width of spatial window.
        num_heads (tuple(int)): Number of attention heads in different residual groups.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        use_chk (bool): Whether to use checkpointing to save memory.
        upscale: Upscale factor. 2/3/4 for image SR
        img_range: Image range. 1. or 255.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        img_size=64,
        in_chans=3,
        embed_dim=180,
        split_size=[2, 4],
        depth=[2, 2, 2, 2],
        num_heads=[2, 2, 2, 2],
        expansion_factor=4.0,
        qkv_bias=True,
        qk_scale: float | None = None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_chk=False,
        upscale=2,
        img_range=1.0,
        resi_connection="1conv",
        upsampler="pixelshuffle",
        regional_compile=False,
    ):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            mean = torch.zeros(1, 1, 1, 1)
        self.register_buffer('mean', mean)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.regional_compile = regional_compile
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        heads = num_heads

        self.before_RG = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"), nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                split_size=split_size,
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]) : sum(depth[: i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                rg_idx=i,
                regional_compile=regional_compile,
            )
            self.layers.append(layer)

        self.norm = norm_layer(curr_dim)
        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # ------------------------- 3, Reconstruction ------------------------- #
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch, (img_size, img_size)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:  # type: ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(
            m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)
        ):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x

    def load_state_dict(self, state_dict, strict=True):
        """
        Override to handle backward compatibility for the 'mean' buffer and attn_mask buffers.
        Old checkpoints don't have 'mean' or attn_mask in their state_dict.
        """
        # Check if 'mean' is missing from the state_dict
        if 'mean' not in state_dict and hasattr(self, 'mean'):
            # Create a copy to avoid modifying the original state_dict
            state_dict = dict(state_dict)
            # Add the current mean buffer value to the state_dict
            # This ensures backward compatibility with old checkpoints
            state_dict['mean'] = self.mean
        
        # Check if we're missing attn_mask keys - these are from old checkpoints
        # If any attn_mask keys are missing, load with strict=False since these
        # masks are dynamically calculated anyway
        missing_attn_masks = False
        for key in list(self.state_dict().keys()):
            if 'attn_mask_' in key and key not in state_dict:
                missing_attn_masks = True
                break
        
        if missing_attn_masks and strict:
            # Load with strict=False for backward compatibility, then verify
            # that only attn_mask keys were missing
            missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)
            
            # Check if any non-attn_mask keys are missing
            non_mask_missing = [k for k in missing_keys if 'attn_mask_' not in k]
            if non_mask_missing:
                raise RuntimeError(f"Missing required keys in state_dict: {non_mask_missing}")
            
            return missing_keys, unexpected_keys
        else:
            # Call the parent's load_state_dict with the potentially modified state_dict
            return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        """
        # Ensure mean is same dtype as input for numerical equivalence
        # Using .to(x.dtype) instead of type_as for CUDA graphs compatibility
        mean = self.mean.to(x.dtype)
        x = (x - mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for image SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        x = x / self.img_range + mean
        return x
    
    def apply_regional_compilation(self):
        """Apply regional compilation to attention and FFN modules after state dict is loaded."""
        if not self.regional_compile:
            return
            
        # Ensure model is in eval mode before compilation
        self.eval()
        
        print("Applying regional compilation to DAT modules...")
        
        # Create compile options with reduce-overhead for faster compilation
        compile_options = {
            "mode": "reduce-overhead",  # Faster compilation, good performance
            "fullgraph": False,
            "dynamic": True,  # Enable dynamic shapes for H, W dimensions
        }
        
        for layer in self.layers:
            for i, blk in enumerate(layer.blocks):
                # Compile attention modules
                blk.attn = torch.compile(blk.attn, **compile_options)
                # Compile FFN modules
                blk.ffn = torch.compile(blk.ffn, **compile_options)
                
        print("Regional compilation applied successfully.")
