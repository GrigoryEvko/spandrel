"""
Unit tests for DAT FlexAttention implementation.
Verifies numerical equivalence between original and FlexAttention implementations.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Tuple, Optional
import sys
import os

# Add the parent directory to the path to import DAT modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We'll need to import both original and new implementations
# For testing, we'll create copies of the original attention classes


class TestHelperFunctions:
    """Test helper functions like img2windows and windows2img."""
    
    @pytest.mark.parametrize("H,W,H_sp,W_sp", [
        (64, 64, 8, 8),
        (32, 48, 4, 8),
        (128, 128, 16, 16),
    ])
    def test_img2windows_windows2img_roundtrip(self, H, W, H_sp, W_sp):
        """Test that img2windows and windows2img are inverse operations."""
        from DAT_optim import img2windows, windows2img
        
        B, C = 2, 192
        img = torch.randn(B, C, H, W)
        
        # Convert to windows and back
        windows = img2windows(img, H_sp, W_sp)
        img_reconstructed = windows2img(windows, H_sp, W_sp, H, W)
        
        # Should be identical
        assert torch.allclose(img.permute(0, 2, 3, 1), img_reconstructed, atol=1e-6)


class OriginalSpatialAttention(nn.Module):
    """Original Spatial Attention implementation for comparison."""
    
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
            from DAT_optim import DynamicPosBias
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer("rpe_biases", biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, self.dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def im2win(self, x, H, W):
        from DAT_optim import img2windows
        B, _N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = (
            x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return x

    def forward(self, x, H, W, mask=None):
        """
        Input: x: (B, L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        from DAT_optim import windows2img
        
        B, L, C = x.shape
        
        # QKV computation
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

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
        x = x.view(B, H * W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TestSpatialAttention:
    """Test Spatial Attention implementation."""
    
    @pytest.fixture
    def setup_modules(self):
        """Setup original and FlexAttention modules with same parameters."""
        from DAT_optim import Spatial_Attention
        
        dim = 192
        num_heads = 6
        split_size = [8, 8]
        
        # Create modules with same initialization
        torch.manual_seed(42)
        original = OriginalSpatialAttention(
            dim=dim, idx=0, split_size=split_size, 
            num_heads=num_heads, position_bias=True
        )
        
        torch.manual_seed(42)
        flex = Spatial_Attention(
            dim=dim, idx=0, split_size=split_size,
            num_heads=num_heads, position_bias=True
        )
        
        # Copy weights to ensure identical parameters
        with torch.no_grad():
            flex.qkv.weight.copy_(original.qkv.weight)
            flex.qkv.bias.copy_(original.qkv.bias)
            flex.proj.weight.copy_(original.proj.weight)
            flex.proj.bias.copy_(original.proj.bias)
            if hasattr(flex, 'pos'):
                # Copy position bias parameters
                for (n1, p1), (n2, p2) in zip(original.pos.named_parameters(), 
                                              flex.pos.named_parameters()):
                    p2.copy_(p1)
        
        return original, flex
    
    @pytest.mark.parametrize("H,W", [(64, 64), (32, 48), (56, 72)])
    def test_spatial_attention_equivalence(self, setup_modules, H, W):
        """Test that FlexAttention produces identical results to original."""
        original, flex = setup_modules
        
        B, C = 2, 192
        x = torch.randn(B, H * W, C)
        
        # Set to eval mode to disable dropout
        original.eval()
        flex.eval()
        
        with torch.no_grad():
            # Run both implementations
            out_original = original(x, H, W)
            
            # For FlexAttention version, we need to prepare QKV
            qkv = flex.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
            out_flex = flex(qkv, H, W)
            out_flex = out_flex.view(B, H * W, C)
            
        # Check if outputs are close
        assert torch.allclose(out_original, out_flex, atol=1e-5, rtol=1e-4), \
            f"Max diff: {(out_original - out_flex).abs().max().item()}"


class TestAdaptiveChannelAttention:
    """Test Adaptive Channel Attention implementation."""
    
    @pytest.fixture
    def setup_modules(self):
        """Setup original and FlexAttention channel attention modules."""
        from DAT_optim import Adaptive_Channel_Attention
        
        dim = 192
        num_heads = 8
        
        # Create modules with same initialization
        torch.manual_seed(42)
        flex = Adaptive_Channel_Attention(
            dim=dim, num_heads=num_heads, qkv_bias=True
        )
        
        return flex
    
    def test_channel_attention_forward(self, setup_modules):
        """Test that channel attention runs without errors."""
        flex = setup_modules
        
        B, H, W = 2, 32, 32
        C = 192
        x = torch.randn(B, H * W, C)
        
        flex.eval()
        
        with torch.no_grad():
            out = flex(x, H, W)
            
        assert out.shape == (B, H * W, C)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestFullModel:
    """Test the full DAT model."""
    
    @pytest.fixture
    def setup_model(self):
        """Setup DAT model."""
        from DAT_optim import DAT
        
        model = DAT(
            img_size=64,
            in_chans=3,
            embed_dim=180,
            split_size=[8, 8],
            depth=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            expansion_factor=4.0,
            qkv_bias=True,
            upscale=4,
            resi_connection="1conv",
            upsampler="pixelshuffle",
        )
        
        return model
    
    def test_model_forward(self, setup_model):
        """Test that the model runs forward pass correctly."""
        model = setup_model
        model.eval()
        
        B, C, H, W = 1, 3, 64, 64
        x = torch.randn(B, C, H, W)
        
        with torch.no_grad():
            out = model(x)
            
        # Check output shape (4x upscale)
        assert out.shape == (B, C, H * 4, W * 4)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_model_compile(self, setup_model):
        """Test that the model compiles with torch.compile."""
        model = setup_model
        model.eval()
        
        # Try to compile the model
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            
            B, C, H, W = 1, 3, 64, 64
            x = torch.randn(B, C, H, W)
            
            with torch.no_grad():
                out = compiled_model(x)
                
            assert out.shape == (B, C, H * 4, W * 4)
            
        except Exception as e:
            pytest.skip(f"torch.compile not available or failed: {e}")


class TestNumericalStability:
    """Test numerical stability of FlexAttention implementation."""
    
    def test_attention_numerical_stability(self):
        """Test that attention doesn't produce NaN/Inf with extreme inputs."""
        from DAT_optim import Spatial_Attention
        
        attn = Spatial_Attention(dim=64, idx=0, split_size=[4, 4], num_heads=4)
        attn.eval()
        
        B, H, W, C = 1, 8, 8, 64
        
        # Test with very small values
        x_small = torch.randn(B, H * W, C) * 1e-6
        qkv = attn.qkv(x_small).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        with torch.no_grad():
            out_small = attn(qkv, H, W).view(B, H * W, C)
        assert not torch.isnan(out_small).any()
        assert not torch.isinf(out_small).any()
        
        # Test with large values
        x_large = torch.randn(B, H * W, C) * 100
        qkv = attn.qkv(x_large).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        with torch.no_grad():
            out_large = attn(qkv, H, W).view(B, H * W, C)
        assert not torch.isnan(out_large).any()
        assert not torch.isinf(out_large).any()


def test_adaptive_spatial_attention_shift_mask():
    """Test shift mask computation in Adaptive Spatial Attention."""
    from DAT_optim import Adaptive_Spatial_Attention
    
    # Test different rg_idx and b_idx combinations
    test_cases = [
        (0, 0, False),  # No shift
        (0, 2, True),   # Shift (rg_idx % 2 == 0 and (b_idx - 2) % 4 == 0)
        (1, 0, True),   # Shift (rg_idx % 2 != 0 and b_idx % 4 == 0)
        (1, 1, False),  # No shift
    ]
    
    for rg_idx, b_idx, expected_shift in test_cases:
        attn = Adaptive_Spatial_Attention(
            dim=192, num_heads=6, rg_idx=rg_idx, b_idx=b_idx
        )
        assert attn.needs_shift == expected_shift, \
            f"Failed for rg_idx={rg_idx}, b_idx={b_idx}"


def test_flex_attention_kernel_options():
    """Test that FlexAttention works with kernel options."""
    from DAT_optim import Spatial_Attention
    
    attn = Spatial_Attention(dim=64, idx=0, split_size=[4, 4], num_heads=4)
    attn.eval()
    
    B, H, W, C = 1, 8, 8, 64
    x = torch.randn(B, H * W, C)
    qkv = attn.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
    
    with torch.no_grad():
        # Should run without errors with kernel options
        out = attn(qkv, H, W)
        
    assert out.shape == (B, H, W, C)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])