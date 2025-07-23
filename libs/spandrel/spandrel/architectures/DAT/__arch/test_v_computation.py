"""
Test to verify v computation for convolution branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import both implementations
from DAT import Adaptive_Channel_Attention as OriginalChannelAttention
from DAT_optim import Adaptive_Channel_Attention as FlexChannelAttention


def test_v_computation():
    """Test that v is computed correctly for the convolution branch."""
    torch.manual_seed(42)
    
    dim = 96
    num_heads = 4
    B, H, W = 1, 8, 8
    N = H * W
    
    # Create input
    x = torch.randn(B, N, dim)
    
    # Create modules
    original = OriginalChannelAttention(dim=dim, num_heads=num_heads, qkv_bias=True)
    flex = FlexChannelAttention(dim=dim, num_heads=num_heads, qkv_bias=True)
    
    # Copy weights
    flex.load_state_dict(original.state_dict())
    
    original.eval()
    flex.eval()
    
    # Get QKV for original
    qkv_orig = original.qkv(x).reshape(B, N, 3, num_heads, dim // num_heads)
    qkv_orig = qkv_orig.permute(2, 0, 3, 1, 4)
    q_orig, k_orig, v_orig = qkv_orig.unbind(0)
    
    # Original computes v for conv after transpose
    v_orig_t = v_orig.transpose(-2, -1)
    v_2d_orig = v_orig_t.reshape(B, dim, N).contiguous().view(B, dim, H, W)
    
    # Get QKV for flex
    qkv_flex = flex.qkv(x).reshape(B, N, 3, num_heads, dim // num_heads)
    qkv_flex = qkv_flex.permute(2, 0, 3, 1, 4)
    q_flex, k_flex, v_flex = qkv_flex.unbind(0)
    
    # Flex computes v for conv after transpose
    v_flex_t = v_flex.transpose(-2, -1)
    v_2d_flex = v_flex_t.reshape(B, dim, N).contiguous().view(B, dim, H, W)
    
    print("=== V Computation Test ===")
    print(f"v_orig shape: {v_orig.shape}")
    print(f"v_orig_t shape: {v_orig_t.shape}")
    print(f"v_2d_orig shape: {v_2d_orig.shape}")
    print()
    print(f"v_flex shape: {v_flex.shape}")
    print(f"v_flex_t shape: {v_flex_t.shape}")
    print(f"v_2d_flex shape: {v_2d_flex.shape}")
    print()
    print(f"v matches: {torch.allclose(v_orig, v_flex)}")
    print(f"v_t matches: {torch.allclose(v_orig_t, v_flex_t)}")
    print(f"v_2d matches: {torch.allclose(v_2d_orig, v_2d_flex)}")
    
    # Test convolution output
    conv_x_orig = original.dwconv(v_2d_orig)
    conv_x_flex = flex.dwconv(v_2d_flex)
    
    print(f"\nconv_x matches: {torch.allclose(conv_x_orig, conv_x_flex)}")
    
    # Now test inside forward pass
    print("\n=== Forward Pass V Computation ===")
    
    # Hook to capture v_2d in forward pass
    v_2d_captured_orig = None
    v_2d_captured_flex = None
    
    def capture_orig_v2d(module, input):
        nonlocal v_2d_captured_orig
        v_2d_captured_orig = input[0].clone()
    
    def capture_flex_v2d(module, input):
        nonlocal v_2d_captured_flex
        v_2d_captured_flex = input[0].clone()
    
    # Register hooks
    handle_orig = original.dwconv.register_forward_pre_hook(capture_orig_v2d)
    handle_flex = flex.dwconv.register_forward_pre_hook(capture_flex_v2d)
    
    # Run forward passes
    with torch.no_grad():
        out_orig = original(x, H, W)
        out_flex = flex(x, H, W)
    
    # Remove hooks
    handle_orig.remove()
    handle_flex.remove()
    
    print(f"Captured v_2d shapes - orig: {v_2d_captured_orig.shape}, flex: {v_2d_captured_flex.shape}")
    print(f"Captured v_2d matches: {torch.allclose(v_2d_captured_orig, v_2d_captured_flex)}")
    
    if not torch.allclose(v_2d_captured_orig, v_2d_captured_flex):
        diff = (v_2d_captured_orig - v_2d_captured_flex).abs()
        print(f"v_2d diff max: {diff.max():.6f}")
        print(f"v_2d diff mean: {diff.mean():.6f}")
        
        # Check a few values
        print("\nSample values:")
        print(f"orig[0,0,0,0]: {v_2d_captured_orig[0,0,0,0]:.6f}")
        print(f"flex[0,0,0,0]: {v_2d_captured_flex[0,0,0,0]:.6f}")


if __name__ == "__main__":
    test_v_computation()