"""
Debug test for Adaptive Spatial Attention shift issues.
"""

import torch
import torch.nn as nn

# Import both implementations
from DAT import Adaptive_Spatial_Attention as OriginalAdaptiveSpatial
from DAT_optim import Adaptive_Spatial_Attention as FlexAdaptiveSpatial


def test_adaptive_spatial_shifts():
    """Test adaptive spatial attention with shifts."""
    print("=== Adaptive Spatial Attention Shift Debug ===\n")
    
    # Test configuration that fails
    dim = 192
    num_heads = 6
    H, W = 64, 64
    B = 1
    N = H * W
    
    # Test case that fails: rg_idx=0, b_idx=2 (should have shift)
    rg_idx, b_idx = 0, 2
    
    print(f"Testing rg_idx={rg_idx}, b_idx={b_idx}")
    
    # Create modules
    torch.manual_seed(42)
    original = OriginalAdaptiveSpatial(
        dim=dim,
        num_heads=num_heads,
        reso=64,
        split_size=[8, 8],
        shift_size=[1, 2],
        qkv_bias=True,
        rg_idx=rg_idx,
        b_idx=b_idx,
    )
    
    torch.manual_seed(42)
    flex = FlexAdaptiveSpatial(
        dim=dim,
        num_heads=num_heads,
        reso=64,
        split_size=[8, 8],
        shift_size=[1, 2],
        qkv_bias=True,
        rg_idx=rg_idx,
        b_idx=b_idx,
    )
    
    # Copy weights
    flex.load_state_dict(original.state_dict())
    
    # Set to eval
    original.eval()
    flex.eval()
    
    # Check shift schedule
    print(f"Original needs shift: {(rg_idx % 2 == 0 and b_idx > 0 and (b_idx - 2) % 4 == 0) or (rg_idx % 2 != 0 and b_idx % 4 == 0)}")
    print(f"Flex needs shift: {flex.needs_shift}")
    print(f"Flex shift_mask: {flex.shift_mask}")
    
    # Create input
    x = torch.randn(B, N, dim)
    
    # Run forward passes
    with torch.no_grad():
        out_original = original(x, H, W)
        out_flex = flex(x, H, W)
    
    # Compare outputs
    diff = (out_original - out_flex).abs()
    print(f"\nOutput shape: {out_original.shape}")
    print(f"Max absolute difference: {diff.max().item():.6f}")
    print(f"Mean absolute difference: {diff.mean().item():.6f}")
    
    # Test the attention branches separately
    print("\n=== Testing Attention Branches ===")
    
    # Get QKV
    qkv_orig = original.qkv(x).reshape(B, -1, 3, dim).permute(2, 0, 1, 3)
    qkv_flex = flex.qkv(x).reshape(B, -1, 3, dim).permute(2, 0, 1, 3)
    
    print(f"QKV matches: {torch.allclose(qkv_orig, qkv_flex, atol=1e-6)}")
    
    # Check if masks are being applied correctly
    if hasattr(original, 'attn_mask_0') and original.attn_mask_0 is not None:
        print(f"\nOriginal attn_mask_0 shape: {original.attn_mask_0.shape}")
        print(f"Flex attn_mask_0 shape: {flex.attn_mask_0.shape if flex.attn_mask_0 is not None else 'None'}")
        
        if flex.attn_mask_0 is not None:
            print(f"Masks match: {torch.allclose(original.attn_mask_0, flex.attn_mask_0)}")
    
    # Test without shifts for comparison
    print("\n=== Testing same config without shifts ===")
    torch.manual_seed(42)
    original_no_shift = OriginalAdaptiveSpatial(
        dim=dim,
        num_heads=num_heads,
        reso=64,
        split_size=[8, 8],
        shift_size=[1, 2],
        qkv_bias=True,
        rg_idx=0,
        b_idx=0,  # No shift
    )
    
    torch.manual_seed(42)
    flex_no_shift = FlexAdaptiveSpatial(
        dim=dim,
        num_heads=num_heads,
        reso=64,
        split_size=[8, 8],
        shift_size=[1, 2],
        qkv_bias=True,
        rg_idx=0,
        b_idx=0,  # No shift
    )
    
    flex_no_shift.load_state_dict(original_no_shift.state_dict())
    original_no_shift.eval()
    flex_no_shift.eval()
    
    with torch.no_grad():
        out_orig_no_shift = original_no_shift(x, H, W)
        out_flex_no_shift = flex_no_shift(x, H, W)
    
    diff_no_shift = (out_orig_no_shift - out_flex_no_shift).abs()
    print(f"No shift - max diff: {diff_no_shift.max().item():.6f}")
    
    # Test the shift operation itself
    print("\n=== Testing Shift Operations ===")
    test_tensor = torch.randn(B, H, W, dim // 2)
    shift_h, shift_w = 1, 2
    
    # Test roll operation
    rolled = torch.roll(test_tensor, shifts=(-shift_h, -shift_w), dims=(1, 2))
    rolled_back = torch.roll(rolled, shifts=(shift_h, shift_w), dims=(1, 2))
    
    print(f"Roll operation is reversible: {torch.allclose(test_tensor, rolled_back)}")
    
    # Check if the issue is in window partitioning after shift
    print("\n=== Window Partitioning Debug ===")
    print(f"Split size: {original.split_size}")
    print(f"Shift size: {original.shift_size}")
    print(f"Image size: ({H}, {W})")
    
    # Check padding
    max_split_size = max(original.split_size[0], original.split_size[1])
    pad_r = (max_split_size - W % max_split_size) % max_split_size
    pad_b = (max_split_size - H % max_split_size) % max_split_size
    print(f"Padding: right={pad_r}, bottom={pad_b}")
    print(f"Padded size: ({H + pad_b}, {W + pad_r})")


if __name__ == "__main__":
    test_adaptive_spatial_shifts()