"""
Test to debug FlexAttention output in channel attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

# Import both implementations
from DAT import Adaptive_Channel_Attention as OriginalChannelAttention
from DAT_optim import Adaptive_Channel_Attention as FlexChannelAttention


def test_flexattention_output():
    """Test FlexAttention output vs manual computation."""
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
    
    # Hook to capture FlexAttention input and output
    flex_q = None
    flex_k = None
    flex_v = None
    flex_out = None
    
    def capture_flex_attention(module, args, output):
        nonlocal flex_q, flex_k, flex_v, flex_out
        # args are (q, k, v, score_mod, scale, enable_gqa)
        flex_q = args[0].clone()
        flex_k = args[1].clone()
        flex_v = args[2].clone()
        flex_out = output.clone()
    
    # Monkey patch flex_attention to capture inputs/outputs
    original_flex_attention = flex_attention
    
    def wrapped_flex_attention(*args, **kwargs):
        output = original_flex_attention(*args, **kwargs)
        capture_flex_attention(None, args, output)
        return output
    
    # Replace flex_attention temporarily
    import DAT_optim
    DAT_optim.flex_attention = wrapped_flex_attention
    
    # Run forward pass
    with torch.no_grad():
        out_flex_full = flex(x, H, W)
    
    # Restore original flex_attention
    DAT_optim.flex_attention = original_flex_attention
    
    print("=== FlexAttention Debug ===")
    print(f"flex_q shape: {flex_q.shape}")
    print(f"flex_k shape: {flex_k.shape}")
    print(f"flex_v shape: {flex_v.shape}")
    print(f"flex_out shape: {flex_out.shape}")
    
    # Manual computation to compare
    scores = torch.matmul(flex_q, flex_k.transpose(-2, -1))
    
    # Apply temperature
    temperature = flex.temperature.squeeze()
    scores_temp = scores * temperature.view(1, -1, 1, 1)
    
    attn = scores_temp.softmax(dim=-1)
    manual_out = torch.matmul(attn, flex_v)
    
    print(f"\nFlexAttention output matches manual: {torch.allclose(flex_out, manual_out, atol=1e-5)}")
    
    if not torch.allclose(flex_out, manual_out, atol=1e-5):
        diff = (flex_out - manual_out).abs()
        print(f"Diff max: {diff.max():.6f}")
        print(f"Diff mean: {diff.mean():.6f}")
    
    # Now compare the full output after transpose and reshape
    flex_out_reshaped = flex_out.transpose(-2, -1).contiguous().reshape(B, N, dim)
    
    # Get original computation
    qkv_orig = original.qkv(x).reshape(B, N, 3, num_heads, dim // num_heads)
    qkv_orig = qkv_orig.permute(2, 0, 3, 1, 4)
    q_orig, k_orig, v_orig = qkv_orig.unbind(0)
    
    q_orig_t = q_orig.transpose(-2, -1)
    k_orig_t = k_orig.transpose(-2, -1)
    v_orig_t = v_orig.transpose(-2, -1)
    
    q_orig_norm = F.normalize(q_orig_t, dim=-1)
    k_orig_norm = F.normalize(k_orig_t, dim=-1)
    
    scores_orig = q_orig_norm @ k_orig_norm.transpose(-2, -1)
    scores_orig_temp = scores_orig * original.temperature
    attn_orig = scores_orig_temp.softmax(dim=-1)
    out_orig = attn_orig @ v_orig_t
    out_orig_reshaped = out_orig.permute(0, 3, 1, 2).reshape(B, N, dim)
    
    print(f"\nAttention output (before AIM) matches: {torch.allclose(flex_out_reshaped, out_orig_reshaped, atol=1e-5)}")
    
    if not torch.allclose(flex_out_reshaped, out_orig_reshaped, atol=1e-5):
        diff = (flex_out_reshaped - out_orig_reshaped).abs()
        print(f"Attention output diff max: {diff.max():.6f}")
        print(f"Attention output diff mean: {diff.mean():.6f}")
    
    # Check intermediate attention computations
    print("\n=== Intermediate Values ===")
    print(f"Q normalized correctly: {torch.allclose(flex_q, q_orig_norm, atol=1e-5)}")
    print(f"K normalized correctly: {torch.allclose(flex_k, k_orig_norm, atol=1e-5)}")
    print(f"V transposed correctly: {torch.allclose(flex_v, v_orig_t, atol=1e-5)}")
    
    # Debug the reshape issue
    print("\n=== Reshape Debug ===")
    print(f"flex_out shape: {flex_out.shape}")  # (B, heads, head_dim, spatial)
    print(f"out_orig shape: {out_orig.shape}")  # (B, heads, head_dim, spatial)
    
    # Original reshapes with permute
    out_orig_permute = out_orig.permute(0, 3, 1, 2)  # (B, spatial, heads, head_dim)
    print(f"out_orig after permute: {out_orig_permute.shape}")
    out_orig_final = out_orig_permute.reshape(B, N, dim)
    
    # Flex reshapes with transpose
    flex_out_transpose = flex_out.transpose(-2, -1)  # (B, heads, spatial, head_dim)
    print(f"flex_out after transpose: {flex_out_transpose.shape}")
    flex_out_final = flex_out_transpose.contiguous().reshape(B, N, dim)
    
    print(f"\nReshape methods produce same result: {torch.allclose(flex_out_final, out_orig_final, atol=1e-5)}")


if __name__ == "__main__":
    test_flexattention_output()