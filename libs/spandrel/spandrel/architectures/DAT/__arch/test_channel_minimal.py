"""
Minimal test to isolate channel attention issue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

# Import both implementations
from DAT import Adaptive_Channel_Attention as OriginalChannelAttention
from DAT_optim import Adaptive_Channel_Attention as FlexChannelAttention


def compare_attention_computation():
    """Compare the exact attention computation between original and flex."""
    torch.manual_seed(42)
    
    dim = 96
    num_heads = 4
    B, H, W = 1, 8, 8
    N = H * W
    
    # Create simple input
    x = torch.randn(B, N, dim)
    
    # Create modules
    original = OriginalChannelAttention(dim=dim, num_heads=num_heads, qkv_bias=True)
    flex = FlexChannelAttention(dim=dim, num_heads=num_heads, qkv_bias=True)
    
    # Copy weights
    flex.load_state_dict(original.state_dict())
    
    original.eval()
    flex.eval()
    
    # Get QKV
    qkv = original.qkv(x).reshape(B, N, 3, num_heads, dim // num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    
    # Original computation
    q_t = q.transpose(-2, -1)
    k_t = k.transpose(-2, -1)
    v_t = v.transpose(-2, -1)
    
    q_norm = F.normalize(q_t, dim=-1)
    k_norm = F.normalize(k_t, dim=-1)
    
    scores = q_norm @ k_norm.transpose(-2, -1)
    scores_temp = scores * original.temperature
    attn = scores_temp.softmax(dim=-1)
    out_orig = attn @ v_t
    out_orig = out_orig.permute(0, 3, 1, 2).reshape(B, N, dim)
    
    # FlexAttention computation step by step
    print("=== Testing FlexAttention Step by Step ===")
    
    # Create the same QKV tensors in the shape FlexAttention expects
    # Shape: (B, heads, head_dim, spatial)
    q_flex = q_norm.clone()
    k_flex = k_norm.clone()
    v_flex = v_t.clone()
    
    # Test 1: FlexAttention without score_mod
    print("\nTest 1: FlexAttention without score_mod")
    out_flex_no_mod = flex_attention(q_flex, k_flex, v_flex, scale=1.0)
    out_flex_no_mod = out_flex_no_mod.permute(0, 3, 1, 2).reshape(B, N, dim)
    
    # Compare with manual computation without temperature
    scores_no_temp = q_norm @ k_norm.transpose(-2, -1)
    attn_no_temp = scores_no_temp.softmax(dim=-1)
    out_manual_no_temp = attn_no_temp @ v_t
    out_manual_no_temp = out_manual_no_temp.permute(0, 3, 1, 2).reshape(B, N, dim)
    
    print(f"FlexAttention (no mod) matches manual (no temp): {torch.allclose(out_flex_no_mod, out_manual_no_temp, atol=1e-5)}")
    
    # Test 2: FlexAttention with identity score_mod
    print("\nTest 2: FlexAttention with identity score_mod")
    def identity_mod(score, b, h, q_idx, kv_idx):
        return score
    
    out_flex_identity = flex_attention(q_flex, k_flex, v_flex, score_mod=identity_mod, scale=1.0)
    out_flex_identity = out_flex_identity.permute(0, 3, 1, 2).reshape(B, N, dim)
    
    print(f"FlexAttention (identity) matches manual (no temp): {torch.allclose(out_flex_identity, out_manual_no_temp, atol=1e-5)}")
    
    # Test 3: FlexAttention with temperature score_mod
    print("\nTest 3: FlexAttention with temperature score_mod")
    temperature = original.temperature.squeeze()
    
    def temp_mod(score, b, h, q_idx, kv_idx):
        return score * temperature[h]
    
    out_flex_temp = flex_attention(q_flex, k_flex, v_flex, score_mod=temp_mod, scale=1.0)
    out_flex_temp = out_flex_temp.permute(0, 3, 1, 2).reshape(B, N, dim)
    
    print(f"FlexAttention (temp mod) matches original: {torch.allclose(out_flex_temp, out_orig, atol=1e-5)}")
    
    if not torch.allclose(out_flex_temp, out_orig, atol=1e-5):
        diff = (out_flex_temp - out_orig).abs()
        print(f"Max diff: {diff.max():.6f}")
        print(f"Mean diff: {diff.mean():.6f}")
    
    # Test 4: Full forward pass
    print("\nTest 4: Full forward pass")
    with torch.no_grad():
        out_original_full = original(x, H, W)
        out_flex_full = flex(x, H, W)
    
    diff_full = (out_original_full - out_flex_full).abs()
    print(f"Full forward pass max diff: {diff_full.max():.6f}")
    print(f"Full forward pass mean diff: {diff_full.mean():.6f}")
    
    # Identify where the difference comes from
    print("\n=== Debugging the difference ===")
    
    # Get attention outputs before AIM
    # For original - need to use the transposed v
    v_transposed = v.transpose(-2, -1)  # Now shape: (B, heads, head_dim, spatial)
    v_2d_correct = v_transposed.reshape(B, dim, N).contiguous().view(B, dim, H, W)
    
    print(f"v shape: {v.shape}")
    print(f"v_transposed shape: {v_transposed.shape}")
    print(f"v_2d shape: {v_2d_correct.shape}")
    
    conv_x_orig = original.dwconv(v_2d_correct)
    attention_reshape = out_orig.transpose(-2, -1).contiguous().view(B, dim, H, W)
    channel_map = original.channel_interaction(attention_reshape)
    spatial_map = original.spatial_interaction(conv_x_orig).permute(0, 2, 3, 1).contiguous().view(B, N, 1)
    
    print(f"conv_x shape: {conv_x_orig.shape}")
    print(f"channel_map shape: {channel_map.shape}")
    print(f"spatial_map shape: {spatial_map.shape}")
    
    # Apply gates
    attn_gated = out_orig * torch.sigmoid(spatial_map)
    conv_gated = conv_x_orig * torch.sigmoid(channel_map)
    conv_reshaped = conv_gated.permute(0, 2, 3, 1).contiguous().view(B, N, dim)
    
    combined = attn_gated + conv_reshaped
    final = original.proj(combined)
    final = original.proj_drop(final)
    
    print(f"Reconstructed output matches original: {torch.allclose(final, out_original_full, atol=1e-5)}")
    
    if not torch.allclose(final, out_original_full, atol=1e-5):
        diff = (final - out_original_full).abs()
        print(f"Reconstruction diff max: {diff.max():.6f}")
        print(f"Reconstruction diff mean: {diff.mean():.6f}")


if __name__ == "__main__":
    compare_attention_computation()