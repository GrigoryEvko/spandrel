"""
Debug test for channel attention to identify the source of numerical differences.
"""

import torch
import torch.nn as nn

# Import both implementations
from DAT import Adaptive_Channel_Attention as OriginalChannelAttention
from DAT_optim import Adaptive_Channel_Attention as FlexChannelAttention


def test_channel_attention_step_by_step():
    """Test channel attention with detailed step-by-step comparison."""
    print("=== Channel Attention Step-by-Step Debug ===\n")
    
    # Configuration
    dim = 96
    num_heads = 4
    B, H, W = 1, 8, 8
    N = H * W
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create modules
    original = OriginalChannelAttention(dim=dim, num_heads=num_heads, qkv_bias=True)
    flex = FlexChannelAttention(dim=dim, num_heads=num_heads, qkv_bias=True)
    
    # Copy weights from original to flex
    flex.load_state_dict(original.state_dict())
    
    # Set to eval mode
    original.eval()
    flex.eval()
    
    # Create input
    x = torch.randn(B, N, dim)
    
    # Print temperature values
    print(f"Temperature shape: {original.temperature.shape}")
    print(f"Temperature values: {original.temperature.squeeze()}")
    print()
    
    # Forward pass with intermediate values
    with torch.no_grad():
        # Original implementation - extract intermediate values
        qkv_orig = original.qkv(x).reshape(B, N, 3, num_heads, dim // num_heads)
        qkv_orig = qkv_orig.permute(2, 0, 3, 1, 4)
        q_orig, k_orig, v_orig = qkv_orig[0], qkv_orig[1], qkv_orig[2]
        
        # Transpose for channel attention
        q_orig_t = q_orig.transpose(-2, -1)
        k_orig_t = k_orig.transpose(-2, -1)
        v_orig_t = v_orig.transpose(-2, -1)
        
        # Normalize
        q_orig_norm = torch.nn.functional.normalize(q_orig_t, dim=-1)
        k_orig_norm = torch.nn.functional.normalize(k_orig_t, dim=-1)
        
        # Compute attention scores
        attn_scores_orig = (q_orig_norm @ k_orig_norm.transpose(-2, -1))
        print(f"Original attention scores shape: {attn_scores_orig.shape}")
        print(f"Original attention scores range: [{attn_scores_orig.min():.4f}, {attn_scores_orig.max():.4f}]")
        
        # Apply temperature
        attn_scores_temp_orig = attn_scores_orig * original.temperature
        print(f"Original scores after temperature range: [{attn_scores_temp_orig.min():.4f}, {attn_scores_temp_orig.max():.4f}]")
        
        # Softmax
        attn_weights_orig = attn_scores_temp_orig.softmax(dim=-1)
        print(f"Original attention weights range: [{attn_weights_orig.min():.4f}, {attn_weights_orig.max():.4f}]")
        print()
        
        # Flex implementation - extract QKV for analysis
        qkv_flex = flex.qkv(x).reshape(B, N, 3, num_heads, dim // num_heads)
        qkv_flex = qkv_flex.permute(2, 0, 3, 1, 4)
        q_flex, k_flex, v_flex = qkv_flex[0], qkv_flex[1], qkv_flex[2]
        
        # Check if QKV match
        print(f"Q matches: {torch.allclose(q_orig, q_flex, atol=1e-6)}")
        print(f"K matches: {torch.allclose(k_orig, k_flex, atol=1e-6)}")
        print(f"V matches: {torch.allclose(v_orig, v_flex, atol=1e-6)}")
        print()
        
        # Run full forward
        out_original = original(x, H, W)
        out_flex = flex(x, H, W)
        
        # Compare outputs
        diff = (out_original - out_flex).abs()
        print(f"Output shape: {out_original.shape}")
        print(f"Max absolute difference: {diff.max().item():.6f}")
        print(f"Mean absolute difference: {diff.mean().item():.6f}")
        
        # Find location of max difference
        max_idx = diff.argmax()
        b_idx = max_idx // (N * dim)
        n_idx = (max_idx % (N * dim)) // dim
        c_idx = max_idx % dim
        
        print(f"\nMax difference at position: B={b_idx}, N={n_idx}, C={c_idx}")
        print(f"Original value: {out_original.view(-1)[max_idx].item():.6f}")
        print(f"Flex value: {out_flex.view(-1)[max_idx].item():.6f}")
        
        # Test the score_mod function separately
        print("\n=== Testing Score Mod Function ===")
        score_mod = flex.create_channel_score_mod()
        
        # Test with sample values
        test_score = torch.tensor(0.5)
        test_b = torch.tensor(0)
        test_h = torch.tensor(0)
        test_q_idx = torch.tensor(0)
        test_kv_idx = torch.tensor(0)
        
        result = score_mod(test_score, test_b, test_h, test_q_idx, test_kv_idx)
        print(f"Score mod result for head 0: {result.item():.6f} (expected: {0.5 * flex.temperature[0, 0, 0].item():.6f})")
        
        # Test for different heads
        for h in range(num_heads):
            test_h = torch.tensor(h)
            result = score_mod(test_score, test_b, test_h, test_q_idx, test_kv_idx)
            expected = test_score.item() * flex.temperature[h, 0, 0].item()
            print(f"Head {h}: result={result.item():.6f}, expected={expected:.6f}, temp={flex.temperature[h, 0, 0].item():.6f}")
        
        # Test normalization differences
        print("\n=== Testing Normalization ===")
        print(f"Original q norm: {q_orig_norm[0, 0, :5, 0]}")
        print(f"Original k norm: {k_orig_norm[0, 0, :5, 0]}")
        
        # Check if the issue is in how FlexAttention handles the transpose
        print("\n=== Testing Transpose and Reshape ===")
        print(f"Original v shape after transpose: {v_orig_t.shape}")
        print(f"Flex should have v shape: (B={B}, heads={num_heads}, head_dim={dim//num_heads}, spatial={N})")
        
        # Test attention computation manually for flex
        print("\n=== Manual FlexAttention Computation ===")
        # This mimics what should happen in FlexAttention
        q_flex_t = q_flex.transpose(-2, -1).contiguous()
        k_flex_t = k_flex.transpose(-2, -1).contiguous()
        v_flex_t = v_flex.transpose(-2, -1).contiguous()
        
        q_flex_norm = torch.nn.functional.normalize(q_flex_t, dim=-1)
        k_flex_norm = torch.nn.functional.normalize(k_flex_t, dim=-1)
        
        # Manual attention computation
        scores_manual = torch.matmul(q_flex_norm, k_flex_norm.transpose(-2, -1))
        scores_manual_temp = scores_manual * flex.temperature
        attn_manual = scores_manual_temp.softmax(dim=-1)
        out_manual = torch.matmul(attn_manual, v_flex_t)
        out_manual = out_manual.transpose(-2, -1).contiguous().reshape(B, N, dim)
        
        # Compare with flex output (before proj and AIM)
        print(f"Manual computation matches original: {torch.allclose(out_manual, out_original, atol=1e-3)}")
        
        # Check if q and k are being normalized twice in flex implementation
        print("\n=== Checking Double Normalization ===")
        print(f"Q norm check - before flex forward: {torch.norm(q_flex[0, 0, 0, :]):.6f}")
        print(f"K norm check - before flex forward: {torch.norm(k_flex[0, 0, 0, :]):.6f}")
        
        # Test AIM components
        print("\n=== Testing AIM (Adaptive Interaction Module) ===")
        
        # Get attention output before AIM for both implementations
        # For original implementation, we need to compute the attention output before AIM
        attened_x_orig = (attn_weights_orig @ v_orig_t).permute(0, 3, 1, 2).reshape(B, N, dim)
        
        # Get convolution output for both
        v_2d_orig = v_orig.reshape(B, dim, N).contiguous().view(B, dim, H, W)
        conv_x_orig = original.dwconv(v_2d_orig)
        
        print(f"Original attention output shape: {attened_x_orig.shape}")
        print(f"Original conv output shape: {conv_x_orig.shape}")
        
        # Test the full forward pass step by step
        print("\n=== Full Forward Pass Comparison ===")
        
        # Let's trace through the exact computation in original
        attention_reshape_orig = attened_x_orig.transpose(-2, -1).contiguous().view(B, dim, H, W)
        channel_map_orig = original.channel_interaction(attention_reshape_orig)
        spatial_map_orig = original.spatial_interaction(conv_x_orig).permute(0, 2, 3, 1).contiguous().view(B, N, 1)
        
        print(f"Channel map shape: {channel_map_orig.shape}")
        print(f"Spatial map shape: {spatial_map_orig.shape}")
        
        # Apply AIM gates
        attened_x_gated = attened_x_orig * torch.sigmoid(spatial_map_orig)
        conv_x_gated = conv_x_orig * torch.sigmoid(channel_map_orig)
        conv_x_reshaped = conv_x_gated.permute(0, 2, 3, 1).contiguous().view(B, N, dim)
        
        x_combined = attened_x_gated + conv_x_reshaped
        x_proj = original.proj(x_combined)
        x_final = original.proj_drop(x_proj)
        
        print(f"\nFinal output comparison:")
        print(f"Manual reconstruction matches original: {torch.allclose(x_final, out_original, atol=1e-5)}")
        
        # The issue might be in how FlexAttention is handling the channel-wise attention
        print("\n=== Potential Issues ===")
        print("1. FlexAttention might be applying softmax along wrong dimension")
        print("2. The transpose operations might not be properly aligned")
        print("3. The normalization might be applied differently")


def test_flexattention_internals():
    """Test FlexAttention internals to understand the computation."""
    print("\n\n=== FlexAttention Internals Test ===\n")
    
    from torch.nn.attention.flex_attention import flex_attention
    
    # Simple test case
    B, H, D, S = 1, 2, 4, 4  # batch, heads, dim, seq_len
    
    # Create simple Q, K, V
    q = torch.randn(B, H, D, S)  # Channel attention: (B, heads, head_dim, spatial)
    k = torch.randn(B, H, D, S)
    v = torch.randn(B, H, D, S)
    
    # Normalize
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    
    # Manual computation
    scores_manual = torch.matmul(q, k.transpose(-2, -1))  # (B, H, D, D)
    attn_manual = scores_manual.softmax(dim=-1)
    out_manual = torch.matmul(attn_manual, v)  # (B, H, D, S)
    
    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Manual scores shape: {scores_manual.shape}")
    print(f"Manual output shape: {out_manual.shape}")
    
    # FlexAttention computation
    out_flex = flex_attention(q, k, v, scale=1.0)
    
    print(f"FlexAttention output shape: {out_flex.shape}")
    print(f"Outputs match: {torch.allclose(out_manual, out_flex, atol=1e-5)}")
    
    if not torch.allclose(out_manual, out_flex, atol=1e-5):
        diff = (out_manual - out_flex).abs()
        print(f"Max difference: {diff.max().item():.6f}")
        print(f"Mean difference: {diff.mean().item():.6f}")


if __name__ == "__main__":
    test_channel_attention_step_by_step()
    test_flexattention_internals()