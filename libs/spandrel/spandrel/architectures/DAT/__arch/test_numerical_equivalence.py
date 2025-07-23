"""
Comprehensive numerical equivalence tests between original DAT and FlexAttention DAT.
This script compares outputs at multiple levels to ensure perfect compatibility.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import gc


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def clone_module_weights(src: nn.Module, dst: nn.Module):
    """Clone weights from source module to destination module."""
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    
    # Copy matching parameters
    for key in dst_state:
        if key in src_state and src_state[key].shape == dst_state[key].shape:
            dst_state[key].copy_(src_state[key])
        else:
            print(f"Warning: Could not copy {key}")
    
    dst.load_state_dict(dst_state)


def compare_outputs(
    out1: torch.Tensor, 
    out2: torch.Tensor, 
    name: str,
    rtol: float = 1e-4,
    atol: float = 1e-5
) -> Dict[str, float]:
    """Compare two tensors and return statistics."""
    abs_diff = (out1 - out2).abs()
    rel_diff = abs_diff / (out1.abs() + 1e-8)
    
    stats = {
        "name": name,
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "allclose": torch.allclose(out1, out2, rtol=rtol, atol=atol),
    }
    
    # Find location of maximum difference
    max_idx = abs_diff.argmax()
    max_idx_tuple = np.unravel_index(max_idx.cpu(), abs_diff.shape)
    stats["max_diff_location"] = max_idx_tuple
    stats["val1_at_max"] = out1.flatten()[max_idx].item()
    stats["val2_at_max"] = out2.flatten()[max_idx].item()
    
    return stats


def test_window_attention_equivalence():
    """Test window-based spatial attention equivalence."""
    print("\n=== Testing Window Attention Equivalence ===")
    
    # Import both implementations
    from DAT_optim import Spatial_Attention as FlexSpatialAttention
    from DAT import Spatial_Attention as OriginalSpatialAttention
    
    # Test configurations
    configs = [
        {"dim": 96, "num_heads": 4, "split_size": [8, 8], "H": 64, "W": 64},
        {"dim": 192, "num_heads": 6, "split_size": [8, 8], "H": 32, "W": 48},
        {"dim": 384, "num_heads": 12, "split_size": [16, 16], "H": 128, "W": 128},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting config: {config}")
        
        # Create modules
        set_random_seed(42)
        original = OriginalSpatialAttention(
            dim=config["dim"],
            idx=0,
            split_size=config["split_size"],
            num_heads=config["num_heads"],
            position_bias=True
        )
        
        set_random_seed(42)
        flex = FlexSpatialAttention(
            dim=config["dim"],
            idx=0,
            split_size=config["split_size"],
            num_heads=config["num_heads"],
            position_bias=True
        )
        
        # Copy weights
        clone_module_weights(original, flex)
        
        # Test input
        B = 2
        H, W = config["H"], config["W"]
        C = config["dim"]
        x = torch.randn(B, H * W, C)
        
        # Set to eval mode
        original.eval()
        flex.eval()
        
        with torch.no_grad():
            # Create QKV for both (neither has internal QKV)
            qkv_layer = torch.nn.Linear(C, C * 3, bias=True)
            torch.nn.init.xavier_uniform_(qkv_layer.weight)
            qkv = qkv_layer(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
            
            # Original forward
            out_original = original(qkv, H, W)
            
            # Flex forward
            out_flex = flex(qkv, H, W)
            
            # Compare
            stats = compare_outputs(
                out_original.view(B, H * W, C),
                out_flex.view(B, H * W, C),
                f"SpatialAttention_{config['dim']}d_{config['num_heads']}h"
            )
            results.append(stats)
            
            print(f"  Max abs diff: {stats['max_abs_diff']:.2e}")
            print(f"  Mean abs diff: {stats['mean_abs_diff']:.2e}")
            print(f"  Allclose: {stats['allclose']}")
    
    return results


def test_adaptive_spatial_attention_equivalence():
    """Test adaptive spatial attention with shifts."""
    print("\n=== Testing Adaptive Spatial Attention Equivalence ===")
    
    from DAT_optim import Adaptive_Spatial_Attention as FlexAdaptiveSpatial
    from DAT import Adaptive_Spatial_Attention as OriginalAdaptiveSpatial
    
    # Test different shift scenarios
    configs = [
        {"rg_idx": 0, "b_idx": 0},  # No shift
        {"rg_idx": 0, "b_idx": 2},  # Shift
        {"rg_idx": 1, "b_idx": 0},  # Shift
        {"rg_idx": 1, "b_idx": 1},  # No shift
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting rg_idx={config['rg_idx']}, b_idx={config['b_idx']}")
        
        dim = 192
        num_heads = 6
        H, W = 64, 64
        
        set_random_seed(42)
        original = OriginalAdaptiveSpatial(
            dim=dim,
            num_heads=num_heads,
            reso=64,
            split_size=[8, 8],
            shift_size=[1, 2],
            qkv_bias=True,
            rg_idx=config["rg_idx"],
            b_idx=config["b_idx"],
        )
        
        set_random_seed(42)
        flex = FlexAdaptiveSpatial(
            dim=dim,
            num_heads=num_heads,
            reso=64,
            split_size=[8, 8],
            shift_size=[1, 2],
            qkv_bias=True,
            rg_idx=config["rg_idx"],
            b_idx=config["b_idx"],
        )
        
        # Copy weights
        clone_module_weights(original, flex)
        
        # Test input
        B = 1
        x = torch.randn(B, H * W, dim)
        
        original.eval()
        flex.eval()
        
        with torch.no_grad():
            out_original = original(x, H, W)
            out_flex = flex(x, H, W)
            
            stats = compare_outputs(
                out_original, out_flex,
                f"AdaptiveSpatial_rg{config['rg_idx']}_b{config['b_idx']}"
            )
            results.append(stats)
            
            print(f"  Max abs diff: {stats['max_abs_diff']:.2e}")
            print(f"  Allclose: {stats['allclose']}")
    
    return results


def test_channel_attention_equivalence():
    """Test adaptive channel attention equivalence."""
    print("\n=== Testing Adaptive Channel Attention Equivalence ===")
    
    from DAT_optim import Adaptive_Channel_Attention as FlexChannelAttention
    from DAT import Adaptive_Channel_Attention as OriginalChannelAttention
    
    configs = [
        {"dim": 96, "num_heads": 4},
        {"dim": 192, "num_heads": 8},
        {"dim": 384, "num_heads": 16},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting config: {config}")
        
        H, W = 32, 32
        
        set_random_seed(42)
        original = OriginalChannelAttention(
            dim=config["dim"],
            num_heads=config["num_heads"],
            qkv_bias=True
        )
        
        set_random_seed(42)
        flex = FlexChannelAttention(
            dim=config["dim"],
            num_heads=config["num_heads"],
            qkv_bias=True
        )
        
        # Copy weights
        clone_module_weights(original, flex)
        
        # Test input
        B = 2
        x = torch.randn(B, H * W, config["dim"])
        
        original.eval()
        flex.eval()
        
        with torch.no_grad():
            out_original = original(x, H, W)
            out_flex = flex(x, H, W)
            
            stats = compare_outputs(
                out_original, out_flex,
                f"ChannelAttention_{config['dim']}d_{config['num_heads']}h"
            )
            results.append(stats)
            
            print(f"  Max abs diff: {stats['max_abs_diff']:.2e}")
            print(f"  Allclose: {stats['allclose']}")
    
    return results


def test_full_model_equivalence():
    """Test full DAT model equivalence."""
    print("\n=== Testing Full Model Equivalence ===")
    
    from DAT_optim import DAT as FlexDAT
    from DAT import DAT as OriginalDAT
    
    # Model configuration
    model_config = {
        "img_size": 64,
        "in_chans": 3,
        "embed_dim": 192,
        "split_size": [8, 8],
        "depth": [2, 2, 2, 2],
        "num_heads": [4, 6, 12, 24],
        "expansion_factor": 4.0,
        "qkv_bias": True,
        "upscale": 4,
        "resi_connection": "1conv",
        "upsampler": "pixelshuffle",
    }
    
    print(f"Model config: {model_config}")
    
    # Create models
    set_random_seed(42)
    original = OriginalDAT(**model_config)
    
    set_random_seed(42)
    flex = FlexDAT(**model_config)
    
    # Copy all weights
    clone_module_weights(original, flex)
    
    # Test inputs
    test_sizes = [(1, 3, 64, 64), (2, 3, 48, 48)]
    
    results = []
    
    for B, C, H, W in test_sizes:
        print(f"\nTesting input shape: ({B}, {C}, {H}, {W})")
        
        x = torch.randn(B, C, H, W)
        
        original.eval()
        flex.eval()
        
        with torch.no_grad():
            # Time the forward passes
            start = time.time()
            out_original = original(x)
            time_original = time.time() - start
            
            start = time.time()
            out_flex = flex(x)
            time_flex = time.time() - start
            
            stats = compare_outputs(
                out_original, out_flex,
                f"FullModel_{B}x{C}x{H}x{W}"
            )
            stats["time_original"] = time_original
            stats["time_flex"] = time_flex
            stats["speedup"] = time_original / time_flex if time_flex > 0 else 0
            
            results.append(stats)
            
            print(f"  Output shape: {out_original.shape}")
            print(f"  Max abs diff: {stats['max_abs_diff']:.2e}")
            print(f"  Mean abs diff: {stats['mean_abs_diff']:.2e}")
            print(f"  Allclose: {stats['allclose']}")
            print(f"  Time original: {time_original:.3f}s")
            print(f"  Time flex: {time_flex:.3f}s")
            print(f"  Speedup: {stats['speedup']:.2f}x")
    
    return results


def test_gradient_equivalence():
    """Test gradient computation equivalence."""
    print("\n=== Testing Gradient Equivalence ===")
    
    from DAT_optim import Spatial_Attention as FlexSpatialAttention
    from DAT import Spatial_Attention as OriginalSpatialAttention
    
    dim = 96
    num_heads = 4
    H, W = 32, 32
    
    # Create modules
    set_random_seed(42)
    original = OriginalSpatialAttention(
        dim=dim, idx=0, split_size=[8, 8], num_heads=num_heads
    )
    
    set_random_seed(42)
    flex = FlexSpatialAttention(
        dim=dim, idx=0, split_size=[8, 8], num_heads=num_heads
    )
    
    clone_module_weights(original, flex)
    
    # Test input
    B = 1
    x = torch.randn(B, H * W, dim, requires_grad=True)
    x_flex = x.clone().detach().requires_grad_(True)
    
    # Forward pass - neither has internal QKV
    qkv_layer = nn.Linear(dim, dim * 3, bias=True)
    torch.nn.init.xavier_uniform_(qkv_layer.weight)
    
    qkv_orig = qkv_layer(x).reshape(B, -1, 3, dim).permute(2, 0, 1, 3)
    out_original = original(qkv_orig, H, W).view(B, H * W, dim)
    
    qkv_flex = qkv_layer(x_flex).reshape(B, -1, 3, dim).permute(2, 0, 1, 3)
    out_flex = flex(qkv_flex, H, W).view(B, H * W, dim)
    
    # Backward pass
    grad_out = torch.randn_like(out_original)
    
    out_original.backward(grad_out)
    out_flex.backward(grad_out)
    
    # Compare gradients
    grad_stats = compare_outputs(
        x.grad, x_flex.grad,
        "Input gradients"
    )
    
    print(f"  Gradient max abs diff: {grad_stats['max_abs_diff']:.2e}")
    print(f"  Gradient allclose: {grad_stats['allclose']}")
    
    return grad_stats


def run_all_tests():
    """Run all numerical equivalence tests."""
    print("=" * 60)
    print("DAT FlexAttention Numerical Equivalence Tests")
    print("=" * 60)
    
    all_results = {}
    
    try:
        # Test individual components
        all_results["window_attention"] = test_window_attention_equivalence()
        all_results["adaptive_spatial"] = test_adaptive_spatial_attention_equivalence()
        all_results["channel_attention"] = test_channel_attention_equivalence()
        
        # Test gradients
        all_results["gradients"] = test_gradient_equivalence()
        
        # Test full model
        all_results["full_model"] = test_full_model_equivalence()
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, results in all_results.items():
            if isinstance(results, list):
                for result in results:
                    total_tests += 1
                    if result.get("allclose", False):
                        passed_tests += 1
                        status = "PASS"
                    else:
                        status = "FAIL"
                    print(f"{result['name']}: {status} (max diff: {result['max_abs_diff']:.2e})")
            elif isinstance(results, dict):
                total_tests += 1
                if results.get("allclose", False):
                    passed_tests += 1
                    status = "PASS"
                else:
                    status = "FAIL"
                print(f"{results['name']}: {status} (max diff: {results['max_abs_diff']:.2e})")
        
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n✅ All tests passed! FlexAttention implementation is numerically equivalent.")
        else:
            print("\n❌ Some tests failed. Check the differences above.")
            
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure both DAT.py (original) and DAT_optim.py (FlexAttention) are available.")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Enable CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on: {device}")
    
    run_all_tests()