"""
Test the loaded model with both original and FlexAttention implementations.
"""

import sys
import os
sys.path.append('/home/grigory/theartisanai/spandrel/libs/spandrel')

import spandrel
import torch
import time
import numpy as np
from DAT import DAT as OriginalDAT
from DAT_optim import DAT as FlexDAT


def benchmark_model(model, input_tensor, num_warmup=5, num_iters=20):
    """Run benchmark on a model."""
    model = model.eval()
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_iters):
        start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    
    return avg_time, std_time, output


def main():
    """Test with spandrel loaded model."""
    model_path = "/home/grigory/theartisanai/model_serving/upscaler/4xNomos2_hq_dat2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load model with spandrel to get exact configuration
    print(f"\nLoading model with spandrel...")
    spandrel_model = spandrel.ModelLoader().load_from_file(model_path)
    original = spandrel_model.model.to(device).eval()
    
    # Get state dict
    state_dict = original.state_dict()
    
    # Create FlexAttention version with same config
    model_config = {
        "img_size": 64,
        "in_chans": 3,
        "embed_dim": 180,
        "split_size": [8, 32],  # Correct split size from spandrel
        "depth": [6, 6, 6, 6, 6, 6],
        "num_heads": [6, 6, 6, 6, 6, 6],
        "expansion_factor": 2.0,
        "qkv_bias": True,
        "upscale": 4,
        "resi_connection": "1conv",
        "upsampler": "pixelshuffle",
    }
    
    print(f"\nCreating FlexAttention model with config: {model_config}")
    flex = FlexDAT(**model_config).to(device)
    
    # Load weights to FlexAttention model
    print("\nLoading weights to FlexAttention model...")
    # Map keys if needed
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if key == "before_RG.1.weight":
            mapped_state_dict["before_RG_norm.weight"] = value
        elif key == "before_RG.1.bias":
            mapped_state_dict["before_RG_norm.bias"] = value
        else:
            mapped_state_dict[key] = value
    
    flex.load_state_dict(mapped_state_dict, strict=True)
    flex = flex.eval()
    
    # Test different input sizes
    input_sizes = [(1, 3, 64, 64), (1, 3, 128, 128)]
    
    print("\n=== Testing correctness ===")
    for input_size in input_sizes:
        print(f"\nInput size: {input_size}")
        x = torch.randn(*input_size, device=device)
        
        with torch.no_grad():
            out_original = original(x)
            out_flex = flex(x)
        
        # Compare outputs
        diff = (out_original - out_flex).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  Output shape: {out_original.shape}")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        print(f"  Outputs match: {torch.allclose(out_original, out_flex, atol=1e-4)}")
    
    print("\n=== Benchmarking ===")
    for input_size in input_sizes:
        print(f"\nInput size: {input_size}")
        x = torch.randn(*input_size, device=device)
        
        # Benchmark original
        orig_time, orig_std, _ = benchmark_model(original, x)
        print(f"  Original: {orig_time:.2f} ± {orig_std:.2f} ms")
        
        # Benchmark FlexAttention
        flex_time, flex_std, _ = benchmark_model(flex, x)
        print(f"  FlexAttention: {flex_time:.2f} ± {flex_std:.2f} ms")
        
        # Calculate speedup
        speedup = orig_time / flex_time
        print(f"  Speedup: {speedup:.2f}x")
    
    # Test on a real-world input
    print("\n=== Testing with larger input ===")
    x_large = torch.randn(1, 3, 256, 256, device=device)
    
    # Time single forward pass
    with torch.no_grad():
        start = time.time()
        out_orig = original(x_large)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        orig_time = (time.time() - start) * 1000
        
        start = time.time()
        out_flex = flex(x_large)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        flex_time = (time.time() - start) * 1000
    
    diff = (out_orig - out_flex).abs()
    print(f"Input: {x_large.shape}, Output: {out_orig.shape}")
    print(f"Max difference: {diff.max().item():.6f}")
    print(f"Original time: {orig_time:.2f} ms")
    print(f"FlexAttention time: {flex_time:.2f} ms")
    print(f"Speedup: {orig_time/flex_time:.2f}x")


if __name__ == "__main__":
    main()