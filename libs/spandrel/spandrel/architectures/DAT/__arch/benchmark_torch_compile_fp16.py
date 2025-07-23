"""
Comprehensive benchmark with torch.compile and float16 support.
"""

import sys
import os
sys.path.append('/home/grigory/theartisanai/spandrel/libs/spandrel')
sys.path.append('/home/grigory/theartisanai/spandrel/libs/spandrel/spandrel/architectures/DAT/__arch')

import spandrel
import torch
import time
import numpy as np
import DAT
import DAT_optim


def benchmark_model(model, input_tensor, name, num_warmup=10, num_runs=20):
    """Benchmark a model with proper warmup and synchronization."""
    model.eval()
    
    times = []
    
    # Warmup
    print(f"  Warming up {name}...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
            torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Benchmarking {name}...")
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
            
            if i == 0:
                print(f"    First pass: {times[0]*1000:.2f} ms")
    
    times = np.array(times)
    return {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000,
        'median': np.median(times) * 1000
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device('cuda')
    torch.set_float32_matmul_precision('high')
    
    # Model configuration
    model_config = {
        "img_size": 64,
        "in_chans": 3,
        "embed_dim": 180,
        "split_size": [8, 32],
        "depth": [6, 6, 6, 6, 6, 6],
        "num_heads": [6, 6, 6, 6, 6, 6],
        "expansion_factor": 2.0,
        "qkv_bias": True,
        "upscale": 4,
        "resi_connection": "1conv",
        "upsampler": "pixelshuffle",
    }
    
    # Load model weights
    model_path = "/home/grigory/theartisanai/model_serving/upscaler/4xNomos2_hq_dat2.pth"
    spandrel_model = spandrel.ModelLoader().load_from_file(model_path)
    state_dict = spandrel_model.model.state_dict()
    
    # Fix state dict key mapping for FlexDAT
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if key == "before_RG.1.weight":
            mapped_state_dict["before_RG_norm.weight"] = value
        elif key == "before_RG.1.bias":
            mapped_state_dict["before_RG_norm.bias"] = value
        else:
            mapped_state_dict[key] = value
    
    print("=" * 80)
    print("Comprehensive Benchmark: torch.compile + float16")
    print("=" * 80)
    
    # Test configurations
    test_configs = [
        ("float32", torch.float32),
        ("float16", torch.float16)
    ]
    
    input_sizes = [
        (1, 3, 64, 64),
        (1, 3, 128, 128),
        (1, 3, 256, 256)
    ]
    
    for dtype_name, dtype in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing with {dtype_name}")
        print(f"{'='*60}")
        
        for input_size in input_sizes:
            print(f"\n\nInput size: {input_size}")
            
            # Create input tensor
            x = torch.randn(input_size, device=device, dtype=dtype)
            
            # Original DAT
            print("\n1. Original DAT (no compile)")
            original = DAT.DAT(**model_config).to(device).to(dtype).eval()
            original.load_state_dict(state_dict, strict=True)
            original_stats = benchmark_model(original, x, "Original DAT")
            
            # Original DAT with torch.compile
            print("\n2. Original DAT (torch.compile)")
            original_compiled = DAT.DAT(**model_config).to(device).to(dtype).eval()
            original_compiled.load_state_dict(state_dict, strict=True)
            original_compiled = torch.compile(original_compiled, mode="max-autotune")
            original_compiled_stats = benchmark_model(original_compiled, x, "Original DAT Compiled")
            
            # FlexAttention DAT
            print("\n3. FlexAttention DAT (no compile)")
            flex = DAT_optim.DAT(**model_config).to(device).to(dtype).eval()
            flex.load_state_dict(mapped_state_dict, strict=True)
            flex_stats = benchmark_model(flex, x, "FlexAttention DAT")
            
            # FlexAttention DAT with torch.compile
            print("\n4. FlexAttention DAT (torch.compile)")
            flex_compiled = DAT_optim.DAT(**model_config).to(device).to(dtype).eval()
            flex_compiled.load_state_dict(mapped_state_dict, strict=True)
            flex_compiled = torch.compile(flex_compiled, mode="max-autotune")
            flex_compiled_stats = benchmark_model(flex_compiled, x, "FlexAttention DAT Compiled")
            
            # Print results table
            print(f"\n{'Model':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Speedup':<10}")
            print("-" * 90)
            
            baseline = original_stats['mean']
            models = [
                ("Original DAT", original_stats, baseline),
                ("Original DAT (compiled)", original_compiled_stats, baseline),
                ("FlexAttention DAT", flex_stats, baseline),
                ("FlexAttention DAT (compiled)", flex_compiled_stats, baseline),
            ]
            
            for name, stats, base in models:
                speedup = base / stats['mean']
                print(f"{name:<30} {stats['mean']:<12.2f} {stats['std']:<12.2f} {stats['min']:<12.2f} {stats['max']:<12.2f} {speedup:<10.2f}x")
            
            # Memory usage
            print("\nMemory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            # Clear memory
            del original, original_compiled, flex, flex_compiled
            torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)


if __name__ == "__main__":
    main()