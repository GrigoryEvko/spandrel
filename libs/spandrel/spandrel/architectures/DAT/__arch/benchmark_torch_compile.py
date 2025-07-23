"""
Comprehensive benchmark comparing original DAT vs FlexAttention DAT with torch.compile.
"""

import sys
import os

# Add the parent directory to sys.path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append('/home/grigory/theartisanai/spandrel/libs/spandrel')

import spandrel
import torch
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import gc

# Import DAT implementations using full paths
from spandrel.architectures.DAT.__arch import DAT
from spandrel.architectures.DAT.__arch import DAT_optim


def measure_inference_time(model, input_tensor, num_warmup=10, num_measure=20):
    """
    Measure inference time with proper warmup and CUDA synchronization.
    
    Args:
        model: The model to benchmark
        input_tensor: Input tensor
        num_warmup: Number of warmup iterations
        num_measure: Number of measurement iterations
    
    Returns:
        tuple: (mean_time_ms, std_time_ms, all_times_ms)
    """
    model = model.eval()
    
    # Warmup phase
    print(f"    Warming up with {num_warmup} iterations...", end='', flush=True)
    for i in range(num_warmup):
        with torch.no_grad():
            _ = model(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print('.', end='', flush=True)
    print(" done")
    
    # Measurement phase
    times = []
    print(f"    Measuring {num_measure} iterations...", end='', flush=True)
    for i in range(num_measure):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # Convert to ms
        if (i + 1) % 5 == 0:
            print('.', end='', flush=True)
    print(" done")
    
    times_np = np.array(times)
    return np.mean(times_np), np.std(times_np), times_np


def compile_model(model, example_input):
    """
    Compile model with torch.compile using max-autotune.
    
    Args:
        model: Model to compile
        example_input: Example input for tracing
    
    Returns:
        Compiled model or None if compilation fails
    """
    try:
        print("    Compiling model with max-autotune...", end='', flush=True)
        start = time.time()
        
        # Compile with max-autotune for best performance
        compiled = torch.compile(
            model,
            mode="max-autotune",
            fullgraph=True,  # Try to compile the entire graph
            dynamic=False,   # No dynamic shapes for better optimization
        )
        
        # Trigger compilation with example input
        with torch.no_grad():
            _ = compiled(example_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        compile_time = time.time() - start
        print(f" done (took {compile_time:.2f}s)")
        
        return compiled
    except Exception as e:
        print(f" failed: {type(e).__name__}: {str(e)}")
        return None


def measure_memory_usage(model, input_tensor):
    """Measure peak memory usage during inference."""
    if not torch.cuda.is_available():
        return 0
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    torch.cuda.synchronize()
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return peak_memory_mb


def create_summary_table(results):
    """Create a summary table of results."""
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    
    # Header
    print(f"{'Input Size':<15} {'Model':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10}")
    print("-"*100)
    
    for input_size, models in results.items():
        size_str = f"{input_size[2]}x{input_size[3]}"
        
        # Get baseline (original) time
        baseline_time = models.get('original', {}).get('mean', float('inf'))
        
        for model_name in ['original', 'flex', 'original_compiled', 'flex_compiled']:
            if model_name in models:
                data = models[model_name]
                speedup = baseline_time / data['mean'] if data['mean'] > 0 else 0
                
                print(f"{size_str:<15} {model_name:<25} "
                      f"{data['mean']:<12.2f} {data['std']:<12.2f} "
                      f"{data.get('memory', 0):<12.1f} {speedup:<10.2f}x")
        
        print()  # Empty line between input sizes


def main():
    """Run comprehensive torch.compile benchmark."""
    # Configuration
    model_path = "/home/grigory/theartisanai/model_serving/upscaler/4xNomos2_hq_dat2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_sizes = [(1, 3, 64, 64), (1, 3, 128, 128), (1, 3, 256, 256)]
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        
    # Set torch settings for better performance
    torch.set_float32_matmul_precision('high')
    
    # Load models
    print(f"\n{'='*60}")
    print("LOADING MODELS")
    print(f"{'='*60}")
    
    # Load original model with spandrel
    print("\nLoading original DAT model...")
    spandrel_model = spandrel.ModelLoader().load_from_file(model_path)
    original = spandrel_model.model.to(device).eval()
    
    # Get state dict and create FlexAttention version
    state_dict = original.state_dict()
    
    # Model configuration (from spandrel loaded model)
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
    
    print("Creating FlexAttention DAT model...")
    flex = DAT_optim.DAT(**model_config).to(device).eval()
    
    # Map state dict keys
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if key == "before_RG.1.weight":
            mapped_state_dict["before_RG_norm.weight"] = value
        elif key == "before_RG.1.bias":
            mapped_state_dict["before_RG_norm.bias"] = value
        else:
            mapped_state_dict[key] = value
    
    flex.load_state_dict(mapped_state_dict, strict=True)
    
    # Results storage
    results = {}
    
    # Run benchmarks
    print(f"\n{'='*60}")
    print("RUNNING BENCHMARKS")
    print(f"{'='*60}")
    
    for input_size in input_sizes:
        print(f"\n\nTesting input size: {input_size}")
        print("-"*60)
        
        results[input_size] = {}
        
        # Create input tensor
        x = torch.randn(*input_size, device=device)
        
        # Force garbage collection before each test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 1. Benchmark original model
        print("\n1. Original DAT:")
        mean_time, std_time, all_times = measure_inference_time(original, x)
        memory = measure_memory_usage(original, x)
        results[input_size]['original'] = {
            'mean': mean_time,
            'std': std_time,
            'times': all_times,
            'memory': memory
        }
        print(f"    Result: {mean_time:.2f} ± {std_time:.2f} ms, Memory: {memory:.1f} MB")
        
        # 2. Benchmark FlexAttention model
        print("\n2. FlexAttention DAT:")
        mean_time, std_time, all_times = measure_inference_time(flex, x)
        memory = measure_memory_usage(flex, x)
        results[input_size]['flex'] = {
            'mean': mean_time,
            'std': std_time,
            'times': all_times,
            'memory': memory
        }
        print(f"    Result: {mean_time:.2f} ± {std_time:.2f} ms, Memory: {memory:.1f} MB")
        
        # 3. Compile and benchmark original model
        print("\n3. Original DAT (torch.compile):")
        original_compiled = compile_model(original, x)
        if original_compiled is not None:
            mean_time, std_time, all_times = measure_inference_time(original_compiled, x)
            memory = measure_memory_usage(original_compiled, x)
            results[input_size]['original_compiled'] = {
                'mean': mean_time,
                'std': std_time,
                'times': all_times,
                'memory': memory
            }
            print(f"    Result: {mean_time:.2f} ± {std_time:.2f} ms, Memory: {memory:.1f} MB")
        
        # 4. Compile and benchmark FlexAttention model
        print("\n4. FlexAttention DAT (torch.compile):")
        flex_compiled = compile_model(flex, x)
        if flex_compiled is not None:
            mean_time, std_time, all_times = measure_inference_time(flex_compiled, x)
            memory = measure_memory_usage(flex_compiled, x)
            results[input_size]['flex_compiled'] = {
                'mean': mean_time,
                'std': std_time,
                'times': all_times,
                'memory': memory
            }
            print(f"    Result: {mean_time:.2f} ± {std_time:.2f} ms, Memory: {memory:.1f} MB")
    
    # Print summary
    create_summary_table(results)
    
    # Additional analysis
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    for input_size in input_sizes:
        size_str = f"{input_size[2]}x{input_size[3]}"
        print(f"\n{size_str}:")
        
        data = results[input_size]
        
        # FlexAttention vs Original
        if 'flex' in data and 'original' in data:
            speedup = data['original']['mean'] / data['flex']['mean']
            print(f"  FlexAttention speedup: {speedup:.2f}x")
        
        # Compilation speedup for original
        if 'original_compiled' in data and 'original' in data:
            speedup = data['original']['mean'] / data['original_compiled']['mean']
            print(f"  Original compilation speedup: {speedup:.2f}x")
        
        # Compilation speedup for FlexAttention
        if 'flex_compiled' in data and 'flex' in data:
            speedup = data['flex']['mean'] / data['flex_compiled']['mean']
            print(f"  FlexAttention compilation speedup: {speedup:.2f}x")
        
        # Best compiled vs original
        if 'flex_compiled' in data and 'original' in data:
            speedup = data['original']['mean'] / data['flex_compiled']['mean']
            print(f"  FlexAttention compiled vs Original: {speedup:.2f}x")


if __name__ == "__main__":
    main()