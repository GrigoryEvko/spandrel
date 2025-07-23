"""
Benchmark script to compare performance between original DAT and FlexAttention DAT.
Measures both eager mode and torch.compile performance.
"""

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import numpy as np
import time
import gc
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_attention_module(
    module: nn.Module,
    input_shape: Tuple[int, ...],
    H: int,
    W: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    desc: str = ""
) -> Dict:
    """Benchmark a single attention module."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device).eval()
    
    # Create input
    if hasattr(module, 'qkv'):
        # For attention modules that expect QKV input
        x = torch.randn(*input_shape, device=device)
        B, L, C = x.shape
        qkv = module.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        input_tensor = qkv
    else:
        input_tensor = torch.randn(*input_shape, device=device)
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = module(input_tensor, H, W)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = module(input_tensor, H, W)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start
    avg_time = total_time / num_iters * 1000  # Convert to ms
    
    # Memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = module(input_tensor, H, W)
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        peak_memory = 0
    
    return {
        "desc": desc,
        "avg_time_ms": avg_time,
        "total_time_s": total_time,
        "peak_memory_mb": peak_memory,
        "num_iters": num_iters,
    }


def benchmark_spatial_attention():
    """Benchmark spatial attention implementations."""
    print("\n=== Benchmarking Spatial Attention ===")
    
    from DAT_optim import Spatial_Attention as FlexSpatialAttention
    from DAT import Spatial_Attention as OriginalSpatialAttention
    
    configs = [
        {"dim": 96, "heads": 3, "H": 64, "W": 64, "split": [8, 8]},
        {"dim": 192, "heads": 6, "H": 128, "W": 128, "split": [8, 8]},
        {"dim": 384, "heads": 12, "H": 256, "W": 256, "split": [16, 16]},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nConfig: dim={config['dim']}, H={config['H']}, W={config['W']}")
        
        # Create modules
        original = OriginalSpatialAttention(
            dim=config["dim"],
            idx=0,
            split_size=config["split"],
            num_heads=config["heads"]
        )
        
        flex = FlexSpatialAttention(
            dim=config["dim"],
            idx=0,
            split_size=config["split"],
            num_heads=config["heads"]
        )
        
        # Input shape
        B = 1
        input_shape = (B, config["H"] * config["W"], config["dim"])
        
        # Benchmark original
        orig_stats = benchmark_attention_module(
            original, input_shape, config["H"], config["W"],
            desc=f"Original_{config['dim']}d"
        )
        
        # Benchmark FlexAttention
        flex_stats = benchmark_attention_module(
            flex, input_shape, config["H"], config["W"],
            desc=f"FlexAttention_{config['dim']}d"
        )
        
        # Benchmark compiled FlexAttention
        try:
            flex_compiled = torch.compile(flex, mode="max-autotune")
            flex_compiled_stats = benchmark_attention_module(
                flex_compiled, input_shape, config["H"], config["W"],
                num_warmup=20,  # More warmup for compilation
                desc=f"FlexCompiled_{config['dim']}d"
            )
        except Exception as e:
            print(f"Compilation failed: {e}")
            flex_compiled_stats = None
        
        # Calculate speedups
        speedup_flex = orig_stats["avg_time_ms"] / flex_stats["avg_time_ms"]
        print(f"  Original: {orig_stats['avg_time_ms']:.2f}ms")
        print(f"  FlexAttention: {flex_stats['avg_time_ms']:.2f}ms (speedup: {speedup_flex:.2f}x)")
        
        if flex_compiled_stats:
            speedup_compiled = orig_stats["avg_time_ms"] / flex_compiled_stats["avg_time_ms"]
            print(f"  FlexCompiled: {flex_compiled_stats['avg_time_ms']:.2f}ms (speedup: {speedup_compiled:.2f}x)")
        
        results.append({
            "config": config,
            "original": orig_stats,
            "flex": flex_stats,
            "flex_compiled": flex_compiled_stats,
        })
    
    return results


def benchmark_full_model():
    """Benchmark full DAT model."""
    print("\n=== Benchmarking Full DAT Model ===")
    
    from DAT_optim import DAT as FlexDAT
    from DAT import DAT as OriginalDAT
    
    # Model configs
    model_configs = [
        {
            "name": "DAT-S",
            "embed_dim": 180,
            "depth": [2, 2, 2, 2],
            "num_heads": [3, 6, 12, 24],
        },
        {
            "name": "DAT-B",
            "embed_dim": 256,
            "depth": [2, 2, 4, 2],
            "num_heads": [4, 8, 16, 32],
        },
    ]
    
    input_sizes = [(1, 3, 64, 64), (1, 3, 128, 128)]
    
    results = []
    
    for model_cfg in model_configs:
        print(f"\n{model_cfg['name']} Model:")
        
        base_config = {
            "img_size": 64,
            "in_chans": 3,
            "split_size": [8, 8],
            "expansion_factor": 4.0,
            "qkv_bias": True,
            "upscale": 4,
            "resi_connection": "1conv",
            "upsampler": "pixelshuffle",
        }
        base_config.update(model_cfg)
        
        # Create models
        original = OriginalDAT(**base_config)
        flex = FlexDAT(**base_config)
        
        for input_size in input_sizes:
            B, C, H, W = input_size
            print(f"\n  Input: {input_size}")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x = torch.randn(B, C, H, W, device=device)
            
            # Benchmark original
            original = original.to(device).eval()
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = original(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            orig_time = (time.time() - start) / 10 * 1000
            
            # Benchmark FlexAttention
            flex = flex.to(device).eval()
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = flex(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            flex_time = (time.time() - start) / 10 * 1000
            
            # Benchmark compiled
            try:
                flex_compiled = torch.compile(flex, mode="max-autotune")
                
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = flex_compiled(x)
                
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                start = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        _ = flex_compiled(x)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                compiled_time = (time.time() - start) / 10 * 1000
            except Exception as e:
                print(f"    Compilation failed: {e}")
                compiled_time = None
            
            print(f"    Original: {orig_time:.2f}ms")
            print(f"    FlexAttention: {flex_time:.2f}ms (speedup: {orig_time/flex_time:.2f}x)")
            if compiled_time:
                print(f"    Compiled: {compiled_time:.2f}ms (speedup: {orig_time/compiled_time:.2f}x)")
            
            results.append({
                "model": model_cfg["name"],
                "input_size": input_size,
                "original_ms": orig_time,
                "flex_ms": flex_time,
                "compiled_ms": compiled_time,
            })
    
    return results


def benchmark_cudagraph_compatibility():
    """Test CUDAGraph compatibility."""
    print("\n=== Testing CUDAGraph Compatibility ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDAGraph tests")
        return None
    
    from DAT_optim import DAT as FlexDAT
    
    model = FlexDAT(
        img_size=64,
        in_chans=3,
        embed_dim=180,
        split_size=[8, 8],
        depth=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        upscale=4,
    ).cuda().eval()
    
    # Static input for CUDAGraph
    static_input = torch.randn(1, 3, 64, 64, device="cuda")
    
    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _ = model(static_input)
    torch.cuda.current_stream().wait_stream(s)
    
    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = model(static_input)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.time()
        g.replay()
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  CUDAGraph average time: {avg_time:.2f} ± {std_time:.2f}ms")
    print("  ✅ CUDAGraph compatible!")
    
    return {"avg_ms": avg_time, "std_ms": std_time}


def plot_results(spatial_results: List, model_results: List):
    """Plot benchmark results."""
    try:
        import matplotlib.pyplot as plt
        
        # Spatial attention results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        dims = [r["config"]["dim"] for r in spatial_results]
        original_times = [r["original"]["avg_time_ms"] for r in spatial_results]
        flex_times = [r["flex"]["avg_time_ms"] for r in spatial_results]
        compiled_times = [r["flex_compiled"]["avg_time_ms"] if r["flex_compiled"] else None for r in spatial_results]
        
        x = np.arange(len(dims))
        width = 0.25
        
        ax1.bar(x - width, original_times, width, label='Original')
        ax1.bar(x, flex_times, width, label='FlexAttention')
        if any(compiled_times):
            ax1.bar(x + width, [t for t in compiled_times if t], width, label='Compiled')
        
        ax1.set_xlabel('Model Dimension')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Spatial Attention Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(dims)
        ax1.legend()
        
        # Model results
        model_data = {}
        for r in model_results:
            key = f"{r['model']}_{r['input_size'][2]}x{r['input_size'][3]}"
            model_data[key] = {
                'original': r['original_ms'],
                'flex': r['flex_ms'],
                'compiled': r['compiled_ms'] if r['compiled_ms'] else r['flex_ms']
            }
        
        models = list(model_data.keys())
        x2 = np.arange(len(models))
        
        original = [model_data[m]['original'] for m in models]
        flex = [model_data[m]['flex'] for m in models]
        compiled = [model_data[m]['compiled'] for m in models]
        
        ax2.bar(x2 - width, original, width, label='Original')
        ax2.bar(x2, flex, width, label='FlexAttention')
        ax2.bar(x2 + width, compiled, width, label='Compiled')
        
        ax2.set_xlabel('Model Configuration')
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('Full Model Performance')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('dat_flexattention_benchmark.png', dpi=150)
        print("\nBenchmark plot saved to 'dat_flexattention_benchmark.png'")
        
    except ImportError:
        print("\nMatplotlib not available, skipping plots")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("DAT FlexAttention Performance Benchmarks")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        # Component benchmarks
        spatial_results = benchmark_spatial_attention()
        
        # Full model benchmarks
        model_results = benchmark_full_model()
        
        # CUDAGraph test
        cudagraph_results = benchmark_cudagraph_compatibility()
        
        # Summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Average speedups
        if spatial_results:
            flex_speedups = []
            compiled_speedups = []
            
            for r in spatial_results:
                orig_time = r["original"]["avg_time_ms"]
                flex_time = r["flex"]["avg_time_ms"]
                flex_speedups.append(orig_time / flex_time)
                
                if r["flex_compiled"]:
                    compiled_time = r["flex_compiled"]["avg_time_ms"]
                    compiled_speedups.append(orig_time / compiled_time)
            
            print(f"\nSpatial Attention:")
            print(f"  Average FlexAttention speedup: {np.mean(flex_speedups):.2f}x")
            if compiled_speedups:
                print(f"  Average Compiled speedup: {np.mean(compiled_speedups):.2f}x")
        
        if model_results:
            model_flex_speedups = []
            model_compiled_speedups = []
            
            for r in model_results:
                model_flex_speedups.append(r["original_ms"] / r["flex_ms"])
                if r["compiled_ms"]:
                    model_compiled_speedups.append(r["original_ms"] / r["compiled_ms"])
            
            print(f"\nFull Model:")
            print(f"  Average FlexAttention speedup: {np.mean(model_flex_speedups):.2f}x")
            if model_compiled_speedups:
                print(f"  Average Compiled speedup: {np.mean(model_compiled_speedups):.2f}x")
        
        # Plot results
        if spatial_results and model_results:
            plot_results(spatial_results, model_results)
        
    except ImportError as e:
        print(f"\nError importing modules: {e}")
        print("Make sure both DAT.py and DAT_optim.py are available")
    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()