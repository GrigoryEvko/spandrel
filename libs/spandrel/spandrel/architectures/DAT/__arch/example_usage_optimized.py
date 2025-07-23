"""
Example usage of optimized DAT model with FlexAttention.
Demonstrates best practices for torch.compile and performance optimization.
"""

import torch
import torch.nn as nn
import time
from typing import Optional, Dict
from DAT_optim import DAT


def load_pretrained_weights(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load pretrained weights into the model."""
    print(f"Loading weights from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict, strict=True)
    print("✅ Weights loaded successfully")
    
    return model


def setup_optimized_dat_model(
    checkpoint_path: Optional[str] = None,
    compile_mode: str = "max-autotune",
    device: str = "cuda"
) -> nn.Module:
    """Setup DAT model with optimizations."""
    
    # Model configuration for 4xNomos2_hq_dat2
    model_config = {
        "img_size": 64,
        "in_chans": 3,
        "embed_dim": 180,
        "split_size": [8, 8],
        "depth": [2, 2, 2, 2],
        "num_heads": [3, 6, 12, 24],
        "expansion_factor": 4.0,
        "qkv_bias": True,
        "upscale": 4,
        "img_range": 1.0,
        "resi_connection": "1conv",
        "upsampler": "pixelshuffle",
    }
    
    print("Creating DAT model with FlexAttention...")
    model = DAT(**model_config)
    
    # Load pretrained weights if provided
    if checkpoint_path:
        model = load_pretrained_weights(model, checkpoint_path)
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    # Compile the model
    print(f"\nCompiling model with mode='{compile_mode}'...")
    print("This may take a few minutes on first run...")
    
    # torch.compile options for different use cases
    compile_options = {
        "default": {},
        "reduce-overhead": {"mode": "reduce-overhead"},
        "max-autotune": {"mode": "max-autotune"},
        "max-autotune-no-cudagraphs": {"mode": "max-autotune-no-cudagraphs"},
    }
    
    if compile_mode in compile_options:
        compiled_model = torch.compile(model, **compile_options[compile_mode])
    else:
        print(f"Unknown compile mode: {compile_mode}, using default")
        compiled_model = torch.compile(model)
    
    return compiled_model


def benchmark_inference(
    model: nn.Module,
    input_shape: tuple = (1, 3, 256, 256),
    num_warmup: int = 3,
    num_iterations: int = 10,
    device: str = "cuda"
) -> Dict[str, float]:
    """Benchmark model inference performance."""
    
    # Create dummy input
    x = torch.randn(*input_shape, device=device)
    
    print(f"\nBenchmarking with input shape: {input_shape}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Benchmark iterations: {num_iterations}")
    
    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for i in range(num_warmup):
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"  Warmup {i+1}/{num_warmup} complete")
    
    # Benchmark
    print("\nBenchmarking...")
    times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            output = model(x)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # Convert to ms
            print(f"  Iteration {i+1}/{num_iterations}: {elapsed*1000:.2f}ms")
    
    # Calculate statistics
    times = times[1:]  # Skip first iteration which might be slower
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Memory usage (CUDA only)
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    else:
        peak_memory = 0
        current_memory = 0
    
    results = {
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "output_shape": list(output.shape),
        "peak_memory_mb": peak_memory,
        "current_memory_mb": current_memory,
    }
    
    return results


def test_cudagraph_capture(model: nn.Module, device: str = "cuda") -> bool:
    """Test if model is compatible with CUDA Graphs."""
    if device != "cuda":
        print("CUDA Graphs require CUDA device")
        return False
    
    print("\nTesting CUDA Graph compatibility...")
    
    try:
        # Static input shape for CUDA Graphs
        static_input = torch.randn(1, 3, 256, 256, device="cuda")
        
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
        
        # Test replay
        g.replay()
        torch.cuda.synchronize()
        
        print("✅ Model is CUDA Graph compatible!")
        return True
        
    except Exception as e:
        print(f"❌ CUDA Graph capture failed: {e}")
        return False


def main():
    """Main example demonstrating optimized DAT usage."""
    print("=" * 60)
    print("DAT Model with FlexAttention - Optimized Usage Example")
    print("=" * 60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Path to pretrained weights (update this path)
    checkpoint_path = "/home/grigory/theartisanai/model_serving/upscaler/4xNomos2_hq_dat2.pth"
    
    # Create and compile model
    try:
        model = setup_optimized_dat_model(
            checkpoint_path=checkpoint_path,
            compile_mode="max-autotune",  # Options: default, reduce-overhead, max-autotune
            device=device
        )
    except FileNotFoundError:
        print(f"\nCheckpoint not found at {checkpoint_path}")
        print("Creating model without pretrained weights...")
        model = setup_optimized_dat_model(
            checkpoint_path=None,
            compile_mode="max-autotune",
            device=device
        )
    
    # Test different input sizes
    test_sizes = [
        (1, 3, 128, 128),
        (1, 3, 256, 256),
        (1, 3, 512, 512),
    ]
    
    print("\n" + "=" * 60)
    print("Performance Benchmarks")
    print("=" * 60)
    
    for input_shape in test_sizes:
        results = benchmark_inference(
            model,
            input_shape=input_shape,
            num_warmup=3,
            num_iterations=10,
            device=device
        )
        
        print(f"\nResults for {input_shape}:")
        print(f"  Average time: {results['avg_time_ms']:.2f}ms")
        print(f"  Min time: {results['min_time_ms']:.2f}ms")
        print(f"  Max time: {results['max_time_ms']:.2f}ms")
        print(f"  Output shape: {results['output_shape']}")
        if device == "cuda":
            print(f"  Peak memory: {results['peak_memory_mb']:.1f}MB")
    
    # Test CUDA Graphs
    if device == "cuda":
        print("\n" + "=" * 60)
        print("CUDA Graph Test")
        print("=" * 60)
        test_cudagraph_capture(model, device)
    
    # Demonstrate actual usage
    print("\n" + "=" * 60)
    print("Example Super-Resolution")
    print("=" * 60)
    
    # Load or create a test image
    test_input = torch.randn(1, 3, 128, 128, device=device)
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Upscale factor: {output.shape[-1] // test_input.shape[-1]}")
    
    print("\n✅ Example completed successfully!")
    
    # Tips for best performance
    print("\n" + "=" * 60)
    print("Performance Tips")
    print("=" * 60)
    print("1. Use torch.compile with mode='max-autotune' for best performance")
    print("2. First run will be slow due to compilation - this is normal")
    print("3. Use consistent input shapes to avoid recompilation")
    print("4. Enable CUDA Graphs for static input shapes")
    print("5. Use batch size 1 for best latency, larger batches for throughput")
    print("6. Ensure GPU has sufficient memory for larger images")


if __name__ == "__main__":
    main()