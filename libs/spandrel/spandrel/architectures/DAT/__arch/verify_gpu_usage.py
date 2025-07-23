"""
Verify that models are actually running on GPU.
"""

import sys
import os
sys.path.append('/home/grigory/theartisanai/spandrel/libs/spandrel')

import spandrel
import torch
import time
from DAT_optim import DAT as FlexDAT


def check_gpu_usage(model, input_tensor, model_name="Model"):
    """Check if model is actually using GPU."""
    print(f"\n=== {model_name} GPU Usage Check ===")
    
    # Check model device
    print(f"Model is on: {next(model.parameters()).device}")
    print(f"Input is on: {input_tensor.device}")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Monitor GPU memory before and after
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Run inference
        start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        torch.cuda.synchronize()
        inference_time = (time.time() - start) * 1000
        
        mem_after = torch.cuda.memory_allocated() / 1024**2  # MB
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"Memory before: {mem_before:.1f} MB")
        print(f"Memory after: {mem_after:.1f} MB")
        print(f"Peak memory: {peak_mem:.1f} MB")
        print(f"Memory increase: {mem_after - mem_before:.1f} MB")
        print(f"Inference time: {inference_time:.2f} ms")
        print(f"Output device: {output.device}")
        
        # Check if computation actually happened on GPU
        if mem_after - mem_before < 1:  # Less than 1MB increase
            print("WARNING: No significant GPU memory increase detected!")
            print("Model might be running on CPU!")
    
    return output


def main():
    # Force CUDA if available
    if not torch.cuda.is_available():
        print("CUDA is not available! Please check your PyTorch installation.")
        return
    
    # Set default device
    torch.set_default_device('cuda')
    device = torch.device('cuda')
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Load models
    model_path = "/home/grigory/theartisanai/model_serving/upscaler/4xNomos2_hq_dat2.pth"
    
    print("\nLoading models...")
    # Load original model
    spandrel_model = spandrel.ModelLoader().load_from_file(model_path)
    original = spandrel_model.model
    
    # IMPORTANT: Ensure model is on GPU
    original = original.cuda()
    original.eval()
    
    # Create FlexAttention model
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
    
    flex = FlexDAT(**model_config)
    
    # Load weights with key mapping
    state_dict = original.state_dict()
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if key == "before_RG.1.weight":
            mapped_state_dict["before_RG_norm.weight"] = value
        elif key == "before_RG.1.bias":
            mapped_state_dict["before_RG_norm.bias"] = value
        else:
            mapped_state_dict[key] = value
    
    flex.load_state_dict(mapped_state_dict, strict=True)
    
    # IMPORTANT: Ensure FlexAttention model is on GPU
    flex = flex.cuda()
    flex.eval()
    
    # Create input on GPU
    input_sizes = [(1, 3, 64, 64), (1, 3, 128, 128)]
    
    for size in input_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with input size: {size}")
        print('='*60)
        
        # Create input tensor directly on GPU
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # Verify input is on GPU
        print(f"Input tensor device: {x.device}")
        print(f"Input tensor shape: {x.shape}")
        print(f"Input tensor dtype: {x.dtype}")
        
        # Test original model
        out1 = check_gpu_usage(original, x, "Original DAT")
        
        # Test FlexAttention model
        out2 = check_gpu_usage(flex, x, "FlexAttention DAT")
        
        # Compare outputs
        diff = (out1 - out2).abs()
        print(f"\nOutput difference: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")
    
    # Test with torch.cuda.nvtx.range for profiling
    print("\n" + "="*60)
    print("Running with NVTX ranges for profiling")
    print("="*60)
    
    x = torch.randn(1, 3, 64, 64, device='cuda')
    
    # Profile original
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("Original_DAT"):
        with torch.no_grad():
            _ = original(x)
    torch.cuda.synchronize()
    
    # Profile FlexAttention
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("FlexAttention_DAT"):
        with torch.no_grad():
            _ = flex(x)
    torch.cuda.synchronize()
    
    print("\nIf models are running on GPU, you should see:")
    print("1. Significant GPU memory usage increase during inference")
    print("2. GPU utilization in nvidia-smi")
    print("3. NVTX ranges visible in Nsight Systems profiler")
    
    # Additional check: force a simple GPU operation
    print("\n" + "="*60)
    print("Testing simple GPU operation for comparison")
    print("="*60)
    
    # Simple matmul on GPU
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')
    
    torch.cuda.synchronize()
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    matmul_time = (time.time() - start) * 1000
    
    print(f"Simple 1000x1000 matmul on GPU: {matmul_time:.2f} ms")
    print("If this is fast (~0.1-1ms), GPU computation is working")


if __name__ == "__main__":
    main()