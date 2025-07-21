#!/usr/bin/env python3
"""
Example: Using ESRGAN models with torch.compile and CUDA graphs

This example shows how to use the new CUDA graph compatibility features
with your production ESRGAN models.
"""

import torch
from PIL import Image
from torchvision import transforms

# Add spandrel to path (remove this if installed via pip)
import sys
sys.path.insert(0, './libs/spandrel')

from spandrel import ModelLoader, make_cuda_graph_compatible, wrap_for_cuda_graphs

def load_and_compile_esrgan(model_path: str, device: str = 'cuda'):
    """Load an ESRGAN model and compile it for maximum performance"""
    
    # Method 1: Load with CUDA graph compatibility enabled
    print(f"Loading model: {model_path}")
    model = ModelLoader(
        device=device, 
        cuda_graph_compatible=True
    ).load_from_file(model_path)
    
    # The model is already prepared for inference due to cuda_graph_compatible=True
    print(f"Model architecture: {model.architecture.name}")
    print(f"Scale: {model.scale}x")
    print(f"Input channels: {model.input_channels}")
    
    # Method 2: Alternative - wrap for CUDA graphs (removes inference_mode)
    wrapped_model = wrap_for_cuda_graphs(model)
    
    # Compile the model
    print("Compiling model with torch.compile...")
    compiled_model = torch.compile(
        wrapped_model,
        mode='max-autotune',  # Now safe to use with CUDA graphs
        fullgraph=False,      # Allow graph breaks for better compatibility
    )
    
    return model, compiled_model


def upscale_with_tiling_example():
    """Example of how to use the compiled model with your existing tiling code"""
    
    # Load and compile the model
    model_path = "./upscaler/8x_NMKD-Superscale_150000_G.pth"  # Your model
    model_descriptor, compiled_model = load_and_compile_esrgan(model_path)
    
    # Example usage with a dummy image
    print("\nTesting compiled model...")
    
    # Create a test tensor (replace with your actual image loading)
    test_tensor = torch.randn(1, 3, 64, 64).cuda().half()
    
    # First run will be slower due to compilation
    print("First run (compilation)...")
    with torch.no_grad():
        output = compiled_model(test_tensor)
    print(f"Output shape: {output.shape}")
    
    # Subsequent runs will be much faster
    print("Second run (using compiled graph)...")
    with torch.no_grad():
        output = compiled_model(test_tensor)
    print(f"Output shape: {output.shape}")
    
    return compiled_model


def integrate_with_your_code():
    """
    Example of how to integrate with your existing upscale_image_with_tiling function
    """
    print("\n" + "="*50)
    print("Integration with your existing code:")
    print("="*50)
    
    code_example = '''
# In your existing code, replace:
upscale_model = get_upscale_model("./upscaler/8x_NMKD-Superscale_150000_G.pth", "cuda")

# With:
from spandrel import ModelLoader, wrap_for_cuda_graphs

# Load with CUDA graph compatibility
model_descriptor = ModelLoader(
    device="cuda",
    cuda_graph_compatible=True
).load_from_file("./upscaler/8x_NMKD-Superscale_150000_G.pth")

# Get the underlying PyTorch model
upscale_model = model_descriptor.model

# Compile with max-autotune (now safe!)
upscale_model = torch.compile(upscale_model, mode='max-autotune')

# Use in your existing esrgan_upscale function as before
'''
    
    print(code_example)


def main():
    """Run the examples"""
    print("ESRGAN CUDA Graph Compilation Example")
    print("=====================================\n")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Examples will use CPU.")
        print("The compilation benefits are primarily for CUDA devices.\n")
    
    # Show integration example
    integrate_with_your_code()
    
    print("\nNote: On your production system with CUDA, this should resolve")
    print("the CUDA graph conflicts when running multiple compiled models.")


if __name__ == "__main__":
    main()