#!/usr/bin/env python3
"""
Example: How to use ESRGAN models with torch.compile and CUDA graphs
"""

import torch

# Add spandrel to path (remove this if installed via pip)
import sys
sys.path.insert(0, './libs/spandrel')

from spandrel import ModelLoader

def main():
    # Step 1: Load model with CUDA graph compatibility enabled
    print("Loading ESRGAN model...")
    model = ModelLoader(
        device='cuda', 
        cuda_graph_compatible=True  # This calls prepare_for_inference() automatically
    ).load_from_file("./upscaler/8x_NMKD-Superscale_150000_G.pth")
    
    # The model is now in eval mode and ready for CUDA graphs
    print(f"Model loaded: {model.architecture.name}")
    print(f"Scale: {model.scale}x")
    
    # Step 2: Extract the pure PyTorch model (no inference_mode wrapper)
    # Option A: Direct access to underlying model
    pure_model = model.model
    
    # Option B: Use the wrapper utility (cleaner approach)
    from spandrel import wrap_for_cuda_graphs
    wrapped_model = wrap_for_cuda_graphs(model)
    
    # Step 3: Compile the model
    print("\nCompiling model with torch.compile...")
    compiled_model = torch.compile(
        pure_model,  # or use wrapped_model
        mode='max-autotune',  # Now safe to use!
        fullgraph=False,      # Allow graph breaks if needed
    )
    
    # Step 4: Use the compiled model
    print("\nTesting compiled model...")
    
    # Create test input
    test_input = torch.randn(1, 3, 64, 64).cuda().half()
    
    # Important: Use torch.no_grad() since we removed inference_mode
    with torch.no_grad():
        # First run will be slower (compilation)
        output = compiled_model(test_input)
        print(f"First run output shape: {output.shape}")
        
        # Subsequent runs will be much faster
        output = compiled_model(test_input)
        print(f"Second run output shape: {output.shape}")
    
    print("\n✓ Model is now compiled and ready for production use!")
    
    # Example integration with your existing code:
    print("\n" + "="*50)
    print("Integration with your existing upscale function:")
    print("="*50)
    print("""
# In your upscale_image_with_tiling function, replace:
model = get_upscale_model(path, 'cuda')

# With:
from spandrel import ModelLoader
model_descriptor = ModelLoader(device='cuda', cuda_graph_compatible=True).load_from_file(path)
model = torch.compile(model_descriptor.model, mode='max-autotune')

# Then use as before in your esrgan_upscale function
    """)

if __name__ == "__main__":
    # Note: This example assumes CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this example")
        sys.exit(1)
    
    main()