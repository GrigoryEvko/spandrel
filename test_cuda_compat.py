#!/usr/bin/env python3
"""
Test script to verify CUDA graph compatibility changes
"""

import torch
import sys
sys.path.insert(0, './libs/spandrel')

from spandrel import (
    ModelLoader,
    make_cuda_graph_compatible,
    compile_with_best_settings,
    wrap_for_cuda_graphs
)

def test_basic_import():
    """Test that all new functions are importable"""
    print("✓ All imports successful")

def test_model_loader_cuda_graph_param():
    """Test ModelLoader with cuda_graph_compatible parameter"""
    try:
        loader = ModelLoader(cuda_graph_compatible=True)
        print("✓ ModelLoader accepts cuda_graph_compatible parameter")
    except Exception as e:
        print(f"✗ ModelLoader cuda_graph_compatible failed: {e}")

def test_prepare_for_inference():
    """Test that prepare_for_inference method exists"""
    # Create a dummy model descriptor to test
    from spandrel import ImageModelDescriptor
    from spandrel.architectures.ESRGAN import ESRGAN
    
    # Create a simple model
    model = ESRGAN(in_nc=3, out_nc=3, num_filters=64, num_blocks=1, scale=4)
    state_dict = model.state_dict()
    
    # Create descriptor
    from spandrel.architectures.ESRGAN import ESRGANArch
    arch = ESRGANArch()
    descriptor = ImageModelDescriptor(
        model=model,
        state_dict=state_dict,
        architecture=arch,
        purpose="SR",
        tags=[],
        supports_half=True,
        supports_bfloat16=True,
        scale=4,
        input_channels=3,
        output_channels=3,
    )
    
    # Test prepare_for_inference
    try:
        descriptor.prepare_for_inference()
        print("✓ prepare_for_inference() method works")
    except Exception as e:
        print(f"✗ prepare_for_inference() failed: {e}")

def test_cuda_compat_functions():
    """Test CUDA compatibility utility functions"""
    from spandrel import ImageModelDescriptor
    from spandrel.architectures.ESRGAN import ESRGAN
    
    # Create a simple model
    model = ESRGAN(in_nc=3, out_nc=3, num_filters=64, num_blocks=1, scale=4)
    state_dict = model.state_dict()
    
    # Create descriptor
    from spandrel.architectures.ESRGAN import ESRGANArch
    arch = ESRGANArch()
    descriptor = ImageModelDescriptor(
        model=model,
        state_dict=state_dict,
        architecture=arch,
        purpose="SR",
        tags=[],
        supports_half=True,
        supports_bfloat16=True,
        scale=4,
        input_channels=3,
        output_channels=3,
    )
    
    # Test make_cuda_graph_compatible
    try:
        compat_model = make_cuda_graph_compatible(descriptor)
        print("✓ make_cuda_graph_compatible() works")
    except Exception as e:
        print(f"✗ make_cuda_graph_compatible() failed: {e}")
    
    # Test wrap_for_cuda_graphs
    try:
        wrapped = wrap_for_cuda_graphs(descriptor)
        print("✓ wrap_for_cuda_graphs() works")
    except Exception as e:
        print(f"✗ wrap_for_cuda_graphs() failed: {e}")
    
    # Test compile_with_best_settings
    try:
        # Only test if CUDA is available
        if torch.cuda.is_available():
            compiled = compile_with_best_settings(model, mode='reduce-overhead')
            print("✓ compile_with_best_settings() works")
        else:
            print("⚠ Skipping compile_with_best_settings() test (no CUDA)")
    except Exception as e:
        print(f"✗ compile_with_best_settings() failed: {e}")

def main():
    print("Testing CUDA graph compatibility changes...\n")
    
    test_basic_import()
    test_model_loader_cuda_graph_param()
    test_prepare_for_inference()
    test_cuda_compat_functions()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()