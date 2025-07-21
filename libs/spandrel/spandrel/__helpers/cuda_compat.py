"""
CUDA Graph Compatibility Utilities for Spandrel

This module provides utilities to make models compatible with torch.compile
and CUDA graphs, addressing common issues that prevent graph capture.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Set

import torch

if TYPE_CHECKING:
    from .model_descriptor import ModelBase

# Architectures known to have CUDA graph compatibility issues
INCOMPATIBLE_ARCHITECTURES: Set[str] = {
    "OmniSR",  # Custom LayerNorm autograd function
    "GFPGAN",  # Custom upfirdn2d and fused_act autograd functions
    "CodeFormer",  # Based on GFPGAN
    "RestoreFormer",  # Based on GFPGAN
}

# Architectures that may have issues but can work with careful usage
PARTIALLY_COMPATIBLE_ARCHITECTURES: Set[str] = {
    "DySample",  # Fixed with pre-computed coordinates, but has size limitations
    "SPAN",  # Dynamic behavior based on training mode
    "PLKSR",  # Dynamic behavior based on training mode
    "RealPLKSR",  # Dynamic behavior based on training mode
    "SeemoRe",  # Dynamic behavior based on training mode
    "RGT",  # Dynamic behavior based on training mode
}


def make_cuda_graph_compatible(model: ModelBase) -> ModelBase:
    """
    Prepare a model for CUDA graph compatibility.
    
    This function:
    1. Sets the model to evaluation mode
    2. Warns about known incompatibilities
    3. Provides recommendations for torch.compile usage
    
    Args:
        model: The model descriptor to prepare
        
    Returns:
        The same model descriptor, prepared for CUDA graphs
    """
    # Get architecture name
    arch_name = model.architecture.name
    
    # Check for known incompatibilities
    if arch_name in INCOMPATIBLE_ARCHITECTURES:
        warnings.warn(
            f"Architecture '{arch_name}' contains custom autograd functions "
            f"that are incompatible with torch.compile. You may experience "
            f"errors or graph breaks. Consider using a different architecture "
            f"or disabling torch.compile for this model.",
            RuntimeWarning,
            stacklevel=2
        )
    elif arch_name in PARTIALLY_COMPATIBLE_ARCHITECTURES:
        warnings.warn(
            f"Architecture '{arch_name}' may have limited compatibility with "
            f"torch.compile. Ensure you use fixed input sizes and call "
            f"model.prepare_for_inference() before compilation.",
            RuntimeWarning,
            stacklevel=2
        )
    
    # Prepare model for inference
    model.prepare_for_inference()
    
    return model


def compile_with_best_settings(
    model: torch.nn.Module,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> torch.nn.Module:
    """
    Compile a model with settings optimized for CUDA graph compatibility.
    
    Args:
        model: The PyTorch module to compile
        mode: Compilation mode. Default is "reduce-overhead" which is more
              compatible than "max-autotune"
        fullgraph: Whether to require full graph capture. Default False allows
                   graph breaks for better compatibility
        dynamic: Whether to allow dynamic shapes. Default False for better
                 CUDA graph compatibility
                 
    Returns:
        The compiled model
    """
    try:
        compiled_model = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        return compiled_model
    except Exception as e:
        warnings.warn(
            f"Failed to compile model: {e}. Returning uncompiled model.",
            RuntimeWarning,
            stacklevel=2
        )
        return model


def wrap_for_cuda_graphs(model: ModelBase) -> torch.nn.Module:
    """
    Create a simple wrapper around a model descriptor that's optimized for CUDA graphs.
    
    This wrapper:
    1. Removes the inference_mode context
    2. Ensures the model stays in eval mode
    3. Provides a simple forward interface
    
    Args:
        model: The model descriptor to wrap
        
    Returns:
        A torch.nn.Module wrapper suitable for CUDA graph compilation
    """
    
    class CUDAGraphWrapper(torch.nn.Module):
        def __init__(self, model_descriptor):
            super().__init__()
            self.model = model_descriptor.model
            self.model.eval()
            
        def forward(self, x):
            # Direct forward pass without any context managers or state changes
            return self.model(x)
    
    # Prepare the model
    model.prepare_for_inference()
    
    return CUDAGraphWrapper(model)