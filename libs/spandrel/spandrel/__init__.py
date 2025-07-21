"""
Spandrel is a library for loading and running pre-trained PyTorch models. It automatically detects the model architecture and hyper parameters from model files, and provides a unified interface for running models.
"""

__version__ = "0.4.1"

from .__helpers.canonicalize import canonicalize_state_dict
from .__helpers.cuda_compat import (
    make_cuda_graph_compatible,
    wrap_for_cuda_graphs,
)
from .__helpers.loader import ModelLoader
from .__helpers.main_registry import MAIN_REGISTRY
from .__helpers.model_descriptor import (
    ArchId,
    Architecture,
    ImageModelDescriptor,
    MaskedImageModelDescriptor,
    ModelBase,
    ModelDescriptor,
    ModelTiling,
    Purpose,
    SizeRequirements,
    StateDict,
    UnsupportedDtypeError,
)
from .__helpers.registry import (
    ArchRegistry,
    ArchSupport,
    DuplicateArchitectureError,
    UnsupportedModelError,
)

__all__ = [
    "ArchId",
    "Architecture",
    "ArchRegistry",
    "ArchSupport",
    "canonicalize_state_dict",
    "DuplicateArchitectureError",
    "ImageModelDescriptor",
    "MAIN_REGISTRY",
    "make_cuda_graph_compatible",
    "MaskedImageModelDescriptor",
    "ModelBase",
    "ModelDescriptor",
    "ModelLoader",
    "ModelTiling",
    "Purpose",
    "SizeRequirements",
    "StateDict",
    "UnsupportedDtypeError",
    "UnsupportedModelError",
    "wrap_for_cuda_graphs",
]
