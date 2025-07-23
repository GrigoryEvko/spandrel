"""
Inspect model using spandrel's default loader.
"""

import sys
import os
sys.path.append('/home/grigory/theartisanai/spandrel/libs/spandrel')

import spandrel
import torch


def main():
    model_path = "/home/grigory/theartisanai/model_serving/upscaler/4xNomos2_hq_dat2.pth"
    
    # Load with spandrel
    print(f"Loading model with spandrel from: {model_path}")
    model = spandrel.ModelLoader().load_from_file(model_path)
    
    print(f"\nModel architecture: {model.architecture}")
    print(f"Model scale: {model.scale}")
    print(f"Model input channels: {model.input_channels}")
    print(f"Model output channels: {model.output_channels}")
    
    # Print model configuration
    if hasattr(model.model, 'state'):
        print(f"\nModel state dict keys: {len(model.model.state)}")
        for key, value in model.model.state.items():
            print(f"  {key}: {value}")
    
    # Get the actual model
    actual_model = model.model
    print(f"\nModel class: {type(actual_model)}")
    
    # Inspect attributes
    if hasattr(actual_model, 'embed_dim'):
        print(f"Embed dim: {actual_model.embed_dim}")
    if hasattr(actual_model, 'num_layers'):
        print(f"Num layers: {actual_model.num_layers}")
    if hasattr(actual_model, 'num_heads'):
        print(f"Num heads: {actual_model.num_heads}")
    if hasattr(actual_model, 'split_size'):
        print(f"Split size: {actual_model.split_size}")
    if hasattr(actual_model, 'expansion_factor'):
        print(f"Expansion factor: {actual_model.expansion_factor}")
    if hasattr(actual_model, 'depth'):
        print(f"Depth: {actual_model.depth}")
    
    # Try to get configuration from the first layer
    if hasattr(actual_model, 'layers'):
        print(f"\nNumber of layer groups: {len(actual_model.layers)}")
        for i, layer in enumerate(actual_model.layers):
            if hasattr(layer, 'blocks'):
                print(f"  Layer {i}: {len(layer.blocks)} blocks")
                # Check first block's attention
                if len(layer.blocks) > 0 and hasattr(layer.blocks[0], 'attn'):
                    attn = layer.blocks[0].attn
                    if hasattr(attn, 'split_size'):
                        print(f"    Split size: {attn.split_size}")
                    if hasattr(attn, 'num_heads'):
                        print(f"    Num heads: {attn.num_heads}")
                    if hasattr(attn, 'attns') and len(attn.attns) > 0:
                        sub_attn = attn.attns[0]
                        if hasattr(sub_attn, 'H_sp'):
                            print(f"    Window H: {sub_attn.H_sp}")
                        if hasattr(sub_attn, 'W_sp'):
                            print(f"    Window W: {sub_attn.W_sp}")


if __name__ == "__main__":
    main()