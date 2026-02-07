"""
Vectorized operations for DAT architecture to eliminate CPU bottlenecks.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


@torch.jit.script
def create_region_masks_vectorized(
    H: int, 
    W: int, 
    split_size_h: int, 
    split_size_w: int,
    shift_size_h: int,
    shift_size_w: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create region masks for shift windows using vectorized operations.
    
    Args:
        H: Height of the image
        W: Width of the image
        split_size_h: Height of the window
        split_size_w: Width of the window
        shift_size_h: Vertical shift size
        shift_size_w: Horizontal shift size
        device: Device to create tensors on
    
    Returns:
        Tuple of (img_mask_0, img_mask_1) with shape (1, H, W, 1)
    """
    # Create coordinate grids
    h_coords = torch.arange(H, device=device).view(-1, 1).expand(H, W)
    w_coords = torch.arange(W, device=device).view(1, -1).expand(H, W)
    
    # For mask_0: split by (split_size_h, split_size_w)
    # Calculate region indices based on position
    h_regions_0 = torch.zeros(H, device=device, dtype=torch.long)
    w_regions_0 = torch.zeros(W, device=device, dtype=torch.long)
    
    # Set region indices for height
    h_regions_0[:H - split_size_h] = 0
    h_regions_0[H - split_size_h:H - shift_size_h] = 1
    h_regions_0[H - shift_size_h:] = 2
    
    # Set region indices for width
    w_regions_0[:W - split_size_w] = 0
    w_regions_0[W - split_size_w:W - shift_size_w] = 1
    w_regions_0[W - shift_size_w:] = 2
    
    # Compute mask_0 using broadcasting
    img_mask_0 = h_regions_0.view(-1, 1) * 3 + w_regions_0.view(1, -1)
    img_mask_0 = img_mask_0.view(1, H, W, 1).float()
    
    # For mask_1: split by (split_size_w, split_size_h) - note the swap
    h_regions_1 = torch.zeros(H, device=device, dtype=torch.long)
    w_regions_1 = torch.zeros(W, device=device, dtype=torch.long)
    
    # Set region indices for height (using split_size_w)
    h_regions_1[:H - split_size_w] = 0
    h_regions_1[H - split_size_w:H - shift_size_w] = 1
    h_regions_1[H - shift_size_w:] = 2
    
    # Set region indices for width (using split_size_h)
    w_regions_1[:W - split_size_h] = 0
    w_regions_1[W - split_size_h:W - shift_size_h] = 1
    w_regions_1[W - shift_size_h:] = 2
    
    # Compute mask_1
    img_mask_1 = h_regions_1.view(-1, 1) * 3 + w_regions_1.view(1, -1)
    img_mask_1 = img_mask_1.view(1, H, W, 1).float()
    
    return img_mask_0, img_mask_1


@torch.jit.script
def calculate_mask_vectorized(
    H: int,
    W: int,
    split_size: Tuple[int, int],
    shift_size: Tuple[int, int],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate attention masks for shift windows using fully vectorized operations.
    
    This replaces the nested loops in the original implementation with
    efficient tensor operations.
    
    Args:
        H: Height of the image
        W: Width of the image
        split_size: (height, width) of the window
        shift_size: (vertical, horizontal) shift size
        device: Device to create tensors on
    
    Returns:
        Tuple of attention masks (attn_mask_0, attn_mask_1)
    """
    # Create region masks
    img_mask_0, img_mask_1 = create_region_masks_vectorized(
        H, W, split_size[0], split_size[1], shift_size[0], shift_size[1], device
    )
    
    # Process mask_0
    # Reshape for window partitioning
    img_mask_0 = img_mask_0.view(
        1,
        H // split_size[0],
        split_size[0],
        W // split_size[1],
        split_size[1],
        1,
    )
    img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous()
    img_mask_0 = img_mask_0.view(-1, split_size[0], split_size[1], 1)
    
    # Create attention mask
    mask_windows_0 = img_mask_0.view(-1, split_size[0] * split_size[1])
    attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
    attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, -100.0)
    attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 == 0, 0.0)
    
    # Process mask_1
    # Note: split dimensions are swapped for mask_1
    img_mask_1 = img_mask_1.view(
        1,
        H // split_size[1],
        split_size[1],
        W // split_size[0],
        split_size[0],
        1,
    )
    img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous()
    img_mask_1 = img_mask_1.view(-1, split_size[1], split_size[0], 1)
    
    # Create attention mask
    mask_windows_1 = img_mask_1.view(-1, split_size[1] * split_size[0])
    attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
    attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, -100.0)
    attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 == 0, 0.0)
    
    return attn_mask_0, attn_mask_1


@torch.jit.script
def window_partition_vectorized(
    x: torch.Tensor,
    window_size: int
) -> torch.Tensor:
    """
    Partition image into windows using efficient tensor operations.
    
    Args:
        x: Input tensor (B, H, W, C)
        window_size: Size of the window
    
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


@torch.jit.script
def window_unpartition_vectorized(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int
) -> torch.Tensor:
    """
    Reverse window partition using efficient tensor operations.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Size of the window
        H: Height of the image
        W: Width of the image
    
    Returns:
        x: (B, H, W, C)
    """
    # JIT-friendly batch size calculation
    num_windows = (H // window_size) * (W // window_size)
    B = windows.shape[0] // num_windows
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


@torch.jit.script
def img2windows(img: torch.Tensor, H_sp: int, W_sp: int) -> torch.Tensor:
    """
    Convert image to windows - vectorized version.
    
    Args:
        img: (B, C, H, W)
        H_sp: Height of window
        W_sp: Width of window
    
    Returns:
        (B*num_windows, H_sp*W_sp, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous()
    img_win = img_perm.view(-1, H_sp * W_sp, C)
    return img_win


@torch.jit.script
def windows2img(img_splits_hw: torch.Tensor, H_sp: int, W_sp: int, H: int, W: int) -> torch.Tensor:
    """
    Convert windows back to image - vectorized version.
    
    Args:
        img_splits_hw: (B*num_windows, H_sp*W_sp, C)
        H_sp: Height of window
        W_sp: Width of window
        H: Height of image
        W: Width of image
    
    Returns:
        img: (B, H, W, C)
    """
    # JIT-friendly batch size calculation
    num_windows = (H // H_sp) * (W // W_sp)
    B = img_splits_hw.shape[0] // num_windows
    
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


@torch.jit.script
def generate_position_bias(H_sp: int, W_sp: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate position bias tensors for relative position encoding.
    
    Args:
        H_sp: Height of spatial window
        W_sp: Width of spatial window
        device: Device to create tensors on
    
    Returns:
        Tuple of (rpe_biases, relative_position_index)
        - rpe_biases: (2*H_sp-1 * 2*W_sp-1, 2)
        - relative_position_index: (H_sp*W_sp, H_sp*W_sp)
    """
    # Generate mother-set
    position_bias_h = torch.arange(1 - H_sp, H_sp, device=device)
    position_bias_w = torch.arange(1 - W_sp, W_sp, device=device)
    biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij'))
    biases = biases.flatten(1).transpose(0, 1).contiguous().float()
    
    # Get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(H_sp, device=device)
    coords_w = torch.arange(W_sp, device=device)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += H_sp - 1
    relative_coords[:, :, 1] += W_sp - 1
    relative_coords[:, :, 0] *= 2 * W_sp - 1
    relative_position_index = relative_coords.sum(-1)
    
    return biases, relative_position_index