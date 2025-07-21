import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        max_size: int = 2048,  # Maximum expected input size for CUDA graph compatibility
    ):
        super().__init__()

        try:
            assert in_channels >= groups and in_channels % groups == 0
        except:  # noqa: E722
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)  # noqa: B904

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        self.max_size = max_size
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())
        
        # Pre-compute coordinate grids for CUDA graph compatibility
        self._precompute_coords()

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )
    
    def _precompute_coords(self):
        """Pre-compute coordinate grids for different sizes to support CUDA graphs."""
        # Pre-compute coordinate ranges for maximum expected size
        coords_h = torch.arange(self.max_size, dtype=torch.float32) + 0.5
        coords_w = torch.arange(self.max_size, dtype=torch.float32) + 0.5
        self.register_buffer("_coords_h", coords_h)
        self.register_buffer("_coords_w", coords_w)

    def forward(self, x):
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        
        # Use pre-computed coordinates (slice to actual size)
        coords_h = self._coords_h[:H]
        coords_w = self._coords_w[:W]

        # Create coordinate grid using pre-computed values
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        
        # Create normalizer more efficiently for CUDA graphs
        # We use the coordinate values themselves to derive W and H
        normalizer_w = coords_w[-1] + 0.5  # This equals W
        normalizer_h = coords_h[-1] + 0.5  # This equals H
        normalizer = torch.stack([normalizer_w, normalizer_h]).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output
