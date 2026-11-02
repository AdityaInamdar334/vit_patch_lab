"""Small, readable Vision Transformer and CNN control baseline.

Independent educational implementation, not Google's original checkpoint.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    """Non-overlapping flattened patches projected to D, implemented as Conv2d."""
    def __init__(self, channels=3, width=48, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(channels, width, kernel_size=patch_size, stride=patch_size)

    def forward(self, images):
        _, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError('Image height and width must be divisible by patch size')
        embedded = self.proj(images)
        grid = (embedded.shape[-2], embedded.shape[-1])
        return embedded.flatten(2).transpose(1, 2), grid


class TransformerBlock(nn.Module):
    def __init__(self, width=48, heads=3, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self, image_size=32, patch_size=8, width=48, depth=2,
                 heads=3, num_classes=2, use_position=True, pooling='cls'):
        super().__init__()
        if image_size % patch_size:
            raise ValueError('image_size must be divisible by patch_size')
        if pooling not in ('cls', 'mean'):
            raise ValueError('pooling must be cls or mean')
        self.patch_size, self.pooling, self.use_position = patch_size, pooling, use_position
        self.grid_size = image_size // patch_size
        self.patch_embed = PatchEmbedding(3, width, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.zeros(1, self.grid_size ** 2 + 1, width))
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def positional_at(self, grid):
        """Bicubic 2D interpolation of patch positions; never interpolate CLS."""
        if grid == (self.grid_size, self.grid_size):
            return self.position
        cls = self.position[:, :1]
        patches = self.position[:, 1:].transpose(1, 2).reshape(1, -1, self.grid_size, self.grid_size)
        resized = F.interpolate(patches.float(), size=grid, mode='bicubic', align_corners=False)
        resized = resized.to(dtype=self.position.dtype).flatten(2).transpose(1, 2)
        return torch.cat([cls, resized], dim=1)

    def forward(self, images):
        x, grid = self.patch_embed(images)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        if self.use_position:
            x = x + self.positional_at(grid)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pooled = x[:, 0] if self.pooling == 'cls' else x[:, 1:].mean(dim=1)
        return self.head(pooled)


class TinyCNN(nn.Module):
    """Locally connected baseline, with a spatial classifier to retain location."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48 * 8 * 8, 48), nn.ReLU(), nn.Linear(48, num_classes))

    def forward(self, x):
        if x.shape[-2:] != (32, 32):
            raise ValueError('TinyCNN is defined for 32 by 32 inputs')
        return self.head(self.features(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
