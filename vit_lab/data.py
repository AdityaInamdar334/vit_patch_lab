"""Balanced controlled two-marker spatial reasoning dataset.

Red marker left of blue -> class 1; red marker right of blue -> class 0.
No label-dependent noise, crop artifacts, or label leaks outside arrangement.
"""
import torch
from torch.utils.data import Dataset


class SpatialMarkers(Dataset):
    def __init__(self, count=2000, seed=0, noise=0.06, shift=0, image_size=32):
        if image_size != 32:
            raise ValueError('The controlled task uses 32x32 images')
        g = torch.Generator().manual_seed(seed)
        imgs = torch.randn(count, 3, 32, 32, generator=g) * noise + 0.05
        labels = torch.arange(count) % 2
        order = torch.randperm(count, generator=g)
        labels = labels[order]
        # Vectorized construction avoids thousands of tiny tensor allocations.
        choices = torch.rand(count, 4, generator=g).argsort(dim=1)[:, :2]
        left = choices.min(dim=1).values
        right = choices.max(dim=1).values
        red_x = torch.where(labels == 1, left, right)
        blue_x = torch.where(labels == 1, right, left)
        ys = torch.randint(0, 4, (count, 2), generator=g)
        if shift:
            jx = torch.randint(-shift, shift + 1, (count,), generator=g)
            jy = torch.randint(-shift, shift + 1, (count,), generator=g)
        else:
            jx = jy = torch.zeros(count, dtype=torch.long)
        rx = (8 * red_x + 1 + jx).clamp(0, 27)
        bx = (8 * blue_x + 1 + jx).clamp(0, 27)
        ry = (8 * ys[:, 0] + 1 + jy).clamp(0, 27)
        by = (8 * ys[:, 1] + 1 + jy).clamp(0, 27)
        ix = torch.arange(count)
        for dy in range(5):
            for dx in range(5):
                imgs[ix, :, ry + dy, rx + dx] = 0
                imgs[ix, 0, ry + dy, rx + dx] = 1
                imgs[ix, :, by + dy, bx + dx] = 0
                imgs[ix, 1, by + dy, bx + dx] = 1
        self.images = imgs.clamp(0, 1)
        self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
