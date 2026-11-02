"""Train and evaluate a model with deterministic torch shuffling."""
import random
from functools import lru_cache
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .model import TinyCNN, TinyViT, count_parameters
from .data import SpatialMarkers


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def make_model(name):
    if name == 'cnn':
        return TinyCNN()
    if name == 'vit8':
        return TinyViT(patch_size=8)
    if name == 'vit4':
        return TinyViT(patch_size=4)
    if name == 'vit8_no_pos':
        return TinyViT(patch_size=8, use_position=False)
    if name == 'vit8_mean':
        return TinyViT(patch_size=8, pooling='mean')
    raise ValueError('Unknown model ' + name)


def evaluate(model, loader, device='cpu'):
    model.eval()
    ok, total, loss_sum = 0, 0, 0.0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
            ok += (logits.argmax(-1) == y).sum().item()
            total += len(y)
    return {'accuracy': ok / total, 'cross_entropy': loss_sum / total}


@lru_cache(maxsize=24)
def dataset(count, seed, noise=0.06, shift=0):
    return SpatialMarkers(count, seed, noise, shift)


def train_one(name='vit8', seed=0, steps=120, train_count=2400, batch_size=64,
              eval_count=512, learning_rate=0.002, device='cpu'):
    set_seed(seed)
    train_ds = dataset(train_count, seed=1000 + seed)
    test_ds = dataset(eval_count, seed=2000 + seed)
    noisy_ds = dataset(eval_count, seed=2000 + seed, noise=0.25)
    shifted_ds = dataset(eval_count, seed=2000 + seed, shift=2)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed + 3000), drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=128)
    noisy_loader = DataLoader(noisy_ds, batch_size=128)
    shifted_loader = DataLoader(shifted_ds, batch_size=128)
    model = make_model(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    start = time.perf_counter()
    iteration = iter(loader)
    model.train()
    for _ in range(steps):
        try:
            x, y = next(iteration)
        except StopIteration:
            iteration = iter(loader)
            x, y = next(iteration)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    elapsed = time.perf_counter() - start
    clean = evaluate(model, test_loader, device)
    noisy = evaluate(model, noisy_loader, device)
    shifted = evaluate(model, shifted_loader, device)
    return {'name':name, 'seed':seed, 'steps':steps, 'train_count':train_count,
            'parameters':count_parameters(model), 'patch_tokens':(32 // model.patch_size) ** 2 if isinstance(model, TinyViT) else None,
            'train_seconds':elapsed, 'clean_accuracy':clean['accuracy'],
            'noisy_accuracy':noisy['accuracy'], 'shifted_accuracy':shifted['accuracy'],
            'clean_ce':clean['cross_entropy'], 'device':device}
