# Vision Transformer From Scratch: Patches, Positions, and Inductive Bias

A compact, tested PyTorch companion to the original Vision Transformer paper and an accompanying technical Medium article.

**The central experiment:** How does a tiny Vision Transformer compare with a CNN when a visual decision depends on where two objects are located? What happens if positional embeddings disappear, patch size changes, or the objects move?

The project implements non-overlapping patch projection, a CLS token, learned positional embeddings, bicubic 2D position interpolation, pre-normalization Transformer blocks, a small spatial CNN baseline, and five controlled model configurations. It has an optional real CIFAR-10 training entry point, too.

**Important:** This is an independent educational implementation. It is not Google's original ViT code, pretrained model, or reproduction of its large-scale ImageNet experiments. The main benchmark uses synthetic 32-by-32 images; no claim about natural-image performance follows from that result.

## What is included

- `vit_lab/model.py`: `PatchEmbedding`, `TransformerBlock`, `TinyViT`, `TinyCNN`.
- `vit_lab/data.py`: deterministic binary classification task with two colored markers.
- `vit_lab/train.py`: training, evaluation, seeds, and variant construction.
- `vit_lab/benchmark.py`: three-seed benchmark with clean, noisy, and shifted evaluations.
- `vit_lab/cifar_train.py`: optional CIFAR-10 training with disjoint train and validation indices, and a held-out test evaluation.
- `tests/test_lab.py`: 15 unit and smoke tests.
- `results/benchmark.json`: saved three-seed results for five models.
- `results/low_data.json`: the 120-example ablation.
- `figures/`: three original figures generated directly from saved results.
- `plot_results.py`: plot generator.
- `BENCHMARK_NOTES.md`: data generation, methodology, observed results, limitations.
- `ARTICLE.md`: full 7,800+ word Medium tutorial.

## Quick start

From inside `vit_patch_lab`:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the complete reproducible benchmark:

```bash
python -m vit_lab.benchmark \
  --seeds 3 \
  --steps 150 \
  --train-count 1200 \
  --eval-count 128 \
  --output results/benchmark.json
```

Run the low-data ablation:

```bash
python -m vit_lab.benchmark \
  --seeds 3 \
  --steps 150 \
  --train-count 120 \
  --eval-count 128 \
  --variants cnn vit4 \
  --output results/low_data.json
```

Regenerate the figures:

```bash
python plot_results.py
```

No GPU, API key, pretrained checkpoint, or data download is required for the default experiments. The experiments have been run on PyTorch 2.10.0+cpu. CPU runtimes vary by host.

## What the saved experiment shows

The primary task labels an image **1 if its red marker is left of its blue marker**, and **0 otherwise**. Marker content and vertical coordinates are randomized. Samples are class-balanced.

Across three seeds with 150 steps per model:

- CNN: 100.00% clean, 99.22% noisy, 98.70% shifted accuracy.
- ViT with 4-pixel patches: 99.74% clean, 97.66% noisy, 73.70% shifted accuracy.
- ViT with 8-pixel patches: 79.17% clean, 76.56% noisy, 60.68% shifted accuracy.
- ViT with 8-pixel patches and no positions: 50.00% on all three evaluations.
- ViT with 8-pixel patches and mean pooling: 80.21% clean, 80.21% noisy, 72.14% shifted accuracy.

These results are **synthetic educational findings**. A full per-seed breakdown and standard deviations are saved in JSON and documented in `BENCHMARK_NOTES.md`. The CNN has more trainable parameters. The ViT-8 and mean-pooling runs have substantial seed variance and may be undertrained under a 150-step budget.

## Verify an architectural invariant

The tests verify that removing position embeddings makes CLS-based ViT classification invariant to patch permutations. They also verify that a non-overlapping convolutional patch projection computes the same output as `unfold` followed by a linear layer, up to floating-point tolerance.

```bash
python -m unittest tests.test_lab.VisionTransformerTests.test_no_position_permutation_invariance -v
python -m unittest tests.test_lab.VisionTransformerTests.test_patch_projection_equals_unfold_plus_linear -v
```

## Optional real-world dataset: CIFAR-10

This path requires an explicit download. It is **not** included in the published synthetic benchmark and its results have not been measured in the provided files.

```bash
python -m vit_lab.cifar_train \
  --model vit8 --download --epochs 5 \
  --max-train 5000 --batch-size 128

python -m vit_lab.cifar_train \
  --model cnn --epochs 5 \
  --max-train 5000 --batch-size 128
```

The script partitions the official 50,000-image CIFAR training split into 45,000 training-eligible and 5,000 validation images. The separate 10,000-image test split is evaluated after the best validation epoch is selected.

For a serious natural-image benchmark, increase epochs, add learning-rate scheduling, match parameter or compute budgets, and compare against pretrained checkpoints with full upstream data disclosure.

## Important limitations

- The test set contains 128 synthetic images per seed, and there are only three seeds. Statistical uncertainty remains substantial.
- The controlled task is much simpler than CIFAR-10 or ImageNet.
- Models are not matched by parameter count, FLOPs, or optimized learning rate.
- A fixed number of gradient steps is not a fixed compute budget.
- The simple CNN has a large spatial classifier and therefore more parameters than the tiny ViTs.
- The 8-pixel ViT is unstable across seeds at the selected short training budget.
- The mean-pooling ablation leaves the CLS token inside the encoder, so it isolates pooling choice imperfectly.
- No claims are made about pretrained ViT robustness, real-world image understanding, or state-of-the-art vision accuracy.

## Primary sources

1. Dosovitskiy et al. *An Image Is Worth 16x16 Words.* https://arxiv.org/abs/2010.11929
2. labml.ai annotated ViT: https://nn.labml.ai/transformers/vit/index.html
3. Google Research official models: https://github.com/google-research/vision_transformer
4. Touvron et al. *DeiT.* https://arxiv.org/abs/2012.12877
5. Steiner et al. *How to Train Your ViT.* https://arxiv.org/abs/2106.10270

## License

Original educational lab code is released under the MIT license in `LICENSE`. Original papers and third-party code retain their respective licenses. No pretrained models or dataset files are redistributed here.
