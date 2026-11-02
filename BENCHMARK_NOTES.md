# Benchmark Methodology and Interpretation

## Scope

This lab tests architectural properties of a small ViT and small CNN on a controlled **synthetic spatial-relation task**. The benchmark is not an attempt to reproduce the original Google Vision Transformer ImageNet/CIFAR results and does not include pretrained weights.

## Dataset

Every 32-by-32 RGB image contains a red and a blue 5-pixel-square marker. The two markers occupy different x positions on an 8-pixel grid and can appear at different y positions. Class 1 means red is left of blue, class 0 means red is right of blue. The dataset is perfectly class-balanced for even sample counts. Background noise is generated independently of the label.

Training data has noise standard deviation 0.06 and 1,200 examples per seed. Independent held-out test sets have 128 examples per seed. Strong-noise evaluation uses noise standard deviation 0.25. Shifted evaluation moves marker positions by an integer offset between -2 and +2 pixels along both axes, with boundary clipping. It retains the same deterministic label rule. The same 128 examples and perturbation seeds are used for every model within a run.

The low-data experiment uses 120 training images but the same number of gradient steps as the main benchmark. Thus its examples are revisited more frequently. This is a data-access comparison, not an epoch-matched one.

## Model configurations

All ViTs: hidden width 48, two blocks, three attention heads, MLP ratio 2.0, learned CLS token, and two learned position types unless ablated. They use non-overlapping convolutional patch projection and pre-LN self-attention with GELU. The normal ViT8 has 48,242 trainable parameters. ViT4 has 43,634. CNN has 166,562 parameters and a flattened spatial head. These models are not parameter matched.

## Training

Three seeds 0, 1, 2. Each run uses 150 AdamW steps, learning rate 0.002, weight decay 0.01, batch size 64, and max gradient norm 1.0. No hyperparameter sweep, pretrained weights, or early stopping. CPU runtime is measured with Python's `perf_counter`, including training updates only, not dataset generation and evaluation. PyTorch 2.10.0+cpu. Run-to-run timings may vary.

## Three-seed summary

All values below are mean percentages across seeds. Full run data and sample standard deviation are in `results/benchmark.json`.

- CNN: clean 100.00%, noise 99.22%, shift 98.70%.
- ViT4: clean 99.74%, noise 97.66%, shift 73.70%.
- ViT8: clean 79.17%, noise 76.56%, shift 60.68%. Clean sample std 25.84 percentage points.
- ViT8 without positions: clean 50.00%, noise 50.00%, shift 50.00%.
- ViT8 mean pooling: clean 80.21%, noise 80.21%, shift 72.14%. Clean sample std 26.58 percentage points.

Mean training wall times for the 150-step runs in this environment: CNN 2.21 seconds; ViT8 1.81 seconds; ViT4 3.96 seconds; ViT8 no positions 1.55 seconds; ViT8 mean pooling 1.62 seconds. Hardware-specific.

## Low-data ablation

With 120 training samples and 150 steps, mean clean CNN accuracy was 100.00% and mean ViT4 accuracy was 98.70%. Mean shifted CNN accuracy was 99.74%; shifted ViT4 accuracy was 61.20%. The task is easy in-distribution, so this does not demonstrate a large sample-efficiency difference in clean accuracy.

## Important invariants

1. Patch projection as `Conv2d(kernel=stride=patch_size)` is numerically equivalent to `unfold + Linear` using identical weights.
2. Without position embeddings, a CLS-based ViT gives the same prediction when whole patches are permuted, within tolerance. The synthetic label depends on the permutation, so that model cannot recover the label in principle on symmetrically constructed inputs.
3. Interpolated positions at a new patch grid have the correct number of tokens and preserve the CLS position exactly.
4. Training, evaluation, and robustness data use independent deterministic seeds and class-balanced labels.

## Interpretation limits

The experiment demonstrates **an inductive-bias distinction**, not a model ranking. The CNN is larger and task-specific. The ViTs are extremely small and not carefully tuned. Three runs do not establish narrow uncertainty bounds, and some ViT settings are unstable at 150 steps. Further work should match parameter budgets, FLOPs, augmentations, and learning-rate searches and should evaluate actual natural images.
