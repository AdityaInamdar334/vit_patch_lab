# Google Replaced the CNN With Image Patches and a Transformer. The Hard Part Was Teaching It What an Image Is.

*An engineer's guide to Vision Transformers, from pixels and positional embeddings to a complete PyTorch implementation, reproducible CNN comparisons, patch-size ablations, and tests that expose what changes when objects move.*

An image classifier is given a picture containing two colored squares. One is red. One is blue. Its job is almost embarrassingly simple: answer whether the red square is to the left of the blue square.

A small convolutional neural network learns the task quickly. A Vision Transformer with sufficiently small patches learns it too. But remove one seemingly unimportant component from that Transformer, its positional embeddings, and the model can no longer distinguish left from right. It receives all the visual information. It just does not know how the pieces are arranged.

In our controlled PyTorch experiment, that position-free model stays at exactly **50% accuracy** across three seeds on a balanced binary task. A tiny ViT using 4-by-4 image patches reaches **99.74%** on the same held-out synthetic task. Then we shift the squares slightly within the image and the ViT drops to **73.70%**. The CNN, trained on the same examples, retains **98.70%**.

There is no paradox. These models have different assumptions about visual structure. And that distinction is the most useful starting point for understanding Vision Transformers.

In 2020, Alexey Dosovitskiy and colleagues at Google Research published *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale* [1]. They showed that a largely standard Transformer encoder could classify images by processing fixed-size image patches as tokens. With sufficiently large-scale pretraining, the approach became competitive with or better than leading convolutional architectures.

The remarkable part was not that attention could process images. Researchers had already used attention in vision. The remarkable part was how little additional image-specific machinery the authors needed, provided they supplied enough data and compute.

This tutorial will build a Vision Transformer from its fundamental tensor operations. We will inspect the annotated labml.ai implementation [2], rebuild the architecture using modern PyTorch, prove that convolutional patch embedding is equivalent to patch flattening plus a linear projection, train multiple baselines, reproduce a positional-invariance counterexample, evaluate robustness, and show how to perform a real CIFAR-10 experiment without accidentally tuning on its test set.

**Everything measured in the companion lab is our independent educational experiment. It is not a replication of Google's pretraining, ImageNet, or CIFAR-10 results.**

## Contents

1. The question the original ViT paper actually answered
2. Turn an image into tokens
3. Why the patch projection uses a convolution
4. The classification token and positional embeddings
5. The Transformer encoder, one operation at a time
6. Compute: what patch size really costs
7. A complete ViT implementation in PyTorch
8. Interpolating positional embeddings correctly
9. Reading the labml.ai implementation critically
10. Constructing an experiment that reveals architectural biases
11. Training the ViT and CNN baselines
12. The main experimental results
13. Ablation one: remove positional embeddings
14. Ablation two: change patch size
15. Ablation three: replace CLS pooling
16. Low-data learning and the right interpretation
17. Distribution shift and robustness
18. Statistical design and benchmark honesty
19. Training on real CIFAR-10
20. Why pretraining changes the outcome
21. Why attention maps are not necessarily explanations
22. Common implementation failures
23. Research extensions worth building
24. Final lessons
25. References

---

## 1. The question the original ViT paper actually answered

Before ViT, convolutional networks dominated image classification for several good reasons.

A convolution applies a small filter across an image. The filter is reused at different spatial locations. That reuse encodes a powerful assumption: a useful feature detected near the top-left corner might also matter near the bottom-right. Local layers build edges, textures, parts, and larger objects. Translation equivariance and local connectivity are baked into the architecture.

The standard Transformer has a very different starting point. It operates on a sequence of vectors. Self-attention lets one token use information from every other token. The architecture does not inherently require neighboring pixels to be processed first or require the same spatial filter to be reused at every image location.

The original paper posed a direct question: can we remove most of the specialized convolutional feature extractor and let a Transformer learn the necessary visual relationships from data?

Their answer was qualified but compelling. Models trained on the smaller image datasets originally available to them often lagged comparably sized ResNets unless substantial regularization was used. Once pretraining expanded to ImageNet-21k, with roughly 14 million images, or the larger JFT-300M dataset, ViT became highly effective after transfer to downstream tasks [1].

The paper reported **88.55% ImageNet top-1 accuracy** for its strongest JFT-pretrained ViT-H/14 configuration. It also reported **94.55% on CIFAR-100** and **77.63% on the 19-task VTAB suite** for that configuration [1]. Those were measurements of large pretrained systems. None should be interpreted as the expected accuracy of a toy ViT trained on a few thousand CIFAR images.

That historical detail matters. ViT did not demonstrate that convolutions were obsolete. It demonstrated that image-specific inductive biases could be traded, to a meaningful degree, for scale, general-purpose attention, and transferable representations.

## 2. Turn an image into tokens

Imagine a 224-by-224 RGB image. It contains three color channels and more than 150,000 scalar pixel values. Treating every pixel as an individual token would produce 50,176 tokens, creating a huge attention matrix.

ViT reduces the sequence length by grouping neighboring pixels into non-overlapping patches.

With **16-by-16 patches**, a 224-by-224 image becomes a grid of 14 patches by 14 patches. That is **196 patch tokens**. Add the classification token and the Transformer processes **197 tokens**.

Each raw patch contains 16 times 16 times 3, or **768** values. A learned linear map converts those 768 numbers into a fixed-size hidden vector, such as 768 dimensions in ViT-B/16.

The patch projection does not initially recognize objects. It translates a local block of pixels into a vector that the rest of the network can manipulate. Later layers determine which combinations are useful.

For a 32-by-32 image, the choice of patch size matters even more:

- A 16-pixel patch produces only 4 image tokens, plus CLS.
- An 8-pixel patch produces 16 image tokens, plus CLS.
- A 4-pixel patch produces 64 image tokens, plus CLS.

This is why blindly copying the original ViT-B/16 configuration into a small CIFAR-10 tutorial often produces a disappointing result. A coarse patch may throw away useful spatial granularity before attention sees the image.

Our lab uses 8-pixel and 4-pixel patches, not 16-pixel patches, because the controlled input images are only 32 pixels wide.

## 3. Why the patch projection uses a convolution

The original mathematical description says: flatten each patch, then multiply it by a learned linear projection.

That can be implemented directly. But a convolutional layer with **kernel size equal to patch size** and **stride equal to patch size** does exactly the same operation. There is no overlapping receptive field and no intermediate convolutional feature hierarchy.

```python
import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, channels=3, width=48, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            channels,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images):
        _, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image dimensions must be divisible by patch size")

        x = self.proj(images)
        grid = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, grid
```

For a batch of 32 RGB images of size 32 by 32 and patch size 8, the convolution produces a tensor of shape `32 x 48 x 4 x 4`. Flattening the spatial dimensions and transposing produces `32 x 16 x 48`, meaning 16 patch tokens of width 48 for each image.

The code contains a convolution, but that does not make the architecture a conventional CNN. The convolution is a convenient implementation of the patch tokenizer. It does not apply a sliding learned filter across overlapping neighborhoods or form a deep multiscale hierarchy.

We can verify the equivalence numerically rather than accept it on faith:

```python
import torch.nn.functional as F

x = torch.randn(2, 3, 32, 32)
patch_embed = PatchEmbedding(channels=3, width=24, patch_size=8)

actual, _ = patch_embed(x)
patches = F.unfold(x, kernel_size=8, stride=8).transpose(1, 2)
expected = F.linear(
    patches,
    patch_embed.proj.weight.flatten(1),
    patch_embed.proj.bias,
)

print(torch.allclose(actual, expected, atol=1e-5))
# True
```

The crucial insight is that the apparent convolutional operation is an exact rearrangement of a linear map over image patches. One of the companion lab's unit tests checks this identity.

## 4. The classification token and positional embeddings

Once patches are converted into vectors, two additional components appear.

First is the learnable **CLS token**. Think of it as a dedicated workspace into which information from the image can flow. It is prepended to the patch sequence. During self-attention, it can interact with every image token. At the final encoder layer, the representation of this token is passed to the classification head.

Second is the set of **positional embeddings**. These are learned vectors associated with token locations. They tell the model which patch came from which region of the image.

Without position information, a standard Transformer applied to patch tokens is permutation equivariant. When CLS is used for classification, permuting the image patches, while leaving the CLS token fixed, does not change the classification result in exact arithmetic. Self-attention sees the same set of patch contents, only in a different order. The CLS representation cannot recover an ordering it was never given.

That is fine if the task is to detect whether a tomato appears somewhere in an image. It is disastrous if the task is to determine whether a tomato is left of a book.

The original ViT learned a one-dimensional vector for each position, even though those positions originate from a two-dimensional grid [1]. The team reported that more elaborate two-dimensional embeddings did not offer a clear performance advantage in its comparisons.

Learned positional embeddings are parameters, not permanent geometric laws. They carry whatever positional relationships training manages to teach them.

## 5. The Transformer encoder, one operation at a time

After tokenization, CLS insertion, and positional addition, the image passes through repeated Transformer blocks.

Each block has two principal components.

**Multihead self-attention** computes how token representations should incorporate information from other tokens. A patch containing a wheel can interact with patches containing a chassis or road. The attention weights depend on the current token features, not merely their absolute positions.

**The feed-forward MLP** independently processes each token's feature vector after attention. It usually expands the hidden dimension, applies a nonlinear activation such as GELU, and projects back to the original width.

Layer normalization is applied before each component. Residual connections preserve the incoming representation and add the new result. This is often called a pre-normalization Transformer block.

```python
class TransformerBlock(nn.Module):
    def __init__(self, width=48, heads=3, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))
```

The use of `batch_first=True` means tensors have shape `batch x tokens x hidden_width`, which is more convenient than the sequence-first layout of the older labml.ai example.

Unlike a text-generation Transformer, image classification ViT uses **bidirectional self-attention**. No causal mask is required. The image is available all at once, so every image patch is allowed to attend to every other image patch.

This difference becomes important if you adapt the implementation for autoregressive image generation. A classifier's unmasked attention cannot simply be copied into a next-token generator without reconsidering information leakage.

## 6. Compute: what patch size really costs

Smaller patches produce more tokens. More tokens produce a larger attention matrix.

The memory and computation associated with dense attention scores scale approximately with the square of sequence length. But the total cost of a Transformer also contains token-wise linear projections and MLP operations. Attention alone does not explain every measured millisecond.

Consider our 32-by-32 experiment. With 8-pixel patches, CLS included, the model has 17 tokens. With 4-pixel patches, it has 65 tokens.

Ignoring CLS for a quick comparison, 64 squared is 16 times 16 squared. The 4-pixel-patch model thus has **16 times as many patch-to-patch attention-score positions** as the 8-pixel-patch model. That does not mean the entire model is exactly 16 times slower. The token projections, feed-forward layers, PyTorch kernel implementations, and CPU overhead all matter.

The spatial resolution of the representation and the price of global attention are in tension. Smaller patches preserve fine detail but increase the cost of global interactions. Larger patches are cheaper but can make small objects and fine spatial structure harder to separate.

This is also why later vision architectures explored hierarchical processing, windowed attention, adaptive patch selection, and alternative tokenization schemes. ViT established a simple reference point. It did not settle the optimal tokenization of images.

## 7. A complete ViT implementation in PyTorch

The runnable project includes `vit_lab/model.py`. Its implementation is intentionally small, with a 48-dimensional hidden state, two Transformer blocks, and three attention heads. This is a teaching model, not ViT-B/16, ViT-L/16, or a pretrained Google checkpoint.

The core model is:

```python
class TinyViT(nn.Module):
    def __init__(self, image_size=32, patch_size=8, width=48,
                 depth=2, heads=3, num_classes=2,
                 use_position=True, pooling="cls"):
        super().__init__()
        self.patch_size = patch_size
        self.pooling = pooling
        self.use_position = use_position
        self.grid_size = image_size // patch_size
        self.patch_embed = PatchEmbedding(3, width, patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(
            torch.zeros(1, self.grid_size ** 2 + 1, width)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(width, heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, images):
        x, grid = self.patch_embed(images)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        if self.use_position:
            x = x + self.positional_at(grid)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        pooled = x[:, 0] if self.pooling == "cls" else x[:, 1:].mean(dim=1)
        return self.head(pooled)
```

The full file also validates patch divisibility and implements `positional_at`, the positional interpolation method introduced in the next section. Running the excerpt above by itself requires that method from the repository.

A few architectural decisions deserve attention.

**CLS and position embeddings are independent learnable parameters.** Neither is derived from raw pixel values. Both receive gradients during training.

**The attention blocks are independent modules.** We create each separately in a `ModuleList`. Reusing exactly the same block object multiple times would accidentally share parameters across depth and change the architecture.

**The classification head produces raw logits.** We pass them directly to `torch.nn.functional.cross_entropy`, which includes log-softmax internally. Adding a softmax before cross-entropy would be unnecessary and could damage numerical behavior.

**The number of output classes is configurable.** The synthetic task is binary. CIFAR-10 requires ten logits. This is handled by constructing the same architecture with `num_classes=10`.

## 8. Interpolating positional embeddings correctly

There is an easy trap when attempting higher-resolution fine-tuning. A model pretrained on 224-by-224 images with 16-pixel patches has a 14-by-14 patch grid. If you fine-tune at 384-by-384 resolution, the patch grid becomes 24 by 24. The original position parameter has 196 patch positions. The new grid needs 576.

You cannot simply slice the first 576 vectors from a parameter containing only 196. Nor should you blindly interpolate the CLS position as if it were a pixel on the image grid.

The original ViT paper uses **two-dimensional interpolation** for patch positions when increasing the fine-tuning resolution [1]. The interpolation respects the original rectangular image layout.

Our implementation follows that idea:

```python
import torch.nn.functional as F

def positional_at(self, grid):
    if grid == (self.grid_size, self.grid_size):
        return self.position

    cls = self.position[:, :1]
    patches = self.position[:, 1:]
    patches = patches.transpose(1, 2).reshape(
        1, -1, self.grid_size, self.grid_size
    )
    resized = F.interpolate(
        patches.float(),
        size=grid,
        mode="bicubic",
        align_corners=False,
    )
    resized = resized.to(self.position.dtype)
    resized = resized.flatten(2).transpose(1, 2)
    return torch.cat([cls, resized], dim=1)
```

This method is included inside `TinyViT` in the project. Its unit test verifies that changing a 4-by-4 patch grid to 6 by 6 creates 37 total token positions, and that the CLS position remains exactly unchanged.

Interpolation makes a model *compatible* with different resolutions. It does not automatically guarantee accurate predictions at those resolutions. The model has still learned its visual representation under particular training conditions.

## 9. Reading the labml.ai implementation critically

The annotated labml.ai code you shared is useful because every component is visible: `PatchEmbeddings`, `LearnedPositionalEmbeddings`, `ClassificationHead`, and `VisionTransformer` [2]. It shows the mechanism without hiding it behind a high-level model factory.

There are several details worth recognizing when moving from that tutorial to a production-quality implementation.

First, the reference code uses `nn.Conv2d` for patch embedding. As we proved earlier, this is mathematically equivalent to flattening each non-overlapping patch and projecting it linearly.

Second, the reference code uses a **sequence-first** tensor layout: `tokens x batch x hidden`. Our implementation uses a batch-first layout because the PyTorch `MultiheadAttention` API supports it naturally and it reduces shape-conversion mistakes.

Third, the reference adds learned positions to the full token sequence, including CLS. That is important. If you prepend CLS but only create positions for image patches, you will introduce a token-count mismatch or shift every patch's position index by one.

Fourth, the supplied example is a small CIFAR-10 experiment. The original labml.ai authors explicitly warn that training ViT on a small dataset from scratch is not the setting in which the original paper's strongest results were obtained. We should preserve that warning, not present the labml example as a failed reproduction of Google's pretrained models.

Finally, the original article discusses interpolation for larger images, but the basic class shown in the excerpt slices positional embeddings rather than performing interpolation. Our implementation adds an explicit tested interpolation path, which is necessary for a meaningful tutorial on higher-resolution fine-tuning.

---
## 10. Constructing an experiment that reveals architectural biases

Benchmarking architecture is difficult because many things change at once: parameter count, augmentation, optimizer, training data, compute budget, tokenization, and pretraining. We need a task small enough to understand completely before asking whether a particular architecture is better at it.

Our first experiment uses a deliberately synthetic image classification problem.

Each image is 32 by 32 pixels. It contains a red marker and a blue marker, each occupying a 5-by-5 square. The markers are placed in different horizontal cells of an 8-pixel grid. Their vertical positions vary independently. Background pixels contain low-amplitude random noise.

The label is simple: class 1 if red is to the left of blue, otherwise class 0.

A crucial property of the construction is that the two markers occupy **different 8-by-8 patch columns**. As a result, knowing the content of each 8-by-8 patch without knowing its global position cannot determine which marker is leftmost.

We generate the two labels in equal proportions, randomize the examples, and create evaluation images using independent seeds. The clean benchmark has 1,200 training images and 128 test images per seed. We use a fixed budget of 150 gradient steps, batch size 64, and three training seeds. The evaluations also include higher-amplitude pixel noise and markers shifted by as much as two pixels along each spatial dimension.

The synthetic task is useful because the answer is perfectly defined and we can systematically vary the conditions that might break the model.

It is not a natural-image benchmark. Its shapes, colors, noise process, and label rule are simple. No real-world image understanding can be inferred from high scores on this task alone.

The implementation is in `vit_lab/data.py`. A simplified version of the construction looks like this:

```python
# Conceptual outline. The repository vectorizes this operation.
image = small_random_background()
red_x, blue_x = choose_distinct_patch_columns()
red_y, blue_y = choose_rows_independently()

place_red_marker(image, red_x, red_y)
place_blue_marker(image, blue_x, blue_y)

label = int(red_x < blue_x)
```

The actual repository includes a deterministic `SpatialMarkers` dataset class, and tests check reproducibility, class balance, and the preservation of labels under shifts.

## 11. Training the ViT and CNN baselines

We compare five configurations.

The first is a small CNN with three convolutional layers and a spatially aware classification head. It contains **166,562 trainable parameters**. Its final spatial head retains the ability to use location information, so we are not unfairly asking a translation-invariant global pooling classifier to solve a task defined by position.

The second is `vit8`, a two-block Vision Transformer with 8-pixel patches and learned positional embeddings. It contains **48,242 parameters** and processes 16 patch tokens plus CLS.

The third is `vit4`, the same hidden width and number of blocks with 4-pixel patches. It contains **43,634 parameters**, and processes 64 patch tokens plus CLS.

The fourth is `vit8_no_pos`, identical to the second configuration except that positional embeddings are not added to the sequence.

The fifth is `vit8_mean`, which replaces CLS-based classification with the average of the final patch-token features. The architecture still contains a CLS token, so this is a pooling comparison rather than a perfectly matched no-CLS redesign.

All models use the same training set for a given seed. The training code uses AdamW, a learning rate of 0.002, weight decay of 0.01, gradient clipping at 1.0, and cross-entropy loss. Each run receives 150 optimizer updates. We fix the training and test generation seeds and record per-run accuracy rather than reporting only a favorable example.

Train the full sweep:

```bash
python -m vit_lab.benchmark \
  --seeds 3 \
  --steps 150 \
  --train-count 1200 \
  --eval-count 128 \
  --output results/benchmark.json
```

The script writes every run and the summary statistics to JSON. There is no pretrained checkpoint, API key, or hidden data source.

The corresponding training step is intentionally ordinary:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = torch.nn.functional.cross_entropy(logits, labels)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

There is no special ViT trick in the optimizer loop. What matters is how the architecture represents the input and how much data the optimizer has available to discover useful patterns.

## 12. The main experimental results

The first result is unsurprising, but it is important.

**The CNN reaches 100.0% average clean accuracy** across the three seeds. The architecture has local feature extraction and a location-aware classifier. For this particular task, that is an effective combination.

The **4-pixel-patch ViT reaches 99.74% average clean accuracy**, with a sample standard deviation of roughly 0.45 percentage points. It successfully learns the relative-position rule with enough spatial granularity.

The **8-pixel-patch ViT reaches 79.17% average clean accuracy**, with a much larger sample standard deviation of **25.84 percentage points**. One seed reached approximately 50%, while another reached almost 100%. The model can solve the task, but under a short, fixed training budget its optimization is not consistently successful.

The **ViT without positional information remains at 50.0%** across all three seeds.

The **8-pixel-patch mean-pooling variant averages 80.21%**, also with high seed-to-seed variation. The pooled architecture can learn the task, but the initial optimization budget does not reliably find the solution.

These results are not an architectural leaderboard. The parameter counts are unequal, and a fixed step budget does not imply equal floating-point compute. More importantly, a model that has not converged after 150 updates has not demonstrated that it lacks the capacity to solve the task.

This makes the experiment more useful than a simplistic claim that one model is better. It shows which representations have access to the necessary information, which models learn it quickly under the selected budget, and where more careful training is needed.

The companion figure `figures/01_clean_noise_translation.png` visualizes the clean, noisy, and shifted accuracies side by side using the saved benchmark rather than manually entered chart values.

## 13. Ablation one: remove positional embeddings

This is the cleanest conceptual experiment in the whole tutorial.

Imagine two images. Both contain exactly one red patch and exactly one blue patch. In the first image, the red patch is to the left. In the second, we exchange the spatial positions of those two patches while retaining the pixels inside each patch.

Without position embeddings, the token sequence contains the same multiset of patch embeddings. The Transformer encoder is permutation equivariant over patch tokens, and the CLS output is invariant to that permutation.

It follows that, without other sources of spatial information, a CLS-based image classifier cannot distinguish the images. This is a structural limitation, not a training bug.

The companion tests verify this with an actual random permutation of the patch grid:

```python
import torch
import torch.nn.functional as F

model = TinyViT(
    width=24, heads=3, depth=1, use_position=False
).eval()

image = torch.randn(1, 3, 32, 32)
patches = F.unfold(image, kernel_size=8, stride=8)
perm = torch.randperm(16)
shuffled = F.fold(
    patches[:, :, perm],
    output_size=(32, 32),
    kernel_size=8,
    stride=8,
)

with torch.no_grad():
    first = model(image)
    second = model(shuffled)

print(torch.allclose(first, second, atol=1e-5))
# True
```

This demonstration says something deeper than the benchmark's 50% accuracy. Even with perfect optimization and unlimited training examples, a permutation-invariant representation cannot solve a label function that distinguishes two permutations of otherwise identical patches.

It also explains why positional embeddings are not merely decorative coordinates. They resolve an information symmetry that would otherwise make certain visual tasks impossible.

There is one caveat. A real implementation can reveal position accidentally through asymmetric padding, cropping artifacts, overlapping patches, border-specific features, or a different tokenization process. Our controlled data and non-overlapping patch construction are designed to avoid those shortcuts.

## 14. Ablation two: change patch size

With positional embeddings enabled, 4-pixel patches learned more consistently than 8-pixel patches under the selected training budget.

The intuitive explanation is that smaller patches preserve more spatial detail. A 5-pixel colored marker may straddle a patch boundary when the image is translated. With coarse patches, the embedding of a single token can change dramatically as the marker moves within it. Finer patches often give the network more explicit local structure to work with.

But that explanation remains a hypothesis about the measured difference. It is not proven simply because one model scored higher. Smaller patches also change the sequence length, the number of patch-embedding parameters, numerical behavior, computational cost, and optimization dynamics.

In our saved runs, the average CPU training time for 150 steps was approximately **1.81 seconds for the 8-pixel ViT** and **3.96 seconds for the 4-pixel ViT**. These timings are specific to the sandbox CPU, model dimensions, batch size, and PyTorch version. They are not hardware-independent speed benchmarks.

A small patch is therefore not a universal improvement. It gives the model a finer visual vocabulary and asks the attention mechanism to process a longer sentence.

The right patch size depends on input resolution, object scale, training data, latency requirements, and the task's tolerance for losing local detail.

## 15. Ablation three: replace CLS pooling

Why should one special token summarize an entire image? Could we simply average the final patch representations?

That is exactly what the `vit8_mean` variant tests.

Mean pooling can work because every patch token has already participated in multiple rounds of attention. After attention, an individual patch representation contains some global context. Averaging those contextualized vectors is not the same as averaging raw pixels or patch embeddings.

In our short-run benchmark, the mean-pooling variant reaches **80.21% average clean accuracy**, compared with **79.17% for CLS pooling**. Both have enormous variation across seeds, so this numerical difference is not evidence of a meaningful superiority.

In the original ViT paper, the authors also studied classification token versus global average pooling and reported that head design could influence outcomes, including under particular learning rate choices [1]. It is not a universal law that CLS is always better.

A stronger comparison would remove the unused CLS token from the mean-pooling architecture, tune learning rates on a validation set, equalize compute budgets, run more seeds, and compare confidence intervals.

The current experiment is a useful ablation, not a definitive judgment on classification heads.

## 16. Low-data learning and the right interpretation

The original ViT paper drew attention to the relationship between dataset size and inductive bias. A natural follow-up is to reduce the amount of data in our synthetic experiment.

We train the same CNN and 4-pixel ViT with **120 training examples instead of 1,200**, preserving the 150-gradient-step budget. This is a data-availability experiment, not an epoch-matched comparison. With only 120 images, the optimizer revisits the available examples many more times.

The CNN still reaches **100.0% average clean accuracy**. The 4-pixel ViT reaches **98.70%** across three seeds, compared with **99.74%** when trained with 1,200 examples.

Neither model faces a severe sample-complexity challenge here. The rule is simple, colors are vivid, and the object count is fixed. Both models can learn from small numbers of examples.

This is a valuable negative result for the original story. It would be incorrect to claim that our low-data experiment establishes a huge ViT sample-efficiency disadvantage. The observed clean-accuracy difference is small on this task. What changes more noticeably is robustness, particularly after objects are shifted.

In the 120-example condition, the 4-pixel ViT's shifted-image accuracy averages **61.20%**, compared with **73.70%** under the larger training set. The CNN's shifted accuracy remains near 100%.

That is more informative than simply counting correct examples in the familiar test distribution. It suggests that the network has acquired a workable rule but is less stable when the pixels move away from familiar locations.

To study sample efficiency rigorously, we would need many more data points, a range of dataset sizes, more training schedules, and ideally multiple tasks with different complexity. The original researchers' conclusions were based on large benchmark suites and pretrained models, not a single two-marker experiment.

## 17. Distribution shift and robustness

An image classifier can be excellent at recognizing examples from its training distribution and still make unexpected mistakes after a small change in the environment.

We create two separate stress conditions.

The first increases the standard deviation of background noise from approximately **0.06 to 0.25**. The label rule does not change, and the same colored objects remain present. This tests sensitivity to a pixel-level nuisance variable.

The second moves marker locations by up to two pixels along each axis, clipping the coordinates to keep the markers inside the image. This tests the effects of local displacement relative to the fixed patch grid. The label rule remains the same.

Across the three seeds, the CNN records **99.22% accuracy with stronger noise** and **98.70% with shifted markers**.

The 4-pixel ViT scores **97.66% with stronger noise** and **73.70% with shifted markers**.

The 8-pixel ViT scores **76.56% with stronger noise** and **60.68% with shifted markers**, although its lower clean accuracy and high variance complicate interpretation. Much of its weakness is already visible before any shift.

This experiment does not prove that Vision Transformers in general are translation fragile. Modern ViTs are often pretrained at huge scale, trained with aggressive augmentation, and designed with substantial improvements over this toy architecture.

It does show that a model can memorize a training coordinate system rather than learn the same geometric relation in a position-robust form. This is exactly the kind of evaluation that disappears when we report only clean accuracy.

The right question is not only whether a model makes the correct prediction. It is whether it still makes that prediction when aspects of the image unrelated to the label change.

## 18. Statistical design and benchmark honesty

We evaluate three seeds, which is better than one, but three runs are not enough to estimate narrow uncertainty intervals reliably. A reported sample standard deviation tells us how much the selected runs varied. It does not guarantee generalization to a new model size, optimizer configuration, dataset, or hardware.

The most unstable configuration is the 8-pixel ViT. Its clean accuracy varies from chance to nearly perfect depending on the seed. It would be misleading to report only its best run or to round the average up into a claim that it nearly always solves the task.

Our comparisons also have unequal parameter budgets: approximately 166,562 parameters for the CNN, 48,242 for the 8-pixel ViT, and 43,634 for the 4-pixel ViT. The CNN's extra parameters come largely from the spatial classification head. The model sizes are not matched.

For a publishable comparative study, I would add two distinct experimental tracks.

**Track A: Equal training budget.** Use the same training images, augmentations, optimizer search ranges, and total gradient steps. Report wall-clock and hardware details. This is close to what our current workshop provides, with a limited hyperparameter search.

**Track B: Equal compute or parameter budget.** Adjust model width, depth, or training steps until effective training FLOPs or parameter count are comparable. Do not confuse this with Track A. Equal steps can still consume very different resources.

For both tracks, the evaluation set should be fixed before comparing models. Hyperparameters should be selected only using a validation set. The final test set should remain untouched until the experiments are finalized.

There should also be an **information budget**. When comparing pretrained ViT with a CNN trained from scratch, the amount of data used during pretraining is part of the model's history. Ignoring it creates an invalid impression that two systems learned from the same evidence.

The companion project saves the configuration, all individual runs, mean values, and sample standard deviations to `results/benchmark.json`. The data-ablation run is saved separately in `results/low_data.json`. These are the records behind the figures and the numbers in this article.

---
## 19. Training on real CIFAR-10

Synthetic tasks make it possible to isolate one idea. They do not replace natural-image benchmarks. The next step is to evaluate on a standard dataset.

CIFAR-10 contains **50,000 training images and 10,000 test images**, each 32 by 32 pixels, across ten object categories [6]. It is small enough for an educational workflow and complicated enough to reveal differences absent from our colored-marker task.

The companion repo includes `vit_lab/cifar_train.py`, an optional CIFAR-10 training entry point. It is not executed by the default benchmark. It requires `torchvision` and an explicit dataset download, and it will not silently contact the internet when running the synthetic tests.

From the project directory:

```bash
# Train a small ViT on a subset of real CIFAR-10 images.
python -m vit_lab.cifar_train \
  --model vit8 \
  --download \
  --epochs 5 \
  --max-train 5000 \
  --batch-size 128

# Run a CNN control with the same training settings.
python -m vit_lab.cifar_train \
  --model cnn \
  --epochs 5 \
  --max-train 5000 \
  --batch-size 128

# Use finer tokens if your machine has more compute.
python -m vit_lab.cifar_train \
  --model vit4 \
  --epochs 5 \
  --max-train 5000
```

The script creates a deterministic partition of the official CIFAR training set: 45,000 images are eligible for training and 5,000 form a validation split. If the `max-train` option is smaller, the training subset is selected from the eligible training indices. Validation examples never overlap with the selected training examples.

The validation set determines which epoch's model is retained. Only then does the script evaluate once on the separate official test set.

That arrangement addresses a surprisingly common tutorial mistake: repeatedly evaluating against the official test set and choosing a model based on those results. Doing so gradually converts your test set into a hyperparameter-tuning set, and the reported final accuracy no longer reflects untouched evaluation data.

The data transform applies random cropping and horizontal flipping to training examples, then converts image pixels to tensors and normalizes channels using standard CIFAR-10 statistics. The validation and test transforms perform normalization without random augmentation.

A five-epoch experiment on 5,000 examples is a **smoke test and learning exercise**, not a serious attempt to reach published CIFAR-10 accuracy. For better natural-image results, run longer, tune schedules on validation data, use stronger regularization, and compare pretrained models against carefully optimized from-scratch baselines.

Do not copy numerical results from the original ViT paper into the results of this script. We have not downloaded CIFAR-10 or run the optional real-data training during preparation of this article.

## 20. Why pretraining changes the outcome

The original paper is sometimes summarized as proof that Transformers outperform CNNs. That summary erases the most informative part of the research.

The authors showed that **pretraining scale** has a profound effect on whether a relatively generic Transformer architecture can learn visual representations that transfer well. Once representations are learned across millions of images, fine-tuning a relatively small downstream dataset becomes a very different optimization problem from training a transformer from random initialization on CIFAR-10 alone.

The released Google Research repository includes JAX/Flax fine-tuning code and multiple pretrained checkpoint families, including ImageNet-21k, augmented regularization, and other training procedures [3]. That is the right source if your goal is to investigate the original checkpoint and large-scale transfer setup, rather than our small PyTorch teaching model.

Later work made the picture more nuanced.

*Training Data-Efficient Image Transformers and Distillation Through Attention*, commonly called DeiT, demonstrated that strong augmentation and distillation could train competitive vision transformers on ImageNet without relying on an external hundreds-of-millions-image dataset. Its authors reported **83.1% ImageNet top-1** for a reference 86-million-parameter ViT and stronger configurations with distillation [4].

*How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers* examined data scale and regularization systematically. It found that stronger augmentation and increased computation can substantially compensate for reduced pretraining data in certain regimes [5].

These findings do not invalidate ViT's data-efficiency concern. They clarify it. Architectural inductive bias is only one contributor to sample efficiency. A training recipe is also an inductive bias, in the practical sense that augmentation, regularization, distillation, and optimizer choices constrain which representations the model is encouraged to learn.

A CNN gets translation-related assumptions for free from weight sharing. A ViT can learn comparable robustness through data and training, but the amount of evidence and optimization required depends on the task and setup.

The best way to evaluate those claims today is not to rerun a five-epoch CIFAR tutorial and declare a winner. It is to document pretrained data, checkpoint provenance, training recipe, compute, resolution, and test-time evaluation protocol, then compare under a clearly stated research question.

## 21. Why attention maps are not necessarily explanations

One of the appealing features of ViT is that attention operates over visual patches. It is tempting to extract the attention weights and paint them over the image as an explanation of a prediction.

Such visualizations can be useful, but there are several reasons to be careful.

Attention weights indicate how information is mixed at a particular layer and head. They do not directly measure the causal contribution of a patch to the final class score. Residual connections and feed-forward components can preserve, introduce, or transform information independently of a single attention map. Different heads may use the same patch for different computations.

Attention rollout and gradient-based attribution methods try to account for these complexities, but neither should be treated as a complete account of the network's reasoning.

For our colored-marker task, a stronger intervention would alter the marker position while keeping its identity and background fixed, then examine the effect on predicted class probabilities. We can also mask a marker, switch colors, or permute patches and see whether the output changes in the predicted direction.

These are causal tests because we manipulate the input and observe the effect on model behavior. They are not the same as proving a particular attention head computes the geometric relation internally. Establishing a specific mechanism would require much more careful tracing and intervention at intermediate representations.

If you add attention visualization to the companion project, start by capturing each head's weights with `need_weights=True`, average only when the averaging operation is justified, and include the CLS token's attention separately from patch-to-patch attention. Remember that attention is a diagnostic view of information flow, not an automatic explanation of decisions.

## 22. Common implementation failures

The most frustrating ViT bugs often produce tensors of valid shape. The model runs, losses decrease, and the architecture quietly does something other than what you intended.

**Mistake: wrong patch order.** Reshaping a convolution output without moving its channel axis can mix the token and feature dimensions. Our implementation explicitly flattens spatial axes and transposes to `[batch, patches, width]`.

**Mistake: forgetting CLS in the position count.** With 16 image patches, learned positional embeddings must cover 17 tokens when CLS is included.

**Mistake: using a causal mask for image classification.** A standard ViT classifier is bidirectional. Imposing a text-style causal mask prevents early tokens from using later image patches and changes the learned representation.

**Mistake: interpolating a flattened position list as one dimension.** Image patches originate from a two-dimensional grid. The correct resolution-adaptation routine separates CLS, reconstructs the old grid, applies two-dimensional interpolation, and flattens the new grid again.

**Mistake: assuming a small number of patches is universally efficient.** Reducing patch count can discard small visual details. Increasing patch count can raise attention cost and make optimization more expensive.

**Mistake: treating a `Conv2d` patch tokenizer as a deep CNN.** A single non-overlapping patch projection is not the same architectural prior as repeated local convolutions and multiscale feature pooling.

**Mistake: forgetting the difference between accuracy and convergence.** Our 8-pixel ViT succeeds under some seeds and not others in 150 steps. That is an optimization finding, not proof that the architecture cannot solve the problem.

**Mistake: reading data-volume results without counting passes over each example.** Equal numbers of gradient steps do not imply equal numbers of examples seen when the datasets have different sizes.

**Mistake: relying only on the best seed.** Saving all runs protects against accidental cherry-picking.

**Mistake: calling a short synthetic benchmark ImageNet-level evidence.** The two-marker task was designed to expose positional structure. It measures neither object recognition nor general visual reasoning across natural photographs.

**Mistake: measuring runtime without stating hardware.** Wall-clock results depend on batch size, CPU threading, accelerator, implementation, and kernel selection. The saved times from this educational lab are useful for reproducing our environment, not for universal performance claims.

The tests in this repository verify patch projection equivalence, model output shapes, positional interpolation, permutation invariance when positions are absent, gradient flow, synthetic dataset determinism, label balance, and a short training run.

## 23. Research extensions worth building

The project is designed to be extended. Here are experiments that would substantially improve its scientific value.

**Run a clean equal-parameter comparison.** Adjust CNN head width and ViT depth until trainable parameters are closely matched. Keep the training protocol identical, then repeat under an approximately matched compute budget. These are distinct experiments and should be labeled separately.

**Measure object-size sensitivity.** Repeat the red-blue task with markers occupying 1, 2, 4, and 8 pixels. Patch-size selection should matter more when objects become smaller than the patch's ability to preserve useful distinguishing information.

**Study the effect of augmentations.** Train models with random translation, crop, color jitter, and noise. Evaluate the same transformations on held-out examples. Determine whether ViT's shifted-marker weakness can be reduced through training alone, or whether architectural changes remain advantageous.

**Try positional encoding alternatives.** Compare learned absolute positions against sinusoidal two-dimensional embeddings and relative positional biases. Use the same model width, number of tokens, and training schedule. Test both clean accuracy and translation robustness.

**Change the pooling interface.** Remove CLS entirely for a true mean-pooling model. Compare that with CLS, learned attention pooling, and a class-attention head. Pay attention to parameter count and optimizer tuning.

**Measure calibration, not just accuracy.** Record negative log-likelihood, expected calibration error, and confidence on corrupted images. A high-accuracy model that becomes confidently wrong under distribution shift may be a poor deployment choice.

**Explore higher-resolution fine-tuning.** Train a small image classifier at one resolution, then test and fine-tune at another using two-dimensional interpolation of the position grid. Include a control where the positional vectors are reinitialized. This directly tests the purpose of the interpolation routine.

**Add representation analysis.** Compare embeddings of translated versions of the same image across layers. Estimate representational similarity and train linear probes for marker identity and position. These are more informative than simply comparing two final logits.

**Introduce real-data transfer learning.** Fine-tune a documented pretrained ViT checkpoint on CIFAR-10 or a larger open dataset. Record the checkpoint, upstream dataset, preprocessing, image resolution, optimizer, scheduler, random seeds, and evaluation procedure. Compare with a pretrained convolutional baseline, not only with a CNN trained from scratch.

Every extension should preserve the distinction between architectural behavior demonstrated in a small controlled environment and broad claims about pretrained vision foundation models.

## 24. Final lessons

ViT's central contribution was architectural simplicity. Split the image into non-overlapping patches, transform patches into token embeddings, add positional information, prepend a CLS token, and let a largely standard Transformer perform the rest.

The original paper showed that this simple pattern, when pretrained at scale, could become a powerful alternative to large convolutional networks [1]. Later research improved small-data ViT training through augmentation, distillation, and regularization [4, 5].

Our companion experiment demonstrates three more focused points.

First, **a model cannot recover spatial order that its representation never encodes**. Removing positional information from a permutation-invariant CLS classifier makes a left-versus-right task structurally unidentifiable.

Second, **patch size changes the information the model can use and the computational price of using it**. In our short synthetic benchmark, 4-pixel patches learn more reliably than 8-pixel patches, at higher measured runtime.

Third, **clean accuracy is not enough**. Our 4-pixel ViT almost perfectly classifies examples drawn from the training distribution, yet performs substantially worse after a small positional shift. A simple CNN is more stable under the same controlled change.

None of those observations means Vision Transformers are inherently unreliable, or that CNNs are always superior. They mean that the most interesting architectural claims emerge only when the experiment is designed to test them.

When we treat an image as a sentence of patches, we inherit both the power and the responsibilities of sequence modeling. We must decide what a token means, where it came from, what relationships it can express, and how the representation should behave when the world changes around it.

That is where the real engineering begins.

---

## 25. References

**[1] Dosovitskiy, A., et al. (2021).** *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. Original paper, architecture, pretraining, and transfer benchmarks. https://arxiv.org/abs/2010.11929

**[2] labml.ai.** *Vision Transformer (ViT): Annotated PyTorch Implementation.* Source of the patch embedding, learned position, CLS, and Transformer walkthrough supplied for this article. https://nn.labml.ai/transformers/vit/index.html

**[3] Google Research.** *Vision Transformer and MLP-Mixer Architectures.* Official code and checkpoint repository, including fine-tuning examples and AugReg checkpoints. https://github.com/google-research/vision_transformer

**[4] Touvron, H., et al. (2021).** *Training Data-Efficient Image Transformers and Distillation Through Attention.* ICML 2021. DeiT's augmentation and distillation-based training approach. https://arxiv.org/abs/2012.12877

**[5] Steiner, A., et al. (2022).** *How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers.* Transactions on Machine Learning Research. https://arxiv.org/abs/2106.10270

**[6] Krizhevsky, A. (2009).** *Learning Multiple Layers of Features from Tiny Images.* CIFAR-10 dataset. Official data description and provenance. https://www.cs.toronto.edu/~kriz/cifar.html

**[7] PyTorch.** `torch.nn.MultiheadAttention` documentation. API semantics, layout, masks, and implementation details. https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

**[8] PyTorch.** `torch.nn.functional.interpolate` documentation. Two-dimensional bicubic interpolation used when adapting the position grid. https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

**[9] Chen, X., et al. (2022).** *An Image Is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels.* Later study of pixel-level tokenization and locality assumptions, published at ICLR 2025. https://arxiv.org/abs/2406.09415

### Companion repository

The accompanying educational PyTorch lab contains the exact synthetic data generator, the mini ViT and CNN, deterministic benchmarks, a separate low-data ablation, an optional CIFAR-10 training path, 15 tests, original figures, and saved JSON results. It does not contain or redistribute Google's pretrained weights.

**Suggested Medium tags:** Vision Transformer, PyTorch, Computer Vision, Deep Learning, Machine Learning.
