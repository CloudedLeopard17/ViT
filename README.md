# Vision Transformer (ViT) on Food101 — From Scratch

A complete implementation of the Vision Transformer (ViT) architecture trained from scratch on the [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset, along with visualizations of the learned positional embeddings and attention maps.

---

## Project Structure

```
ViT/
├── Vit.ipynb                        # Model definition, training & evaluation
├── visualizations.ipynb             # Position embedding & attention visualizations
├── models_vit_food101/
│   └── vit_base_patch16_224_best.pt # Best saved model checkpoint
└── logs_vit_food101/                # TensorBoard training logs
```

---

## Dataset

**[Food101](https://pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html)** — 101 food categories, 101,000 images total.

| Split | Samples |
|-------|---------|
| Train | 75,750  |
| Test  | 25,250  |

---

## Model Architecture

A ViT built entirely from scratch using PyTorch, without using any pre-built ViT implementations.

| Hyperparameter     | Value              |
|--------------------|--------------------|
| Image size         | 224 × 224          |
| Patch size         | 16 × 16            |
| Number of patches  | 196 (14 × 14 grid) |
| Embedding dim      | 512                |
| Attention heads    | 8                  |
| Encoder layers     | 12                 |
| FFN hidden dim     | 2048 (4× dim)      |
| Dropout            | 0.1                |
| Classes            | 101                |

**Pipeline:**
1. Image → non-overlapping 16×16 patches (196 patches)
2. Linear projection to embedding dim (512)
3. Prepend `[CLS]` token
4. Add learnable 1D positional embeddings
5. Pass through 12 Transformer encoder blocks (Pre-LN: LayerNorm → MHA → residual → LayerNorm → FFN → residual)
6. Classify using `[CLS]` token output

**Key implementation details:**
- Uses `F.scaled_dot_product_attention` (Flash Attention) during training for memory efficiency
- Weight initialization: Xavier uniform for weight matrices, zeros for biases
- Model compiled with `torch.compile` for additional speedup

---

## Training

### Data Augmentation

**Training transforms:**
- RandomResizedCrop (224, scale 0.6–1.0, BICUBIC)
- RandomHorizontalFlip
- RandomRotation (±20°)
- RandomGrayscale (p=0.1)
- RandomAutocontrast (p=0.1)
- RandAugment (2 ops, magnitude 15)
- RandomErasing (p=0.25)
- MixUp (α=0.2) or CutMix (α=1.0) — applied randomly per batch (50/50)

**Validation transforms:** Resize to 224 → ImageNet normalize.

### Optimizer & Scheduler

| Setting            | Value                              |
|--------------------|------------------------------------|
| Optimizer          | AdamW                              |
| Learning rate      | 1e-3                               |
| Weight decay       | 0.1                                |
| Warmup             | Linear LR: 1% → 100% over 25 epochs |
| Main schedule      | CosineAnnealingLR (η_min = 1e-5)   |
| Total epochs       | 250                                |
| Batch size         | 256                                |
| Precision          | bfloat16 (autocast)                |

### Logging

Training metrics (loss, accuracy, learning rate, elapsed time, ETA) are logged to TensorBoard:
```bash
tensorboard --logdir logs_vit_food101
```

---

## Evaluation

The best model (highest validation accuracy) is saved to `models_vit_food101/vit_base_patch16_224_best.pt`.

Post-training evaluation includes:
- Confusion matrix across all 101 classes
- Identification of the top-5 most confused class pairs

![Confusion Matrix](assets%20/confusion_matrix.png)

---

## Visualizations (`visualizations.ipynb`)

### 1. Positional Embedding Cosine Similarity

The learned positional embeddings are extracted and pairwise cosine similarities are computed across all 196 patch positions (14×14 grid). Each subplot shows how similar one patch's position embedding is to all others — the model learns spatially coherent structure: nearby patches have higher similarity.

```python
pos_embed = model._orig_mod.embedding[0, 1:, :]  # (196, 512)
sim = cosine_similarity(pos_embed[i], pos_embed).reshape(14, 14)
```

![Positional Embedding Cosine Similarity](assets%20/pos_embedding.png)

### 2. Attention Map Visualization

Attention scores are extracted from a specific encoder block (block 10) using PyTorch forward hooks. Since Flash Attention is used during training (which does not return attention scores), Q/K/V matrices are manually projected and the softmax attention scores are recomputed for visualization.

The visualization shows:
- The original image
- Attention weight maps for 8 randomly sampled patch tokens — each heatmap shows where that patch "looks" across the image

```python
_, scores = MultiHeadAttention.attention(query, key, value, mask=None, dropout=None)
# scores shape: (1, heads, seq_len, seq_len)
at = scores[0, head, patch_idx, 1:].reshape(14, 14)
```

![Attention Visualization](assets%20/attention_visualization.png)

---

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
tensorboard
```

---

## Usage

### Training

Open and run `Vit.ipynb`. The notebook will:
1. Download Food101 automatically via `torchvision.datasets.Food101`
2. Build and initialize the ViT model
3. Run training for 250 epochs with warmup + cosine annealing
4. Save the best checkpoint to `models_vit_food101/`

### Visualization

Open and run `visualizations.ipynb`. It loads the saved checkpoint and produces:
- Position embedding similarity heatmaps
- Per-head attention maps for a randomly sampled test image

---

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., 2020
- [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — Touvron et al., 2021 (DeiT, training recipes)
