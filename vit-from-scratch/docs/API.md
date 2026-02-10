# Vision Transformer API Documentation

## Module: `src.main`

### Classes

#### `LayerNorm`

Layer normalization over the last dimension.

- `forward(x)`: Normalize and scale input.
- `backward(grad_out, lr)`: Backpropagate and update parameters.

#### `MultiHeadSelfAttention`

Multi-head scaled dot-product self-attention.

- Constructor: `MultiHeadSelfAttention(dim_model, num_heads)`.
- `forward(x)`: Input `(batch, seq_len, dim_model)`, output same shape.
- `backward(grad_out, lr)`: Returns gradient w.r.t. input and updates Q/K/V/O.

#### `FeedForward`

Position-wise feed-forward network with ReLU.

- Constructor: `FeedForward(dim_model, dim_hidden)`.
- `forward(x)`, `backward(grad_out, lr)`.

#### `ViTEncoderLayer`

Single Vision Transformer encoder block:

- Multi-head attention + residual + layer norm.
- Feed-forward + residual + layer norm.

- Constructor: `ViTEncoderLayer(dim_model, num_heads, dim_ff)`.
- `forward(x)`, `backward(grad_out, lr)`.

#### `PatchEmbedding`

Converts images into a sequence of patch embeddings.

- Constructor: `PatchEmbedding(image_size, patch_size, in_channels, dim_model)`.
- `forward(images)`: Input `(batch, H, W, C)`, output `(batch, num_patches, dim_model)`.
- `backward(grad_out, lr)`: Updates projection matrix and bias.

#### `ViTClassifier`

End-to-end Vision Transformer classifier with [CLS] token.

- Constructor:

```python
ViTClassifier(
    image_size: int,
    patch_size: int,
    in_channels: int,
    dim_model: int,
    num_heads: int,
    dim_ff: int,
    num_layers: int,
    num_classes: int,
)
```

- `forward(images)`: Returns logits of shape `(batch, num_classes)`.
- `backward(grad_logits, lr)`: Backpropagates through classifier head, encoder, and patch embedding.
- `cross_entropy_loss(logits, labels)`: Static; returns `(loss, grad_logits)`.

### Data utilities

#### `generate_synthetic_images(n_samples, image_size, in_channels, num_classes, random_seed=None)`

Returns:

- `images`: `(n_samples, image_size, image_size, in_channels)`.
- `labels`: `(n_samples,)` integer class labels.

### Runner

#### `ViTRunner(config_path: Optional[Path] = None)`

Orchestrates configuration loading, data generation, model training, and evaluation.

- `run()`: Returns dict with:
  - `train_loss`: Final training loss.
  - `test_accuracy`: Accuracy on held-out synthetic images.

### Entry point

#### `main()`

Command-line interface:

- `--config`: path to YAML configuration.
- `--output`: optional JSON path for saving results.

