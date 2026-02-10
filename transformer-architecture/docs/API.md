# Transformer Encoder API Documentation

## Module: `src.main`

### Classes

#### `LayerNorm`

Layer normalization module applied over the last feature dimension.

**Constructor**: `LayerNorm(features: int, eps: float = 1e-5)`

- `features`: Number of feature dimensions.
- `eps`: Small constant for numerical stability.

**Methods**:

- `forward(x)`: Apply layer normalization to input tensor.
- `backward(grad_out, lr)`: Backpropagate gradients and update scale and bias.

#### `PositionalEncoding`

Sinusoidal positional encoding as introduced in the original Transformer
architecture. Encodings are precomputed up to a maximum sequence length.

**Constructor**: `PositionalEncoding(dim_model: int, max_len: int = 5000)`

- `dim_model`: Embedding dimension.
- `max_len`: Maximum supported sequence length.

**Methods**:

- `forward(x)`: Add positional encodings to token embeddings. Expects input
  shape `(batch_size, sequence_length, dim_model)`.

#### `MultiHeadSelfAttention`

Multi-head scaled dot-product self-attention layer.

**Constructor**: `MultiHeadSelfAttention(dim_model: int, num_heads: int)`

- `dim_model`: Embedding dimension.
- `num_heads`: Number of attention heads; must divide `dim_model`.

**Methods**:

- `forward(x)`: Compute self-attention for input embeddings of shape
  `(batch_size, sequence_length, dim_model)`.
- `backward(grad_out, lr)`: Backpropagate gradients with respect to the
  attention parameters and return gradient with respect to the inputs.

#### `FeedForward`

Position-wise feed-forward network applied independently at each sequence
position.

**Constructor**: `FeedForward(dim_model: int, dim_hidden: int)`

- `dim_model`: Embedding dimension.
- `dim_hidden`: Hidden dimension of the intermediate layer.

**Methods**:

- `forward(x)`: Apply two-layer MLP with ReLU activation in the hidden layer.
- `backward(grad_out, lr)`: Backpropagate gradients and update weights.

#### `TransformerEncoderLayer`

Single Transformer encoder layer composed of:

- Multi-head self-attention with residual connection and layer normalization.
- Position-wise feed-forward network with residual connection and layer
  normalization.

**Constructor**:
`TransformerEncoderLayer(dim_model: int, num_heads: int, dim_ff: int)`

- `dim_model`: Embedding dimension.
- `num_heads`: Number of attention heads.
- `dim_ff`: Hidden dimension of the feed-forward network.

**Methods**:

- `forward(x)`: Run the encoder layer on input of shape
  `(batch_size, sequence_length, dim_model)`.
- `backward(grad_out, lr)`: Backpropagate gradients through the layer.

#### `TransformerEncoderClassifier`

Transformer encoder-based classifier using the representation of the first
token as a classification token.

**Constructor**:

```python
TransformerEncoderClassifier(
    vocab_size: int,
    dim_model: int,
    num_heads: int,
    dim_ff: int,
    num_layers: int,
    num_classes: int,
    max_seq_len: int,
)
```

- `vocab_size`: Size of the discrete token vocabulary.
- `dim_model`: Embedding dimension.
- `num_heads`: Number of attention heads per layer.
- `dim_ff`: Hidden dimension of the feed-forward network in each layer.
- `num_layers`: Number of stacked encoder layers.
- `num_classes`: Number of output classes for classification.
- `max_seq_len`: Maximum supported sequence length.

**Methods**:

- `forward(tokens)`: Forward pass from integer token ids of shape
  `(batch_size, sequence_length)` to logits of shape
  `(batch_size, num_classes)`.
- `backward(grad_logits, lr)`: Backpropagate gradients from classification
  logits through the encoder stack and update classifier weights.

### Runner

#### `TransformerRunner`

High-level orchestration class that loads configuration, generates synthetic
data, trains the Transformer encoder classifier, and evaluates accuracy.

**Constructor**: `TransformerRunner(config_path: Optional[Path] = None)`

**Methods**:

- `run()`: Returns a dictionary with:
  - `train_loss`: Final training loss after the last epoch.
  - `test_accuracy`: Classification accuracy on held-out data.

### Functions

#### `generate_synthetic_classification_data(n_samples, seq_len, vocab_size, random_seed)`

Generate synthetic token sequences and binary labels where each label is the
parity (even or odd) of the sum of token ids in the sequence.

Returns:

- `tokens`: Integer token ids, shape `(n_samples, seq_len)`.
- `labels`: Integer labels in `{0, 1}`, shape `(n_samples,)`.

#### `main()`

Command-line entry point. Supports:

- `--config`: Path to a configuration YAML file.
- `--output`: Path to a JSON file for saving results.

