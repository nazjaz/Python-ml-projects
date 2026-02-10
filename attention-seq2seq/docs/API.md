# Attention-based Seq2Seq API Documentation

## Module: `src.main`

### Functions

#### `scaled_dot_product_attention(query, key, value, mask=None)`

Compute scaled dot-product attention.

- `query`: Array of shape `(batch_size, target_len, d_k)`
- `key`: Array of shape `(batch_size, source_len, d_k)`
- `value`: Array of shape `(batch_size, source_len, d_v)`
- `mask`: Optional boolean or 0/1 mask of shape `(batch_size, target_len, source_len)`

Returns:

- `output`: Attention output of shape `(batch_size, target_len, d_v)`
- `weights`: Attention weights of shape `(batch_size, target_len, source_len)`

### Classes

#### `Seq2SeqAttentionModel`

Minimal attention-based encoder-decoder model for copy-style sequence tasks.

**Constructor**: `Seq2SeqAttentionModel(vocab_size, src_length, tgt_length, d_model, d_k, d_v)`

- `vocab_size`: Size of the token vocabulary.
- `src_length`: Length of source sequences.
- `tgt_length`: Length of target sequences.
- `d_model`: Embedding and hidden dimension.
- `d_k`: Dimension for attention queries and keys.
- `d_v`: Dimension for attention values.

**Methods**:

- `train(x_src, x_tgt, epochs, learning_rate, batch_size, verbose)`: Train model on integer token sequences using teacher forcing. Returns training history with per-epoch loss and accuracy.
- `evaluate(x_src, x_tgt)`: Evaluate model on a dataset and return loss and accuracy.
- `predict(x_src)`: Generate predictions (token sequences) for input sources.

#### `AttentionRunner`

Configuration-driven runner for training and evaluating the attention-based seq2seq model.

**Constructor**: `AttentionRunner(config_path: Optional[Path] = None)`

- Loads configuration from a YAML file and configures logging.

**Methods**:

- `run()`: Generate synthetic data, construct the model, train it, evaluate on a held-out test set, and return a dictionary with training and test metrics.

### Data Utilities

#### `generate_copy_task_data(n_samples, src_length, tgt_length, vocab_size, random_seed)`

Generate synthetic integer sequences for a copy task.

- `n_samples`: Number of sequence pairs to generate.
- `src_length`: Length of source sequences.
- `tgt_length`: Length of target sequences (typically equal to `src_length`).
- `vocab_size`: Size of the vocabulary.
- `random_seed`: Seed for reproducibility.

Returns:

- `x_src`: Source sequences of shape `(n_samples, src_length)`.
- `x_tgt`: Target sequences of shape `(n_samples, tgt_length)` equal to `x_src`.

### Entry Point

#### `main()`

Command-line entry point.

Supported arguments:

- `--config`: Path to YAML configuration file.
- `--output`: Path to JSON file where results will be written.

