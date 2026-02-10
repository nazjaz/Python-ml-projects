# BERT-from-Scratch API Documentation

## Module: `src.main`

### Constants

- `PAD_ID`: 0
- `CLS_ID`: 1
- `SEP_ID`: 2
- `MASK_ID`: 3

### Classes

#### `LayerNorm`

Layer normalization over the last dimension. Used inside the transformer encoder.

- `forward(x)`: Normalize and scale; returns same shape as `x`.
- `backward(grad_out, lr)`: Backpropagate and update gamma/beta.

#### `MultiHeadSelfAttention`

Multi-head scaled dot-product self-attention.

- Constructor: `MultiHeadSelfAttention(dim_model, num_heads)`; `dim_model` must be divisible by `num_heads`.
- `forward(x)`: Input `(batch, seq_len, dim_model)`; output same shape.
- `backward(grad_out, lr)`: Returns gradient w.r.t. input; updates Q/K/V/O projections.

#### `FeedForward`

Position-wise two-layer MLP with ReLU.

- Constructor: `FeedForward(dim_model, dim_hidden)`.
- `forward(x)`, `backward(grad_out, lr)`.

#### `BertEmbeddings`

Token + position + segment embeddings.

- Constructor: `BertEmbeddings(vocab_size, dim_model, max_seq_len, num_segment_types=2)`.
- `forward(token_ids, segment_ids)`: Returns embeddings of shape `(batch, seq_len, dim_model)`.
- `backward_embed(grad_out, lr)`: Updates token embedding matrix only.

#### `BertEncoder`

Stack of transformer encoder layers (attention + norm + FF + norm with residuals).

- Constructor: `BertEncoder(dim_model, num_heads, dim_ff, num_layers)`.
- `forward(x)`: Returns hidden states same shape as `x`.
- `backward(grad_out, lr)`: Returns gradient w.r.t. encoder input.

#### `BertPooler`

Dense + tanh on the [CLS] token (first position).

- Constructor: `BertPooler(dim_model)`.
- `forward(hidden_states)`: Returns `(batch, dim_model)`.
- `backward(grad_out, lr)`: Returns gradient w.r.t. hidden states (only [CLS] position filled).

#### `BertMLMHead`

Linear projection from hidden size to vocab size for masked positions.

- Constructor: `BertMLMHead(dim_model, vocab_size)`.
- `forward(hidden_states, masked_positions)`: Returns logits shape `(batch, num_masked, vocab_size)`.
- `backward(grad_logits, lr)`: Returns gradient w.r.t. hidden states at masked positions.

#### `BertNSPHead`

Binary classifier from pooler output.

- Constructor: `BertNSPHead(dim_model)`.
- `forward(pooler_output)`: Returns logits shape `(batch, 2)`.
- `backward(grad_logits, lr)`: Returns gradient w.r.t. pooler output.

#### `BertModel`

Full BERT-like model with MLM and NSP heads.

- Constructor: `BertModel(vocab_size, dim_model, num_heads, dim_ff, num_layers, max_seq_len)`.
- `forward(input_ids, segment_ids, masked_positions)`: Returns `(mlm_logits, nsp_logits)`.
- `backward(grad_mlm_logits, grad_nsp_logits, lr)`: Updates all parameters.
- `mlm_loss_and_grad(logits, labels, mask)`: Static; returns `(loss, grad_logits)` for MLM cross-entropy over masked positions.
- `nsp_loss_and_grad(logits, labels)`: Static; returns `(loss, grad_logits)` for NSP cross-entropy.

### Data functions

#### `create_mlm_batch(batch_size, seq_len, vocab_size, mask_prob=0.15, random_seed=None)`

Creates a single-segment batch with [CLS] ... [SEP], randomly masks tokens with probability `mask_prob`, and returns:

- `input_ids`: (batch, seq_len)
- `segment_ids`: (batch, seq_len), zeros
- `masked_positions`: (batch, max_masked), -1 for padding
- `mlm_labels`: (batch, max_masked), -1 for padding
- `nsp_labels`: (batch,) all 1 for single-segment

#### `create_nsp_pair_batch(batch_size, seq_len, vocab_size, random_seed=None)`

Creates [CLS] sent_a [SEP] sent_b [SEP] with segment_ids 0 for A and 1 for B. NSP label 1 when B is a copy of A, 0 when B is random. Returns same five arrays; MLM positions/labels are padding (-1).

#### `generate_synthetic_bert_batches(num_batches, batch_size, seq_len, vocab_size, mask_prob=0.15, random_seed=None)`

Returns a list of batches; each batch is either from `create_mlm_batch` or `create_nsp_pair_batch` (randomly chosen per batch).

### Runner

#### `BertRunner(config_path=None)`

Loads YAML config, sets up logging, and runs training and evaluation.

- `run()`: Returns dict with `train_mlm_loss`, `train_nsp_loss`, `test_mlm_loss`, `test_nsp_loss`, `test_nsp_accuracy`.

### Entry point

#### `main()`

Parses `--config` and `--output`, runs `BertRunner().run()`, prints results and optionally writes JSON.
