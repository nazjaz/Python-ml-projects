"""BERT-like model from scratch with MLM and NSP.

This module implements a BERT-style encoder using only NumPy: token, position,
and segment embeddings; stacked transformer encoder layers; masked language
modeling (MLM) and next sentence prediction (NSP) heads with joint training.
"""

import argparse
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

# Special token ids (must match data generation).
PAD_ID = 0
CLS_ID = 1
SEP_ID = 2
MASK_ID = 3


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax along a given axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x - x_max, -500, 500))
    return e / np.sum(e, axis=axis, keepdims=True)


class LayerNorm:
    """Layer normalization over the last dimension."""

    def __init__(self, features: int, eps: float = 1e-5) -> None:
        self.features = features
        self.eps = eps
        self.gamma = np.ones((features,), dtype=np.float32)
        self.beta = np.zeros((features,), dtype=np.float32)
        self._cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        self._cache = (x_norm, std)
        return self.gamma * x_norm + self.beta

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        x_norm, std = self._cache
        dgamma = np.sum(
            grad_out * x_norm, axis=tuple(range(grad_out.ndim - 1))
        )
        dbeta = np.sum(grad_out, axis=tuple(range(grad_out.ndim - 1)))
        self.gamma -= lr * dgamma.astype(np.float32)
        self.beta -= lr * dbeta.astype(np.float32)
        dx_norm = grad_out * self.gamma
        dx = (
            dx_norm
            - np.mean(dx_norm, axis=-1, keepdims=True)
            - x_norm * np.mean(dx_norm * x_norm, axis=-1, keepdims=True)
        ) / std
        return dx.astype(np.float32)


class MultiHeadSelfAttention:
    """Multi-head scaled dot-product self-attention."""

    def __init__(self, dim_model: int, num_heads: int) -> None:
        if dim_model % num_heads != 0:
            raise ValueError("dim_model must be divisible by num_heads")
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        limit = np.sqrt(1.0 / max(1, dim_model))
        self.w_q = np.random.uniform(-limit, limit, (dim_model, dim_model)).astype(
            np.float32
        )
        self.w_k = np.random.uniform(-limit, limit, (dim_model, dim_model)).astype(
            np.float32
        )
        self.w_v = np.random.uniform(-limit, limit, (dim_model, dim_model)).astype(
            np.float32
        )
        self.w_o = np.random.uniform(-limit, limit, (dim_model, dim_model)).astype(
            np.float32
        )
        self._cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = (
            None
        )

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        q, k, v = x @ self.w_q, x @ self.w_k, x @ self.w_v
        q_h = self._split_heads(q)
        k_h = self._split_heads(k)
        v_h = self._split_heads(v)
        dk = float(self.head_dim)
        scores = (q_h @ np.transpose(k_h, (0, 1, 3, 2))) / np.sqrt(dk)
        weights = _softmax(scores, axis=-1)
        attn = weights @ v_h
        context = self._combine_heads(attn)
        out = context @ self.w_o
        self._cache = (x, weights, v_h, context)
        return out

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        x, weights, v_h, context = self._cache
        batch_size, seq_len, _ = x.shape
        d_w_o = context.reshape(-1, self.dim_model).T @ grad_out.reshape(
            -1, self.dim_model
        )
        d_context = grad_out @ self.w_o.T
        d_context_h = self._split_heads(d_context)
        d_weights = d_context_h @ np.transpose(v_h, (0, 1, 3, 2))
        d_v_h = np.transpose(weights, (0, 1, 3, 2)) @ d_context_h
        d_scores = d_weights * weights - weights * np.sum(
            d_weights * weights, axis=-1, keepdims=True
        )
        d_scores_scaled = d_scores / np.sqrt(float(self.head_dim))
        q_h = self._split_heads(x @ self.w_q)
        k_h = self._split_heads(x @ self.w_k)
        d_q_h = d_scores_scaled @ k_h
        d_k_h = np.transpose(d_scores_scaled, (0, 1, 3, 2)) @ q_h
        d_q = self._combine_heads(d_q_h)
        d_k = self._combine_heads(d_k_h)
        d_v = self._combine_heads(d_v_h)
        d_w_q = x.reshape(-1, self.dim_model).T @ d_q.reshape(-1, self.dim_model)
        d_w_k = x.reshape(-1, self.dim_model).T @ d_k.reshape(-1, self.dim_model)
        d_w_v = x.reshape(-1, self.dim_model).T @ d_v.reshape(-1, self.dim_model)
        dx = d_q @ self.w_q.T + d_k @ self.w_k.T + d_v @ self.w_v.T
        scale = 1.0 / float(batch_size * max(1, seq_len))
        for w, dw in [
            (self.w_o, d_w_o),
            (self.w_q, d_w_q),
            (self.w_k, d_w_k),
            (self.w_v, d_w_v),
        ]:
            w -= lr * dw.astype(np.float32) * scale
        return dx.astype(np.float32)


class FeedForward:
    """Position-wise feed-forward network with ReLU."""

    def __init__(self, dim_model: int, dim_hidden: int) -> None:
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden
        limit1 = np.sqrt(1.0 / max(1, dim_model))
        limit2 = np.sqrt(1.0 / max(1, dim_hidden))
        self.w1 = np.random.uniform(-limit1, limit1, (dim_model, dim_hidden)).astype(
            np.float32
        )
        self.b1 = np.zeros((dim_hidden,), dtype=np.float32)
        self.w2 = np.random.uniform(-limit2, limit2, (dim_hidden, dim_model)).astype(
            np.float32
        )
        self.b2 = np.zeros((dim_model,), dtype=np.float32)
        self._cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        self._cache = (x, a1)
        return z2

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        x, a1 = self._cache
        da1 = grad_out @ self.w2.T
        dz1 = da1 * (a1 > 0.0).astype(np.float32)
        batch_size, seq_len, _ = x.shape
        d_w2 = a1.reshape(-1, self.dim_hidden).T @ grad_out.reshape(
            -1, self.dim_model
        )
        d_b2 = np.sum(grad_out, axis=(0, 1))
        d_w1 = x.reshape(-1, self.dim_model).T @ dz1.reshape(-1, self.dim_hidden)
        d_b1 = np.sum(dz1, axis=(0, 1))
        dx = dz1 @ self.w1.T
        scale = 1.0 / float(batch_size * max(1, seq_len))
        self.w1 -= lr * d_w1.astype(np.float32) * scale
        self.b1 -= lr * d_b1.astype(np.float32) * scale
        self.w2 -= lr * d_w2.astype(np.float32) * scale
        self.b2 -= lr * d_b2.astype(np.float32) * scale
        return dx.astype(np.float32)


class BertEmbeddings:
    """Token, position, and segment embeddings summed together."""

    def __init__(
        self,
        vocab_size: int,
        dim_model: int,
        max_seq_len: int,
        num_segment_types: int = 2,
    ) -> None:
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        limit = np.sqrt(1.0 / max(1, dim_model))
        self.token_embed = np.random.uniform(
            -limit, limit, (vocab_size, dim_model)
        ).astype(np.float32)
        self.position_embed = np.random.uniform(
            -limit, limit, (max_seq_len, dim_model)
        ).astype(np.float32)
        self.segment_embed = np.random.uniform(
            -limit, limit, (num_segment_types, dim_model)
        ).astype(np.float32)
        self._token_input_cache: Optional[np.ndarray] = None

    def forward(
        self,
        token_ids: np.ndarray,
        segment_ids: np.ndarray,
    ) -> np.ndarray:
        batch_size, seq_len = token_ids.shape
        x = self.token_embed[token_ids]
        x = x + self.position_embed[:seq_len][None, :, :]
        x = x + self.segment_embed[segment_ids]
        self._token_input_cache = token_ids
        return x

    def backward_embed(self, grad_out: np.ndarray, lr: float) -> None:
        """Accumulate gradients into token embedding only (position/segment fixed for simplicity)."""
        if self._token_input_cache is None:
            raise RuntimeError("Forward must be called before backward.")
        token_ids = self._token_input_cache
        for b in range(grad_out.shape[0]):
            for s in range(grad_out.shape[1]):
                tid = int(token_ids[b, s])
                self.token_embed[tid] -= lr * grad_out[b, s, :].astype(np.float32)


class BertEncoder:
    """Stack of transformer encoder layers."""

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_ff: int,
        num_layers: int,
    ) -> None:
        self.layers: List[Tuple[MultiHeadSelfAttention, LayerNorm, FeedForward, LayerNorm]] = []
        for _ in range(num_layers):
            attn = MultiHeadSelfAttention(dim_model=dim_model, num_heads=num_heads)
            norm1 = LayerNorm(features=dim_model)
            ff = FeedForward(dim_model=dim_model, dim_hidden=dim_ff)
            norm2 = LayerNorm(features=dim_model)
            self.layers.append((attn, norm1, ff, norm2))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for attn, norm1, ff, norm2 in self.layers:
            x = x + attn.forward(norm1.forward(x))
            x = x + ff.forward(norm2.forward(x))
        return x

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        for attn, norm1, ff, norm2 in reversed(self.layers):
            grad_ff = grad_out
            grad_norm2 = ff.backward(grad_ff, lr=lr)
            grad_x1 = grad_out + norm2.backward(grad_norm2, lr=lr)
            grad_attn = attn.backward(grad_x1, lr=lr)
            grad_out = grad_x1 + norm1.backward(grad_attn, lr=lr)
        return grad_out


class BertPooler:
    """Dense + tanh on [CLS] representation (BERT-style)."""

    def __init__(self, dim_model: int) -> None:
        limit = np.sqrt(1.0 / max(1, dim_model))
        self.w = np.random.uniform(-limit, limit, (dim_model, dim_model)).astype(
            np.float32
        )
        self.b = np.zeros((dim_model,), dtype=np.float32)
        self._cache: Optional[np.ndarray] = None

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        cls_repr = hidden_states[:, 0, :]
        out = np.tanh(cls_repr @ self.w + self.b)
        self._cache = (cls_repr, out)
        return out

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        cls_repr, out = self._cache
        d_tanh = grad_out * (1.0 - out**2)
        d_w = cls_repr.T @ d_tanh
        d_b = np.sum(d_tanh, axis=0)
        d_cls = d_tanh @ self.w.T
        batch_size = grad_out.shape[0]
        scale = 1.0 / float(batch_size)
        self.w -= lr * d_w.astype(np.float32) * scale
        self.b -= lr * d_b.astype(np.float32) * scale
        grad_enc = np.zeros(
            (batch_size, 1, self.w.shape[0]), dtype=np.float32
        )
        grad_enc[:, 0, :] = d_cls
        return grad_enc


class BertMLMHead:
    """Linear projection from hidden size to vocab size for MLM."""

    def __init__(self, dim_model: int, vocab_size: int) -> None:
        limit = np.sqrt(1.0 / max(1, dim_model))
        self.w = np.random.uniform(-limit, limit, (dim_model, vocab_size)).astype(
            np.float32
        )
        self.b = np.zeros((vocab_size,), dtype=np.float32)
        self._cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def forward(self, hidden_states: np.ndarray, masked_positions: np.ndarray) -> np.ndarray:
        """Return logits for masked positions only. Shape (batch, num_masked, vocab_size)."""
        batch_size, seq_len, dim = hidden_states.shape
        num_masked = masked_positions.shape[1]
        gathered = np.zeros((batch_size, num_masked, dim), dtype=np.float32)
        for b in range(batch_size):
            for m in range(num_masked):
                pos = int(masked_positions[b, m])
                if pos >= 0:
                    gathered[b, m, :] = hidden_states[b, pos, :]
        logits = gathered @ self.w + self.b
        self._cache = (gathered, masked_positions, hidden_states.shape)
        return logits

    def backward(self, grad_logits: np.ndarray, lr: float) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        gathered, masked_positions, (batch_size, seq_len, dim) = self._cache
        num_masked = gathered.shape[1]
        d_gathered = grad_logits @ self.w.T
        d_w = gathered.reshape(-1, dim).T @ grad_logits.reshape(-1, self.w.shape[1])
        d_b = np.sum(grad_logits, axis=(0, 1))
        scale = 1.0 / float(batch_size * max(1, num_masked))
        self.w -= lr * d_w.astype(np.float32) * scale
        self.b -= lr * d_b.astype(np.float32) * scale
        grad_hidden = np.zeros((batch_size, seq_len, dim), dtype=np.float32)
        for b in range(batch_size):
            for m in range(num_masked):
                pos = int(masked_positions[b, m])
                if pos >= 0:
                    grad_hidden[b, pos, :] += d_gathered[b, m, :]
        return grad_hidden


class BertNSPHead:
    """Binary classifier for next sentence prediction from pooler output."""

    def __init__(self, dim_model: int) -> None:
        limit = np.sqrt(1.0 / max(1, dim_model))
        self.w = np.random.uniform(-limit, limit, (dim_model, 2)).astype(np.float32)
        self.b = np.zeros((2,), dtype=np.float32)
        self._cache: Optional[np.ndarray] = None

    def forward(self, pooler_output: np.ndarray) -> np.ndarray:
        logits = pooler_output @ self.w + self.b
        self._cache = pooler_output
        return logits

    def backward(self, grad_logits: np.ndarray, lr: float) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        pooler_output = self._cache
        batch_size = grad_logits.shape[0]
        d_w = pooler_output.T @ grad_logits
        d_b = np.sum(grad_logits, axis=0)
        d_pooler = grad_logits @ self.w.T
        scale = 1.0 / float(batch_size)
        self.w -= lr * d_w.astype(np.float32) * scale
        self.b -= lr * d_b.astype(np.float32) * scale
        return d_pooler


class BertModel:
    """BERT-like model with MLM and NSP heads, joint forward/backward."""

    def __init__(
        self,
        vocab_size: int,
        dim_model: int,
        num_heads: int,
        dim_ff: int,
        num_layers: int,
        max_seq_len: int,
    ) -> None:
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            dim_model=dim_model,
            max_seq_len=max_seq_len,
            num_segment_types=2,
        )
        self.encoder = BertEncoder(
            dim_model=dim_model,
            num_heads=num_heads,
            dim_ff=dim_ff,
            num_layers=num_layers,
        )
        self.pooler = BertPooler(dim_model=dim_model)
        self.mlm_head = BertMLMHead(dim_model=dim_model, vocab_size=vocab_size)
        self.nsp_head = BertNSPHead(dim_model=dim_model)
        self._hidden_cache: Optional[np.ndarray] = None
        self._pooler_out_cache: Optional[np.ndarray] = None
        self._mlm_positions_cache: Optional[np.ndarray] = None

    def forward(
        self,
        input_ids: np.ndarray,
        segment_ids: np.ndarray,
        masked_positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mlm_logits, nsp_logits). mlm_logits shape (batch, num_masked, vocab)."""
        x = self.embeddings.forward(input_ids, segment_ids)
        hidden = self.encoder.forward(x)
        self._hidden_cache = hidden
        pooler_out = self.pooler.forward(hidden)
        self._pooler_out_cache = pooler_out
        self._mlm_positions_cache = masked_positions
        mlm_logits = self.mlm_head.forward(hidden, masked_positions)
        nsp_logits = self.nsp_head.forward(pooler_out)
        return mlm_logits, nsp_logits

    def backward(
        self,
        grad_mlm_logits: np.ndarray,
        grad_nsp_logits: np.ndarray,
        lr: float,
    ) -> None:
        if self._hidden_cache is None or self._pooler_out_cache is None:
            raise RuntimeError("Forward must be called before backward.")
        grad_pooler = self.nsp_head.backward(grad_nsp_logits, lr=lr)
        grad_hidden_from_pooler = self.pooler.backward(grad_pooler, lr=lr)
        grad_hidden_from_mlm = self.mlm_head.backward(
            grad_mlm_logits, lr=lr
        )
        grad_hidden = grad_hidden_from_pooler + grad_hidden_from_mlm
        grad_enc = self.encoder.backward(grad_hidden, lr=lr)
        self.embeddings.backward_embed(grad_enc, lr=lr)

    @staticmethod
    def mlm_loss_and_grad(
        logits: np.ndarray, labels: np.ndarray, mask: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Cross-entropy over masked positions only. labels (batch, num_masked)."""
        batch_size, num_masked, vocab_size = logits.shape
        probs = _softmax(logits, axis=-1)
        labels_clipped = np.clip(labels, 0, vocab_size - 1).astype(np.int64)
        one_hot = np.eye(vocab_size, dtype=np.float32)[labels_clipped]
        log_probs = np.log(probs + 1e-12)
        loss_per_pos = -np.sum(one_hot * log_probs, axis=-1)
        loss_per_pos = loss_per_pos * mask
        n_valid = max(1, np.sum(mask))
        loss = float(np.sum(loss_per_pos) / n_valid)
        grad_logits = (probs - one_hot) * mask[:, :, None]
        grad_logits = grad_logits / n_valid
        return loss, grad_logits

    @staticmethod
    def nsp_loss_and_grad(
        logits: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Cross-entropy for binary NSP labels."""
        probs = _softmax(logits, axis=1)
        batch_size = logits.shape[0]
        one_hot = np.eye(2, dtype=np.float32)[labels]
        loss = -float(np.sum(one_hot * np.log(probs + 1e-12)) / batch_size)
        grad_logits = (probs - one_hot) / float(batch_size)
        return loss, grad_logits


def create_mlm_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a batch for MLM+NSP. All sequences are single-segment for simplicity.

    Returns:
        input_ids: (batch, seq_len) with [CLS] ... [SEP], some tokens replaced by MASK_ID.
        segment_ids: (batch, seq_len) all zeros.
        masked_positions: (batch, max_masked) indices of masked positions, -1 padding.
        mlm_labels: (batch, max_masked) true token id at each masked position, -1 padding.
        nsp_labels: (batch,) binary 0/1 (we use 1 for "same segment" for this synthetic setup).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    first_real = 4
    nsp_labels = np.ones((batch_size,), dtype=np.int64)

    input_ids = np.full((batch_size, seq_len), PAD_ID, dtype=np.int32)
    input_ids[:, 0] = CLS_ID
    input_ids[:, -1] = SEP_ID
    for b in range(batch_size):
        for s in range(1, seq_len - 1):
            input_ids[b, s] = np.random.randint(first_real, vocab_size)

    num_masked_max = 0
    masked_positions_list: List[List[int]] = []
    mlm_labels_list: List[List[int]] = []

    for b in range(batch_size):
        positions = []
        labels = []
        for s in range(1, seq_len - 1):
            if np.random.rand() < mask_prob:
                positions.append(s)
                labels.append(int(input_ids[b, s]))
                input_ids[b, s] = MASK_ID
        num_masked_max = max(num_masked_max, len(positions))
        masked_positions_list.append(positions)
        mlm_labels_list.append(labels)

    max_masked = num_masked_max if num_masked_max > 0 else 1
    masked_positions = np.full((batch_size, max_masked), -1, dtype=np.int32)
    mlm_labels = np.full((batch_size, max_masked), -1, dtype=np.int64)
    for b in range(batch_size):
        for i, (pos, lab) in enumerate(zip(masked_positions_list[b], mlm_labels_list[b])):
            masked_positions[b, i] = pos
            mlm_labels[b, i] = lab

    segment_ids = np.zeros((batch_size, seq_len), dtype=np.int32)
    return input_ids, segment_ids, masked_positions, mlm_labels, nsp_labels


def create_nsp_pair_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create batch with [CLS] sent_a [SEP] sent_b [SEP]. NSP label 1 = next, 0 = random.

    When label is 1, segment B is a copy of segment A so the model can learn
    to predict "next". When label is 0, segment B is random.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    first_real = 4
    half_len = (seq_len - 3) // 2

    input_ids = np.full((batch_size, seq_len), PAD_ID, dtype=np.int32)
    segment_ids = np.zeros((batch_size, seq_len), dtype=np.int32)
    input_ids[:, 0] = CLS_ID
    input_ids[:, half_len + 1] = SEP_ID
    segment_ids[:, half_len + 2 :] = 1
    input_ids[:, -1] = SEP_ID

    nsp_labels = np.random.randint(0, 2, size=(batch_size,), dtype=np.int64)
    for b in range(batch_size):
        for s in range(1, half_len + 1):
            input_ids[b, s] = np.random.randint(first_real, vocab_size)
        if nsp_labels[b] == 1:
            for s in range(half_len + 2, seq_len - 1):
                src_s = 1 + (s - (half_len + 2)) % (half_len)
                input_ids[b, s] = input_ids[b, src_s]
        else:
            for s in range(half_len + 2, seq_len - 1):
                input_ids[b, s] = np.random.randint(first_real, vocab_size)

    masked_positions = np.full((batch_size, 1), -1, dtype=np.int32)
    mlm_labels = np.full((batch_size, 1), -1, dtype=np.int64)
    return input_ids, segment_ids, masked_positions, mlm_labels, nsp_labels


def generate_synthetic_bert_batches(
    num_batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    random_seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Generate list of (input_ids, segment_ids, masked_positions, mlm_labels, nsp_labels)."""
    batches = []
    rng = np.random.default_rng(random_seed)
    for _ in range(num_batches):
        use_mlm = rng.random() > 0.5
        seed = int(rng.integers(0, 2**31))
        if use_mlm:
            batches.append(
                create_mlm_batch(
                    batch_size, seq_len, vocab_size, mask_prob, seed
                )
            )
        else:
            batches.append(
                create_nsp_pair_batch(batch_size, seq_len, vocab_size, seed)
            )
    return batches


class BertRunner:
    """Train and evaluate BERT-like model from config."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Config not found: %s, using defaults", config_path)
            return {}

    def _setup_logging(self) -> None:
        log_cfg = self.config.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO"))
        log_file = log_cfg.get("file", "logs/app.log")
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.setLevel(level)
        logger.addHandler(handler)

    def run(self) -> Dict[str, float]:
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        batch_size = data_cfg.get("batch_size", 8)
        seq_len = data_cfg.get("sequence_length", 32)
        vocab_size = data_cfg.get("vocab_size", 64)
        mask_prob = data_cfg.get("mask_probability", 0.15)
        num_train_batches = data_cfg.get("num_train_batches", 100)
        num_test_batches = data_cfg.get("num_test_batches", 20)
        random_seed = data_cfg.get("random_seed", 42)

        dim_model = model_cfg.get("dim_model", 32)
        num_heads = model_cfg.get("num_heads", 4)
        dim_ff = model_cfg.get("dim_ff", 64)
        num_layers = model_cfg.get("num_layers", 2)

        epochs = train_cfg.get("epochs", 5)
        learning_rate = train_cfg.get("learning_rate", 0.001)
        mlm_weight = train_cfg.get("mlm_loss_weight", 1.0)
        nsp_weight = train_cfg.get("nsp_loss_weight", 1.0)

        model = BertModel(
            vocab_size=vocab_size,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_ff=dim_ff,
            num_layers=num_layers,
            max_seq_len=seq_len,
        )

        train_batches = generate_synthetic_bert_batches(
            num_batches=num_train_batches,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            mask_prob=mask_prob,
            random_seed=random_seed,
        )

        for epoch in range(epochs):
            total_mlm_loss = 0.0
            total_nsp_loss = 0.0
            n_mlm = 0
            n_nsp = 0
            for (
                input_ids,
                segment_ids,
                masked_positions,
                mlm_labels,
                nsp_labels,
            ) in train_batches:
                mlm_logits, nsp_logits = model.forward(
                    input_ids, segment_ids, masked_positions
                )
                mlm_mask = (masked_positions >= 0).astype(np.float32)
                if np.sum(mlm_mask) > 0:
                    mlm_loss, grad_mlm = model.mlm_loss_and_grad(
                        mlm_logits, mlm_labels, mlm_mask
                    )
                    total_mlm_loss += mlm_loss
                    n_mlm += 1
                else:
                    grad_mlm = np.zeros_like(mlm_logits)
                nsp_loss, grad_nsp = model.nsp_loss_and_grad(nsp_logits, nsp_labels)
                total_nsp_loss += nsp_loss
                n_nsp += 1
                model.backward(
                    grad_mlm * mlm_weight,
                    grad_nsp * nsp_weight,
                    lr=learning_rate,
                )
            avg_mlm = total_mlm_loss / max(1, n_mlm)
            avg_nsp = total_nsp_loss / max(1, n_nsp)
            if (epoch + 1) % max(1, epochs // 5) == 0:
                logger.info(
                    "Epoch %d/%d - mlm_loss: %.4f, nsp_loss: %.4f",
                    epoch + 1,
                    epochs,
                    avg_mlm,
                    avg_nsp,
                )

        test_batches = generate_synthetic_bert_batches(
            num_batches=num_test_batches,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            mask_prob=mask_prob,
            random_seed=random_seed + 9999,
        )
        test_mlm_loss = 0.0
        test_nsp_loss = 0.0
        n_mlm = 0
        n_nsp = 0
        nsp_correct = 0
        nsp_total = 0
        for (
            input_ids,
            segment_ids,
            masked_positions,
            mlm_labels,
            nsp_labels,
        ) in test_batches:
            mlm_logits, nsp_logits = model.forward(
                input_ids, segment_ids, masked_positions
            )
            mlm_mask = (masked_positions >= 0).astype(np.float32)
            if np.sum(mlm_mask) > 0:
                mlm_loss, _ = model.mlm_loss_and_grad(
                    mlm_logits, mlm_labels, mlm_mask
                )
                test_mlm_loss += mlm_loss
                n_mlm += 1
            nsp_loss, _ = model.nsp_loss_and_grad(nsp_logits, nsp_labels)
            test_nsp_loss += nsp_loss
            n_nsp += 1
            preds = np.argmax(nsp_logits, axis=1)
            nsp_correct += int(np.sum(preds == nsp_labels))
            nsp_total += nsp_logits.shape[0]

        results = {
            "train_mlm_loss": float(avg_mlm),
            "train_nsp_loss": float(avg_nsp),
            "test_mlm_loss": float(test_mlm_loss / max(1, n_mlm)),
            "test_nsp_loss": float(test_nsp_loss / max(1, n_nsp)),
            "test_nsp_accuracy": float(nsp_correct / max(1, nsp_total)),
        }
        logger.info(
            "Final - test_mlm_loss: %.4f, test_nsp_loss: %.4f, nsp_acc: %.4f",
            results["test_mlm_loss"],
            results["test_nsp_loss"],
            results["test_nsp_accuracy"],
        )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BERT-like model with MLM and NSP from scratch"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = BertRunner(config_path=Path(args.config) if args.config else None)
    results = runner.run()

    print("\nFinal Results:")
    print("=" * 40)
    for key, val in results.items():
        print(f"  {key}: {val:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
