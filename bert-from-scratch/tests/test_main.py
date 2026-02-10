"""Tests for BERT-like model with MLM and NSP."""

import numpy as np
import pytest

from src.main import (
    PAD_ID,
    CLS_ID,
    SEP_ID,
    MASK_ID,
    BertModel,
    create_mlm_batch,
    create_nsp_pair_batch,
    generate_synthetic_bert_batches,
)


class TestConstants:
    """Special token ids."""

    def test_special_token_ids_distinct(self) -> None:
        assert PAD_ID != CLS_ID != SEP_ID != MASK_ID


class TestMLMBatch:
    """Tests for MLM batch creation."""

    def test_create_mlm_batch_shapes(self) -> None:
        input_ids, segment_ids, masked_positions, mlm_labels, nsp_labels = (
            create_mlm_batch(
                batch_size=4,
                seq_len=16,
                vocab_size=32,
                mask_prob=0.2,
                random_seed=42,
            )
        )
        assert input_ids.shape == (4, 16)
        assert segment_ids.shape == (4, 16)
        assert input_ids[:, 0].tolist() == [CLS_ID] * 4
        assert input_ids[:, -1].tolist() == [SEP_ID] * 4
        assert nsp_labels.shape == (4,)

    def test_create_mlm_batch_has_masks(self) -> None:
        input_ids, _, masked_positions, mlm_labels, _ = create_mlm_batch(
            batch_size=8,
            seq_len=20,
            vocab_size=40,
            mask_prob=0.25,
            random_seed=7,
        )
        assert np.any(input_ids == MASK_ID)
        assert masked_positions.shape[1] >= 1
        assert np.any(masked_positions >= 0)


class TestNSPBatch:
    """Tests for NSP pair batch creation."""

    def test_create_nsp_pair_batch_shapes(self) -> None:
        input_ids, segment_ids, masked_positions, mlm_labels, nsp_labels = (
            create_nsp_pair_batch(
                batch_size=4,
                seq_len=20,
                vocab_size=32,
                random_seed=42,
            )
        )
        assert input_ids.shape == (4, 20)
        assert segment_ids.shape == (4, 20)
        assert nsp_labels.shape == (4,)
        assert np.all(np.isin(nsp_labels, [0, 1]))

    def test_nsp_label_one_has_matching_segments(self) -> None:
        input_ids, _, _, _, nsp_labels = create_nsp_pair_batch(
            batch_size=20,
            seq_len=24,
            vocab_size=32,
            random_seed=123,
        )
        half_len = (24 - 3) // 2
        for b in range(20):
            if nsp_labels[b] == 1:
                for s in range(half_len + 2, 23):
                    src_s = 1 + (s - (half_len + 2)) % half_len
                    assert input_ids[b, s] == input_ids[b, src_s]


class TestBertModel:
    """Tests for BERT model forward and loss."""

    def test_forward_output_shapes(self) -> None:
        np.random.seed(0)
        batch_size = 2
        seq_len = 12
        vocab_size = 32
        model = BertModel(
            vocab_size=vocab_size,
            dim_model=16,
            num_heads=4,
            dim_ff=32,
            num_layers=1,
            max_seq_len=seq_len,
        )
        input_ids, segment_ids, masked_positions, _, _ = create_mlm_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            mask_prob=0.2,
            random_seed=1,
        )
        mlm_logits, nsp_logits = model.forward(
            input_ids, segment_ids, masked_positions
        )
        assert mlm_logits.shape[0] == batch_size
        assert mlm_logits.shape[2] == vocab_size
        assert nsp_logits.shape == (batch_size, 2)

    def test_mlm_loss_and_grad_finite(self) -> None:
        np.random.seed(2)
        batch_size = 4
        num_masked = 3
        vocab_size = 20
        logits = np.random.randn(batch_size, num_masked, vocab_size).astype(
            np.float32
        ) * 0.1
        labels = np.random.randint(0, vocab_size, (batch_size, num_masked))
        mask = np.ones((batch_size, num_masked), dtype=np.float32)
        loss, grad = BertModel.mlm_loss_and_grad(logits, labels, mask)
        assert np.isfinite(loss)
        assert np.all(np.isfinite(grad))

    def test_nsp_loss_and_grad_finite(self) -> None:
        np.random.seed(3)
        batch_size = 4
        logits = np.random.randn(batch_size, 2).astype(np.float32) * 0.1
        labels = np.random.randint(0, 2, (batch_size,))
        loss, grad = BertModel.nsp_loss_and_grad(logits, labels)
        assert np.isfinite(loss)
        assert np.all(np.isfinite(grad))

    def test_backward_updates_parameters(self) -> None:
        np.random.seed(4)
        model = BertModel(
            vocab_size=24,
            dim_model=12,
            num_heads=4,
            dim_ff=24,
            num_layers=1,
            max_seq_len=16,
        )
        input_ids, segment_ids, masked_positions, mlm_labels, nsp_labels = (
            create_mlm_batch(2, 16, 24, 0.2, random_seed=4)
        )
        mlm_mask = (masked_positions >= 0).astype(np.float32)
        w_before = model.mlm_head.w.copy()
        mlm_logits, nsp_logits = model.forward(
            input_ids, segment_ids, masked_positions
        )
        if np.sum(mlm_mask) > 0:
            _, grad_mlm = model.mlm_loss_and_grad(
                mlm_logits, mlm_labels, mlm_mask
            )
        else:
            grad_mlm = np.zeros_like(mlm_logits)
        _, grad_nsp = model.nsp_loss_and_grad(nsp_logits, nsp_labels)
        model.backward(grad_mlm, grad_nsp, lr=0.01)
        assert not np.allclose(model.mlm_head.w, w_before)


class TestSyntheticBatches:
    """Tests for batch generator."""

    def test_generate_synthetic_bert_batches_length(self) -> None:
        batches = generate_synthetic_bert_batches(
            num_batches=5,
            batch_size=4,
            seq_len=16,
            vocab_size=32,
            random_seed=10,
        )
        assert len(batches) == 5
        for (input_ids, segment_ids, masked_positions, mlm_labels, nsp_labels) in batches:
            assert input_ids.shape == (4, 16)
            assert segment_ids.shape == (4, 16)
            assert nsp_labels.shape == (4,)
