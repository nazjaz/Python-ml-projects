"""Tests for neural style transfer (content and style loss optimization)."""

import numpy as np
import pytest

from src.main import (
    FeatureExtractor,
    StyleTransferConfig,
    content_loss,
    content_loss_grad,
    gram_matrix,
    gram_matrix_backward,
    load_content_style_images,
    run_pipeline,
    run_style_transfer,
    style_loss,
    style_loss_grad,
)


class TestGramMatrix:
    def test_gram_shape(self) -> None:
        f = np.random.randn(4, 4, 8).astype(np.float32)
        g = gram_matrix(f)
        assert g.shape == (8, 8)

    def test_gram_symmetric(self) -> None:
        f = np.random.randn(3, 3, 4).astype(np.float32)
        g = gram_matrix(f)
        assert np.allclose(g, g.T)


class TestGramBackward:
    def test_grad_shape(self) -> None:
        f = np.random.randn(2, 2, 3).astype(np.float32)
        g = gram_matrix(f)
        grad_g = np.ones_like(g)
        grad_f = gram_matrix_backward(grad_g, f)
        assert grad_f.shape == f.shape


class TestFeatureExtractor:
    def test_forward_shape(self) -> None:
        ext = FeatureExtractor(in_channels=1, out_channels=8, random_seed=0)
        x = np.random.randn(8, 8, 1).astype(np.float32) * 0.1
        out = ext.forward(x)
        assert out.shape == (8, 8, 8)

    def test_backward_shape(self) -> None:
        ext = FeatureExtractor(in_channels=1, out_channels=4, random_seed=0)
        x = np.random.randn(6, 6, 1).astype(np.float32) * 0.1
        ext.forward(x)
        grad_out = np.ones((6, 6, 4), dtype=np.float32)
        grad_in = ext.backward(grad_out)
        assert grad_in.shape == x.shape


class TestContentStyleLoss:
    def test_content_loss_non_negative(self) -> None:
        a = np.random.randn(2, 2, 4).astype(np.float32)
        b = np.random.randn(2, 2, 4).astype(np.float32)
        assert content_loss(a, b) >= 0

    def test_style_loss_non_negative(self) -> None:
        g1 = np.random.randn(4, 4).astype(np.float32)
        g2 = np.random.randn(4, 4).astype(np.float32)
        assert style_loss(g1, g2) >= 0

    def test_content_loss_grad_shape(self) -> None:
        a = np.random.randn(2, 2, 3).astype(np.float32)
        b = np.random.randn(2, 2, 3).astype(np.float32)
        g = content_loss_grad(a, b)
        assert g.shape == a.shape

    def test_style_loss_grad_shape(self) -> None:
        g1 = np.random.randn(3, 3).astype(np.float32)
        g2 = np.random.randn(3, 3).astype(np.float32)
        grad = style_loss_grad(g1, g2)
        assert grad.shape == g1.shape


class TestLoadContentStyle:
    def test_shapes(self) -> None:
        c, s = load_content_style_images(size=8, random_seed=0)
        assert c.shape == (8, 8, 1)
        assert s.shape == (8, 8, 1)
        assert np.all(c >= -1.0) and np.all(c <= 1.0)


class TestRunStyleTransfer:
    def test_returns_image_and_history(self) -> None:
        c, s = load_content_style_images(size=8, random_seed=0)
        ext = FeatureExtractor(out_channels=8, random_seed=0)
        gen, hist = run_style_transfer(
            c, s, ext, num_steps=5,
            content_weight=1.0, style_weight=100.0, lr=1.0, random_seed=0,
        )
        assert gen.shape == c.shape
        assert len(hist) == 5
        assert "total" in hist[0]


class TestRunPipeline:
    def test_returns_metrics(self) -> None:
        cfg = StyleTransferConfig(num_steps=5, random_seed=0)
        results = run_pipeline(cfg)
        assert "final_content_loss" in results
        assert "final_style_loss" in results
        assert "final_total_loss" in results
        assert "generated_shape" in results
