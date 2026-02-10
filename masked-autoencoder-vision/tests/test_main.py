"""Tests for masked autoencoder vision module."""

import tempfile
from pathlib import Path

import torch

from src.main import (
    MaskedAutoencoder,
    _load_config,
    generate_synthetic_images,
    patchify,
    unpatchify,
)


class TestLoadConfig:
    """Test cases for _load_config."""

    def test_load_config_returns_dict(self) -> None:
        """Test that _load_config returns a dict with expected keys."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  image_size: 32\n")
            path = f.name
        try:
            cfg = _load_config(path)
            assert isinstance(cfg, dict)
            assert "model" in cfg
            assert cfg["model"]["image_size"] == 32
        finally:
            Path(path).unlink(missing_ok=True)


class TestPatchifyUnpatchify:
    """Test cases for patchify and unpatchify."""

    def test_patchify_and_unpatchify_inverse(self) -> None:
        """Test that unpatchify(patchify(x)) reconstructs original images."""
        images = torch.rand(2, 3, 32, 32)
        patches = patchify(images, patch_size=4)
        recon = unpatchify(patches, patch_size=4, channels=3, height=32, width=32)
        assert torch.allclose(images, recon, atol=1e-6)

    def test_patchify_shape(self) -> None:
        """Test that patchify returns correct patch shape."""
        images = torch.rand(1, 3, 16, 16)
        patches = patchify(images, patch_size=4)
        assert patches.shape == (1, (16 // 4) * (16 // 4), 3 * 4 * 4)


class TestMAEModel:
    """Test cases for MaskedAutoencoder."""

    def test_forward_returns_loss_and_reconstruction(self) -> None:
        """Test that MAE forward returns a scalar loss and reconstructed images."""
        device = torch.device("cpu")
        model = MaskedAutoencoder(
            image_size=16,
            patch_size=4,
            in_channels=3,
            embed_dim=32,
            encoder_layers=2,
            decoder_layers=1,
            mask_ratio=0.5,
        ).to(device)
        images = torch.rand(4, 3, 16, 16, device=device)
        loss, recon = model(images)
        assert loss.dim() == 0
        assert recon.shape == images.shape

    def test_mask_ratio_effect(self) -> None:
        """Test that changing mask_ratio does not break forward pass."""
        device = torch.device("cpu")
        for mask_ratio in (0.25, 0.5, 0.75):
            model = MaskedAutoencoder(
                image_size=16,
                patch_size=4,
                in_channels=3,
                embed_dim=16,
                encoder_layers=1,
                decoder_layers=1,
                mask_ratio=mask_ratio,
            ).to(device)
            images = torch.rand(2, 3, 16, 16, device=device)
            loss, _ = model(images)
            assert loss.item() >= 0.0


class TestSyntheticImages:
    """Test cases for generate_synthetic_images."""

    def test_image_shapes(self) -> None:
        """Test that synthetic images have correct shapes."""
        device = torch.device("cpu")
        images = generate_synthetic_images(
            num_images=5,
            channels=3,
            height=16,
            width=16,
            device=device,
            seed=123,
        )
        assert images.shape == (5, 3, 16, 16)

