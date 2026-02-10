"""Masked autoencoder (MAE) for self-supervised vision learning.

This module implements a simple MAE-style model: images are split into patches,
a random subset of patches is masked, and the model learns to reconstruct
the original pixels of masked patches. The encoder operates on visible
patch tokens, and the decoder reconstructs all patches.
"""

import argparse
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _setup_logging(level: str, log_file: Optional[str]) -> None:
    """Configure logging to console and optionally to file."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        )
        logging.getLogger().addHandler(file_handler)


def patchify(
    images: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Convert images into a sequence of flattened patches.

    Args:
        images: Tensor of shape (N, C, H, W).
        patch_size: Patch side length; must divide H and W.

    Returns:
        Patches of shape (N, L, C * patch_size * patch_size) where
        L = (H / patch_size) * (W / patch_size).
    """
    n, c, h, w = images.shape
    assert h % patch_size == 0 and w % patch_size == 0
    ph = h // patch_size
    pw = w // patch_size
    x = images.reshape(n, c, ph, patch_size, pw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = x.reshape(n, ph * pw, c * patch_size * patch_size)
    return patches


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reconstruct images from flattened patches."""
    n, l, dim = patches.shape
    ph = height // patch_size
    pw = width // patch_size
    assert l == ph * pw
    x = patches.reshape(n, ph, pw, channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    images = x.reshape(n, channels, height, width)
    return images


class MaskedAutoencoder(nn.Module):
    """Simple transformer-based masked autoencoder for image patches."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        encoder_layers: int,
        decoder_layers: int,
        mask_ratio: float,
    ) -> None:
        """Initialize MAE model."""
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers
        )

        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_layers
        )
        self.patch_reconstruct = nn.Linear(embed_dim, self.patch_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _random_mask(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random patch masks for each image.

        Returns:
            ids_keep: Indices of visible patches (N, L_keep).
            ids_mask: Indices of masked patches (N, L_mask).
            bool_mask: Boolean mask over L patches (N, L) where True=masked.
        """
        l = self.num_patches
        num_mask = int(self.mask_ratio * l)
        noise = torch.rand(n, l, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_mask = ids_shuffle[:, :num_mask]
        ids_keep = ids_shuffle[:, num_mask:]
        bool_mask = torch.zeros(n, l, dtype=torch.bool, device=device)
        bool_mask.scatter_(1, ids_mask, True)
        return ids_keep, ids_mask, bool_mask

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: returns reconstruction loss and reconstructed images.

        Loss is mean squared error on masked patches only.
        """
        device = images.device
        patches = patchify(images, self.patch_size)
        n, l, dim = patches.shape
        ids_keep, ids_mask, bool_mask = self._random_mask(n, device)

        x = self.patch_embed(patches) + self.pos_embed
        # Gather only visible tokens per sample.
        batch_indices = torch.arange(n, device=device).unsqueeze(-1)
        x_vis = x[batch_indices, ids_keep]
        x_vis = x_vis.transpose(0, 1)
        encoded = self.encoder(x_vis).transpose(0, 1)

        # Prepare decoder input: place encoded tokens and mask tokens.
        dec_tokens = self.decoder_embed(encoded)
        l_keep = dec_tokens.size(1)
        l_mask = l - l_keep

        mask_tokens = self.mask_token.expand(n, l_mask, -1)
        full_tokens = torch.zeros(n, l, dec_tokens.size(2), device=device)
        full_tokens[batch_indices, ids_keep] = dec_tokens
        full_tokens[batch_indices, ids_mask] = mask_tokens
        full_tokens = full_tokens + self.decoder_pos_embed

        full_tokens = full_tokens.transpose(0, 1)
        decoded = self.decoder(full_tokens).transpose(0, 1)
        pred_patches = self.patch_reconstruct(decoded)

        target = patches
        mask = bool_mask.unsqueeze(-1).type_as(pred_patches)
        loss = F.mse_loss(pred_patches * mask, target * mask)

        recon_images = unpatchify(
            pred_patches, self.patch_size, self.in_channels, self.image_size, self.image_size
        )
        return loss, recon_images


def generate_synthetic_images(
    num_images: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate random images for self-supervised training."""
    if seed is not None:
        torch.manual_seed(seed)
    images = torch.rand(num_images, channels, height, width, device=device)
    return images


def run_training(config: Dict[str, Any]) -> None:
    """Run MAE self-supervised training on synthetic images."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    image_size = model_cfg.get("image_size", 32)
    patch_size = model_cfg.get("patch_size", 4)
    in_channels = model_cfg.get("in_channels", 3)
    embed_dim = model_cfg.get("embed_dim", 64)
    encoder_layers = model_cfg.get("encoder_layers", 4)
    decoder_layers = model_cfg.get("decoder_layers", 2)
    mask_ratio = float(model_cfg.get("mask_ratio", 0.75))

    model = MaskedAutoencoder(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        mask_ratio=mask_ratio,
    ).to(device)

    epochs = train_cfg.get("epochs", 20)
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 0.001)
    num_train = data_cfg.get("num_train", 1000)
    seed = data_cfg.get("random_seed", 42)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    images = generate_synthetic_images(
        num_images=num_train,
        channels=in_channels,
        height=image_size,
        width=image_size,
        device=device,
        seed=seed,
    )

    steps_per_epoch = max(1, num_train // batch_size)
    logger.info(
        "Training MAE for %d epochs, %d steps/epoch, batch_size=%d",
        epochs,
        steps_per_epoch,
        batch_size,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            batch = images[start:end]
            optimizer.zero_grad()
            loss, _ = model(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / steps_per_epoch
        logger.info("Epoch %d/%d MAE reconstruction loss: %.4f", epoch + 1, epochs, avg)


def main() -> None:
    """Entry point: load config, set up logging, run MAE training."""
    parser = argparse.ArgumentParser(
        description="Masked autoencoder (MAE) self-supervised vision training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    log_cfg = config.get("logging", {})
    _setup_logging(
        log_cfg.get("level", "INFO"),
        log_cfg.get("file"),
    )

    logger.info("Starting MAE self-supervised training")
    run_training(config)
    logger.info("Done")


if __name__ == "__main__":
    main()

