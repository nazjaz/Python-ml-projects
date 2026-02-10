"""Semantic segmentation using U-Net with encoder-decoder and skip connections.

This module implements a U-Net architecture: an encoder downsamples the input
via repeated conv blocks and max pooling, a bottleneck captures high-level
features, and a decoder upsamples with skip connections from the encoder
to restore spatial resolution. The final layer outputs per-pixel class logits
for semantic segmentation.
"""

import argparse
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _setup_logging(level: str, log_file: Optional[str]) -> None:
    """Configure logging to console and optionally to file.

    Args:
        level: Log level (e.g. INFO, DEBUG).
        log_file: Optional path to log file.
    """
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


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU (no down/upsample)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize double convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Spatial dimensions preserved (with padding=1)."""
        return self.block(x)


class EncoderBlock(nn.Module):
    """One encoder stage: DoubleConv then MaxPool. Exposes pre-pool output for skip."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize encoder block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels after conv (and after pool).
        """
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            pooled: Output after MaxPool (for next encoder level).
            skip: Output after DoubleConv (for decoder skip connection).
        """
        skip = self.conv(x)
        pooled = self.pool(skip)
        return pooled, skip


class DecoderBlock(nn.Module):
    """One decoder stage: upsample, concatenate skip, then DoubleConv."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize decoder block.

        Args:
            in_channels: Channels from previous decoder (or bottleneck).
            skip_channels: Channels from encoder skip (concatenated).
            out_channels: Output channels after DoubleConv.
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: upsample x, concat with skip, then DoubleConv.

        Skip is cropped to match upsampled spatial size if needed.
        """
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for semantic segmentation: encoder, bottleneck, decoder, skip connections."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 64,
        depth: int = 4,
    ) -> None:
        """Initialize U-Net.

        Args:
            in_channels: Input image channels (e.g. 3 for RGB).
            num_classes: Number of segmentation classes (output channels).
            base_channels: Channels at first encoder level; doubled each level.
            depth: Number of encoder/decoder levels (excluding bottleneck).
        """
        super().__init__()
        self.depth = depth
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Encoder: each level doubles channels and halves spatial size
        enc_ch = [in_channels] + [base_channels * (2 ** i) for i in range(depth)]
        for i in range(depth):
            self.encoder_blocks.append(
                EncoderBlock(enc_ch[i], enc_ch[i + 1])
            )

        # Bottleneck: same spatial size as encoder output, double channels
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = DoubleConv(
            enc_ch[depth],
            bottleneck_ch,
        )

        # Decoder: each level halves channels and doubles spatial size
        dec_in = bottleneck_ch
        for i in range(depth):
            skip_ch = enc_ch[depth - i]
            out_ch = enc_ch[depth - 1 - i] if i < depth - 1 else base_channels
            self.decoder_blocks.append(
                DecoderBlock(dec_in, skip_ch, out_ch)
            )
            dec_in = out_ch

        self.final = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (N, in_channels, H, W).

        Returns:
            Logits (N, num_classes, H, W). Spatial size matches input if H,W divisible by 2^depth.
        """
        skips: List[torch.Tensor] = []
        h = x
        for block in self.encoder_blocks:
            h, skip = block(h)
            skips.append(skip)
        h = self.bottleneck(h)
        for i, block in enumerate(self.decoder_blocks):
            h = block(h, skips[-(i + 1)])
        return self.final(h)


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy loss for semantic segmentation.

    Args:
        logits: (N, num_classes, H, W) model output.
        targets: (N, H, W) integer class indices in [0, num_classes-1] or ignore_index.
        num_classes: Number of classes (unused; for API consistency).
        ignore_index: Target value to ignore in loss (default -100).

    Returns:
        Scalar loss tensor.
    """
    del num_classes
    return F.cross_entropy(
        logits,
        targets.long(),
        ignore_index=ignore_index,
    )


def generate_synthetic_batch(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    num_classes: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of random images and random per-pixel labels.

    Args:
        batch_size: N.
        channels: Image channels.
        height, width: Spatial size.
        num_classes: Number of segmentation classes.
        device: Torch device.
        seed: Optional random seed.

    Returns:
        images: (N, C, H, W).
        labels: (N, H, W) integer class indices in [0, num_classes-1].
    """
    if seed is not None:
        torch.manual_seed(seed)
    images = torch.rand(batch_size, channels, height, width, device=device)
    labels = torch.randint(
        0, num_classes, (batch_size, height, width), device=device
    )
    return images, labels


def run_training(config: Dict[str, Any]) -> None:
    """Train U-Net on synthetic data according to config.

    Args:
        config: Full config dict with model, training, data, logging.
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    net = UNet(
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=model_cfg.get("num_classes", 2),
        base_channels=model_cfg.get("base_channels", 64),
        depth=model_cfg.get("depth", 4),
    ).to(device)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
    )
    epochs = train_cfg.get("epochs", 20)
    batch_size = train_cfg.get("batch_size", 8)
    num_train = data_cfg.get("num_train", 400)
    seed = data_cfg.get("random_seed", 42)
    height = data_cfg.get("image_height", 128)
    width = data_cfg.get("image_width", 128)
    num_classes = model_cfg.get("num_classes", 2)

    steps_per_epoch = max(1, num_train // batch_size)
    logger.info(
        "Training for %d epochs, %d steps/epoch, batch_size=%d",
        epochs, steps_per_epoch, batch_size,
    )

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        for step in range(steps_per_epoch):
            images, labels = generate_synthetic_batch(
                batch_size, 3, height, width, num_classes, device,
                seed=seed + epoch * 1000 + step,
            )
            optimizer.zero_grad()
            logits = net(images)
            loss = segmentation_loss(logits, labels, num_classes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / steps_per_epoch
        logger.info("Epoch %d/%d loss: %.4f", epoch + 1, epochs, avg)


def main() -> None:
    """Entry point: load config, set up logging, run training."""
    parser = argparse.ArgumentParser(
        description="U-Net semantic segmentation with encoder-decoder and skip connections."
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

    logger.info("Starting U-Net semantic segmentation")
    run_training(config)
    logger.info("Done")


if __name__ == "__main__":
    main()
