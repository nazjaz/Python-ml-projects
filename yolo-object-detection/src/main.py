"""YOLO object detection with bounding box regression and class prediction.

This module implements a YOLO-style architecture: a convolutional backbone
extracts features, and a detection head predicts per grid cell bounding box
offsets (tx, ty, tw, th), objectness, and class logits. Bounding boxes are
decoded from grid coordinates and post-processed with non-maximum suppression.
"""

import argparse
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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


class ConvBlock(nn.Module):
    """Conv2d, BatchNorm2d, and LeakyReLU block for backbone feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
    ) -> None:
        """Initialize convolution block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Convolution kernel size.
            stride: Stride (default 1).
            padding: Padding; defaults to kernel_size // 2 when stride == 1.
        """
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.act(self.bn(self.conv(x)))


class Backbone(nn.Module):
    """Darknet-style backbone: stacked ConvBlocks with downsampling."""

    def __init__(
        self,
        in_channels: int = 3,
        channel_list: List[int] = (32, 64, 128, 256, 512),
        block_repeats: List[int] = (1, 2, 2, 2),
    ) -> None:
        """Build backbone with channel_list and block_repeats.

        Args:
            in_channels: Input image channels (e.g. 3 for RGB).
            channel_list: Channel sizes per stage.
            block_repeats: Number of ConvBlocks per stage (first is stride 2).
        """
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_channels
        for i, (ch, reps) in enumerate(zip(channel_list, block_repeats)):
            layers.append(ConvBlock(prev, ch, kernel_size=3, stride=2, padding=1))
            prev = ch
            for _ in range(reps - 1):
                layers.append(ConvBlock(prev, ch, kernel_size=3, stride=1))
                prev = ch
        self.net = nn.Sequential(*layers)
        self.out_channels = channel_list[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; output shape (N, out_channels, H, W)."""
        return self.net(x)


class YOLODetectionHead(nn.Module):
    """YOLO detection head: one prediction per grid cell per anchor.

    For each cell predicts num_anchors * (5 + num_classes): tx, ty, tw, th,
    objectness, and num_classes class logits. Bounding box regression uses
    center offsets (tx, ty) and log-scale (tw, th) relative to cell/anchor.
    """

    def __init__(
        self,
        in_channels: int,
        grid_size: int,
        num_anchors: int,
        num_classes: int,
    ) -> None:
        """Initialize detection head.

        Args:
            in_channels: Backbone feature channels.
            grid_size: Spatial grid size (S x S).
            num_anchors: Number of boxes per cell (B).
            num_classes: Number of class labels (C).
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Per anchor: 4 bbox + 1 objectness + num_classes
        out_per_cell = num_anchors * (5 + num_classes)
        self.conv = nn.Conv2d(in_channels, out_per_cell, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Backbone features (N, in_channels, grid_size, grid_size).

        Returns:
            Predictions (N, grid_size, grid_size, num_anchors, 5 + num_classes).
        """
        n, c, h, w = x.shape
        out = self.conv(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(
            n, h, w, self.num_anchors, 5 + self.num_classes
        )
        return out


class YOLONet(nn.Module):
    """Full YOLO model: backbone plus detection head with bbox and class output."""

    def __init__(
        self,
        in_channels: int = 3,
        grid_size: int = 13,
        num_anchors: int = 5,
        num_classes: int = 20,
        backbone_channels: List[int] = (32, 64, 128, 256, 512),
        backbone_blocks: List[int] = (1, 2, 2, 2, 2),
    ) -> None:
        """Initialize YOLO network.

        Args:
            in_channels: Input image channels.
            grid_size: Detection grid size S.
            num_anchors: Anchors per cell B.
            num_classes: Number of classes C.
            backbone_channels: Channel list for backbone.
            backbone_blocks: Block repeats per stage.
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.backbone = Backbone(
            in_channels, backbone_channels, backbone_blocks
        )
        self.head = YOLODetectionHead(
            self.backbone.out_channels,
            grid_size,
            num_anchors,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Images (N, C, H, W). H and W should match config (e.g. 416).

        Returns:
            Predictions (N, S, S, B, 5+C): tx, ty, tw, th, obj, class_logits.
        """
        features = self.backbone(x)
        return self.head(features)


def decode_predictions(
    raw: torch.Tensor,
    grid_size: int,
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode raw YOLO outputs to box coordinates, objectness, and class scores.

    Box regression: bx = sigmoid(tx) + cell_x, by = sigmoid(ty) + cell_y,
    bw = exp(tw), bh = exp(th), then scale to image size.

    Args:
        raw: (N, S, S, B, 5+C) raw logits.
        grid_size: S.
        image_size: (height, width) of input image.

    Returns:
        boxes: (N, S*S*B, 4) in xyxy format normalized to [0, 1].
        objectness: (N, S*S*B) sigmoid objectness.
        class_scores: (N, S*S*B, C) class probabilities (softmax).
    """
    device = raw.device
    n, s, _, b, dim = raw.shape
    cell_w = 1.0 / grid_size
    cell_h = 1.0 / grid_size

    tx = raw[..., 0].sigmoid()
    ty = raw[..., 1].sigmoid()
    tw = raw[..., 2].clamp(max=10).exp()
    th = raw[..., 3].clamp(max=10).exp()
    obj = raw[..., 4].sigmoid()
    class_logits = raw[..., 5:]

    yi = torch.arange(grid_size, device=device, dtype=raw.dtype)
    xi = torch.arange(grid_size, device=device, dtype=raw.dtype)
    cy = (yi.view(1, -1, 1, 1) + 0.5) * cell_h
    cx = (xi.view(1, 1, -1, 1) + 0.5) * cell_w

    by = (ty * cell_h) + cy
    bx = (tx * cell_w) + cx
    bw = tw * cell_w
    bh = th * cell_h

    x1 = (bx - bw / 2).reshape(n, -1)
    y1 = (by - bh / 2).reshape(n, -1)
    x2 = (bx + bw / 2).reshape(n, -1)
    y2 = (by + bh / 2).reshape(n, -1)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)
    objectness = obj.reshape(n, -1)
    class_scores = F.softmax(class_logits, dim=-1).reshape(
        n, grid_size * grid_size * b, -1
    )
    return boxes, objectness, class_scores


def yolo_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    grid_size: int,
    num_anchors: int,
    num_classes: int,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5,
) -> torch.Tensor:
    """YOLO loss: bbox regression (MSE), objectness BCE, class cross-entropy.

    targets shape: (N, S, S, B, 5+C) with (tx, ty, tw, th, obj, one_hot_class).
    Only cells with obj=1 contribute to bbox and class loss.

    Args:
        predictions: (N, S, S, B, 5+C) raw model output.
        targets: (N, S, S, B, 5+C) target values.
        grid_size: S.
        num_anchors: B.
        num_classes: C.
        lambda_coord: Weight for bbox loss.
        lambda_noobj: Weight for no-object confidence loss.

    Returns:
        Scalar loss tensor.
    """
    obj_mask = targets[..., 4] >= 0.5
    noobj_mask = ~obj_mask

    bbox_pred = predictions[..., :4]
    bbox_tgt = targets[..., :4]
    obj_pred = predictions[..., 4].sigmoid()
    obj_tgt = targets[..., 4]
    class_pred = predictions[..., 5:]
    class_tgt = targets[..., 5:]

    bbox_loss = F.mse_loss(
        bbox_pred[obj_mask], bbox_tgt[obj_mask], reduction="sum"
    )
    obj_loss_pos = F.binary_cross_entropy_with_logits(
        predictions[..., 4][obj_mask],
        obj_tgt[obj_mask],
        reduction="sum",
    )
    obj_loss_neg = F.binary_cross_entropy_with_logits(
        predictions[..., 4][noobj_mask],
        obj_tgt[noobj_mask],
        reduction="sum",
    )
    n_obj = max(obj_mask.sum().item(), 1)
    n_noobj = max(noobj_mask.sum().item(), 1)
    class_loss = F.cross_entropy(
        class_pred[obj_mask].view(-1, num_classes),
        class_tgt[obj_mask].argmax(dim=-1),
        reduction="sum",
    )

    total = (
        lambda_coord * bbox_loss / n_obj
        + obj_loss_pos / n_obj
        + lambda_noobj * obj_loss_neg / n_noobj
        + class_loss / n_obj
    )
    return total


def _box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU; boxes in xyxy format (M, 4) and (N, 4). Returns (M, N)."""
    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    xa = torch.max(x11.unsqueeze(1), x21.unsqueeze(0))
    ya = torch.max(y11.unsqueeze(1), y21.unsqueeze(0))
    xb = torch.min(x12.unsqueeze(1), x22.unsqueeze(0))
    yb = torch.min(y12.unsqueeze(1), y22.unsqueeze(0))
    inter = torch.clamp(xb - xa, min=0) * torch.clamp(yb - ya, min=0)
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    return inter / torch.clamp(union, min=1e-6)


def non_max_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45,
) -> List[torch.Tensor]:
    """Apply NMS per batch item. boxes (N, M, 4) xyxy, scores (N, M).

    Returns:
        List of length N; each element is indices of kept boxes.
    """
    results: List[torch.Tensor] = []
    for i in range(boxes.shape[0]):
        b = boxes[i]
        s = scores[i]
        if b.shape[0] == 0:
            results.append(torch.tensor([], dtype=torch.long, device=b.device))
            continue
        order = torch.argsort(s, descending=True)
        keep: List[int] = []
        while order.numel() > 0:
            idx = order[0].item()
            keep.append(idx)
            if order.numel() == 1:
                break
            iou = _box_iou_xyxy(b[idx: idx + 1], b[order[1:]])
            mask = iou.squeeze(0) <= iou_threshold
            order = order[1:][mask]
        results.append(torch.tensor(keep, dtype=torch.long, device=b.device))
    return results


def generate_synthetic_batch(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    grid_size: int,
    num_anchors: int,
    num_classes: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of random images and matching YOLO targets.

    Targets have one random object per image: one cell has obj=1 and bbox/class.

    Args:
        batch_size: N.
        channels: Image channels.
        height, width: Image size.
        grid_size: S.
        num_anchors: B.
        num_classes: C.
        device: Torch device.
        seed: Random seed.

    Returns:
        images: (N, C, H, W).
        targets: (N, S, S, B, 5+C).
    """
    if seed is not None:
        torch.manual_seed(seed)
    images = torch.rand(batch_size, channels, height, width, device=device)
    targets = torch.zeros(
        batch_size, grid_size, grid_size, num_anchors, 5 + num_classes,
        device=device,
    )
    cell_size_h = height / grid_size
    cell_size_w = width / grid_size
    for i in range(batch_size):
        gi = torch.randint(0, grid_size, (1,)).item()
        gj = torch.randint(0, grid_size, (1,)).item()
        anchor_idx = torch.randint(0, num_anchors, (1,)).item()
        cls = torch.randint(0, num_classes, (1,)).item()
        targets[i, gi, gj, anchor_idx, 0] = 0.5
        targets[i, gi, gj, anchor_idx, 1] = 0.5
        targets[i, gi, gj, anchor_idx, 2] = 0.0
        targets[i, gi, gj, anchor_idx, 3] = 0.0
        targets[i, gi, gj, anchor_idx, 4] = 1.0
        targets[i, gi, gj, anchor_idx, 5 + cls] = 1.0
    return images, targets


def run_training(config: Dict[str, Any]) -> None:
    """Train YOLO on synthetic data according to config.

    Args:
        config: Full config dict with model, training, data, logging.
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    net = YOLONet(
        in_channels=3,
        grid_size=model_cfg.get("grid_size", 13),
        num_anchors=model_cfg.get("num_anchors", 5),
        num_classes=model_cfg.get("num_classes", 20),
        backbone_channels=model_cfg.get("backbone_channels", [32, 64, 128, 256, 512]),
        backbone_blocks=model_cfg.get("backbone_blocks", [1, 2, 2, 2]),
    ).to(device)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
    )
    epochs = train_cfg.get("epochs", 30)
    batch_size = train_cfg.get("batch_size", 8)
    num_train = data_cfg.get("num_train", 500)
    seed = data_cfg.get("random_seed", 42)
    h = model_cfg.get("input_height", 416)
    w = model_cfg.get("input_width", 416)
    s = model_cfg.get("grid_size", 13)
    b = model_cfg.get("num_anchors", 5)
    c = model_cfg.get("num_classes", 20)
    lambda_coord = train_cfg.get("lambda_coord", 5.0)
    lambda_noobj = train_cfg.get("lambda_noobj", 0.5)

    steps_per_epoch = max(1, num_train // batch_size)
    logger.info(
        "Training for %d epochs, %d steps/epoch, batch_size=%d",
        epochs, steps_per_epoch, batch_size,
    )

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        for step in range(steps_per_epoch):
            images, targets = generate_synthetic_batch(
                batch_size, 3, h, w, s, b, c, device,
                seed=seed + epoch * 1000 + step,
            )
            optimizer.zero_grad()
            pred = net(images)
            loss = yolo_loss(
                pred, targets, s, b, c,
                lambda_coord=lambda_coord,
                lambda_noobj=lambda_noobj,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / steps_per_epoch
        logger.info("Epoch %d/%d loss: %.4f", epoch + 1, epochs, avg)


def main() -> None:
    """Entry point: load config, set up logging, run training or inference."""
    parser = argparse.ArgumentParser(
        description="YOLO object detection with bounding box regression and class prediction."
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

    logger.info("Starting YOLO object detection")
    run_training(config)
    logger.info("Done")


if __name__ == "__main__":
    main()
