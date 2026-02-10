# YOLO Object Detection API Documentation

## Module: src.main

### Classes

#### ConvBlock

Conv2d, BatchNorm2d, and LeakyReLU block for backbone feature extraction.

**Constructor**: `ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=None)`

- `in_channels`: Input channel count
- `out_channels`: Output channel count
- `kernel_size`: Convolution kernel size (default: 3)
- `stride`: Stride (default: 1)
- `padding`: Padding; defaults to kernel_size // 2 when stride == 1

**Methods**:
- `forward(x)`: Forward pass. Input (N, C_in, H, W), output (N, C_out, H', W')

#### Backbone

Darknet-style backbone of stacked ConvBlocks with downsampling.

**Constructor**: `Backbone(in_channels=3, channel_list=(32, 64, 128, 256, 512), block_repeats=(1, 2, 2, 2))`

- `in_channels`: Input image channels (e.g. 3 for RGB)
- `channel_list`: Channel sizes per stage
- `block_repeats`: Number of ConvBlocks per stage (first uses stride 2)

**Methods**:
- `forward(x)`: Forward pass. Output shape (N, channel_list[-1], H', W')

**Attributes**:
- `out_channels`: channel_list[-1]

#### YOLODetectionHead

Detection head: one prediction per grid cell per anchor (tx, ty, tw, th, objectness, class logits).

**Constructor**: `YOLODetectionHead(in_channels, grid_size, num_anchors, num_classes)`

- `in_channels`: Backbone feature channels
- `grid_size`: Spatial grid size S
- `num_anchors`: Number of boxes per cell B
- `num_classes`: Number of classes C

**Methods**:
- `forward(x)`: Input (N, in_channels, grid_size, grid_size). Output (N, S, S, B, 5+C).

#### YOLONet

Full YOLO model: backbone plus detection head.

**Constructor**: `YOLONet(in_channels=3, grid_size=13, num_anchors=5, num_classes=20, backbone_channels=(32, 64, 128, 256, 512), backbone_blocks=(1, 2, 2, 2))`

- `in_channels`: Input image channels
- `grid_size`: Detection grid size S
- `num_anchors`: Anchors per cell B
- `num_classes`: Number of classes C
- `backbone_channels`: Channel list for backbone
- `backbone_blocks`: Block repeats per stage

**Methods**:
- `forward(x)`: Input (N, C, H, W). Output (N, S, S, B, 5+C): tx, ty, tw, th, obj_logit, class_logits.

### Functions

#### _load_config(config_path)

Load YAML configuration file. Returns configuration dict. Raises FileNotFoundError if path does not exist.

#### _setup_logging(level, log_file)

Configure logging to console and optionally to a file. `level` is e.g. "INFO"; `log_file` is optional path.

#### decode_predictions(raw, grid_size, image_size)

Decode raw YOLO outputs to box coordinates, objectness, and class scores.

- **raw**: (N, S, S, B, 5+C) raw logits
- **grid_size**: S
- **image_size**: (height, width) of input image

Returns:
- **boxes**: (N, S*S*B, 4) in xyxy format, normalized to [0, 1]
- **objectness**: (N, S*S*B) sigmoid objectness
- **class_scores**: (N, S*S*B, C) class probabilities (softmax)

#### yolo_loss(predictions, targets, grid_size, num_anchors, num_classes, lambda_coord=5.0, lambda_noobj=0.5)

Compute YOLO loss: bbox MSE, objectness BCE, class cross-entropy. Targets shape (N, S, S, B, 5+C) with (tx, ty, tw, th, obj, one_hot_class). Returns scalar loss tensor.

#### _box_iou_xyxy(boxes1, boxes2)

Compute pairwise IoU for boxes in xyxy format. boxes1 (M, 4), boxes2 (N, 4). Returns (M, N).

#### non_max_suppression(boxes, scores, iou_threshold=0.45)

Apply NMS per batch item. **boxes**: (N, M, 4) xyxy; **scores**: (N, M). Returns list of length N; each element is a tensor of indices of kept boxes.

#### generate_synthetic_batch(batch_size, channels, height, width, grid_size, num_anchors, num_classes, device, seed=None)

Generate random images and YOLO targets (one object per image in a random cell). Returns (images, targets); images (N, C, H, W), targets (N, S, S, B, 5+C).

#### run_training(config)

Train YOLO on synthetic data. **config**: full config dict with model, training, data, logging.

#### main()

CLI entry point. Parses --config and runs training.
