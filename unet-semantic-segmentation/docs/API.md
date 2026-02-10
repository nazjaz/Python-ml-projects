# U-Net Semantic Segmentation API Documentation

## Module: src.main

### Classes

#### DoubleConv

Two 3x3 convolutions with BatchNorm and ReLU. Preserves spatial size (padding=1).

**Constructor**: `DoubleConv(in_channels, out_channels)`

- `in_channels`: Input channel count
- `out_channels`: Output channel count

**Methods**:
- `forward(x)`: Forward pass. Input (N, in_channels, H, W), output (N, out_channels, H, W)

#### EncoderBlock

One encoder stage: DoubleConv then MaxPool. Exposes pre-pool output for skip connection.

**Constructor**: `EncoderBlock(in_channels, out_channels)`

- `in_channels`: Input channels
- `out_channels`: Output channels after conv and after pool

**Methods**:
- `forward(x)`: Returns (pooled, skip). pooled for next level, skip for decoder.

#### DecoderBlock

One decoder stage: upsample (ConvTranspose2d), concatenate skip, DoubleConv.

**Constructor**: `DecoderBlock(in_channels, skip_channels, out_channels)`

- `in_channels`: Channels from previous decoder or bottleneck
- `skip_channels`: Channels from encoder skip (concatenated)
- `out_channels`: Output channels after DoubleConv

**Methods**:
- `forward(x, skip)`: Upsample x, concat with skip (cropped if needed), DoubleConv. Returns (N, out_channels, H, W).

#### UNet

Full U-Net for semantic segmentation: encoder, bottleneck, decoder, skip connections.

**Constructor**: `UNet(in_channels=3, num_classes=2, base_channels=64, depth=4)`

- `in_channels`: Input image channels (e.g. 3 for RGB)
- `num_classes`: Number of segmentation classes (output channels)
- `base_channels`: Channels at first encoder level; doubled each level
- `depth`: Number of encoder/decoder levels (excluding bottleneck)

**Methods**:
- `forward(x)`: Input (N, in_channels, H, W). Output (N, num_classes, H, W) logits. Input H,W should be divisible by 2^depth for same output size.

### Functions

#### _load_config(config_path)

Load YAML configuration file. Returns configuration dict. Raises FileNotFoundError if path does not exist.

#### _setup_logging(level, log_file)

Configure logging to console and optionally to a file.

#### segmentation_loss(logits, targets, num_classes, ignore_index=-100)

Cross-entropy loss for semantic segmentation. **logits**: (N, num_classes, H, W). **targets**: (N, H, W) integer class indices. Pixels with value ignore_index are ignored. Returns scalar loss tensor.

#### generate_synthetic_batch(batch_size, channels, height, width, num_classes, device, seed=None)

Generate random images and per-pixel labels. Returns (images, labels); images (N, C, H, W), labels (N, H, W).

#### run_training(config)

Train U-Net on synthetic data. **config**: full config dict with model, training, data, logging.

#### main()

CLI entry point. Parses --config and runs training.
