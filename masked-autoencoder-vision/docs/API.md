# Masked Autoencoder Vision API Documentation

## Module: src.main

### Functions

#### _load_config(config_path)

Load YAML configuration file. Returns configuration dict. Raises
FileNotFoundError if the path does not exist.

#### _setup_logging(level, log_file)

Configure logging to console and optionally to a rotating log file.

#### patchify(images, patch_size)

Convert images `(N, C, H, W)` into patches `(N, L, C * patch_size^2)` where
`L = (H / patch_size) * (W / patch_size)`.

#### unpatchify(patches, patch_size, channels, height, width)

Reconstruct images `(N, C, H, W)` from flattened patches `(N, L, C * patch_size^2)`.

#### generate_synthetic_images(num_images, channels, height, width, device, seed=None)

Generate random images for self-supervised training. Returns tensor of shape
`(num_images, channels, height, width)`.

#### run_training(config)

Run MAE self-supervised training on synthetic images according to `config`. Uses
`config["model"]`, `config["training"]`, and `config["data"]`.

#### main()

CLI entry point. Parses `--config` and runs training.

### Classes

#### MaskedAutoencoder

Simple transformer-based masked autoencoder for image patches.

**Constructor**:

`MaskedAutoencoder(image_size, patch_size, in_channels, embed_dim, encoder_layers, decoder_layers, mask_ratio)`

- `image_size`: Input image size (assumes square images).
- `patch_size`: Patch side length (must divide image_size).
- `in_channels`: Number of image channels.
- `embed_dim`: Embedding dimension for patch tokens.
- `encoder_layers`: Number of transformer encoder layers in the encoder.
- `decoder_layers`: Number of transformer encoder layers in the decoder.
- `mask_ratio`: Fraction of patches to mask.

**Methods**:

- `forward(images)`: Input `(N, C, H, W)` images. Returns `(loss, recon_images)`
  where `loss` is the mean squared error on masked patches and
  `recon_images` has the same shape as input images.

