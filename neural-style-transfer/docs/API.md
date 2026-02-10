# API Reference

Neural style transfer with content and style loss optimization.

## Conv and Gram helpers

- **conv2d_forward(x, w, pad)**  
  Conv2D with same padding; x (H, W, C_in), w (kH, kW, C_in, C_out).

- **conv2d_backward_input(grad_out, w, x_shape, pad)**  
  Gradient of conv output w.r.t. input (weights fixed).

- **gram_matrix(features)**  
  From (H, W, C) to (C, C) Gram matrix.

- **gram_matrix_backward(grad_gram, features)**  
  Gradient of Gram w.r.t. features.

## Data

- **load_content_style_images(size?, content_index?, style_index?, random_seed?)**  
  Returns (content, style) as (H, W, 1) in [-1, 1]. Uses digits or synthetic.

## Feature extractor

### FeatureExtractor

Small CNN (one Conv2D + ReLU) with fixed weights.

- **forward(x)**  
  Returns feature map (H, W, C_out).
- **backward(grad_out)**  
  Returns gradient w.r.t. input image.

## Losses

- **content_loss(feat_gen, feat_content)**  
  MSE between feature maps; returns scalar.
- **content_loss_grad(feat_gen, feat_content)**  
  Gradient of content loss w.r.t. feat_gen.
- **style_loss(gram_gen, gram_style)**  
  MSE between Gram matrices.
- **style_loss_grad(gram_gen, gram_style)**  
  Gradient of style loss w.r.t. gram_gen.

## Optimization

- **run_style_transfer(content, style, extractor, num_steps, content_weight, style_weight, lr, random_seed?)**  
  Optimizes generated image by gradient descent. Returns (generated_image, history).

## Config and pipeline

- **StyleTransferConfig**  
  num_steps, content_weight, style_weight, learning_rate, extractor_channels, image_size, content_index, style_index, random_seed.

- **run_pipeline(config)**  
  Loads content/style, builds extractor, runs style transfer, returns metrics dict.

- **main()**  
  CLI: --config, --output. Runs pipeline and prints or writes JSON.
