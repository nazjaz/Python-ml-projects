# API Reference

SimCLR-style contrastive learning for self-supervised representation learning.

## Augmentations

### make_two_views(x, noise_std?, scale_range?, random_seed?) -> (v1, v2)

Creates two augmented views of each sample: scale by random factor and add Gaussian noise. Returns two arrays of shape (N, D) for positive pairs.

## Model

### MLPBlock

Two-layer ReLU MLP: forward(x), backward(grad_out, lr). Used as encoder and projection building blocks.

### SimCLRNet

Encoder + projection head.

- **__init__(in_dim, repr_dim, proj_dim, encoder_hidden?, proj_hidden?, random_seed?)**
- **encode(x) -> repr**  
  Representation (before projection); shape (batch, repr_dim).
- **forward(x) -> z**  
  Encode then project; shape (batch, proj_dim).
- **backward_projection(grad_proj, lr) -> grad_repr**  
  Backward through projection head.
- **backward_encoder(grad_repr, lr)**  
  Backward through encoder.

## Loss

### l2_normalize(x, axis=-1)

L2-normalize x along axis.

### nt_xent_loss(z, temperature, pair_indices?) -> (loss, grad_z)

NT-Xent (InfoNCE) contrastive loss. z has shape (2*N, proj_dim); rows 2k and 2k+1 are the two views of sample k. Returns scalar loss and gradient w.r.t. z (for backprop through projection and encoder).

## Data and training

### load_data(max_samples?, random_seed?) -> x

Loads digits (sklearn) or synthetic data; returns feature matrix (no labels).

### train_simclr(net, data, epochs, batch_size, lr, temperature, noise_std, scale_range, random_seed?) -> List[float]

Trains SimCLR with NT-Xent; creates two views per batch, forward, loss, backward. Returns per-epoch mean loss.

### evaluate_representation(net, x, y?) -> repr

Returns encoder output (representation vectors) for downstream use.

## Config and run

### SimCLRConfig

repr_dim, proj_dim, encoder_hidden, proj_hidden, epochs, batch_size, learning_rate, temperature, noise_std, scale_low, scale_high, max_samples, random_seed.

### run_simclr(config) -> Dict

Loads data, builds net, runs train_simclr, returns final_loss, num_samples, in_dim, repr_dim, proj_dim.

### main()

CLI: --config, --output. Runs SimCLR and prints or writes JSON.
