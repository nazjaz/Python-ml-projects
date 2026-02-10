# Variational Autoencoder API Documentation

## Module: `src.main`

### Classes

#### `VariationalAutoencoder`

Fully connected variational autoencoder with:

- Encoder: input → hidden → latent mean/log-variance
- Decoder: latent sample → hidden → reconstruction

**Constructor**:

```python
VariationalAutoencoder(input_dim: int, hidden_dim: int, latent_dim: int)
```

**Methods**:

- `encode(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`  
  Returns `(mu, log_var)` for input batch `x` of shape `(batch_size, input_dim)`.
- `reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray`  
  Samples latent `z` using the reparameterization trick.
- `decode(z: np.ndarray) -> np.ndarray`  
  Reconstructs inputs from latent samples of shape `(batch_size, latent_dim)`.
- `forward(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`  
  Full VAE forward pass: returns `(x_recon, mu, log_var)`.
- `compute_loss_and_gradients(x: np.ndarray) -> Tuple[float, float, float, Dict[str, np.ndarray]]`  
  Computes total loss, reconstruction loss, KL divergence, and gradients.
- `apply_gradients(grads: Dict[str, np.ndarray], learning_rate: float) -> None`  
  Updates all parameters with gradient descent.

### Functions

#### `generate_synthetic_data(n_samples, n_features, random_seed)`

Generate synthetic Gaussian data for VAE training.

Returns an array `x` of shape `(n_samples, n_features)`.

#### `train_vae(model, x, epochs, learning_rate, batch_size)`

Train a variational autoencoder using mini-batch gradient descent.

Returns a history dictionary with lists per epoch:

- `total_loss`
- `recon_loss`
- `kl_loss`

### Runner

#### `VAERunner`

Loads configuration, builds the VAE, trains it, and reports final losses.

**Constructor**:

```python
VAERunner(config_path: Optional[Path] = None)
```

**Methods**:

- `run() -> Dict[str, float]`  
  Returns:
  - `final_total_loss`
  - `final_recon_loss`
  - `final_kl_loss`

### Entry Point

#### `main()`

Command-line entry point.

Arguments:

- `--config`: Path to YAML configuration file.
- `--output`: Path to JSON file where metrics are written.

