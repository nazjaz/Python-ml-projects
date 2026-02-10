# Autoencoder API Documentation

## Module: `src.main`

### Classes

#### `Autoencoder`

Fully connected autoencoder with one hidden encoder layer, a latent layer, and a symmetric decoder.

**Constructor**:

```python
Autoencoder(input_dim: int, hidden_dim: int, latent_dim: int)
```

- `input_dim`: Number of input features.
- `hidden_dim`: Size of encoder and decoder hidden layers.
- `latent_dim`: Size of latent representation.

**Methods**:

- `encode(x: np.ndarray) -> np.ndarray`  
  Maps inputs of shape `(batch_size, input_dim)` into latent space `(batch_size, latent_dim)`.
- `decode(z: np.ndarray) -> np.ndarray`  
  Reconstructs inputs from latent codes of shape `(batch_size, latent_dim)`.
- `forward(x: np.ndarray) -> np.ndarray`  
  Full autoencoder pass: returns reconstructions with shape `(batch_size, input_dim)`.
- `compute_loss_and_gradients(x: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]`  
  Computes mean squared error reconstruction loss and gradients for all parameters.
- `apply_gradients(grads: Dict[str, np.ndarray], learning_rate: float) -> None`  
  Updates parameters with gradient descent.

### Functions

#### `generate_synthetic_data(n_samples, n_features, random_seed)`

Generate synthetic Gaussian data for autoencoder training.

Returns a NumPy array of shape `(n_samples, n_features)`.

#### `train_autoencoder(model, x, epochs, learning_rate, batch_size)`

Train an autoencoder using mini-batch gradient descent.

Returns a history dictionary:

- `loss`: List of reconstruction losses per epoch.

### Runner

#### `AutoencoderRunner`

Loads configuration, builds the model, trains it, and reports final loss.

**Constructor**:

```python
AutoencoderRunner(config_path: Optional[Path] = None)
```

**Methods**:

- `run() -> Dict[str, float]`  
  Returns a dictionary with:
  - `final_loss`: Final reconstruction loss on the full dataset.

### Entry Point

#### `main()`

Command-line entry point.

Arguments:

- `--config`: Path to YAML configuration file.
- `--output`: Path to JSON file where metrics are written.

