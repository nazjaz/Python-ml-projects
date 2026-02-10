# GAN API Documentation

## Module: `src.main`

### Classes

#### `Generator`

Fully connected generator network.

**Constructor**:

```python
Generator(noise_dim: int, hidden_dim: int, data_dim: int)
```

- `noise_dim`: Dimension of input noise vector.
- `hidden_dim`: Size of hidden layer.
- `data_dim`: Dimension of generated data.

**Methods**:

- `forward(z: np.ndarray) -> np.ndarray`  
  Maps noise `z` of shape `(batch_size, noise_dim)` to generated samples of shape `(batch_size, data_dim)`.
- `parameters() -> Dict[str, np.ndarray]`  
  Returns parameter arrays (`w1`, `b1`, `w2`, `b2`).
- `update(params: Dict[str, np.ndarray]) -> None`  
  Replaces internal parameters with those in `params`.

#### `Discriminator`

Fully connected discriminator network.

**Constructor**:

```python
Discriminator(data_dim: int, hidden_dim: int)
```

**Methods**:

- `forward(x: np.ndarray) -> np.ndarray`  
  Returns probabilities `D(x)` of shape `(batch_size, 1)`.
- `parameters() -> Dict[str, np.ndarray]`  
  Returns parameter arrays.
- `update(params: Dict[str, np.ndarray]) -> None`  
  Replaces internal parameters with those in `params`.

#### `GANTrainer`

Coordinates GAN training loop for generator and discriminator.

**Constructor**:

```python
GANTrainer(
    generator: Generator,
    discriminator: Discriminator,
)
```

**Methods**:

- `train(x_real, epochs, batch_size, lr_g, lr_d, d_steps, g_steps) -> Dict[str, List[float]]`  
  Trains GAN on real data `x_real` and returns per-epoch histories for:
  - `d_loss`
  - `g_loss`

### Functions

#### `generate_real_data(n_samples, data_dim, random_seed)`

Generate synthetic 2D data from a mixture of Gaussians.

Returns `x_real` of shape `(n_samples, data_dim)`.

### Runner

#### `GANRunner`

Loads configuration, builds generator and discriminator, runs training, and reports final losses.

**Constructor**:

```python
GANRunner(config_path: Optional[Path] = None)
```

**Methods**:

- `run() -> Dict[str, float]`  
  Returns:
  - `final_d_loss`
  - `final_g_loss`

### Entry Point

#### `main()`

Command-line entry point.

Arguments:

- `--config`: Path to YAML configuration file.
- `--output`: Path to JSON file where metrics are written.

