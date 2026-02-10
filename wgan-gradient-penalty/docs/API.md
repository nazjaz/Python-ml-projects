# WGAN-GP API Documentation

## Module: `src.main`

### Classes

#### `WGANGenerator`

Fully connected generator network for WGAN-GP.

**Constructor**:

```python
WGANGenerator(noise_dim: int, hidden_dim: int, data_dim: int)
```

**Methods**:

- `forward(z: np.ndarray) -> np.ndarray`  
  Maps noise `z` of shape `(batch_size, noise_dim)` to generated samples of shape `(batch_size, data_dim)`.
- `parameters() -> Dict[str, np.ndarray]`  
  Returns generator parameters.
- `update(params: Dict[str, np.ndarray]) -> None`  
  Updates generator parameters from a dictionary.

#### `WGANCritic`

Fully connected critic network (discriminator without sigmoid).

**Constructor**:

```python
WGANCritic(data_dim: int, hidden_dim: int)
```

**Methods**:

- `forward(x: np.ndarray) -> np.ndarray`  
  Returns critic scores of shape `(batch_size, 1)`.
- `parameters() -> Dict[str, np.ndarray]`  
  Returns critic parameters.
- `update(params: Dict[str, np.ndarray]) -> None`  
  Updates critic parameters from a dictionary.
- `backward_params(grad_scores: np.ndarray) -> Dict[str, np.ndarray]`  
  Computes gradients of parameters with respect to critic scores.
- `backward_input(grad_scores: np.ndarray) -> np.ndarray`  
  Computes gradients of scores with respect to inputs (used for gradient penalty).

#### `WGANTrainer`

Coordinates WGAN-GP training loop for generator and critic.

**Constructor**:

```python
WGANTrainer(
    generator: WGANGenerator,
    critic: WGANCritic,
    noise_dim: int,
    lambda_gp: float,
)
```

**Methods**:

- `train(x_real, epochs, batch_size, lr_g, lr_c, critic_iters) -> Dict[str, List[float]]`  
  Trains WGAN-GP on real data `x_real` and returns histories for:
  - `critic_loss`
  - `generator_loss`

### Functions

#### `generate_real_data(n_samples, data_dim, random_seed)`

Generate synthetic real data from a mixture of Gaussians.

Returns `x_real` of shape `(n_samples, data_dim)`.

### Runner

#### `WGANRunner`

Loads configuration, builds generator and critic, runs WGAN-GP training, and reports final losses.

**Constructor**:

```python
WGANRunner(config_path: Optional[Path] = None)
```

**Methods**:

- `run() -> Dict[str, float]`  
  Returns:
  - `final_critic_loss`
  - `final_generator_loss`

### Entry Point

#### `main()`

Command-line entry point.

Arguments:

- `--config`: Path to YAML configuration file.
- `--output`: Path to JSON file where metrics are written.

