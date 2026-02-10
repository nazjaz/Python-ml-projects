# Transfer Learning and Fine-Tuning API Documentation

## Module: `src.main`

### Classes

#### `MLPBase`

Two-layer base model used for source and target tasks.

**Constructor**:

```python
MLPBase(input_dim: int, hidden_dim: int, n_classes: int)
```

**Methods**:

- `forward(x: np.ndarray) -> np.ndarray`  
  Returns logits of shape `(batch_size, n_classes)`.
- `compute_loss_and_gradients(x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]`  
  Computes cross-entropy loss and gradients with respect to parameters.
- `get_params() -> Dict[str, np.ndarray]`  
  Returns a mapping of parameter names to arrays.
- `set_params(params: Dict[str, np.ndarray]) -> None`  
  Sets internal parameters from a dict.
- `predict(x: np.ndarray) -> np.ndarray`  
  Predicts class labels.

#### `TransferLearningRunner`

Coordinates base training and transfer strategies.

**Constructor**:

```python
TransferLearningRunner(config_path: Optional[Path] = None)
```

**Methods**:

- `run(strategy: str) -> Dict[str, float]`  
  Runs the full workflow for the specified strategy:
  - `"scratch"`: train target model from random initialization
  - `"feature_extractor"`: train base on source, freeze base, train new head on target
  - `"head_finetune"`: initialize from base, train all layers on target
  - `"full_finetune"`: same as head fine-tuning but reported separately

Returns a results dictionary containing:

- `source_loss`, `source_accuracy` (for strategies using a base model)
- `target_loss`, `target_accuracy`

### Functions

#### `generate_source_and_target_data(base_n_samples, target_n_samples, n_features, n_classes, random_seed)`

Generate related source and target datasets for transfer learning experiments.

Returns:

- `(x_base, y_base, x_target, y_target)`.

#### `train_model(model, x, y, epochs, learning_rate, batch_size)`

Train a given model on `(x, y)` using mini-batch gradient descent.

Returns a history dictionary with per-epoch loss and accuracy.

### Entry Point

#### `main()`

Command-line entry point.

Arguments:

- `--strategy`: `"scratch"`, `"feature_extractor"`, `"head_finetune"`, or `"full_finetune"`.
- `--config`: Path to YAML configuration file.
- `--output`: Path to JSON file where metrics are written.

