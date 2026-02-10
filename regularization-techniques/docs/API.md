# Regularization Techniques API Documentation

## Module: `src.main`

### Classes

#### `Dropout`

Implements inverted dropout for fully connected layers.

**Constructor**: `Dropout(rate: float)`

- `rate`: Drop probability in \[0, 1). A value of 0.5 means half of the activations are dropped on average.

**Methods**:

- `forward(x: np.ndarray, training: bool = True) -> np.ndarray`  
  Applies dropout during training and returns scaled activations. During evaluation, returns the input unchanged.
- `backward(grad_output: np.ndarray) -> np.ndarray`  
  Backpropagates gradient through the stored dropout mask (expects `forward` to have been called in training mode).

#### `RegularizedMLP`

Two-layer feedforward neural network with optional dropout and L2 regularization.

**Constructor**:  
`RegularizedMLP(input_dim: int, hidden_dim: int, n_classes: int, dropout_rate: float, l2_lambda: float, mode: str)`

- `input_dim`: Number of input features.
- `hidden_dim`: Hidden layer size.
- `n_classes`: Number of output classes.
- `dropout_rate`: Drop probability used by the `Dropout` layer.
- `l2_lambda`: L2 weight decay coefficient.
- `mode`: One of `"none"`, `"dropout"`, `"l2"`, or `"dropout_l2"`.

**Methods**:

- `forward(x: np.ndarray, training: bool = True) -> np.ndarray`  
  Returns logits of shape `(batch_size, n_classes)`.
- `compute_loss_and_gradients(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]`  
  Computes cross-entropy loss plus L2 penalty (depending on mode) and returns the scalar loss and gradient of the loss with respect to logits.
- `backward(grad_logits: np.ndarray, learning_rate: float) -> None`  
  Backpropagates gradients through the network and updates parameters, including L2 contributions when enabled.
- `predict(x: np.ndarray) -> np.ndarray`  
  Predicts integer class labels from input features.

#### `RegularizationRunner`

Configuration-driven training runner.

**Constructor**: `RegularizationRunner(config_path: Optional[Path] = None)`

**Methods**:

- `run(mode: str) -> Dict[str, float]`  
  Builds a `RegularizedMLP` with the specified mode, trains it on synthetic data, evaluates on a held-out test set, and returns metrics:
  - `train_loss`, `train_accuracy`, `test_loss`, `test_accuracy`.

### Functions

#### `generate_classification_data(n_samples, n_features, n_classes, random_seed)`

Generate a synthetic multi-class classification dataset.

Returns:

- `x_train`, `x_test`: Features.
- `y_train`, `y_test`: Integer labels.

#### `main()`

Command-line entry point.

Arguments:

- `--mode`: `"none"`, `"dropout"`, `"l2"`, or `"dropout_l2"`.
- `--config`: Path to YAML configuration file.
- `--output`: Path to JSON file where metrics are written.

