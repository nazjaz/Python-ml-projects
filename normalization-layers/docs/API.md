# Normalization Layers API Documentation

## Module: `src.main`

### Classes

#### `BatchNorm1D`

Batch normalization layer for fully connected networks (1D feature vectors).

**Constructor**: `BatchNorm1D(num_features: int, momentum: float = 0.9, eps: float = 1e-5)`

- `num_features`: Number of input features.
- `momentum`: Momentum for running mean and variance.
- `eps`: Small constant for numerical stability.

**Methods**:

- `forward(x: np.ndarray, training: bool = True) -> np.ndarray`  
  Input shape: `(batch_size, num_features)`.  
  Returns normalized output of the same shape.
- `backward(grad_output: np.ndarray) -> np.ndarray`  
  Backpropagates gradient through the layer (expects `forward` to have been called in training mode).

#### `LayerNorm1D`

Layer normalization over feature dimension for each sample.

**Constructor**: `LayerNorm1D(num_features: int, eps: float = 1e-5)`

- `num_features`: Number of input features.
- `eps`: Small constant for numerical stability.

**Methods**:

- `forward(x: np.ndarray) -> np.ndarray`  
  Input shape: `(batch_size, num_features)`.  
  Returns normalized output of the same shape.
- `backward(grad_output: np.ndarray) -> np.ndarray`  
  Backpropagates gradient through layer normalization.

#### `SimpleMLP`

Two-layer feedforward neural network with optional normalization after the first affine layer.

**Constructor**: `SimpleMLP(input_dim: int, hidden_dim: int, n_classes: int, norm: str = "none")`

- `input_dim`: Number of input features.
- `hidden_dim`: Hidden layer size.
- `n_classes`: Number of output classes.
- `norm`: One of `"none"`, `"batchnorm"`, or `"layernorm"`.

**Methods**:

- `forward(x: np.ndarray, training: bool = True) -> np.ndarray`: Returns logits of shape `(batch_size, n_classes)`.
- `backward(grad_logits: np.ndarray, learning_rate: float) -> None`: Backpropagates gradient and updates parameters.
- `predict(x: np.ndarray) -> np.ndarray`: Returns predicted class labels.

#### `NormalizationRunner`

Configuration-driven training runner.

**Constructor**: `NormalizationRunner(config_path: Optional[Path] = None)`

**Methods**:

- `run(mode: str) -> Dict[str, float]`:  
  Builds and trains a `SimpleMLP` with the selected normalization mode (`"none"`, `"batchnorm"`, or `"layernorm"`), evaluates on a held-out test set, and returns metrics:
  - `train_loss`, `train_accuracy`, `test_loss`, `test_accuracy`.

### Functions

#### `generate_classification_data(n_samples, n_features, n_classes, random_seed)`

Generate a synthetic multi-class classification dataset.

Returns:

- `x_train`, `x_test`: Feature arrays.
- `y_train`, `y_test`: Integer labels.

#### `main()`

Command-line entry point.

Arguments:

- `--mode`: `"none"`, `"batchnorm"`, or `"layernorm"`.
- `--config`: Path to YAML configuration file.
- `--output`: Path to JSON file where metrics are written.

