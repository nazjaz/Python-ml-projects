# Optimization Algorithms API Documentation

## Module: `src.main`

### Optimizer Classes

All optimizers implement a common interface:

- `step(params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], learning_rate: float) -> None`

where:

- `params`: Mapping from parameter names to parameter arrays.
- `grads`: Mapping from parameter names to gradient arrays (same keys as `params`).

#### `SGDOptimizer`

Standard stochastic gradient descent without momentum.

Update rule:

```text
theta = theta - learning_rate * grad
```

#### `AdaGradOptimizer`

Implements the AdaGrad algorithm with per-parameter learning rate adaptation.

State:

- Accumulated squared gradients `G` per parameter.

Update rule:

```text
G = G + grad^2
theta = theta - learning_rate * grad / (sqrt(G) + eps)
```

#### `RMSpropOptimizer`

Implements RMSprop using an exponential moving average of squared gradients.

State:

- Exponential moving average `E[g^2]` per parameter.

Hyperparameters:

- `rho`: Decay rate.
- `eps`: Small constant for numerical stability.

Update rule:

```text
E = rho * E + (1 - rho) * grad^2
theta = theta - learning_rate * grad / (sqrt(E) + eps)
```

#### `AdamOptimizer`

Implements Adam (adaptive moment estimation) with bias correction.

State:

- First moment estimate `m`
- Second moment estimate `v`
- Time step `t`

Hyperparameters:

- `beta1`, `beta2`: Exponential decay rates for moments.
- `eps`: Small constant for numerical stability.

Update rule:

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
theta = theta - learning_rate * m_hat / (sqrt(v_hat) + eps)
```

### Model Class

#### `MLPClassifier`

Two-layer feedforward classifier shared across all optimizers.

**Constructor**:

```python
MLPClassifier(input_dim: int, hidden_dim: int, n_classes: int)
```

**Methods**:

- `forward(x: np.ndarray) -> np.ndarray`  
  Returns logits of shape `(batch_size, n_classes)`.
- `compute_loss_and_gradients(x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]`  
  Computes cross-entropy loss and gradients with respect to model parameters.
- `get_params() -> Dict[str, np.ndarray]`  
  Returns a mapping of parameter names to arrays.
- `set_params(params: Dict[str, np.ndarray]) -> None`  
  Sets internal parameters from a dict.
- `predict(x: np.ndarray) -> np.ndarray`  
  Returns predicted class labels.

### Runner Class

#### `OptimizationRunner`

Configuration-driven training runner.

**Constructor**:

```python
OptimizationRunner(config_path: Optional[Path] = None)
```

**Methods**:

- `run(optimizer_name: str) -> Dict[str, float]`  
  Trains `MLPClassifier` with the selected optimizer (`"sgd"`, `"adagrad"`, `"rmsprop"`, `"adam"`), evaluates on a held-out test set, and returns metrics:
  - `train_loss`, `train_accuracy`, `test_loss`, `test_accuracy`.

### Data Utility

#### `generate_classification_data(n_samples, n_features, n_classes, random_seed)`

Generate a synthetic multi-class classification dataset.

Returns `(x_train, x_test, y_train, y_test)`.

### Entry Point

#### `main()`

Command-line entry point.

Arguments:

- `--optimizer`: `"sgd"`, `"adagrad"`, `"rmsprop"`, or `"adam"`.
- `--config`: Path to YAML configuration file.
- `--output`: Path to JSON file where metrics are written.

