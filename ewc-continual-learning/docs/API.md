# EWC Continual Learning API Documentation

## Module: src.main

### Functions

#### _load_config(config_path)

Load YAML configuration file. Returns configuration dict. Raises FileNotFoundError
if the path does not exist.

#### _setup_logging(level, log_file)

Configure logging to console and optionally to a rotating log file.

#### generate_task_dataset(task_id, num_samples, input_dim, num_classes, device, seed=None)

Generate synthetic classification dataset for a given task. Each task has its
own class prototypes (Gaussian means). Returns `(features, labels)` where:

- `features`: (num_samples, input_dim)
- `labels`: (num_samples,) integer labels in [0, num_classes-1]

#### compute_accuracy(model, x, y, batch_size)

Compute classification accuracy of `model` on tensors `x`, `y` using
mini-batches of size `batch_size`. Returns a float in [0, 1].

#### estimate_fisher_diagonal(model, x, y, batch_size)

Estimate diagonal Fisher information matrix for a trained task using squared
gradients of the loss. Returns a list of tensors matching model parameters.

#### snapshot_parameters(model)

Create a detached copy of the model parameters. Returns a list of tensors.

#### ewc_penalty(model, fisher_diags, prev_params, lambda_ewc)

Compute EWC quadratic penalty given current `model` parameters, `fisher_diags`
from a previous task, and `prev_params` (snapshot from that task). Returns a
scalar tensor.

#### train_task(model, x, y, optimizer, epochs, batch_size, fisher_diags=None, prev_params=None, lambda_ewc=0.0)

Train `model` on a single task for the specified number of epochs. If
`fisher_diags` and `prev_params` are provided, adds EWC regularization with
weight `lambda_ewc`.

#### run_training(config)

Run continual learning experiment with EWC on synthetic tasks according to
`config`. Trains sequential tasks, computing Fisher information and parameter
snapshots after each task.

#### main()

CLI entry point. Parses `--config` and runs training.

### Classes

#### SimpleMLP

Simple multi-layer perceptron for classification.

**Constructor**: `SimpleMLP(input_dim, hidden_dim, num_classes)`

- `input_dim`: Input feature dimension.
- `hidden_dim`: Hidden layer dimension.
- `num_classes`: Number of output classes.

**Methods**:

- `forward(x)`: Standard forward pass; returns logits of shape
  `(N, num_classes)`.

