# Multi-Task Learning API Documentation

## Module: src.main

### Functions

#### _load_config(config_path)

Load YAML configuration file. Returns configuration dict. Raises FileNotFoundError if path does not exist.

#### _setup_logging(level, log_file)

Configure logging to console and optionally to a rotating log file.

#### multi_task_loss(logits_list, labels_list, weights=None)

Compute weighted sum of cross-entropy losses for each task. **logits_list**: list of logits tensors (N, num_classes_i). **labels_list**: list of label tensors (N,) long. **weights**: optional list of per-task weights. Returns scalar loss tensor.

#### generate_synthetic_multi_task_data(num_samples, input_dim, task_num_classes, device, seed=None)

Generate synthetic data: one feature matrix and one label tensor per task. Returns (features, labels_list). features: (num_samples, input_dim). labels_list: list of (num_samples,) integer class indices.

#### run_training(config)

Train multi-task model on synthetic data. Uses config sections: model (input_dim, shared_hidden, shared_dim, tasks), training (epochs, batch_size, learning_rate, loss_weights), data (num_train, random_seed).

#### main()

CLI entry point. Parses --config and runs training.

### Classes

#### SharedEncoder

MLP that maps input to shared representation.

**Constructor**: `SharedEncoder(input_dim, hidden_dims, shared_dim)`

- input_dim: Input feature dimension
- hidden_dims: List of hidden layer sizes
- shared_dim: Output representation dimension

**Methods**:
- `forward(x)`: Returns tensor of shape (N, shared_dim).

**Attributes**:
- shared_dim: Output dimension

#### TaskHead

Task-specific linear head for classification.

**Constructor**: `TaskHead(shared_dim, num_classes)`

**Methods**:
- `forward(shared)`: Returns logits (N, num_classes).

**Attributes**:
- num_classes: Number of classes

#### MultiTaskModel

Full model: shared encoder plus one task head per task.

**Constructor**: `MultiTaskModel(input_dim, shared_hidden, shared_dim, task_configs)`

- task_configs: List of dicts with "type": "classification" and "num_classes".

**Methods**:
- `forward(x)`: Returns list of logits tensors, one per task.
- `num_tasks()`: Returns number of tasks.
