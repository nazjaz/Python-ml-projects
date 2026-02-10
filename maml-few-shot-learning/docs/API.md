# MAML Few-Shot Learning API Documentation

## Module: src.main

### Functions

#### _load_config(config_path)

Load YAML configuration file. Returns configuration dict. Raises FileNotFoundError if path does not exist.

#### _setup_logging(level, log_file)

Configure logging to console and optionally to a rotating log file.

#### sample_few_shot_task(n_way, k_shot, query_size, input_dim, device, seed=None)

Sample one N-way K-shot classification task. Each class is a random Gaussian prototype; support and query points are prototype + noise. Returns `(support_x, support_y, query_x, query_y)`:

- support_x: (n_way * k_shot, input_dim)
- support_y: (n_way * k_shot,) long, class indices in [0, n_way - 1]
- query_x: (query_size, input_dim)
- query_y: (query_size,) long

#### maml_inner_outer_step(model, support_x, support_y, query_x, query_y, inner_lr, inner_steps, first_order)

Perform one MAML inner adaptation and return query loss. Inner loop clones parameters and takes `inner_steps` gradient steps on support cross-entropy. Query loss is computed with adapted parameters. If `first_order` is False, meta-gradient flows through inner updates; if True (FOMAML), gradient through inner loop is stopped. Returns scalar query loss tensor.

#### run_training(config)

Run MAML meta-training on synthetic few-shot tasks. Uses config sections: model, maml, training, data, logging.

#### main()

CLI entry point. Parses `--config` and runs training.

### Classes

#### FewShotMLP

MLP that supports forward with an explicit parameter list for MAML inner loop.

**Constructor**: `FewShotMLP(input_dim, hidden_dim, num_classes, num_layers=2)`

- input_dim: Input feature dimension
- hidden_dim: Hidden layer dimension
- num_classes: Output class count
- num_layers: Number of hidden layers (>= 1)

**Methods**:

- `forward(x)`: Standard forward using module parameters. Returns logits (N, num_classes).
- `forward_with_params(x, params)`: Forward using list of (weight, bias) tensors in layer order. Returns logits (N, num_classes).
- `get_param_list()`: Return list of parameters in the order expected by forward_with_params.
