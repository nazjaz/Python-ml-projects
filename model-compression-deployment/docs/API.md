# Model Compression Deployment API Documentation

## Module: src.main

### Functions

#### _load_config(config_path)

Load YAML configuration file. Returns configuration dict. Raises FileNotFoundError if path does not exist.

#### _setup_logging(level, log_file)

Configure logging to console and optionally to a rotating log file.

#### apply_quantization(model, dtype="qint8")

Apply dynamic quantization to Linear layers. Uses torch.quantization.quantize_dynamic. Returns a new quantized model. **dtype**: e.g. "qint8".

#### apply_pruning(model, amount, make_permanent=True)

Apply L1 unstructured pruning to all Linear layers. **amount**: fraction in (0, 1) of parameters to prune. **make_permanent**: if True, remove reparametrization so masks are permanent. Modifies model in place.

#### distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)

Compute distillation loss: alpha * KL(soft_teacher || soft_student) * T^2 + (1 - alpha) * CE(student, labels). **temperature**: softmax temperature. **alpha**: weight for soft term. Returns scalar loss tensor.

#### count_parameters(model)

Return total number of trainable parameters.

#### generate_synthetic_data(num_samples, input_dim, num_classes, device, seed=None)

Generate synthetic classification data. Returns (features, labels).

#### evaluate_accuracy(model, x, y, batch_size)

Compute classification accuracy of model on (x, y). Returns float in [0, 1].

#### run_demo(config)

Train teacher MLP, optionally apply quantization and pruning, train student via distillation. Logs accuracies and parameter counts.

#### main()

CLI entry point. Parses --config and runs run_demo.

### Classes

#### MLP

MLP classifier with configurable hidden layers.

**Constructor**: `MLP(input_dim, hidden_dims, num_classes)`

- input_dim: Input feature dimension
- hidden_dims: List of hidden layer sizes
- num_classes: Number of output classes

**Methods**: `forward(x)` returns logits (N, num_classes).
