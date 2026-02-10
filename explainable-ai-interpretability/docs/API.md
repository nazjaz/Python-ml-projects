# Explainable AI Interpretability API Documentation

## Module: src.main

### Functions

#### _load_config(config_path)

Load YAML configuration file. Returns configuration dict. Raises FileNotFoundError if path does not exist.

#### _setup_logging(level, log_file)

Configure logging to console and optionally to a rotating log file.

#### integrated_gradients(model, x, target_class, baseline=None, steps=50)

Compute integrated gradients attribution for the given target class. **model**: differentiable model returning logits. **x**: input (1, D) or (N, D); first row used. **target_class**: class index. **baseline**: baseline input (default zeros). **steps**: interpolation steps. Returns attribution tensor of shape (D,) for one sample.

#### lime_explain(model, x, num_samples, num_features, kernel_width=0.25, device=None)

LIME: fit a local weighted linear model to approximate predictions. **model**: model returning logits. **x**: single instance (1, D) or (D,). **num_samples**: number of perturbed samples. **num_features**: feature count D. **kernel_width**: exponential kernel width for weighting. Returns 1D numpy array of feature importance (length num_features).

#### get_attention_weights(model)

Return the last stored attention weights if the model has a `last_attention` attribute (e.g. ModelWithAttention). Returns tensor or None.

#### format_attention_for_visualization(attention, num_decimals=3)

Format an attention tensor as a string for logging. **attention**: (num_heads, seq, seq) or (seq, seq). Returns multi-line string.

#### generate_synthetic_data(num_samples, input_dim, num_classes, device, seed=None)

Generate synthetic classification data. Returns (features, labels).

#### run_demo(config)

Train a SimpleClassifier on synthetic data, then run integrated gradients, LIME, and attention visualization (using ModelWithAttention) and log results.

#### main()

CLI entry point. Parses --config and runs run_demo.

### Classes

#### SimpleClassifier

MLP classifier for use with IG and LIME.

**Constructor**: `SimpleClassifier(input_dim, hidden_dim, num_classes)`

**Methods**: `forward(x)` returns logits (N, num_classes).

#### ModelWithAttention

Classifier with one self-attention layer over the feature dimension; stores last attention for visualization.

**Constructor**: `ModelWithAttention(input_dim, embed_dim, num_heads, num_classes)`

**Methods**: `forward(x)` returns logits. After forward, `last_attention` is set (batch, num_heads, seq, seq).

**Attributes**: `last_attention`: last attention tensor or None.
