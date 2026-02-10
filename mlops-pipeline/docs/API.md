# MLOps Pipeline API Documentation

## Module: src.main

### Functions

#### _load_config(config_path)

Load YAML configuration file. Returns configuration dictionary. Raises FileNotFoundError if path does not exist.

#### _setup_logging(level, log_file)

Configure logging to console and optionally to a rotating log file.

#### generate_synthetic_data(n_samples, n_features, random_state)

Generate synthetic binary classification DataFrame for pipeline demo. Returns DataFrame with numeric feature columns and "target" column.

#### train_model(X_train, y_train, config)

Train a classifier from config (training.model_type, e.g. "logistic"). Returns fitted sklearn estimator.

#### run_pipeline(config_path)

Run end-to-end MLOps pipeline: ingest and version data, check monitor, retrain if needed (no model, low accuracy, or drift), evaluate and record metrics. Returns summary dict with data_version, model_version, retrained, accuracy.

#### main()

CLI entry point. Parses --config and runs run_pipeline.

### Classes

#### DataVersioner

Content-addressed data versioning with metadata.

**Constructor**: `DataVersioner(base_dir, hash_algorithm="sha256")`

- base_dir: Root directory for versioned data
- hash_algorithm: Hash algorithm for content addressing

**Methods**:

- `register(df, version_id=None)`: Register DataFrame; returns version id (content hash or provided id).
- `list_versions()`: List version ids, newest first.
- `load(version_id)`: Load (DataFrame, metadata) for a version.
- `get_latest_version()`: Return latest version id or None.

#### ModelRegistry

Versioned model storage with metadata.

**Constructor**: `ModelRegistry(base_dir)`

**Methods**:

- `save(model, version_id, metadata=None)`: Save model and metadata.
- `load(version_id)`: Load (model, metadata).
- `list_versions()`: List version ids, newest first.
- `get_latest_version()`: Return latest version id or None.

#### ModelMonitor

Metrics history and drift-based retrain triggers.

**Constructor**: `ModelMonitor(monitoring_dir, metrics_file, drift_file, accuracy_min_threshold, drift_threshold, drift_min_samples)`

**Methods**:

- `record_metrics(accuracy, data_version, model_version, extra)`: Append metrics record.
- `get_latest_accuracy()`: Return most recent accuracy or None.
- `update_reference_distribution(feature_means)`: Set reference feature means for drift.
- `compute_drift(feature_means)`: Return relative L1 drift vs reference.
- `should_retrain(current_accuracy, current_feature_means, no_model_yet)`: Return True if retrain recommended.
