# Model Persistence and Versioning

A comprehensive Python implementation for model persistence, versioning, serialization, loading, and metadata tracking for machine learning models.

## Description

This project provides a robust system for managing machine learning models throughout their lifecycle. It includes model serialization (pickle and joblib), version management, comprehensive metadata tracking, and a model registry for easy discovery and comparison of model versions.

Key features include:
- **Model Serialization**: Support for both pickle and joblib formats with optional compression
- **Version Management**: Semantic versioning with automatic version tracking
- **Metadata Tracking**: Comprehensive metadata including model parameters, performance metrics, and custom fields
- **Model Registry**: Centralized registry for all models and versions
- **Model Comparison**: Compare metadata across different model versions
- **CLI Interface**: Command-line tools for model management

## Features

- **Multiple Serialization Formats**: Support for pickle and joblib with compression options
- **Semantic Versioning**: Flexible version string support (e.g., "1.0.0", "v1.2.3")
- **Comprehensive Metadata**: Track model type, parameters, performance metrics, training data info, and custom fields
- **Model Registry**: Centralized JSON-based registry for all models and versions
- **Version Comparison**: Compare metadata across versions using pandas DataFrames
- **Automatic Metadata**: Automatically captures model parameters and type information
- **Scikit-learn Compatible**: Works with any scikit-learn BaseEstimator
- **CLI Tools**: Command-line interface for save, load, list, info, compare, and delete operations
- **YAML Configuration**: Flexible configuration management
- **Detailed Logging**: Comprehensive logging for debugging and audit trails
- **Error Handling**: Robust error handling and validation

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. Clone or navigate to the project directory:
```bash
cd model-persistence-versioning
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses a YAML configuration file (`config.yaml`) for default settings:

```yaml
logging:
  level: INFO
  file: logs/app.log

persistence:
  models_dir: models
  registry_path: null
  serialization_format: joblib
  compress: true
```

### Configuration Parameters

- **models_dir**: Directory to store models (default: "models")
- **registry_path**: Path to registry directory (default: models_dir)
- **serialization_format**: Format to use - "pickle" or "joblib" (default: "joblib")
- **compress**: Whether to compress models (default: true)

## Usage

### Command-Line Interface

#### Save a Model

```bash
python src/main.py save \
  --model-name my_model \
  --version 1.0.0 \
  --model-file model.pkl \
  --metadata metadata.json
```

#### Load a Model

```bash
python src/main.py load \
  --model-name my_model \
  --version 1.0.0 \
  --output loaded_model.pkl
```

#### List All Models

```bash
python src/main.py list
```

#### Get Model Information

```bash
python src/main.py info \
  --model-name my_model \
  --version 1.0.0
```

#### Compare Model Versions

```bash
python src/main.py compare \
  --model-name my_model
```

#### Delete a Model Version

```bash
python src/main.py delete \
  --model-name my_model \
  --version 1.0.0
```

### Programmatic Usage

#### Basic Save and Load

```python
from sklearn.linear_model import LinearRegression
from src.main import ModelPersistence
import numpy as np

# Create and train a model
model = LinearRegression()
X = np.random.randn(100, 5)
y = np.random.randn(100)
model.fit(X, y)

# Initialize persistence manager
persistence = ModelPersistence(models_dir="models")

# Save model with metadata
model_path = persistence.save(
    model,
    model_name="regression_model",
    version="1.0.0",
    accuracy=0.95,
    training_samples=100,
    features=["feature1", "feature2", "feature3"]
)

# Load model
loaded_model = persistence.load("regression_model", "1.0.0")
```

#### Save with Custom Metadata

```python
persistence.save(
    model,
    model_name="my_model",
    version="1.0.0",
    accuracy=0.95,
    f1_score=0.92,
    precision=0.94,
    recall=0.91,
    training_date="2024-01-01",
    dataset_version="v2.1",
    hyperparameters={"n_estimators": 100, "max_depth": 10}
)
```

#### Load with Metadata

```python
model, metadata = persistence.load(
    "my_model",
    version="1.0.0",
    return_metadata=True
)

print(f"Model: {metadata.model_name}")
print(f"Version: {metadata.version}")
print(f"Accuracy: {metadata.metadata['accuracy']}")
print(f"Created: {metadata.created_at}")
```

#### Load Latest Version

```python
# Automatically loads the latest version
model = persistence.load("my_model")
```

#### List Models and Versions

```python
# List all models
models = persistence.list_models()
print(f"Registered models: {models}")

# List versions for a model
versions = persistence.list_versions("my_model")
print(f"Versions: {versions}")
```

#### Get Metadata

```python
metadata = persistence.get_metadata("my_model", "1.0.0")
if metadata:
    print(f"Model type: {metadata.model_type}")
    print(f"Model parameters: {metadata.metadata.get('model_params')}")
```

#### Compare Versions

```python
comparison = persistence.compare_versions("my_model")
print(comparison[['version', 'accuracy', 'f1_score', 'created_at']])
```

#### Using ModelPersistenceManager (with config)

```python
from pathlib import Path
from src.main import ModelPersistenceManager

manager = ModelPersistenceManager(config_path=Path("config.yaml"))

# Save model
model_path = manager.save_model(
    model,
    "my_model",
    "1.0.0",
    accuracy=0.95
)

# Load model
loaded_model = manager.load_model("my_model", "1.0.0")

# Get metadata
metadata = manager.get_model_metadata("my_model", "1.0.0")
```

#### Different Serialization Formats

```python
# Use pickle format
persistence_pickle = ModelPersistence(
    models_dir="models",
    serialization_format="pickle",
    compress=True
)

# Use joblib format (default, better for scikit-learn)
persistence_joblib = ModelPersistence(
    models_dir="models",
    serialization_format="joblib",
    compress=True
)
```

## Project Structure

```
model-persistence-versioning/
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
├── src/
│   └── main.py
├── tests/
│   └── test_main.py
├── docs/
│   └── API.md (if applicable)
├── logs/
│   └── .gitkeep
└── models/
    └── (model files stored here)
```

## Model Storage Structure

Models are stored in a hierarchical structure:

```
models/
├── registry.json
└── model_name/
    └── version/
        ├── model_name_vversion.joblib
        └── model_name_vversion_metadata.json
```

## Metadata Structure

Model metadata includes:

- **Basic Information**: model_name, model_type, version
- **Timestamps**: created_at, updated_at
- **Model Parameters**: Automatically captured from model.get_params()
- **Custom Fields**: Any additional metadata provided during save
- **Performance Metrics**: accuracy, f1_score, precision, recall, etc.
- **Training Information**: training_samples, dataset_version, etc.

Example metadata:

```json
{
  "model_name": "regression_model",
  "model_type": "LinearRegression",
  "version": "1.0.0",
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:00:00",
  "model_params": {
    "fit_intercept": true,
    "normalize": false
  },
  "accuracy": 0.95,
  "training_samples": 1000,
  "features": ["feature1", "feature2", "feature3"]
}
```

## Serialization Formats

### Joblib (Recommended)

- **Advantages**: Optimized for NumPy arrays, faster for scikit-learn models
- **Compression**: Supports gzip compression
- **File Extension**: `.joblib`

### Pickle

- **Advantages**: Standard Python serialization, works with any Python object
- **Compression**: Supports gzip compression (`.pkl.gz`)
- **File Extension**: `.pkl` or `.pkl.gz`

## Version Management

The system supports flexible version strings:

- Semantic versioning: `1.0.0`, `1.1.0`, `2.0.0`
- Custom versions: `v1.2.3`, `alpha-1`, `production-v1`
- Latest version: Automatically determined by sorting versions

## Model Registry

The registry (`registry.json`) maintains a centralized index of all models:

```json
{
  "model_name": {
    "1.0.0": {
      "model_path": "models/model_name/1.0.0/model_name_v1.0.0.joblib",
      "metadata_path": "models/model_name/1.0.0/model_name_v1.0.0_metadata.json",
      "metadata": {...},
      "registered_at": "2024-01-01T12:00:00"
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **"Model must be a scikit-learn BaseEstimator"**
   - Ensure your model inherits from sklearn.base.BaseEstimator
   - Use scikit-learn models or custom estimators that follow the sklearn interface

2. **"Model file not found"**
   - Check that the model was saved successfully
   - Verify the model_name and version are correct
   - Check file permissions

3. **"No versions found for model"**
   - Ensure the model has been saved
   - Check the registry.json file
   - Verify model_name spelling

4. **Serialization errors**
   - Ensure all model dependencies are installed
   - Check that the model is fully trained before saving
   - Try a different serialization format

### Performance Tips

- Use joblib format for scikit-learn models (faster, optimized)
- Enable compression for smaller file sizes
- Use semantic versioning for easier version management
- Regularly clean up old model versions
- Store models on fast storage (SSD) for better performance

## Best Practices

1. **Version Naming**: Use semantic versioning (MAJOR.MINOR.PATCH)
2. **Metadata**: Include comprehensive metadata (metrics, parameters, dataset info)
3. **Backup**: Regularly backup the models directory and registry
4. **Documentation**: Document model versions and their use cases
5. **Testing**: Test model loading after saving
6. **Cleanup**: Remove unused model versions to save space
7. **Security**: Protect model files and registry from unauthorized access

## Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Write docstrings for all public functions and classes
4. Add tests for new features
5. Update README.md if adding new features

## License

This project is part of the Python ML Projects collection.
