# K-Nearest Neighbors (KNN) Classifier

A Python implementation of k-nearest neighbors algorithm from scratch with different distance metrics and k-value optimization. This is the twenty-fifth project in the ML learning series, focusing on understanding instance-based learning and the KNN algorithm.

## Project Title and Description

The K-Nearest Neighbors Classifier tool provides a complete implementation of the KNN algorithm from scratch, including multiple distance metrics (Euclidean, Manhattan, Minkowski, Hamming, Cosine) and k-value optimization using cross-validation. It helps users understand how instance-based learning works and how to choose optimal hyperparameters.

This tool solves the problem of learning KNN fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates distance metrics, k-value selection, and cross-validation for hyperparameter optimization.

**Target Audience**: Beginners learning machine learning, students studying instance-based learning algorithms, and anyone who needs to understand KNN and distance metrics from scratch.

## Features

- KNN classifier implementation from scratch
- Multiple distance metrics:
  - Euclidean distance
  - Manhattan (L1) distance
  - Minkowski distance (configurable p)
  - Hamming distance (for categorical data)
  - Cosine distance
- K-value optimization using cross-validation
- Class predictions
- Probability predictions
- Feature scaling (standardization)
- Multi-class classification support
- Cross-validation for hyperparameter tuning
- Optimization results visualization
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for various input types (lists, numpy arrays, pandas DataFrames)

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/knn-classifier
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python src/main.py --input sample.csv --target label
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  k: 5
  distance_metric: "euclidean"
  metric_params:
    p: 2.0
  scale_features: true
  k_range: [1, 3, 5, 7, 9, 11, 13, 15]
  cv_folds: 5
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.k`: Number of neighbors (default: 5)
- `model.distance_metric`: Distance metric. Options: "euclidean", "manhattan", "minkowski", "hamming", "cosine" (default: "euclidean")
- `model.metric_params.p`: Power parameter for Minkowski distance (default: 2.0)
- `model.scale_features`: Whether to scale features (default: true)
- `model.k_range`: List of k values to test during optimization (default: [1, 3, 5, 7, 9, 11, 13, 15])
- `model.cv_folds`: Number of cross-validation folds for optimization (default: 5)

## Usage

### Basic Usage

```python
from src.main import KNNClassifier
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Initialize and fit model
knn = KNNClassifier(k=3)
knn.fit(X, y)

# Make predictions
predictions = knn.predict(X)
print(f"Predictions: {predictions}")

# Get probabilities
probabilities = knn.predict_proba(X)
print(f"Probabilities: {probabilities}")

# Calculate accuracy
accuracy = knn.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### With Different Distance Metrics

```python
from src.main import KNNClassifier
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Euclidean distance (default)
knn_euclidean = KNNClassifier(k=3, distance_metric="euclidean")
knn_euclidean.fit(X, y)

# Manhattan distance
knn_manhattan = KNNClassifier(k=3, distance_metric="manhattan")
knn_manhattan.fit(X, y)

# Cosine distance
knn_cosine = KNNClassifier(k=3, distance_metric="cosine")
knn_cosine.fit(X, y)
```

### K-Value Optimization

```python
from src.main import KNNOptimizer
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Optimize k-value
optimizer = KNNOptimizer(
    k_range=[1, 3, 5, 7, 9],
    cv_folds=5,
    scale_features=True,
)
results = optimizer.optimize(X, y)

print(f"Best k: {results['best_k']}")
print(f"Best score: {results['best_score']:.4f}")

# Plot optimization results
optimizer.plot_optimization_results()
```

### With Pandas DataFrame

```python
from src.main import KNNClassifier
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]]
y = df["target"]

knn = KNNClassifier(k=5)
knn.fit(X, y)

predictions = knn.predict(X)
probabilities = knn.predict_proba(X)
```

### Command-Line Usage

Basic training:

```bash
python src/main.py --input data.csv --target label
```

With specific k value:

```bash
python src/main.py --input data.csv --target label --k 7
```

With different distance metric:

```bash
python src/main.py --input data.csv --target label --distance manhattan
```

Optimize k-value:

```bash
python src/main.py --input data.csv --target label --optimize-k
```

Optimize with custom k range:

```bash
python src/main.py --input data.csv --target label --optimize-k --k-range "1:20"
```

Plot optimization results:

```bash
python src/main.py --input data.csv --target label --optimize-k --plot
```

Save predictions:

```bash
python src/main.py --input data.csv --target label --output predictions.csv
```

### Complete Example

```python
from src.main import KNNClassifier, KNNOptimizer
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2) * 2
y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

# Optimize k-value
print("Optimizing k-value...")
optimizer = KNNOptimizer(
    k_range=list(range(1, 21)),
    cv_folds=5,
    scale_features=True,
)
results = optimizer.optimize(X, y)

print(f"\nBest k: {results['best_k']}")
print(f"Best CV score: {results['best_score']:.4f}")

# Train with optimal k
knn = KNNClassifier(
    k=results["best_k"],
    distance_metric="euclidean",
    scale_features=True,
)
knn.fit(X, y)

print(f"\nAccuracy: {knn.score(X, y):.4f}")

# Plot optimization results
optimizer.plot_optimization_results()

# Compare different distance metrics
print("\nComparing distance metrics:")
for metric in ["euclidean", "manhattan", "cosine"]:
    knn = KNNClassifier(k=results["best_k"], distance_metric=metric)
    knn.fit(X, y)
    score = knn.score(X, y)
    print(f"{metric}: {score:.4f}")
```

## Project Structure

```
knn-classifier/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── src/
│   └── main.py              # Main implementation
├── tests/
│   └── test_main.py         # Unit tests
├── docs/
│   └── API.md               # API documentation
└── logs/
    └── .gitkeep             # Log directory
```

### File Descriptions

- `src/main.py`: Core implementation with `KNNClassifier` and `KNNOptimizer` classes
- `config.yaml`: Configuration file for model and optimization settings
- `tests/test_main.py`: Comprehensive unit tests
- `docs/API.md`: Detailed API documentation
- `logs/`: Directory for application logs

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

Tests cover:
- Distance metrics (Euclidean, Manhattan, Minkowski, Hamming, Cosine)
- Model initialization
- Training and fitting
- Prediction
- Probability prediction
- Accuracy calculation
- K-value optimization
- Cross-validation
- Error handling
- Different input types

## Understanding K-Nearest Neighbors

### KNN Algorithm

KNN is an instance-based learning algorithm that classifies samples based on the majority class of their k nearest neighbors:

1. Calculate distance from test sample to all training samples
2. Select k nearest neighbors
3. Predict class based on majority vote

### Distance Metrics

**Euclidean Distance:**
```
d(x, y) = √(Σ(x_i - y_i)²)
```
- Most common metric
- Works well for continuous features
- Sensitive to scale

**Manhattan Distance (L1):**
```
d(x, y) = Σ|x_i - y_i|
```
- Less sensitive to outliers
- Good for high-dimensional data
- City-block distance

**Minkowski Distance:**
```
d(x, y) = (Σ|x_i - y_i|^p)^(1/p)
```
- Generalization of Euclidean (p=2) and Manhattan (p=1)
- Configurable p parameter

**Hamming Distance:**
```
d(x, y) = (1/n) * Σ(x_i ≠ y_i)
```
- For categorical data
- Proportion of differing elements

**Cosine Distance:**
```
d(x, y) = 1 - (x·y) / (||x|| ||y||)
```
- Measures angle between vectors
- Good for high-dimensional sparse data
- Normalized by vector magnitudes

### K-Value Selection

Choosing the right k is crucial:
- **Small k (k=1)**: High variance, sensitive to noise
- **Large k**: High bias, smoother decision boundaries
- **Optimal k**: Balance between bias and variance

**Optimization Strategy:**
- Use cross-validation to test different k values
- Choose k with highest cross-validation score
- Consider odd k values to avoid ties in binary classification

## Troubleshooting

### Common Issues

**Issue**: Poor predictions

**Solution**: 
- Scale features (enabled by default)
- Optimize k-value using cross-validation
- Try different distance metrics
- Check data quality

**Issue**: Slow predictions

**Solution**: 
- KNN is inherently slow for large datasets
- Consider reducing number of features
- Use approximate nearest neighbor methods for very large datasets

**Issue**: All predictions same class

**Solution**: 
- Check class balance
- Try different k values
- Verify feature scaling
- Check for data leakage

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: k cannot be greater than number of samples`: Reduce k value
- `ValueError: k must be at least 1`: Use k >= 1

## Best Practices

1. **Always scale features**: Distance metrics are sensitive to scale
2. **Optimize k-value**: Use cross-validation to find optimal k
3. **Try different metrics**: Different metrics work better for different data types
4. **Use odd k for binary classification**: Avoid ties
5. **Consider computational cost**: KNN is slow for large datasets
6. **Handle class imbalance**: May need special handling for imbalanced classes

## Real-World Applications

- Classification problems (image recognition, text classification)
- Recommendation systems
- Anomaly detection
- Pattern recognition
- Medical diagnosis
- Educational purposes for learning instance-based learning

## Contributing

### Development Setup

1. Clone the repository
2. Create virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Run tests: `pytest tests/`

### Code Style Guidelines

- Follow PEP 8 strictly
- Maximum line length: 88 characters
- Use type hints for all functions
- Include docstrings for all public functions and classes
- Write tests for all new functionality

### Pull Request Process

1. Create feature branch
2. Implement changes with tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit pull request with clear description

## License

This project is part of the Python ML Projects collection. See LICENSE file in parent directory for details.
