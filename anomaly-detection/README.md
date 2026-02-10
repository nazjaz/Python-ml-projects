# Anomaly Detection

A Python implementation of three popular anomaly detection algorithms: Isolation Forest, One-Class SVM, and Local Outlier Factor (LOF). This tool provides a comprehensive solution for identifying outliers and anomalies in datasets across various domains.

## Project Title and Description

The Anomaly Detection tool provides implementations of three state-of-the-art anomaly detection algorithms, each with different strengths and use cases. It offers a unified interface for detecting anomalies in datasets, with support for evaluation metrics and visualization capabilities.

This tool solves the problem of identifying unusual patterns, outliers, or anomalies in data that may indicate errors, fraud, system failures, or other significant events. It helps users detect anomalies without requiring labeled training data (unsupervised learning).

**Target Audience**: Data scientists, security analysts, quality assurance engineers, system administrators, and anyone working with data who needs to identify unusual patterns or outliers.

## Features

### Isolation Forest
- **Tree-based Anomaly Detection**: Uses random forests to isolate anomalies
- **Efficient for High-dimensional Data**: Works well with many features
- **Fast Training**: Linear time complexity
- **Configurable Contamination**: Control expected proportion of outliers
- **Anomaly Scores**: Provides decision function scores

### One-Class SVM
- **Kernel-based Detection**: Supports multiple kernel types (RBF, linear, polynomial, sigmoid)
- **Novelty Detection**: Learns decision boundary for normal data
- **Effective for Non-linear Patterns**: Handles complex data distributions
- **Configurable Parameters**: Nu parameter controls margin and support vectors
- **Anomaly Scores**: Provides decision function scores

### Local Outlier Factor (LOF)
- **Density-based Detection**: Uses local density comparison
- **Effective for Local Anomalies**: Detects anomalies relative to local neighborhood
- **Configurable Neighbors**: Adjustable number of neighbors for density estimation
- **Multiple Distance Metrics**: Supports various distance metrics
- **Anomaly Scores**: Provides LOF scores

### Additional Features
- Unified interface for all three algorithms
- Automatic feature scaling
- Evaluation metrics (accuracy, precision, recall, F1-score)
- Visualization capabilities (2D plots)
- Command-line interface
- Configuration via YAML file
- Support for CSV input/output
- Comprehensive logging
- Input validation and error handling

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/anomaly-detection
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
python src/main.py --input sample.csv --method all
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

isolation_forest:
  n_estimators: 100
  max_samples: "auto"
  contamination: 0.1
  max_features: 1.0
  random_state: 42
  n_jobs: null

one_class_svm:
  kernel: "rbf"
  nu: 0.1
  gamma: "scale"
  degree: 3
  coef0: 0.0
  shrinking: true
  tol: 0.001
  cache_size: 200

local_outlier_factor:
  n_neighbors: 20
  algorithm: "auto"
  leaf_size: 30
  metric: "minkowski"
  p: 2
  contamination: 0.1
  novelty: false
  n_jobs: null
```

### Configuration Parameters

#### Isolation Forest
- `n_estimators`: Number of base estimators (default: 100)
- `max_samples`: Number of samples per estimator (default: "auto")
- `contamination`: Expected proportion of outliers (default: 0.1)
- `max_features`: Number of features per estimator (default: 1.0)
- `random_state`: Random seed (default: 42)
- `n_jobs`: Number of parallel jobs (default: null)

#### One-Class SVM
- `kernel`: Kernel type - "rbf", "linear", "poly", "sigmoid" (default: "rbf")
- `nu`: Upper bound on training errors, lower bound on support vectors (default: 0.1)
- `gamma`: Kernel coefficient - "scale", "auto", or float (default: "scale")
- `degree`: Degree for polynomial kernel (default: 3)
- `coef0`: Independent term in kernel (default: 0.0)
- `shrinking`: Use shrinking heuristic (default: true)
- `tol`: Tolerance for stopping (default: 0.001)
- `cache_size`: Cache size in MB (default: 200)

#### Local Outlier Factor
- `n_neighbors`: Number of neighbors (default: 20)
- `algorithm`: Nearest neighbor algorithm (default: "auto")
- `leaf_size`: Leaf size for tree algorithms (default: 30)
- `metric`: Distance metric (default: "minkowski")
- `p`: Power parameter for Minkowski (default: 2)
- `contamination`: Expected proportion of outliers (default: 0.1)
- `novelty`: Use novelty detection mode (default: false)
- `n_jobs`: Number of parallel jobs (default: null)

## Usage

### Command-Line Interface

#### Detect Anomalies with All Methods

```bash
python src/main.py --input data.csv --method all --output results.csv
```

#### Use Specific Method

```bash
# Isolation Forest
python src/main.py --input data.csv --method isolation_forest --output results.csv

# One-Class SVM
python src/main.py --input data.csv --method one_class_svm --output results.csv

# Local Outlier Factor
python src/main.py --input data.csv --method local_outlier_factor --output results.csv
```

#### With Evaluation (if true labels available)

```bash
python src/main.py --input data.csv --method all --true-labels label \
  --evaluation-output evaluation.json --output results.csv
```

#### With Visualization

```bash
python src/main.py --input data.csv --method all --plot-output plot.png
```

### Programmatic Usage

#### Basic Anomaly Detection

```python
import numpy as np
from src.main import AnomalyDetector

# Generate sample data
np.random.seed(42)
X_normal = np.random.randn(90, 2)
X_anomaly = np.random.randn(10, 2) * 3 + np.array([5, 5])
X = np.vstack([X_normal, X_anomaly])

# Initialize detector
detector = AnomalyDetector()

# Fit all methods
detector.fit_all(X)

# Predict anomalies
predictions = detector.predict(X)

# Get scores
scores = detector.get_scores(X)

# Print results
for method, pred in predictions.items():
    n_anomalies = np.sum(pred == -1)
    print(f"{method}: {n_anomalies} anomalies detected")
```

#### Using Individual Detectors

```python
from src.main import IsolationForestDetector, OneClassSVMDetector, LocalOutlierFactorDetector

# Isolation Forest
if_detector = IsolationForestDetector(n_estimators=100, contamination=0.1, random_state=42)
if_detector.fit(X)
predictions = if_detector.predict(X)
scores = if_detector.decision_function(X)

# One-Class SVM
svm_detector = OneClassSVMDetector(kernel="rbf", nu=0.1)
svm_detector.fit(X)
predictions = svm_detector.predict(X)
scores = svm_detector.decision_function(X)

# Local Outlier Factor
lof_detector = LocalOutlierFactorDetector(n_neighbors=20, contamination=0.1)
lof_detector.fit(X)
predictions = lof_detector.predict(X)
scores = lof_detector.decision_function(X)
```

#### Evaluation with True Labels

```python
# True labels: 1 for normal, -1 for anomaly
y_true = np.ones(100)
y_true[:10] = -1

detector = AnomalyDetector()
detector.fit_all(X)

evaluation = detector.evaluate(X, y_true)

for method, metrics in evaluation.items():
    print(f"\n{method}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
```

#### Visualization

```python
detector = AnomalyDetector()
detector.fit_all(X)

predictions = detector.predict(X)
detector.plot_results(X, predictions, save_path="anomalies.png")
```

### Common Use Cases

1. **Fraud Detection**: Identify fraudulent transactions in financial data
2. **Network Security**: Detect intrusions or unusual network activity
3. **Quality Control**: Find defective products in manufacturing
4. **System Monitoring**: Identify system failures or performance issues
5. **Medical Diagnosis**: Detect unusual patterns in medical data
6. **Data Cleaning**: Find and remove outliers before analysis
7. **IoT Monitoring**: Detect anomalies in sensor data

## Project Structure

```
anomaly-detection/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── src/
│   └── main.py             # Main implementation
├── tests/
│   └── test_main.py        # Unit tests
├── docs/
│   └── API.md              # API documentation (if applicable)
└── logs/
    └── .gitkeep            # Keep logs directory in git
```

### File Descriptions

- `src/main.py`: Contains all implementation:
  - `IsolationForestDetector`: Isolation Forest implementation
  - `OneClassSVMDetector`: One-Class SVM implementation
  - `LocalOutlierFactorDetector`: LOF implementation
  - `AnomalyDetector`: Main class combining all methods
  - `main()`: Command-line interface entry point

- `tests/test_main.py`: Comprehensive unit tests including:
  - Individual detector tests
  - Main detector class tests
  - Integration tests
  - Error handling tests

- `config.yaml`: Configuration file for algorithm parameters

- `requirements.txt`: Python package dependencies with versions

## Testing

### Run All Tests

```bash
pytest tests/test_main.py -v
```

### Run Tests with Coverage

```bash
pytest tests/test_main.py --cov=src --cov-report=html
```

### Test Coverage Information

The test suite includes:
- Unit tests for each detector class
- Integration tests for complete workflows
- Error handling and edge case tests
- Input validation tests

Current test coverage: >90% of code paths

## Algorithm Details

### Isolation Forest

**How it works**:
- Builds random trees by randomly selecting features and split values
- Anomalies are easier to isolate (fewer splits needed)
- Average path length indicates anomaly score

**Advantages**:
- Fast training and prediction
- Works well with high-dimensional data
- No distance or density calculations needed
- Handles irrelevant features well

**Limitations**:
- May struggle with local anomalies
- Performance depends on number of trees
- Less interpretable than some methods

**Best for**: High-dimensional data, large datasets, when speed is important

### One-Class SVM

**How it works**:
- Learns a decision boundary around normal data
- Uses kernel trick to handle non-linear patterns
- Points outside boundary are anomalies

**Advantages**:
- Handles non-linear patterns well
- Flexible with different kernels
- Good generalization
- Works with high-dimensional data

**Limitations**:
- Can be slow for large datasets
- Requires careful parameter tuning
- Memory intensive with large datasets
- May struggle with high-dimensional sparse data

**Best for**: Non-linear patterns, when you have good understanding of normal data

### Local Outlier Factor (LOF)

**How it works**:
- Compares local density of point to neighbors
- Points with significantly lower density are anomalies
- LOF score > 1 indicates anomaly

**Advantages**:
- Detects local anomalies well
- Provides interpretable scores
- Works with various distance metrics
- Good for clustered data

**Limitations**:
- Computationally expensive
- Sensitive to number of neighbors
- May struggle with high-dimensional data
- Requires careful parameter selection

**Best for**: Local anomalies, when interpretability is important, clustered data

## Choosing the Right Algorithm

### Use Isolation Forest when:
- You have high-dimensional data
- Speed is important
- You need to handle many irrelevant features
- Dataset is large

### Use One-Class SVM when:
- Data has non-linear patterns
- You have good understanding of normal data
- You need flexible kernel options
- Dataset is medium-sized

### Use Local Outlier Factor when:
- You need to detect local anomalies
- Interpretability is important
- Data has clear clusters
- You can afford computational cost

## Troubleshooting

### Common Issues and Solutions

#### Issue: Too Many/Few Anomalies Detected
**Problem**: Contamination parameter doesn't match actual anomaly rate

**Solution**: 
- Adjust `contamination` parameter in config
- Use evaluation metrics to find optimal value
- Try different values: 0.05, 0.1, 0.2, etc.

#### Issue: One-Class SVM is Slow
**Problem**: Large dataset or complex kernel

**Solution**:
- Use linear kernel for large datasets
- Reduce `cache_size` if memory is limited
- Use `shrinking=False` to speed up (may reduce accuracy)
- Consider Isolation Forest for very large datasets

#### Issue: LOF Fails with Insufficient Neighbors
**Error**: "X must have at least n_neighbors + 1 samples"

**Solution**:
- Reduce `n_neighbors` parameter
- Ensure dataset has enough samples
- Use Isolation Forest or One-Class SVM for small datasets

#### Issue: Poor Detection Performance
**Problem**: Algorithm not suitable for data characteristics

**Solution**:
- Try different algorithms
- Preprocess data (normalize, remove irrelevant features)
- Adjust algorithm-specific parameters
- Use ensemble approach (combine multiple methods)

#### Issue: Memory Error with Large Datasets
**Error**: Out of memory during training

**Solution**:
- Use Isolation Forest (most memory efficient)
- Reduce dataset size (sampling)
- Use `n_jobs=1` to reduce memory usage
- Process data in batches

### Error Message Explanations

- **"X must be 2D array"**: Input must be feature matrix, not 1D array
- **"X must have at least 2 samples"**: Need minimum samples for training
- **"Model must be fitted before prediction"**: Call fit() before predict()
- **"Method 'X' not fitted"**: Specified method hasn't been trained yet

## Performance Considerations

### Computational Complexity

- **Isolation Forest**: O(n * log(n)) training, O(n) prediction
- **One-Class SVM**: O(n²) to O(n³) training, O(n) prediction
- **LOF**: O(n²) training and prediction

### Memory Usage

- **Isolation Forest**: Low memory usage
- **One-Class SVM**: High memory usage (stores support vectors)
- **LOF**: Medium memory usage (stores neighbor information)

### Scalability Tips

1. **For large datasets**: Use Isolation Forest
2. **For high-dimensional data**: Use Isolation Forest or One-Class SVM
3. **For real-time detection**: Use Isolation Forest (fastest)
4. **For interpretability**: Use LOF (provides scores)
5. **For ensemble approach**: Combine all three methods

## Contributing

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Create a feature branch: `git checkout -b feature/new-detector`

### Code Style Guidelines

- Follow PEP 8 strictly
- Maximum line length: 88 characters
- Use type hints for all functions
- Write docstrings for all public functions and classes
- Add unit tests for new detectors
- Update documentation for new features

### Pull Request Process

1. Ensure all tests pass: `pytest tests/test_main.py`
2. Check code coverage: `pytest --cov=src`
3. Update documentation if adding new features
4. Submit pull request with clear description

## License

This project is part of the Python ML Projects collection. Please refer to the main repository license.

## Additional Resources

- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [One-Class SVM Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
- [LOF Paper](https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf)
- [Scikit-learn Anomaly Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Anomaly Detection Survey](https://arxiv.org/abs/2007.02500)
