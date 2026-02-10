# Support Vector Machine (SVM) for Classification

A Python implementation of Support Vector Machine (SVM) from scratch for classification with support for linear, polynomial, and RBF kernels. This is the thirty-third project in the ML learning series, focusing on understanding SVM, kernel methods, and the SMO optimization algorithm.

## Project Title and Description

The Support Vector Machine tool provides a complete implementation of SVM from scratch, including the SMO (Sequential Minimal Optimization) algorithm, multiple kernel functions, and decision boundary visualization. It helps users understand how SVM works, how different kernels affect classification, and how to tune hyperparameters for optimal performance.

This tool solves the problem of learning SVM fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates kernel methods, margin maximization, support vectors, and optimization techniques from scratch.

**Target Audience**: Beginners learning machine learning, students studying classification algorithms, and anyone who needs to understand SVM and kernel methods from scratch.

## Features

- SVM implementation from scratch
- SMO (Sequential Minimal Optimization) algorithm
- Multiple kernel types: linear, polynomial, RBF
- Binary classification support
- Class probability prediction
- Decision boundary visualization (2D features)
- Support vector identification
- Configurable regularization parameter C
- Automatic gamma calculation for RBF kernel
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
cd /path/to/Python-ml-projects/svm-classification
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
python src/main.py --input sample.csv --target class_label --kernel linear
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  kernel: "linear"
  C: 1.0
  degree: 3
  gamma: null
  coef0: 0.0
  tol: 0.001
  max_iter: 1000
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.kernel`: Kernel type. Options: "linear", "poly", "rbf" (default: "linear")
- `model.C`: Regularization parameter (default: 1.0)
- `model.degree`: Degree for polynomial kernel (default: 3)
- `model.gamma`: Kernel coefficient for RBF and polynomial. If null, uses 1/n_features (default: null)
- `model.coef0`: Independent term for polynomial kernel (default: 0.0)
- `model.tol`: Tolerance for stopping criterion (default: 0.001)
- `model.max_iter`: Maximum number of iterations (default: 1000)

## Usage

### Basic Usage

```python
from src.main import SVM
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize and fit model
svm = SVM(kernel="linear", C=1.0)
svm.fit(X, y)

# Predict
predictions = svm.predict(X)
print(f"Predictions: {predictions}")

# Evaluate
accuracy = svm.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### With Different Kernels

```python
from src.main import SVM
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Linear kernel
svm_linear = SVM(kernel="linear", C=1.0)
svm_linear.fit(X, y)

# Polynomial kernel
svm_poly = SVM(kernel="poly", C=1.0, degree=3, gamma=0.1)
svm_poly.fit(X, y)

# RBF kernel
svm_rbf = SVM(kernel="rbf", C=1.0, gamma=0.1)
svm_rbf.fit(X, y)
```

### Class Probabilities

```python
from src.main import SVM
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

svm = SVM(kernel="linear")
svm.fit(X, y)

# Predict probabilities
probabilities = svm.predict_proba(X)
print(f"Probabilities shape: {probabilities.shape}")
```

### Decision Boundary Visualization

```python
from src.main import SVM
import numpy as np

# 2D features required for visualization
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

svm = SVM(kernel="rbf", C=1.0, gamma=0.1)
svm.fit(X, y)

# Plot decision boundary
svm.plot_decision_boundary(X=X, y=y)
```

### With Pandas DataFrame

```python
from src.main import SVM
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2"]].values
y = df["target"].values

svm = SVM(kernel="linear", C=1.0)
svm.fit(X, y)

predictions = svm.predict(X)
```

### Command-Line Usage

Basic SVM with linear kernel:

```bash
python src/main.py --input data.csv --target class_label --kernel linear
```

With polynomial kernel:

```bash
python src/main.py --input data.csv --target class_label --kernel poly --degree 3
```

With RBF kernel:

```bash
python src/main.py --input data.csv --target class_label --kernel rbf --gamma 0.1
```

With custom C parameter:

```bash
python src/main.py --input data.csv --target class_label --kernel linear --C 0.5
```

Plot decision boundary:

```bash
python src/main.py --input data.csv --target class_label --kernel rbf --plot
```

Save predictions:

```bash
python src/main.py --input data.csv --target class_label --kernel linear --output predictions.csv
```

Make predictions on new data:

```bash
python src/main.py --input train.csv --target class_label --kernel linear --predict test.csv --output-predictions predictions.csv
```

### Complete Example

```python
from src.main import SVM
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int)

# Fit with RBF kernel
svm = SVM(kernel="rbf", C=1.0, gamma=0.1)
svm.fit(X, y)

print(f"Support vectors: {len(svm.support_vectors_)}")
print(f"Classes: {svm.classes_}")

# Evaluate
accuracy = svm.score(X, y)
print(f"Accuracy: {accuracy:.4f}")

# Predict
predictions = svm.predict(X)
probabilities = svm.predict_proba(X)

# Visualize
svm.plot_decision_boundary(X=X, y=y)
```

## Project Structure

```
svm-classification/
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

- `src/main.py`: Core implementation with `SVM` class
- `config.yaml`: Configuration file for model settings
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
- Model initialization
- Fitting with different kernels
- Prediction
- Class probabilities
- Kernel functions
- Different C values
- Error handling
- Different input types
- Decision boundary visualization

## Understanding SVM

### Support Vector Machine

SVM finds the optimal hyperplane that separates classes with maximum margin:

**Objective:**
- Maximize margin between classes
- Minimize classification error
- Support vectors are data points closest to decision boundary

### Mathematical Foundation

**Primal Problem:**
```
Minimize: (1/2)||w||² + CΣξᵢ
Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

**Dual Problem:**
```
Maximize: Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼK(xᵢ, xⱼ)
Subject to: 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0
```

Where:
- `w`: Weight vector
- `b`: Bias term
- `C`: Regularization parameter
- `αᵢ`: Lagrange multipliers
- `K(xᵢ, xⱼ)`: Kernel function

### Kernel Functions

**Linear Kernel:**
```
K(xᵢ, xⱼ) = xᵢ · xⱼ
```

**Polynomial Kernel:**
```
K(xᵢ, xⱼ) = (γ(xᵢ · xⱼ) + r)ᵈ
```

Where:
- `γ`: Gamma parameter
- `r`: Coef0 parameter
- `d`: Degree

**RBF (Gaussian) Kernel:**
```
K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)
```

Where:
- `γ`: Gamma parameter (controls influence of each training example)

### SMO Algorithm

Sequential Minimal Optimization (SMO) solves the dual problem:
1. Select two Lagrange multipliers to optimize
2. Optimize these two multipliers
3. Repeat until convergence

**Key Steps:**
- Choose violating pair (αᵢ, αⱼ)
- Optimize while keeping others fixed
- Update bias term
- Check convergence

### Hyperparameters

**C (Regularization Parameter):**
- Small C: Larger margin, more misclassifications allowed
- Large C: Smaller margin, fewer misclassifications
- Typical range: 0.01 to 100

**Gamma (for RBF/Polynomial):**
- Small gamma: Larger influence radius, smoother decision boundary
- Large gamma: Smaller influence radius, more complex boundary
- Auto: 1/n_features

**Degree (for Polynomial):**
- Controls polynomial complexity
- Typical range: 2 to 5

## Troubleshooting

### Common Issues

**Issue**: Poor classification accuracy

**Solution**: 
- Try different kernels (RBF for non-linear data)
- Tune C parameter
- Adjust gamma for RBF kernel
- Check data quality and scaling
- Ensure classes are balanced

**Issue**: Too many support vectors

**Solution**: 
- Increase C parameter
- Use simpler kernel (linear instead of RBF)
- Check for outliers
- Normalize features

**Issue**: Slow training

**Solution**: 
- Reduce max_iter
- Increase tol (tolerance)
- Use linear kernel for large datasets
- Reduce number of samples

**Issue**: Decision boundary not smooth

**Solution**: 
- For RBF: reduce gamma
- For polynomial: reduce degree
- Increase C parameter

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: SVM currently supports binary classification only`: Use binary classification data
- `ValueError: X and y must have the same length`: Ensure matching lengths
- `ValueError: Unknown kernel`: Use valid kernel type ("linear", "poly", "rbf")

## Best Practices

1. **Scale features**: SVM is sensitive to feature scale, normalize features
2. **Choose appropriate kernel**: Linear for linear data, RBF for non-linear
3. **Tune C parameter**: Use cross-validation to find optimal C
4. **Tune gamma for RBF**: Smaller gamma for smoother boundaries
5. **Visualize decision boundary**: Helps understand model behavior (2D only)
6. **Check support vectors**: Few support vectors indicate good generalization
7. **Use appropriate degree**: For polynomial kernel, start with degree 3
8. **Handle imbalanced classes**: Consider class weights or resampling

## Real-World Applications

- Text classification
- Image classification
- Bioinformatics
- Handwriting recognition
- Face detection
- Medical diagnosis
- Educational purposes for learning classification algorithms

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
