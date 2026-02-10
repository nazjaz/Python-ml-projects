# Linear Discriminant Analysis (LDA) for Dimensionality Reduction and Classification

A Python implementation of Linear Discriminant Analysis (LDA) from scratch for dimensionality reduction and classification with class separation analysis. This is the thirty-first project in the ML learning series, focusing on understanding supervised dimensionality reduction and LDA algorithms.

## Project Title and Description

The Linear Discriminant Analysis tool provides a complete implementation of LDA from scratch, including dimensionality reduction, classification, and class separation visualization. It helps users understand how LDA works, how to reduce dimensions while maximizing class separation, and how to use LDA for both dimensionality reduction and classification tasks.

This tool solves the problem of learning supervised dimensionality reduction fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates scatter matrix computation, eigenvalue decomposition, class separation maximization, and how to choose the optimal number of components.

**Target Audience**: Beginners learning machine learning, students studying supervised dimensionality reduction techniques, and anyone who needs to understand LDA and class separation from scratch.

## Features

- LDA implementation from scratch
- Supervised dimensionality reduction
- Classification capability
- Class separation maximization
- Explained variance ratio calculation
- Multiple solvers: eigen and SVD
- Shrinkage regularization support
- Class probability prediction
- Flexible n_components selection
- Component visualization with class separation
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for various input types (lists, numpy arrays, pandas DataFrames)
- Multiclass classification support

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/lda-classification
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
python src/main.py --input sample.csv --target class_label --n-components 1
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  n_components: null
  solver: "eigen"
  shrinkage: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.n_components`: Number of components. Options: null (auto: min(n_features, n_classes - 1)), int (top n) (default: null)
- `model.solver`: Solver to use. Options: "eigen" (default), "svd"
- `model.shrinkage`: Shrinkage parameter for regularization (0-1). If null, no shrinkage is applied (default: null)

## Usage

### Basic Usage

```python
from src.main import LDA
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize and fit model
lda = LDA(n_components=1)
lda.fit(X, y)

# Transform data
X_transformed = lda.transform(X)
print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_transformed.shape}")

# Predict classes
predictions = lda.predict(X)
print(f"Predictions: {predictions}")

# Get accuracy
accuracy = lda.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### Classification

```python
from src.main import LDA
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

lda = LDA(n_components=1)
lda.fit(X, y)

# Predict class labels
predictions = lda.predict(X)
print(f"Predictions: {predictions}")

# Predict class probabilities
probabilities = lda.predict_proba(X)
print(f"Probabilities shape: {probabilities.shape}")
```

### Fit and Transform

```python
from src.main import LDA
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

lda = LDA(n_components=1)
X_transformed = lda.fit_transform(X, y)
```

### With SVD Solver

```python
from src.main import LDA
import numpy as np

X = np.random.randn(100, 10)
y = np.random.randint(0, 3, 100)

lda = LDA(n_components=2, solver="svd")
lda.fit(X, y)

X_transformed = lda.transform(X)
print(f"Transformed shape: {X_transformed.shape}")
```

### With Shrinkage Regularization

```python
from src.main import LDA
import numpy as np

X = np.random.randn(100, 10)
y = np.random.randint(0, 3, 100)

lda = LDA(n_components=2, solver="eigen", shrinkage=0.5)
lda.fit(X, y)

X_transformed = lda.transform(X)
```

### Explained Variance Analysis

```python
from src.main import LDA
import numpy as np

X = np.random.randn(100, 10)
y = np.random.randint(0, 3, 100)

lda = LDA(n_components=None)
lda.fit(X, y)

# Get explained variance ratio
explained_variance_ratio = lda.get_explained_variance_ratio()

print("Explained variance ratio by component:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"LD{i+1}: {ratio:.4f}")
```

### With Pandas DataFrame

```python
from src.main import LDA
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]].values
y = df["target"].values

lda = LDA(n_components=2)
lda.fit(X, y)

X_transformed = lda.transform(X)
df_transformed = pd.DataFrame(
    X_transformed, columns=[f"LD{i+1}" for i in range(lda.n_components_)]
)
```

### Command-Line Usage

Basic LDA:

```bash
python src/main.py --input data.csv --target class_label --n-components 1
```

With SVD solver:

```bash
python src/main.py --input data.csv --target class_label --solver svd --n-components 2
```

With shrinkage:

```bash
python src/main.py --input data.csv --target class_label --shrinkage 0.5 --n-components 1
```

Plot components:

```bash
python src/main.py --input data.csv --target class_label --n-components 2 --plot
```

Save transformed data:

```bash
python src/main.py --input data.csv --target class_label --n-components 2 --output transformed.csv
```

Make predictions:

```bash
python src/main.py --input train.csv --target class_label --n-components 1 --predict test.csv --output-predictions predictions.csv
```

### Complete Example

```python
from src.main import LDA
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 100
n_features = 5
n_classes = 3

X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, n_samples)

# Apply LDA
lda = LDA(n_components=2)
lda.fit(X, y)

print(f"Original dimensions: {X.shape}")
print(f"Reduced dimensions: {lda.n_components_}")
print(f"Classes: {lda.classes_}")

# Transform data
X_transformed = lda.transform(X)
print(f"Transformed shape: {X_transformed.shape}")

# Predict classes
predictions = lda.predict(X)
accuracy = lda.score(X, y)
print(f"Classification accuracy: {accuracy:.4f}")

# Analyze variance
if lda.explained_variance_ratio_ is not None:
    print("\nExplained variance ratio by component:")
    for i, ratio in enumerate(lda.explained_variance_ratio_):
        print(f"LD{i+1}: {ratio:.4f}")

# Visualize
lda.plot_components(X=X, y=y)
```

## Project Structure

```
lda-classification/
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

- `src/main.py`: Core implementation with `LDA` class
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
- Fitting with different n_components
- Data transformation
- Classification (predict, predict_proba, score)
- Explained variance calculation
- Different solvers (eigen, svd)
- Shrinkage regularization
- Error handling
- Different input types
- Multiclass classification

## Understanding LDA

### Linear Discriminant Analysis

LDA is a supervised dimensionality reduction technique that finds directions of maximum class separation:

1. **Compute scatter matrices**: Calculate within-class and between-class scatter
2. **Solve generalized eigenvalue problem**: Find directions maximizing class separation
3. **Select components**: Choose top components based on eigenvalues
4. **Transform data**: Project data onto discriminant components
5. **Classify**: Use transformed space for classification

### Mathematical Foundation

**Within-Class Scatter Matrix (Sw):**
```
Sw = (1/n) * Σ Σ (x - μ_c)(x - μ_c)^T
     c  x∈c
```

**Between-Class Scatter Matrix (Sb):**
```
Sb = Σ n_c * (μ_c - μ)(μ_c - μ)^T
    c
```

**Generalized Eigenvalue Problem:**
```
Sb @ v = λ * Sw @ v
```

Or equivalently:
```
Sw^(-1) @ Sb @ v = λ * v
```

Where:
- `v` is eigenvector (discriminant direction)
- `λ` is eigenvalue (separation measure)

**Transformation:**
```
X_transformed = (X - μ) @ V^T
```

Where `V` is matrix of selected eigenvectors.

### Key Differences from PCA

1. **Supervised vs Unsupervised**: LDA uses class labels, PCA does not
2. **Objective**: LDA maximizes class separation, PCA maximizes variance
3. **Components**: LDA limited to min(n_features, n_classes - 1), PCA limited to n_features
4. **Use case**: LDA for classification, PCA for general dimensionality reduction

### Choosing Number of Components

**Maximum Components:**
- Limited to `min(n_features, n_classes - 1)`
- For 2 classes: maximum 1 component
- For 3 classes: maximum 2 components
- For n classes: maximum n-1 components

**Methods:**
1. **Auto**: Use all available components (default)
2. **Fixed number**: Specify exact number of components
3. **Explained variance**: Choose components explaining most separation

### Solver Options

**Eigen Solver:**
- Computes Sw^(-1) @ Sb and finds eigenvalues/eigenvectors
- More intuitive but requires matrix inversion
- Can use shrinkage for regularization

**SVD Solver:**
- Uses singular value decomposition
- More numerically stable
- Avoids matrix inversion

### Shrinkage Regularization

Shrinkage helps when Sw is singular or near-singular:
```
Sw_shrunk = (1 - α) * Sw + α * (trace(Sw) / n) * I
```

Where α is shrinkage parameter (0-1).

## Troubleshooting

### Common Issues

**Issue**: Singular matrix error

**Solution**: 
- Use shrinkage regularization
- Use SVD solver instead of eigen
- Check for linearly dependent features
- Ensure sufficient samples per class

**Issue**: Maximum components limited

**Solution**: 
- LDA is limited to min(n_features, n_classes - 1) components
- This is a mathematical constraint, not a bug
- For more components, use PCA or other methods

**Issue**: Low classification accuracy

**Solution**: 
- Check if classes are linearly separable
- Try different number of components
- Use shrinkage if overfitting
- Ensure balanced classes
- Check data quality

**Issue**: Predictions don't make sense

**Solution**: 
- Verify class labels are correct
- Check feature scaling
- Ensure model was fitted properly
- Use predict_proba to see confidence

### Error Messages

- `ValueError: Model must be fitted before transformation`: Call `fit()` before `transform()` or `predict()`
- `ValueError: LDA requires at least 2 classes`: Provide data with at least 2 classes
- `ValueError: X and y must have the same length`: Ensure feature matrix and labels have matching lengths
- `ValueError: shrinkage must be between 0 and 1`: Use shrinkage value between 0 and 1

## Best Practices

1. **Scale features**: LDA benefits from feature scaling (though not strictly required)
2. **Balance classes**: Imbalanced classes can affect results
3. **Use appropriate components**: Don't exceed min(n_features, n_classes - 1)
4. **Try both solvers**: Compare eigen and SVD solvers
5. **Use shrinkage for regularization**: Helpful when Sw is singular
6. **Visualize components**: Helps understand class separation
7. **Check explained variance**: Understand how much separation each component captures

## Real-World Applications

- Face recognition
- Medical diagnosis
- Document classification
- Feature extraction for classification
- Dimensionality reduction before classification
- Pattern recognition
- Educational purposes for learning supervised dimensionality reduction

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
