# Principal Component Analysis (PCA) for Dimensionality Reduction

A Python implementation of Principal Component Analysis (PCA) from scratch for dimensionality reduction with explained variance analysis. This is the thirtieth project in the ML learning series, focusing on understanding dimensionality reduction and PCA algorithms.

## Project Title and Description

The Principal Component Analysis tool provides a complete implementation of PCA from scratch, including dimensionality reduction, explained variance calculation, and data reconstruction. It helps users understand how PCA works, how to reduce dimensions while preserving variance, and how to analyze the importance of each principal component.

This tool solves the problem of learning dimensionality reduction fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates eigenvalue decomposition, variance preservation, and how to choose the optimal number of components.

**Target Audience**: Beginners learning machine learning, students studying dimensionality reduction techniques, and anyone who needs to understand PCA and explained variance from scratch.

## Features

- PCA implementation from scratch
- Dimensionality reduction
- Explained variance calculation
- Explained variance ratio
- Cumulative explained variance
- Data transformation to lower dimensions
- Inverse transformation (reconstruction)
- Component whitening option
- Flexible n_components selection:
  - Integer: Keep top n components
  - Float (0-1): Keep components explaining at least that variance
  - None: Keep all components
- Explained variance visualization
- Cumulative variance visualization
- Component visualization (for 2D data)
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
cd /path/to/Python-ml-projects/pca-dimensionality-reduction
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
python src/main.py --input sample.csv --n-components 2
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
  whiten: false
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.n_components`: Number of components. Options: null (all), int (top n), float 0-1 (variance threshold) (default: null)
- `model.whiten`: Whether to whiten components (default: false)

## Usage

### Basic Usage

```python
from src.main import PCA
import numpy as np

# Create sample data
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])

# Initialize and fit model
pca = PCA(n_components=2)
pca.fit(X)

# Transform data
X_transformed = pca.transform(X)
print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_transformed.shape}")

# Get explained variance
explained_variance = pca.get_explained_variance()
explained_variance_ratio = pca.get_explained_variance_ratio()
print(f"Explained variance ratio: {explained_variance_ratio}")
```

### With Variance Threshold

```python
from src.main import PCA
import numpy as np

X = np.random.randn(100, 10)

# Keep components explaining 95% of variance
pca = PCA(n_components=0.95)
pca.fit(X)

print(f"Number of components: {pca.n_components_}")
print(f"Explained variance: {np.sum(pca.explained_variance_ratio):.4f}")
```

### Fit and Transform

```python
from src.main import PCA
import numpy as np

X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)
```

### Inverse Transformation (Reconstruction)

```python
from src.main import PCA
import numpy as np

X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])

pca = PCA(n_components=2)
pca.fit(X)

X_transformed = pca.transform(X)
X_reconstructed = pca.inverse_transform(X_transformed)

print(f"Reconstruction error: {np.mean((X - X_reconstructed) ** 2):.6f}")
```

### Explained Variance Analysis

```python
from src.main import PCA
import numpy as np

X = np.random.randn(100, 10)

pca = PCA(n_components=None)
pca.fit(X)

# Get explained variance
explained_variance = pca.get_explained_variance()
explained_variance_ratio = pca.get_explained_variance_ratio()
cumulative_variance = pca.get_cumulative_variance()

print("Explained variance by component:")
for i, (var, ratio, cum) in enumerate(
    zip(explained_variance, explained_variance_ratio, cumulative_variance)
):
    print(f"PC{i+1}: var={var:.4f}, ratio={ratio:.4f}, cumulative={cum:.4f}")

# Plot explained variance
pca.plot_explained_variance()

# Plot cumulative variance
pca.plot_explained_variance(cumulative=True)
```

### With Pandas DataFrame

```python
from src.main import PCA
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]].values

pca = PCA(n_components=2)
pca.fit(X)

X_transformed = pca.transform(X)
df_transformed = pd.DataFrame(
    X_transformed, columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)
```

### Command-Line Usage

Basic PCA:

```bash
python src/main.py --input data.csv --n-components 2
```

With variance threshold:

```bash
python src/main.py --input data.csv --n-components 0.95
```

Plot explained variance:

```bash
python src/main.py --input data.csv --n-components 2 --plot-variance
```

Plot cumulative variance:

```bash
python src/main.py --input data.csv --n-components 2 --plot-cumulative
```

Save transformed data:

```bash
python src/main.py --input data.csv --n-components 2 --output transformed.csv
```

Save reconstructed data:

```bash
python src/main.py --input data.csv --n-components 2 --output-original reconstructed.csv
```

### Complete Example

```python
from src.main import PCA
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 100
n_features = 5
X = np.random.randn(n_samples, n_features)

# Apply PCA
pca = PCA(n_components=0.95)
pca.fit(X)

print(f"Original dimensions: {X.shape}")
print(f"Reduced dimensions: {pca.n_components_}")
print(f"Explained variance: {np.sum(pca.explained_variance_ratio):.4f}")

# Transform data
X_transformed = pca.transform(X)
print(f"Transformed shape: {X_transformed.shape}")

# Reconstruct data
X_reconstructed = pca.inverse_transform(X_transformed)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"Reconstruction MSE: {reconstruction_error:.6f}")

# Analyze variance
print("\nExplained variance by component:")
for i, ratio in enumerate(pca.explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.4f}")

cumulative = pca.get_cumulative_variance()
print(f"\nCumulative variance:")
for i, cum in enumerate(cumulative):
    print(f"PC{i+1}: {cum:.4f}")

# Visualize
pca.plot_explained_variance()
pca.plot_explained_variance(cumulative=True)
```

## Project Structure

```
pca-dimensionality-reduction/
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

- `src/main.py`: Core implementation with `PCA` class
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
- Inverse transformation
- Explained variance calculation
- Variance ratio calculation
- Cumulative variance
- Error handling
- Different input types

## Understanding PCA

### Principal Component Analysis

PCA is a dimensionality reduction technique that finds directions of maximum variance:

1. **Center data**: Subtract mean from each feature
2. **Compute covariance matrix**: Calculate feature relationships
3. **Eigenvalue decomposition**: Find principal directions (eigenvectors) and variances (eigenvalues)
4. **Select components**: Choose top components based on variance
5. **Transform data**: Project data onto principal components

### Mathematical Foundation

**Covariance Matrix:**
```
C = (1/(n-1)) * X_centered^T @ X_centered
```

**Eigenvalue Decomposition:**
```
C @ v = λ @ v
```

Where:
- `v` is eigenvector (principal component)
- `λ` is eigenvalue (explained variance)

**Transformation:**
```
X_transformed = X_centered @ V^T
```

Where `V` is matrix of selected eigenvectors.

### Explained Variance

**Explained Variance:**
- Variance captured by each principal component
- Equal to corresponding eigenvalue

**Explained Variance Ratio:**
```
ratio_i = λ_i / Σλ_j
```

**Cumulative Explained Variance:**
```
cumulative_i = Σ(ratio_j) for j=1 to i
```

### Choosing Number of Components

**Methods:**
1. **Fixed number**: Specify exact number of components
2. **Variance threshold**: Keep components explaining at least X% variance
3. **Scree plot**: Look for "elbow" in explained variance plot
4. **Cumulative variance**: Keep components until cumulative variance reaches threshold

**Common Thresholds:**
- 95% variance: Good balance
- 99% variance: Very high retention
- 80% variance: Aggressive reduction

## Troubleshooting

### Common Issues

**Issue**: Low explained variance with few components

**Solution**: 
- Data may have many independent features
- Consider feature selection before PCA
- Use more components or higher variance threshold

**Issue**: Reconstruction error too high

**Solution**: 
- Use more components
- Check if data is suitable for PCA (linear relationships)
- Consider if dimensionality reduction is appropriate

**Issue**: Components don't make sense

**Solution**: 
- PCA finds directions of maximum variance, not necessarily interpretable
- Consider feature scaling
- Check data quality

### Error Messages

- `ValueError: Model must be fitted before transformation`: Call `fit()` before `transform()`
- `ValueError: n_components must be between 0 and 1 when float`: Use float between 0 and 1
- `ValueError: Need at least 2 samples for PCA`: Provide at least 2 samples

## Best Practices

1. **Scale features**: PCA is sensitive to feature scale (centering is automatic)
2. **Use variance threshold**: More flexible than fixed number
3. **Visualize explained variance**: Helps choose optimal number of components
4. **Check reconstruction error**: Verify information loss is acceptable
5. **Consider whitening**: If components need unit variance
6. **Interpret components carefully**: PCA components may not be interpretable

## Real-World Applications

- Dimensionality reduction for visualization
- Feature extraction
- Noise reduction
- Data compression
- Preprocessing for other ML algorithms
- Exploratory data analysis
- Educational purposes for learning dimensionality reduction

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
