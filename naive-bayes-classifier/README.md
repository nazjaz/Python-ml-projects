# Naive Bayes Classifier

A Python implementation of naive Bayes classifiers from scratch with Gaussian, Multinomial, and Bernoulli variants for different data types. This is the twenty-sixth project in the ML learning series, focusing on understanding probabilistic classification and the naive Bayes algorithm.

## Project Title and Description

The Naive Bayes Classifier tool provides complete implementations of three naive Bayes variants from scratch: Gaussian (for continuous data), Multinomial (for count data), and Bernoulli (for binary data). It helps users understand how probabilistic classification works and how different variants handle different data types.

This tool solves the problem of learning naive Bayes fundamentals by providing clear, educational implementations without relying on external ML libraries. It demonstrates probability calculations, feature independence assumption, and how different distributions model different data types.

**Target Audience**: Beginners learning machine learning, students studying probabilistic classification algorithms, and anyone who needs to understand naive Bayes and its variants from scratch.

## Features

- Three naive Bayes variants:
  - **GaussianNB**: For continuous features (assumes Gaussian distribution)
  - **MultinomialNB**: For count data (word counts, frequencies)
  - **BernoulliNB**: For binary features (presence/absence)
- Probability-based classification
- Class probability predictions
- Multi-class classification support
- Laplace smoothing for Multinomial and Bernoulli
- Smoothing parameter for Gaussian
- Automatic feature binarization for Bernoulli
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
cd /path/to/Python-ml-projects/naive-bayes-classifier
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
python src/main.py --input sample.csv --target label --variant gaussian
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  variant: "gaussian"
  alpha: 1.0
  binarize: 0.0
  smoothing: 1e-9
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.variant`: Naive Bayes variant. Options: "gaussian", "multinomial", "bernoulli" (default: "gaussian")
- `model.alpha`: Smoothing parameter for Multinomial and Bernoulli (Laplace smoothing, default: 1.0)
- `model.binarize`: Binarization threshold for Bernoulli (default: 0.0, None to disable)
- `model.smoothing`: Smoothing parameter for Gaussian (to prevent division by zero, default: 1e-9)

## Usage

### Gaussian Naive Bayes (Continuous Data)

```python
from src.main import GaussianNB
import numpy as np

# Continuous features
X = np.array([[1.5], [2.3], [3.1], [4.7], [5.2]])
y = np.array([0, 0, 1, 1, 1])

# Initialize and fit model
model = GaussianNB()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")

# Get probabilities
probabilities = model.predict_proba(X)
print(f"Probabilities: {probabilities}")

# Calculate accuracy
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### Multinomial Naive Bayes (Count Data)

```python
from src.main import MultinomialNB
import numpy as np

# Count data (e.g., word frequencies)
X = np.array([[5, 2, 0], [3, 4, 1], [1, 6, 2], [0, 3, 5], [2, 1, 4]])
y = np.array([0, 0, 1, 1, 1])

# Initialize and fit model
model = MultinomialNB(alpha=1.0)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

### Bernoulli Naive Bayes (Binary Data)

```python
from src.main import BernoulliNB
import numpy as np

# Binary features
X = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 1]])
y = np.array([0, 0, 1, 1, 1])

# Initialize and fit model
model = BernoulliNB(alpha=1.0, binarize=0.5)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

### With Pandas DataFrame

```python
from src.main import GaussianNB
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]]
y = df["target"]

model = GaussianNB()
model.fit(X, y)

predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

### Command-Line Usage

Gaussian Naive Bayes:

```bash
python src/main.py --input data.csv --target label --variant gaussian
```

Multinomial Naive Bayes:

```bash
python src/main.py --input data.csv --target label --variant multinomial --alpha 1.0
```

Bernoulli Naive Bayes:

```bash
python src/main.py --input data.csv --target label --variant bernoulli --alpha 1.0 --binarize 0.5
```

Save predictions:

```bash
python src/main.py --input data.csv --target label --variant gaussian --output predictions.csv
```

### Complete Example

```python
from src.main import GaussianNB, MultinomialNB, BernoulliNB
import numpy as np

# Generate sample data
np.random.seed(42)

# Continuous data for Gaussian
X_gaussian = np.random.randn(100, 2) * 2
y_gaussian = ((X_gaussian[:, 0] + X_gaussian[:, 1]) > 0).astype(int)

# Count data for Multinomial
X_multinomial = np.random.poisson(3, size=(100, 5)).astype(int)
y_multinomial = np.random.randint(0, 2, 100)

# Binary data for Bernoulli
X_bernoulli = np.random.randint(0, 2, size=(100, 5))
y_bernoulli = np.random.randint(0, 2, 100)

# Compare all three variants
models = {
    "Gaussian": GaussianNB(),
    "Multinomial": MultinomialNB(alpha=1.0),
    "Bernoulli": BernoulliNB(alpha=1.0),
}

datasets = {
    "Gaussian": (X_gaussian, y_gaussian),
    "Multinomial": (X_multinomial, y_multinomial),
    "Bernoulli": (X_bernoulli, y_bernoulli),
}

for name, model in models.items():
    X, y = datasets[name]
    model.fit(X, y)
    score = model.score(X, y)
    print(f"{name} Naive Bayes: Accuracy = {score:.4f}")
```

## Project Structure

```
naive-bayes-classifier/
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

- `src/main.py`: Core implementation with `GaussianNB`, `MultinomialNB`, and `BernoulliNB` classes
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
- GaussianNB (continuous data)
- MultinomialNB (count data)
- BernoulliNB (binary data)
- Model initialization
- Training and fitting
- Prediction
- Probability prediction
- Accuracy calculation
- Error handling
- Different input types

## Understanding Naive Bayes

### Naive Bayes Theorem

Naive Bayes is based on Bayes' theorem with the "naive" assumption of feature independence:

```
P(y|x₁, x₂, ..., xₙ) ∝ P(y) * Π P(xᵢ|y)
```

Where:
- `P(y|x₁, x₂, ..., xₙ)` is the posterior probability
- `P(y)` is the prior probability of class y
- `P(xᵢ|y)` is the likelihood of feature xᵢ given class y

### Gaussian Naive Bayes

Assumes features follow a Gaussian (normal) distribution:

```
P(xᵢ|y) = (1/√(2πσ²)) * exp(-(xᵢ - μ)² / (2σ²))
```

**Use cases:**
- Continuous features
- Real-valued data
- Normally distributed features

### Multinomial Naive Bayes

Assumes features follow a multinomial distribution:

```
P(xᵢ|y) = (count(xᵢ, y) + α) / (count(y) + α * n_features)
```

**Use cases:**
- Count data (word frequencies, document-term matrices)
- Non-negative integer features
- Text classification
- Discrete count features

### Bernoulli Naive Bayes

Assumes features follow a Bernoulli distribution:

```
P(xᵢ|y) = (count(xᵢ=1, y) + α) / (count(y) + 2α)
```

**Use cases:**
- Binary features (0/1, presence/absence)
- Boolean features
- Text classification (word presence)
- Categorical features converted to binary

### Feature Independence Assumption

The "naive" assumption is that features are conditionally independent given the class:

```
P(x₁, x₂, ..., xₙ|y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y)
```

This simplifies calculations but may not hold in practice. Despite this, naive Bayes often performs well.

## Troubleshooting

### Common Issues

**Issue**: Poor predictions with GaussianNB

**Solution**: 
- Check if features are normally distributed
- Consider feature scaling
- Verify data is continuous

**Issue**: MultinomialNB requires non-negative values

**Solution**: 
- Ensure all feature values are non-negative integers
- Use count data or frequencies
- Don't use negative values

**Issue**: BernoulliNB predictions inconsistent

**Solution**: 
- Ensure features are binary (0/1)
- Adjust binarization threshold
- Check feature encoding

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: MultinomialNB requires non-negative feature values`: Use non-negative values for MultinomialNB
- `ValueError: Length mismatch`: X and y have different lengths

## Best Practices

1. **Choose the right variant**: Match variant to data type
2. **Use GaussianNB for continuous data**: Assumes normal distribution
3. **Use MultinomialNB for count data**: Word frequencies, document counts
4. **Use BernoulliNB for binary data**: Presence/absence features
5. **Adjust smoothing parameters**: Alpha parameter affects predictions
6. **Handle feature independence**: Be aware of the naive assumption

## Real-World Applications

- **Text classification**: Spam detection, sentiment analysis
- **Document classification**: News categorization, topic modeling
- **Medical diagnosis**: Disease prediction
- **Email filtering**: Spam vs. ham classification
- **Recommendation systems**: User preference prediction
- **Educational purposes**: Learning probabilistic classification

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
