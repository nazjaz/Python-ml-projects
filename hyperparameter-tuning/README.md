# Hyperparameter Tuning with Grid Search, Random Search, and Bayesian Optimization

A Python implementation of hyperparameter tuning from scratch including grid search, random search, and Bayesian optimization methods. This is the forty-fourth project in the ML learning series, focusing on understanding hyperparameter optimization techniques for machine learning models.

## Project Title and Description

The Hyperparameter Tuning tool provides complete implementations of three major hyperparameter optimization methods from scratch: grid search (exhaustive search), random search (random sampling), and Bayesian optimization (Gaussian Process-based). It includes cross-validation support, scoring functions, and result analysis. It helps users understand how hyperparameters are optimized, how different search strategies work, and how to find optimal model configurations.

This tool solves the problem of finding optimal hyperparameters for machine learning models by providing clear, educational implementations without relying on external optimization libraries. It demonstrates grid search, random search, and Bayesian optimization from scratch.

**Target Audience**: Beginners learning machine learning, students studying model optimization, and anyone who needs to understand hyperparameter tuning from scratch.

## Features

- Grid Search with exhaustive parameter search
- Random Search with random sampling
- Bayesian Optimization with Gaussian Process
- Cross-validation support (k-fold)
- Configurable scoring functions
- Result analysis and reporting
- Best parameter extraction
- CV results storage
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for CSV input files

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/hyperparameter-tuning
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
python src/main.py --input sample.csv --target label --method grid --param-grid param_grid.json --output results.csv
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

tuning:
  cv: 5
  n_iter: 10
  n_initial: 5
  random_state: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `tuning.cv`: Number of cross-validation folds (default: 5)
- `tuning.n_iter`: Number of iterations for random/bayesian search (default: 10)
- `tuning.n_initial`: Number of initial random samples for Bayesian optimization (default: 5)
- `tuning.random_state`: Random seed (default: null)

## Usage

### Grid Search

```python
from src.main import GridSearchCV
from src.example_estimator import SimpleClassifier
import numpy as np

X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

estimator = SimpleClassifier()
param_grid = {
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10],
    "learning_rate": [0.01, 0.1, 0.5],
}

gs = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    cv=5,
    verbose=1,
)
gs.fit(X, y)

print(f"Best parameters: {gs.best_params_}")
print(f"Best score: {gs.best_score_:.6f}")
```

### Random Search

```python
from src.main import RandomSearchCV
from src.example_estimator import SimpleClassifier
import numpy as np

X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

estimator = SimpleClassifier()
param_distributions = {
    "max_depth": [3, 5, 7, 10, 15],
    "min_samples_split": [2, 5, 10, 20],
    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
}

rs = RandomSearchCV(
    estimator=estimator,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    verbose=1,
)
rs.fit(X, y)

print(f"Best parameters: {rs.best_params_}")
print(f"Best score: {rs.best_score_:.6f}")
```

### Bayesian Optimization

```python
from src.main import BayesianOptimization
from src.example_estimator import SimpleRegressor
import numpy as np

X = np.random.randn(100, 5)
y = np.random.randn(100)

estimator = SimpleRegressor()
param_space = {
    "alpha": (0.1, 10.0),
    "learning_rate": (0.001, 0.1),
    "max_iter": (50, 200),
}

bo = BayesianOptimization(
    estimator=estimator,
    param_space=param_space,
    n_iter=20,
    n_initial=5,
    cv=5,
    verbose=1,
)
bo.fit(X, y)

print(f"Best parameters: {bo.best_params_}")
print(f"Best score: {bo.best_score_:.6f}")
```

### Accessing CV Results

```python
gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
gs.fit(X, y)

# Access all results
cv_results = gs.cv_results_
print(f"Mean scores: {cv_results['mean_test_score']}")
print(f"Std scores: {cv_results['std_test_score']}")

# Access best estimator
best_estimator = gs.best_estimator_
predictions = best_estimator.predict(X_test)
```

### Complete Example

```python
from src.main import GridSearchCV, RandomSearchCV, BayesianOptimization
from src.example_estimator import SimpleClassifier
import numpy as np
import pandas as pd

# Generate sample data
X = np.random.randn(200, 10)
y = np.random.randint(0, 2, 200)

# Grid Search
estimator = SimpleClassifier()
param_grid = {
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5],
}

gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, verbose=1)
gs.fit(X, y)

print(f"\nGrid Search Results:")
print(f"Best parameters: {gs.best_params_}")
print(f"Best score: {gs.best_score_:.6f}")

# Random Search
param_distributions = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
}

rs = RandomSearchCV(
    estimator=estimator,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    verbose=1,
)
rs.fit(X, y)

print(f"\nRandom Search Results:")
print(f"Best parameters: {rs.best_params_}")
print(f"Best score: {rs.best_score_:.6f}")

# Bayesian Optimization
param_space = {
    "max_depth": (3, 10),
    "min_samples_split": (2, 10),
}

bo = BayesianOptimization(
    estimator=estimator,
    param_space=param_space,
    n_iter=10,
    n_initial=3,
    cv=5,
    verbose=1,
)
bo.fit(X, y)

print(f"\nBayesian Optimization Results:")
print(f"Best parameters: {bo.best_params_}")
print(f"Best score: {bo.best_score_:.6f}")
```

### Command-Line Usage

Grid search:

```bash
python src/main.py --input data.csv --target label --method grid --param-grid param_grid.json --output results.csv
```

Random search:

```bash
python src/main.py --input data.csv --target label --method random --param-grid param_grid.json --n-iter 20 --output results.csv
```

Bayesian optimization:

```bash
python src/main.py --input data.csv --target label --method bayesian --param-grid param_space.json --n-iter 20 --output results.csv
```

With custom CV:

```bash
python src/main.py --input data.csv --target label --method grid --param-grid param_grid.json --cv 10 --output results.csv
```

### Parameter Grid JSON Format

For grid search and random search:

```json
{
  "estimator": "SimpleClassifier",
  "param_grid": {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "learning_rate": [0.01, 0.1, 0.5]
  }
}
```

For Bayesian optimization:

```json
{
  "estimator": "SimpleRegressor",
  "param_grid": {
    "alpha": [0.1, 10.0],
    "learning_rate": [0.001, 0.1],
    "max_iter": [50, 200]
  }
}
```

## Project Structure

```
hyperparameter-tuning/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── src/
│   ├── main.py              # Main implementation
│   └── example_estimator.py  # Example estimators
├── tests/
│   └── test_main.py         # Unit tests
├── docs/
│   └── API.md               # API documentation
└── logs/
    └── .gitkeep             # Log directory
```

### File Descriptions

- `src/main.py`: Core implementation with tuning classes
- `src/example_estimator.py`: Example estimators for demonstration
- `config.yaml`: Configuration file for tuning settings
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
- Grid search functionality
- Random search functionality
- Bayesian optimization functionality
- Parameter combination generation
- Cross-validation
- Error handling

## Understanding Hyperparameter Tuning

### Grid Search

**Concept:**
- Exhaustive search over parameter grid
- Tests all combinations
- Guaranteed to find best in grid

**Process:**
1. Generate all parameter combinations
2. For each combination:
   - Perform cross-validation
   - Calculate mean score
3. Select best parameters

**Use Cases:**
- Small parameter spaces
- When you need exhaustive search
- Discrete parameters

**Pros:**
- Guaranteed to find best in grid
- Simple and intuitive
- No randomness

**Cons:**
- Computationally expensive
- Doesn't scale to large spaces
- Can't explore outside grid

### Random Search

**Concept:**
- Random sampling from parameter space
- Tests random combinations
- More efficient than grid search

**Process:**
1. Sample random parameter combinations
2. For each combination:
   - Perform cross-validation
   - Calculate mean score
3. Select best parameters

**Use Cases:**
- Large parameter spaces
- When grid search is too expensive
- Continuous and discrete parameters

**Pros:**
- More efficient than grid search
- Can explore larger spaces
- Often finds good solutions faster

**Cons:**
- No guarantee of finding best
- May miss optimal parameters
- Random sampling

### Bayesian Optimization

**Concept:**
- Uses probabilistic model (Gaussian Process)
- Balances exploration and exploitation
- Learns from previous evaluations

**Process:**
1. Start with random samples
2. Build probabilistic model of objective
3. Use acquisition function to select next point
4. Evaluate and update model
5. Repeat until convergence

**Use Cases:**
- Expensive evaluations
- Continuous parameter spaces
- When you need efficient search

**Pros:**
- Most efficient for expensive evaluations
- Learns from previous results
- Balances exploration/exploitation

**Cons:**
- More complex implementation
- Requires more iterations to start
- May get stuck in local optima

### Comparison

| Method | Efficiency | Guarantee | Use Case |
|--------|-----------|-----------|----------|
| Grid Search | Low | Best in grid | Small spaces |
| Random Search | Medium | None | Large spaces |
| Bayesian | High | None | Expensive evaluations |

## Troubleshooting

### Common Issues

**Issue**: Grid search too slow

**Solution**: 
- Reduce parameter grid size
- Use random search instead
- Reduce CV folds
- Use smaller dataset for tuning

**Issue**: Random search not finding good parameters

**Solution**: 
- Increase n_iter
- Check parameter ranges
- Use Bayesian optimization
- Ensure good parameter space

**Issue**: Bayesian optimization not improving

**Solution**: 
- Increase n_initial
- Increase n_iter
- Check parameter space bounds
- Adjust acquisition function

### Error Messages

- `ValueError: Parameter grid is empty`: Check parameter grid configuration
- `ValueError: Target column not found`: Check target column name
- `ValueError: Feature columns not found`: Check feature column names

## Best Practices

1. **Start with grid search**: For small spaces
2. **Use random search**: For larger spaces
3. **Use Bayesian optimization**: For expensive evaluations
4. **Cross-validation**: Always use CV for reliable estimates
5. **Parameter ranges**: Choose appropriate ranges
6. **Iterations**: Balance between time and quality
7. **Validation set**: Keep separate validation set
8. **Early stopping**: Consider early stopping for long training

## Real-World Applications

- Model selection
- Feature engineering
- Algorithm comparison
- Production model optimization
- Educational purposes for learning hyperparameter tuning

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
