# Decision Tree Classifier with Information Gain and Gini Impurity

A Python implementation of Decision Tree Classifier from scratch with information gain (entropy-based), Gini impurity, and tree visualization. This is the thirty-fourth project in the ML learning series, focusing on understanding decision trees, splitting criteria, and tree-based classification.

## Project Title and Description

The Decision Tree Classifier tool provides a complete implementation of decision trees from scratch, including information gain calculation, Gini impurity, recursive tree building, and comprehensive visualization. It helps users understand how decision trees work, how different splitting criteria affect tree structure, and how to interpret tree decisions.

This tool solves the problem of learning decision tree fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates entropy calculation, Gini impurity, information gain, recursive tree building, and tree visualization from scratch.

**Target Audience**: Beginners learning machine learning, students studying classification algorithms, and anyone who needs to understand decision trees and tree-based methods from scratch.

## Features

- Decision tree implementation from scratch
- Information gain (entropy-based) splitting criterion
- Gini impurity splitting criterion
- Recursive tree building algorithm
- Tree depth and node count statistics
- Text-based tree visualization
- Graphical tree visualization
- Multiclass classification support
- Configurable stopping criteria (max_depth, min_samples_split, min_samples_leaf)
- Class probability prediction
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
cd /path/to/Python-ml-projects/decision-tree-classifier
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
python src/main.py --input sample.csv --target class_label --criterion gini
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  criterion: "gini"
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  min_impurity_decrease: 0.0
  random_state: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.criterion`: Splitting criterion. Options: "gini", "entropy" (default: "gini")
- `model.max_depth`: Maximum depth of tree. If null, nodes expanded until pure (default: null)
- `model.min_samples_split`: Minimum samples required to split node (default: 2)
- `model.min_samples_leaf`: Minimum samples required at leaf node (default: 1)
- `model.min_impurity_decrease`: Minimum impurity decrease for split (default: 0.0)
- `model.random_state`: Random seed (default: null)

## Usage

### Basic Usage

```python
from src.main import DecisionTreeClassifier
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize and fit model
tree = DecisionTreeClassifier(criterion="gini")
tree.fit(X, y)

# Predict
predictions = tree.predict(X)
print(f"Predictions: {predictions}")

# Evaluate
accuracy = tree.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### With Different Criteria

```python
from src.main import DecisionTreeClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Gini impurity
tree_gini = DecisionTreeClassifier(criterion="gini")
tree_gini.fit(X, y)

# Information gain (entropy)
tree_entropy = DecisionTreeClassifier(criterion="entropy")
tree_entropy.fit(X, y)
```

### With Depth Control

```python
from src.main import DecisionTreeClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Limit tree depth
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X, y)

print(f"Tree depth: {tree.get_depth()}")
print(f"Number of nodes: {tree.get_n_nodes()}")
```

### Tree Visualization

```python
from src.main import DecisionTreeClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

tree = DecisionTreeClassifier()
tree.fit(X, y)

# Print tree structure
tree.print_tree()

# Plot tree visualization
tree.plot_tree()
```

### Class Probabilities

```python
from src.main import DecisionTreeClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

tree = DecisionTreeClassifier()
tree.fit(X, y)

# Predict probabilities
probabilities = tree.predict_proba(X)
print(f"Probabilities shape: {probabilities.shape}")
```

### With Pandas DataFrame

```python
from src.main import DecisionTreeClassifier
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2"]].values
y = df["target"].values

tree = DecisionTreeClassifier(criterion="gini")
tree.fit(X, y)

predictions = tree.predict(X)
```

### Command-Line Usage

Basic decision tree:

```bash
python src/main.py --input data.csv --target class_label --criterion gini
```

With entropy criterion:

```bash
python src/main.py --input data.csv --target class_label --criterion entropy
```

With max depth:

```bash
python src/main.py --input data.csv --target class_label --max-depth 5
```

Print tree structure:

```bash
python src/main.py --input data.csv --target class_label --print-tree
```

Plot tree visualization:

```bash
python src/main.py --input data.csv --target class_label --plot-tree
```

Save tree plot:

```bash
python src/main.py --input data.csv --target class_label --plot-tree --save-plot tree.png
```

Save predictions:

```bash
python src/main.py --input data.csv --target class_label --output predictions.csv
```

Make predictions on new data:

```bash
python src/main.py --input train.csv --target class_label --predict test.csv --output-predictions predictions.csv
```

### Complete Example

```python
from src.main import DecisionTreeClassifier
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int)

# Fit with Gini criterion
tree = DecisionTreeClassifier(criterion="gini", max_depth=5)
tree.fit(X, y)

print(f"Tree depth: {tree.get_depth()}")
print(f"Number of nodes: {tree.get_n_nodes()}")
print(f"Classes: {tree.classes_}")

# Evaluate
accuracy = tree.score(X, y)
print(f"Accuracy: {accuracy:.4f}")

# Predict
predictions = tree.predict(X)
probabilities = tree.predict_proba(X)

# Visualize
tree.print_tree()
tree.plot_tree()
```

## Project Structure

```
decision-tree-classifier/
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

- `src/main.py`: Core implementation with `DecisionTreeClassifier` class
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
- Fitting with different criteria
- Prediction
- Class probabilities
- Entropy and Gini calculations
- Information gain
- Tree depth and node count
- Stopping criteria
- Error handling
- Different input types
- Multiclass classification
- Tree visualization

## Understanding Decision Trees

### Decision Tree Algorithm

Decision trees recursively partition data based on feature values:

1. **Start with root node** containing all data
2. **Find best split** that maximizes information gain or minimizes impurity
3. **Create child nodes** based on split
4. **Repeat recursively** for each child node
5. **Stop when** stopping criteria met (pure node, max depth, etc.)

### Splitting Criteria

**Gini Impurity:**
```
Gini = 1 - Σ(pᵢ)²
```

Where `pᵢ` is probability of class `i`.

**Entropy:**
```
Entropy = -Σ(pᵢ * log₂(pᵢ))
```

**Information Gain:**
```
IG = Impurity(parent) - Weighted_Impurity(children)
```

### Best Split Selection

For each feature and threshold:
1. Split data into left and right subsets
2. Calculate information gain
3. Select split with maximum gain

### Stopping Criteria

**Pure Node:**
- All samples belong to same class
- No further splitting needed

**Max Depth:**
- Tree reaches maximum allowed depth
- Prevents overfitting

**Min Samples Split:**
- Node has fewer samples than minimum required
- Prevents overfitting on small subsets

**Min Samples Leaf:**
- Split would create leaf with fewer samples than minimum
- Ensures meaningful leaf nodes

### Tree Interpretation

**Internal Nodes:**
- Feature and threshold for splitting
- Number of samples
- Impurity value

**Leaf Nodes:**
- Predicted class
- Number of samples

## Troubleshooting

### Common Issues

**Issue**: Tree too deep (overfitting)

**Solution**: 
- Set max_depth parameter
- Increase min_samples_split
- Increase min_samples_leaf
- Use pruning techniques

**Issue**: Tree too shallow (underfitting)

**Solution**: 
- Increase max_depth
- Decrease min_samples_split
- Decrease min_samples_leaf
- Check data quality

**Issue**: Poor classification accuracy

**Solution**: 
- Try different criterion (gini vs entropy)
- Tune stopping criteria
- Check feature quality
- Ensure balanced classes
- Consider feature engineering

**Issue**: Tree visualization too large

**Solution**: 
- Limit max_depth for visualization
- Use print_tree() for text output
- Save plot to file for zooming

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: Unknown criterion`: Use "gini" or "entropy"
- `ValueError: X and y must have the same length`: Ensure matching lengths

## Best Practices

1. **Choose appropriate criterion**: Gini is faster, entropy may be more accurate
2. **Control tree depth**: Use max_depth to prevent overfitting
3. **Set minimum samples**: Use min_samples_split and min_samples_leaf
4. **Visualize tree**: Helps understand model decisions
5. **Check tree statistics**: Depth and node count indicate complexity
6. **Handle imbalanced classes**: Consider class weights
7. **Feature scaling**: Not required for trees, but can help
8. **Cross-validation**: Use CV to tune hyperparameters

## Real-World Applications

- Medical diagnosis
- Credit risk assessment
- Customer segmentation
- Fraud detection
- Quality control
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
