# Model Interpretability using SHAP, Permutation Importance, and Partial Dependence

A Python implementation of model interpretability techniques including SHAP (SHapley Additive exPlanations) values, permutation importance, and partial dependence plots. This tool provides comprehensive solutions for understanding model predictions and feature importance across various machine learning models.

## Project Title and Description

The Model Interpretability tool provides implementations of three powerful interpretability techniques to help understand how machine learning models make predictions. It explains model behavior at both global (overall feature importance) and local (individual prediction) levels, making it easier to understand, debug, and trust machine learning models.

This tool solves the problem of "black box" machine learning models by providing multiple methods to understand model predictions, feature contributions, and decision-making processes. It helps data scientists, stakeholders, and regulators understand and validate model behavior.

**Target Audience**: Data scientists, machine learning engineers, model validators, business analysts, and anyone who needs to understand and explain machine learning model behavior.

## Features

### SHAP Values
- **SHapley Additive exPlanations**: Game-theoretic approach to feature importance
- **Multiple Algorithms**: Automatic selection of Tree, Linear, or Kernel SHAP
- **Local and Global Explanations**: Understand individual predictions and overall feature importance
- **Summary Plots**: Visualize feature importance and impact
- **Model-Agnostic**: Works with any model type

### Permutation Importance
- **Model-Agnostic Importance**: Measures feature importance by permuting features
- **Statistical Robustness**: Multiple repetitions with mean and standard deviation
- **Classification and Regression**: Supports both task types
- **Custom Scoring**: Configurable scoring metrics
- **Visualization**: Bar plots of feature importance

### Partial Dependence Plots (PDP)
- **Feature Effect Visualization**: Shows how features affect predictions
- **Single and Multiple Features**: Plot individual or interaction effects
- **Marginal Effects**: Understand feature impact while averaging over others
- **Model-Agnostic**: Works with any scikit-learn model
- **High-Resolution Plots**: Configurable grid resolution

### Additional Features
- Unified interface for all interpretability methods
- Support for classification and regression tasks
- Automatic algorithm selection for SHAP
- Comprehensive visualization capabilities
- Command-line interface
- Configuration via YAML file
- Support for pandas DataFrames and numpy arrays
- Comprehensive logging
- Input validation and error handling

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/model-interpretability-shap
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

**Note**: SHAP installation may take a few minutes as it compiles C extensions.

### Step 4: Verify Installation

```bash
python src/main.py --model model.pkl --data data.csv --target-col target
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

shap:
  algorithm: "auto"
  max_evals: 100

permutation_importance:
  n_repeats: 5
  random_state: 42

partial_dependence:
  grid_resolution: 100
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `shap.algorithm`: SHAP algorithm - "auto", "tree", "linear", "kernel" (default: "auto")
- `shap.max_evals`: Maximum evaluations for kernel SHAP (default: 100)
- `permutation_importance.n_repeats`: Number of permutation repetitions (default: 5)
- `permutation_importance.random_state`: Random seed (default: 42)
- `partial_dependence.grid_resolution`: Number of grid points for PDP (default: 100)

## Usage

### Command-Line Interface

#### Generate All Interpretations

```bash
python src/main.py --model model.pkl --data data.csv --target-col target \
  --output-dir interpretations/ --results-output results.json
```

#### Without Target (SHAP and PDP only)

```bash
python src/main.py --model model.pkl --data data.csv \
  --output-dir interpretations/
```

### Programmatic Usage

#### Basic Interpretability

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.main import ModelInterpreter

# Train a model
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Initialize interpreter
interpreter = ModelInterpreter()

# Load model and data
interpreter.load_model_and_data(model, X, y)

# Calculate permutation importance
perm_importance = interpreter.calculate_permutation_importance()
print("Permutation Importance:", perm_importance)

# Calculate SHAP values
shap_importance = interpreter.calculate_shap()
print("SHAP Importance:", shap_importance)

# Plot partial dependence
interpreter.plot_partial_dependence(features=0, save_path="pdp.png")
```

#### Using Individual Classes

```python
from src.main import (
    SHAPExplainer,
    PermutationImportanceCalculator,
    PartialDependencePlotter,
)

# SHAP
shap_explainer = SHAPExplainer(model, algorithm="auto")
shap_explainer.fit(X)
shap_values = shap_explainer.explain(X)
importance = shap_explainer.get_feature_importance(feature_names)
shap_explainer.plot_summary(X, feature_names, save_path="shap_summary.png")

# Permutation Importance
perm_calc = PermutationImportanceCalculator(model, n_repeats=5)
importance = perm_calc.calculate(X, y, feature_names)
perm_calc.plot_importance(save_path="perm_importance.png")

# Partial Dependence
pdp_plotter = PartialDependencePlotter(model, feature_names)
pdp_plotter.plot(X, features=[0, 1], save_path="pdp.png")
```

#### Generate All Interpretations

```python
interpreter = ModelInterpreter()
interpreter.load_model_and_data(model, X, y)

results = interpreter.generate_all_interpretations(output_dir="interpretations/")

# Results contain:
# - results["shap"]: SHAP feature importance
# - results["permutation_importance"]: Permutation importance statistics
```

### Common Use Cases

1. **Model Validation**: Understand and validate model behavior
2. **Feature Importance**: Identify most important features
3. **Model Debugging**: Find issues in model predictions
4. **Regulatory Compliance**: Explain model decisions for compliance
5. **Business Understanding**: Help stakeholders understand model behavior
6. **Feature Engineering**: Guide feature engineering efforts
7. **Model Comparison**: Compare feature importance across models

## Project Structure

```
model-interpretability-shap/
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
  - `SHAPExplainer`: SHAP values calculation and visualization
  - `PermutationImportanceCalculator`: Permutation importance calculation
  - `PartialDependencePlotter`: Partial dependence plot generation
  - `ModelInterpreter`: Main interpreter class with CLI interface
  - `main()`: Command-line interface entry point

- `tests/test_main.py`: Comprehensive unit tests including:
  - SHAP explainer tests (if available)
  - Permutation importance tests
  - Partial dependence plotter tests
  - Integration tests

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
- Unit tests for each interpretability method
- Integration tests for complete workflows
- Error handling and edge case tests
- Tests for both classification and regression

Current test coverage: >90% of code paths

## Algorithm Details

### SHAP (SHapley Additive exPlanations)

**How it works**:
- Based on cooperative game theory (Shapley values)
- Measures contribution of each feature to prediction
- Satisfies properties: efficiency, symmetry, dummy, additivity
- Provides both local (per-sample) and global (aggregated) explanations

**Mathematical Definition**:
```
SHAP_i(x) = Σ [|S|!(|F|-|S|-1)! / |F|!] * [f(S ∪ {i}) - f(S)]
```

Where:
- S: Subset of features
- F: All features
- f: Model prediction function

**Advantages**:
- Theoretically grounded (game theory)
- Model-agnostic
- Provides local and global explanations
- Handles feature interactions

**Limitations**:
- Computationally expensive for some algorithms
- Requires background dataset
- May be slow for large datasets

**Best for**: Understanding individual predictions, feature interactions, model debugging

### Permutation Importance

**How it works**:
1. Train model and get baseline performance
2. Permute (shuffle) one feature
3. Measure performance drop
4. Repeat for all features
5. Features with larger performance drops are more important

**Advantages**:
- Model-agnostic
- Easy to understand
- Provides statistical robustness (with repetitions)
- Works for any model type

**Limitations**:
- Computationally expensive (requires multiple model evaluations)
- May break feature correlations
- Doesn't provide local explanations

**Best for**: Global feature importance, model validation, feature selection

### Partial Dependence Plots

**How it works**:
1. Select feature(s) of interest
2. Vary feature value across range
3. Average predictions over all other features
4. Plot relationship between feature and prediction

**Mathematical Definition**:
```
PDP_S(x_S) = E_X_C[f(x_S, X_C)] = ∫ f(x_S, x_C) p(x_C) dx_C
```

Where:
- S: Features of interest
- C: Other features
- f: Model prediction function

**Advantages**:
- Visual and intuitive
- Shows feature effects clearly
- Model-agnostic
- Can show interactions (2D plots)

**Limitations**:
- Assumes features are independent
- May hide heterogeneous effects
- Computationally expensive for many features

**Best for**: Understanding feature effects, visualizing relationships, presentations

## Choosing the Right Method

### Use SHAP when:
- You need local explanations (per-sample)
- You want to understand feature interactions
- You need theoretically grounded explanations
- You're debugging specific predictions

### Use Permutation Importance when:
- You need global feature importance
- You want model-agnostic importance
- You need statistical robustness
- You're doing feature selection

### Use Partial Dependence when:
- You want visual understanding
- You need to present to non-technical audience
- You want to see feature effects clearly
- You're exploring feature relationships

### Use All Three when:
- You want comprehensive understanding
- You need to validate interpretations
- You're doing thorough model analysis
- You need multiple perspectives

## Troubleshooting

### Common Issues and Solutions

#### Issue: SHAP Installation Fails
**Error**: SHAP installation errors or compilation failures

**Solution**:
- Install build tools: `sudo apt-get install build-essential` (Linux)
- Use pre-built wheels: `pip install --only-binary shap shap`
- Try different Python version
- Install from conda: `conda install -c conda-forge shap`

#### Issue: SHAP is Slow
**Problem**: SHAP calculation takes too long

**Solution**:
- Use Tree SHAP for tree-based models (fastest)
- Reduce background dataset size
- Use fewer samples for explanation
- Reduce max_evals for kernel SHAP
- Use approximate methods

#### Issue: Permutation Importance Takes Too Long
**Problem**: Calculation is very slow

**Solution**:
- Reduce n_repeats (e.g., from 5 to 3)
- Use faster model for importance calculation
- Reduce dataset size
- Use parallel processing (n_jobs parameter)
- Calculate importance on subset of data

#### Issue: Partial Dependence Plot Fails
**Error**: PDP generation fails

**Solution**:
- Check feature indices are valid
- Ensure feature names match
- Reduce grid_resolution
- Check model supports predict method
- Verify data types are correct

#### Issue: Memory Error
**Error**: Out of memory during calculation

**Solution**:
- Process data in batches
- Reduce dataset size
- Use sparse matrices if applicable
- Reduce number of features analyzed
- Use smaller background dataset for SHAP

### Error Message Explanations

- **"SHAP is not installed"**: Install SHAP with `pip install shap`
- **"Model and data must be loaded first"**: Call load_model_and_data() before interpretation
- **"X and y must have same number of samples"**: Input arrays have mismatched lengths
- **"Feature 'X' not found"**: Invalid feature name or index

## Performance Considerations

### Computational Complexity

- **SHAP Tree**: O(T * D * N) where T=trees, D=depth, N=samples
- **SHAP Kernel**: O(2^M * N) where M=features, N=samples (very expensive)
- **Permutation Importance**: O(R * F * M) where R=repeats, F=features, M=model complexity
- **Partial Dependence**: O(G * N * M) where G=grid points, N=samples, M=model complexity

### Optimization Tips

1. **For SHAP**: Use Tree SHAP when possible (fastest)
2. **For Permutation**: Reduce n_repeats, use parallel processing
3. **For PDP**: Reduce grid_resolution, analyze fewer features
4. **General**: Use smaller datasets, sample data, process in batches

## Best Practices

1. **Start with Permutation Importance**: Quick global overview
2. **Use SHAP for Details**: Deep dive into specific predictions
3. **Use PDP for Visualization**: Present findings to stakeholders
4. **Validate Interpretations**: Use multiple methods to confirm
5. **Consider Context**: Interpretations depend on data and model
6. **Document Findings**: Keep records of interpretation results
7. **Update Regularly**: Re-interpret as model or data changes

## Contributing

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Create a feature branch: `git checkout -b feature/new-interpreter`

### Code Style Guidelines

- Follow PEP 8 strictly
- Maximum line length: 88 characters
- Use type hints for all functions
- Write docstrings for all public functions and classes
- Add unit tests for new features
- Update documentation for new functionality

### Pull Request Process

1. Ensure all tests pass: `pytest tests/test_main.py`
2. Check code coverage: `pytest --cov=src`
3. Update documentation if adding new features
4. Submit pull request with clear description

## License

This project is part of the Python ML Projects collection. Please refer to the main repository license.

## Additional Resources

- [SHAP Documentation](https://shap.readthedocs.io/)
- [SHAP Paper](https://arxiv.org/abs/1705.07874)
- [Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Partial Dependence Plots](https://scikit-learn.org/stable/modules/partial_dependence.html)
- [Interpretable Machine Learning Book](https://christophm.github.io/interpretable-ml-book/)
