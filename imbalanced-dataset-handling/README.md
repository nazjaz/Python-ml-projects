# Imbalanced Dataset Handling

A Python implementation of techniques for handling imbalanced datasets in machine learning, including SMOTE (Synthetic Minority Oversampling Technique), multiple undersampling strategies, and class weighting methods. This tool helps address the common problem of class imbalance in classification tasks.

## Project Title and Description

The Imbalanced Dataset Handling tool provides comprehensive solutions for dealing with imbalanced datasets, which occur when one or more classes have significantly fewer samples than others. This imbalance can lead to biased models that perform poorly on minority classes. The tool implements three main approaches: oversampling (SMOTE), undersampling (Random, Tomek Links, ENN), and class weighting.

This tool solves the problem of training machine learning models on imbalanced datasets by providing multiple resampling and weighting strategies. It helps improve model performance on minority classes and provides flexibility to choose the most appropriate technique for specific use cases.

**Target Audience**: Machine learning practitioners, data scientists, students learning about imbalanced learning, and researchers working with skewed datasets.

## Features

### Oversampling Techniques
- **SMOTE (Synthetic Minority Oversampling Technique)**: Generates synthetic samples for minority classes by interpolating between existing samples and their nearest neighbors
- Configurable k-neighbors parameter
- Adjustable sampling strategy to control class balance ratio
- Optional feature scaling before resampling

### Undersampling Techniques
- **Random Undersampling**: Randomly removes samples from majority class
- **Tomek Links**: Removes Tomek link pairs (samples from different classes that are each other's nearest neighbors)
- **Edited Nearest Neighbours (ENN)**: Removes samples whose class label differs from the majority of its k nearest neighbors
- Configurable sampling strategy for random undersampling

### Class Weighting
- **Balanced Weights**: Automatically calculate weights inversely proportional to class frequencies
- **Inverse Frequency Weights**: Weight classes based on inverse frequency
- **Custom Weights**: Apply user-defined class weights
- Compatible with scikit-learn models

### Additional Features
- Class distribution analysis and statistics
- Imbalance ratio calculation
- Command-line interface for batch processing
- Configuration via YAML file
- Support for numpy arrays and pandas DataFrames
- Comprehensive logging
- Input validation and error handling
- Integration with scikit-learn preprocessing

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/imbalanced-dataset-handling
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
python src/main.py --input sample.csv --target-col target --method all
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

random_state: 42

smote:
  k_neighbors: 5
  sampling_strategy: 1.0

undersampling:
  sampling_strategy: 1.0
  method: "random"

class_weights:
  method: "balanced"
  custom_weights:
    0: 1.0
    1: 2.0
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `random_state`: Random seed for reproducibility (default: 42)
- `smote.k_neighbors`: Number of nearest neighbors for SMOTE (default: 5)
- `smote.sampling_strategy`: Desired ratio of minority to majority class after SMOTE (default: 1.0)
- `undersampling.sampling_strategy`: Desired ratio after undersampling (default: 1.0)
- `undersampling.method`: Default undersampling method (random, tomek, enn)
- `class_weights.method`: Default class weighting method (balanced, inverse, custom)
- `class_weights.custom_weights`: Custom weight dictionary for each class

## Usage

### Command-Line Interface

#### Apply SMOTE Oversampling

```bash
python src/main.py --input data.csv --target-col target --method smote --output resampled.csv
```

#### Apply Undersampling

```bash
# Random undersampling
python src/main.py --input data.csv --target-col target --method undersample --undersample-method random --output resampled.csv

# Tomek Links undersampling
python src/main.py --input data.csv --target-col target --method undersample --undersample-method tomek --output resampled.csv

# ENN undersampling
python src/main.py --input data.csv --target-col target --method undersample --undersample-method enn --output resampled.csv
```

#### Compute Class Weights

```bash
python src/main.py --input data.csv --target-col target --method class_weights --weights-output weights.json
```

#### Apply All Techniques

```bash
python src/main.py --input data.csv --target-col target --method all --output resampled.csv --weights-output weights.json
```

#### With Feature Scaling

```bash
python src/main.py --input data.csv --target-col target --method smote --scale --output resampled.csv
```

### Programmatic Usage

#### SMOTE Oversampling

```python
import numpy as np
import pandas as pd
from src.main import SMOTE, ImbalancedDatasetHandler

# Using SMOTE directly
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 0, 0, 1])

smote = SMOTE(k_neighbors=5, sampling_strategy=1.0, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Using handler
handler = ImbalancedDatasetHandler()
X_resampled, y_resampled = handler.apply_smote(X, y, scale_features=True)
```

#### Undersampling

```python
from src.main import Undersampler, ImbalancedDatasetHandler

# Random undersampling
X_resampled, y_resampled = Undersampler.random_undersample(
    X, y, sampling_strategy=1.0, random_state=42
)

# Tomek Links
X_resampled, y_resampled = Undersampler.tomek_links_undersample(X, y)

# ENN
X_resampled, y_resampled = Undersampler.edited_nearest_neighbours_undersample(X, y)

# Using handler
handler = ImbalancedDatasetHandler()
X_resampled, y_resampled = handler.apply_undersampling(X, y, method="random")
```

#### Class Weighting

```python
from src.main import ClassWeightCalculator, ImbalancedDatasetHandler

# Balanced weights
weights = ClassWeightCalculator.balanced_weights(y)

# Inverse frequency weights
weights = ClassWeightCalculator.compute_class_weight(y, method="inverse")

# Custom weights
custom_weights = {0: 1.0, 1: 2.0, 2: 1.5}
weights = ClassWeightCalculator.custom_weights(y, custom_weights)

# Using handler
handler = ImbalancedDatasetHandler()
weights = handler.compute_class_weights(y, method="balanced")
```

#### Class Distribution Analysis

```python
handler = ImbalancedDatasetHandler()
distribution = handler.get_class_distribution(y)

print(f"Total samples: {distribution['total_samples']}")
print(f"Number of classes: {distribution['n_classes']}")
print(f"Class counts: {distribution['class_counts']}")
print(f"Imbalance ratio: {distribution.get('imbalance_ratio', 'N/A')}")
```

#### Complete Workflow Example

```python
import pandas as pd
from src.main import ImbalancedDatasetHandler

# Load data
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1).values
y = df["target"].values

# Initialize handler
handler = ImbalancedDatasetHandler()

# Analyze original distribution
distribution = handler.get_class_distribution(y)
print("Original distribution:", distribution)

# Apply SMOTE
X_resampled, y_resampled = handler.apply_smote(X, y, scale_features=True)
print("After SMOTE:", handler.get_class_distribution(y_resampled))

# Compute class weights
weights = handler.compute_class_weights(y, method="balanced")
print("Class weights:", weights)

# Use weights with scikit-learn model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight=weights)
model.fit(X, y)
```

### Common Use Cases

1. **Binary Classification with Severe Imbalance**: Use SMOTE to balance classes before training
2. **Multi-class Imbalance**: Apply SMOTE or undersampling to balance multiple minority classes
3. **Large Datasets**: Use undersampling to reduce dataset size while maintaining balance
4. **Model Training**: Apply class weights directly in scikit-learn models without resampling
5. **Data Exploration**: Analyze class distribution and imbalance ratios
6. **Pipeline Integration**: Integrate resampling into ML pipelines

## Project Structure

```
imbalanced-dataset-handling/
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
  - `SMOTE`: Class for SMOTE oversampling
  - `Undersampler`: Static methods for various undersampling techniques
  - `ClassWeightCalculator`: Static methods for computing class weights
  - `ImbalancedDatasetHandler`: Main handler class with CLI interface
  - `main()`: Command-line interface entry point

- `tests/test_main.py`: Comprehensive unit tests including:
  - SMOTE functionality tests
  - Undersampling method tests
  - Class weight calculation tests
  - Handler integration tests
  - Error handling tests

- `config.yaml`: Configuration file for logging and algorithm parameters

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
- Unit tests for SMOTE oversampling
- Unit tests for all undersampling methods
- Unit tests for class weight calculation
- Integration tests for the handler
- Error handling and edge case tests
- Input validation tests

Current test coverage: >90% of code paths

## Algorithm Details

### SMOTE (Synthetic Minority Oversampling Technique)

SMOTE generates synthetic samples for minority classes by:
1. Finding k nearest neighbors for each minority sample
2. Randomly selecting one neighbor
3. Creating a synthetic sample along the line segment between the sample and its neighbor
4. Repeating until desired class balance is achieved

**Advantages**:
- Creates diverse synthetic samples
- Reduces overfitting compared to simple oversampling
- Works well with continuous features

**Limitations**:
- May create noise in high-dimensional spaces
- Can be computationally expensive for large datasets
- Less effective with discrete/categorical features

### Undersampling Techniques

#### Random Undersampling
- Randomly removes majority class samples
- Simple and fast
- May lose important information

#### Tomek Links
- Removes ambiguous samples (Tomek link pairs)
- Helps clean decision boundaries
- Preserves more information than random undersampling

#### Edited Nearest Neighbours (ENN)
- Removes misclassified samples
- Focuses on removing noisy samples
- Can be conservative in sample removal

### Class Weighting

Class weights adjust the loss function during training:
- **Balanced**: Weights inversely proportional to class frequency
- **Inverse**: Weights based on inverse frequency
- **Custom**: User-defined weights for specific requirements

**Advantages**:
- No data modification required
- Preserves all original samples
- Easy to integrate with existing models

## Troubleshooting

### Common Issues and Solutions

#### Issue: SMOTE Fails with Insufficient Neighbors
**Error**: `ValueError: Not enough neighbors`

**Solution**: Reduce `k_neighbors` parameter or increase minority class samples. For very small datasets, consider using simpler oversampling or class weighting instead.

#### Issue: Undersampling Removes Too Many Samples
**Error**: Dataset becomes too small after undersampling

**Solution**: Adjust `sampling_strategy` to keep more majority samples, or use SMOTE instead of undersampling for small datasets.

#### Issue: Memory Error with Large Datasets
**Error**: Out of memory during SMOTE or undersampling

**Solution**: 
- Use undersampling instead of SMOTE for very large datasets
- Process data in batches
- Reduce `k_neighbors` for SMOTE
- Use class weighting instead of resampling

#### Issue: Class Weights Not Working with Model
**Error**: Model doesn't accept class_weight parameter

**Solution**: Check model documentation. Some models require `sample_weight` instead of `class_weight`. Convert class weights to sample weights:
```python
sample_weights = np.array([weights[y[i]] for i in range(len(y))])
model.fit(X, y, sample_weight=sample_weights)
```

### Error Message Explanations

- **"X and y must have same number of samples"**: Feature matrix and labels have mismatched lengths
- **"SMOTE requires at least 2 classes"**: Dataset has only one class
- **"Not enough neighbors"**: Minority class has fewer samples than k_neighbors
- **"Unknown method"**: Invalid method name provided

## Performance Considerations

### SMOTE
- Time complexity: O(n_minority * k * n_features)
- Memory complexity: O(n_samples * n_features)
- Best for: Medium-sized datasets (< 100K samples)

### Undersampling
- Time complexity: O(n_samples * log(n_samples)) for Tomek/ENN
- Memory complexity: O(n_samples * n_features)
- Best for: Large datasets where reducing size is beneficial

### Class Weighting
- Time complexity: O(1) for weight calculation
- Memory complexity: O(n_classes)
- Best for: Any dataset size, when resampling is not desired

## Contributing

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Create a feature branch: `git checkout -b feature/new-sampling-method`

### Code Style Guidelines

- Follow PEP 8 strictly
- Maximum line length: 88 characters
- Use type hints for all functions
- Write docstrings for all public functions and classes
- Add unit tests for new methods
- Update documentation for new features

### Pull Request Process

1. Ensure all tests pass: `pytest tests/test_main.py`
2. Check code coverage: `pytest --cov=src`
3. Update documentation if adding new features
4. Submit pull request with clear description

## License

This project is part of the Python ML Projects collection. Please refer to the main repository license.

## Additional Resources

- [SMOTE Paper](https://www.jair.org/index.php/jair/article/view/10302)
- [Imbalanced Learning Book](https://www.amazon.com/Imbalanced-Learning-Foundations-Algorithms-Applications/dp/1118074629)
- [scikit-learn Imbalanced Learning](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils)
- [Class Imbalance Problem](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)
