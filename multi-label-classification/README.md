# Multi-Label Classification

A Python implementation of multi-label classification methods including Binary Relevance, Classifier Chains, and Label Powerset. This tool provides comprehensive solutions for handling classification problems where each sample can belong to multiple classes simultaneously.

## Project Title and Description

The Multi-Label Classification tool provides implementations of three fundamental approaches to multi-label classification: Binary Relevance, Classifier Chains, and Label Powerset. These methods transform the multi-label problem into different formulations, each with unique advantages for different scenarios.

This tool solves the problem of classifying instances into multiple categories simultaneously, which is common in applications like text categorization, image tagging, music genre classification, and medical diagnosis where items can have multiple labels.

**Target Audience**: Data scientists, machine learning engineers, researchers working with multi-label problems, and anyone needing to classify data into multiple categories simultaneously.

## Features

### Binary Relevance
- **Independent Classifiers**: Trains one binary classifier per label
- **Simple and Fast**: Easy to implement and parallelize
- **No Label Dependencies**: Treats each label independently
- **Scalable**: Works well with many labels
- **Probability Support**: Provides probability estimates

### Classifier Chains
- **Label Dependencies**: Captures dependencies between labels
- **Chain-Based Prediction**: Uses previous predictions as features
- **Configurable Order**: Custom or random chain ordering
- **Better Accuracy**: Often outperforms Binary Relevance
- **Probability Support**: Provides probability estimates

### Label Powerset
- **Label Combination Learning**: Learns label combinations as classes
- **Captures Correlations**: Naturally handles label correlations
- **Multi-Class Transformation**: Converts to multi-class problem
- **Fewer Unique Combinations**: Efficient when label combinations are limited
- **Probability Support**: Provides probability estimates

### Additional Features
- Unified interface for all three methods
- Support for various base estimators (Logistic Regression, Random Forest)
- Multi-label evaluation metrics (Hamming loss, accuracy, F1, Jaccard)
- Support for list of label sets or binary matrix input
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
cd /path/to/Python-ml-projects/multi-label-classification
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
python src/main.py --input sample.csv --target-cols label1 label2 label3 --method all
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

binary_relevance:
  estimator: "logistic_regression"
  n_estimators: 100
  random_state: 42
  max_iter: 1000

classifier_chain:
  estimator: "logistic_regression"
  n_estimators: 100
  random_state: 42
  max_iter: 1000
  order: null

label_powerset:
  estimator: "random_forest"
  n_estimators: 100
  random_state: 42
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `binary_relevance.estimator`: Base estimator - "logistic_regression" or "random_forest"
- `binary_relevance.n_estimators`: Number of trees for Random Forest (default: 100)
- `binary_relevance.random_state`: Random seed (default: 42)
- `classifier_chain.estimator`: Base estimator for chain (default: "logistic_regression")
- `classifier_chain.order`: Custom chain order (default: null, random)
- `label_powerset.estimator`: Base estimator (default: "random_forest")

## Usage

### Command-Line Interface

#### Train and Evaluate All Methods

```bash
python src/main.py --input train.csv --target-cols label1 label2 label3 \
  --method all --test-data test.csv --evaluation-output evaluation.json
```

#### Use Specific Method

```bash
# Binary Relevance
python src/main.py --input data.csv --target-cols label1 label2 \
  --method binary_relevance --output predictions.csv

# Classifier Chain
python src/main.py --input data.csv --target-cols label1 label2 \
  --method classifier_chain --output predictions.csv

# Label Powerset
python src/main.py --input data.csv --target-cols label1 label2 \
  --method label_powerset --output predictions.csv
```

### Programmatic Usage

#### Basic Multi-Label Classification

```python
import numpy as np
from src.main import MultiLabelClassifier

# Generate sample data
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, (100, 3))  # 3 labels

# Initialize classifier
classifier = MultiLabelClassifier()

# Load data
classifier.load_data(X, y)

# Fit Binary Relevance
classifier.fit_binary_relevance()

# Predict
X_test = np.random.randn(10, 5)
predictions = classifier.predict(X_test, method="binary_relevance")
print("Predictions shape:", predictions.shape)
```

#### Using Individual Classes

```python
from src.main import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.linear_model import LogisticRegression

# Binary Relevance
br = BinaryRelevance(base_estimator=LogisticRegression(random_state=42))
br.fit(X, y)
predictions = br.predict(X_test)
probabilities = br.predict_proba(X_test)

# Classifier Chain
cc = ClassifierChain(
    base_estimator=LogisticRegression(random_state=42),
    order=[2, 0, 1]  # Custom order
)
cc.fit(X, y)
predictions = cc.predict(X_test)

# Label Powerset
lp = LabelPowerset()
lp.fit(X, y)
predictions = lp.predict(X_test)
```

#### With List of Label Sets

```python
# Input as list of label sets
X = np.random.randn(100, 5)
y = [[0, 1], [1], [0, 1, 2], [2], [0]] * 20

classifier = MultiLabelClassifier()
classifier.load_data(X, y)
classifier.fit_binary_relevance()

predictions = classifier.predict(X_test, method="binary_relevance")
```

#### Evaluation

```python
from src.main import MultiLabelEvaluator

# Evaluate predictions
metrics = MultiLabelEvaluator.evaluate(y_true, y_pred)

print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print(f"Jaccard Score: {metrics['jaccard_score']:.4f}")

# Or use classifier's evaluate method
evaluation = classifier.evaluate(X_test, y_test, method="binary_relevance")
```

#### Compare All Methods

```python
classifier = MultiLabelClassifier()
classifier.load_data(X, y)

# Fit all methods
classifier.fit_all()

# Evaluate all
br_eval = classifier.evaluate(X_test, y_test, method="binary_relevance")
cc_eval = classifier.evaluate(X_test, y_test, method="classifier_chain")
lp_eval = classifier.evaluate(X_test, y_test, method="label_powerset")

print("Binary Relevance F1:", br_eval["f1_score"])
print("Classifier Chain F1:", cc_eval["f1_score"])
print("Label Powerset F1:", lp_eval["f1_score"])
```

### Common Use Cases

1. **Text Categorization**: Classify documents into multiple topics
2. **Image Tagging**: Tag images with multiple labels
3. **Music Genre Classification**: Classify songs into multiple genres
4. **Medical Diagnosis**: Diagnose multiple conditions simultaneously
5. **Protein Function Prediction**: Predict multiple protein functions
6. **Emotion Recognition**: Recognize multiple emotions in text/speech
7. **Content Recommendation**: Recommend multiple content types

## Project Structure

```
multi-label-classification/
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
  - `BinaryRelevance`: Binary Relevance classifier
  - `ClassifierChain`: Classifier Chain classifier
  - `LabelPowerset`: Label Powerset classifier
  - `MultiLabelEvaluator`: Evaluation metrics
  - `MultiLabelClassifier`: Main classifier class with CLI interface
  - `main()`: Command-line interface entry point

- `tests/test_main.py`: Comprehensive unit tests including:
  - Binary Relevance tests
  - Classifier Chain tests
  - Label Powerset tests
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
- Unit tests for each classification method
- Evaluation metric tests
- Integration tests for complete workflows
- Error handling and edge case tests

Current test coverage: >90% of code paths

## Algorithm Details

### Binary Relevance

**How it works**:
1. Transform multi-label problem into L independent binary problems
2. Train one binary classifier for each label
3. Each classifier predicts presence/absence of one label
4. Combine predictions to get multi-label output

**Advantages**:
- Simple and intuitive
- Fast training and prediction
- Easy to parallelize
- Works well with many labels
- No label dependency assumptions

**Limitations**:
- Ignores label correlations
- May predict incompatible label combinations
- Each classifier trained independently

**Best for**: When labels are independent, large number of labels, fast training needed

### Classifier Chains

**How it works**:
1. Order labels in a chain
2. Train binary classifier for first label using original features
3. Train classifier for second label using original features + first label prediction
4. Continue for all labels in chain order
5. Predict labels sequentially using previous predictions

**Advantages**:
- Captures label dependencies
- Often better accuracy than Binary Relevance
- Flexible chain ordering
- Can use domain knowledge for ordering

**Limitations**:
- Order-dependent (different orders give different results)
- Error propagation (early errors affect later predictions)
- Slower than Binary Relevance
- Requires careful ordering

**Best for**: When labels are correlated, moderate number of labels, accuracy is priority

### Label Powerset

**How it works**:
1. Treat each unique label combination as a separate class
2. Transform multi-label problem into multi-class problem
3. Train one multi-class classifier
4. Map predictions back to label sets

**Advantages**:
- Naturally captures label correlations
- Treats label combinations as atomic units
- Can learn complex label interactions
- Good when label combinations are limited

**Limitations**:
- Exponential growth in number of classes
- May have many classes with few samples
- Less scalable for many labels
- Requires sufficient data for each combination

**Best for**: When label combinations are limited, strong label correlations, small to medium number of labels

## Choosing the Right Method

### Use Binary Relevance when:
- Labels are independent
- You have many labels (> 50)
- Speed is important
- Labels don't have strong correlations
- You need simple, interpretable solution

### Use Classifier Chains when:
- Labels are correlated
- You have moderate number of labels (5-50)
- Accuracy is more important than speed
- You can determine good label ordering
- You want to capture label dependencies

### Use Label Powerset when:
- Labels have strong correlations
- Number of unique label combinations is limited
- You have sufficient data for each combination
- Label interactions are important
- Number of labels is small to medium (< 20)

## Evaluation Metrics

### Hamming Loss
- Measures fraction of labels incorrectly predicted
- Range: [0, 1], lower is better
- Formula: `(1/n) * (1/L) * Σ |y_true XOR y_pred|`

### Accuracy (Exact Match)
- Fraction of samples with all labels correctly predicted
- Range: [0, 1], higher is better
- Strict metric (all labels must match)

### Precision, Recall, F1-Score
- Can be macro-averaged (per label) or micro-averaged (per sample)
- Macro: Average across labels
- Micro: Aggregate across all labels
- Range: [0, 1], higher is better

### Jaccard Score
- Intersection over union of predicted and true labels
- Range: [0, 1], higher is better
- Measures similarity between label sets

## Troubleshooting

### Common Issues and Solutions

#### Issue: Too Many Label Combinations in Label Powerset
**Problem**: Label Powerset creates too many classes

**Solution**:
- Use Binary Relevance or Classifier Chains instead
- Filter rare label combinations
- Increase minimum samples per combination
- Reduce number of labels

#### Issue: Poor Performance with Binary Relevance
**Problem**: Labels are correlated but treated independently

**Solution**:
- Try Classifier Chains to capture dependencies
- Use Label Powerset if combinations are limited
- Consider feature engineering to capture correlations

#### Issue: Classifier Chain Order Matters Too Much
**Problem**: Performance varies significantly with chain order

**Solution**:
- Try multiple random orders and average
- Use domain knowledge to determine order
- Order by label frequency or importance
- Consider ensemble of multiple chains

#### Issue: Memory Error with Label Powerset
**Problem**: Too many unique label combinations

**Solution**:
- Use Binary Relevance or Classifier Chains
- Filter rare combinations
- Reduce number of labels
- Use sparse representations

#### Issue: Slow Training
**Problem**: Training takes too long

**Solution**:
- Use Binary Relevance (fastest)
- Reduce number of labels
- Use faster base estimators
- Reduce training data size
- Use parallel processing

### Error Message Explanations

- **"y must be 2D array or list of label sets"**: Labels must be multi-label format
- **"X and y must have same number of samples"**: Input arrays have mismatched lengths
- **"Model must be fitted before prediction"**: Call fit() before predict()
- **"Data must be loaded before fitting"**: Load data before fitting model
- **"Unknown method"**: Invalid method name

## Performance Considerations

### Computational Complexity

- **Binary Relevance**: O(L * T) where L=labels, T=training time per classifier
- **Classifier Chains**: O(L * T) but sequential (can't parallelize easily)
- **Label Powerset**: O(C * T) where C=unique combinations (can be large)

### Optimization Tips

1. **For many labels**: Use Binary Relevance
2. **For correlated labels**: Use Classifier Chains
3. **For limited combinations**: Use Label Powerset
4. **For speed**: Use Binary Relevance with fast base estimator
5. **For accuracy**: Try all three and compare

## Best Practices

1. **Start with Binary Relevance**: Baseline method, fast and simple
2. **Try Classifier Chains**: If labels are correlated
3. **Use Label Powerset Sparingly**: Only when combinations are limited
4. **Evaluate with Multiple Metrics**: Hamming loss, F1, Jaccard
5. **Consider Label Ordering**: For Classifier Chains, order matters
6. **Handle Imbalanced Labels**: Some labels may be rare
7. **Validate on Test Set**: Always evaluate on held-out data

## Contributing

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Create a feature branch: `git checkout -b feature/new-method`

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

- [Multi-Label Classification Wikipedia](https://en.wikipedia.org/wiki/Multi-label_classification)
- [Classifier Chains Paper](https://www.cs.waikato.ac.nz/~ml/publications/2009/chains.pdf)
- [Label Powerset Paper](https://link.springer.com/chapter/10.1007/978-3-540-74958-5_12)
- [Scikit-learn Multi-Label](https://scikit-learn.org/stable/modules/multiclass.html)
- [Multi-Label Learning Survey](https://link.springer.com/article/10.1007/s10994-013-5341-z)
