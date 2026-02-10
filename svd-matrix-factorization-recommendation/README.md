# SVD Matrix Factorization for Recommendation Systems

A Python implementation of matrix factorization for recommendation systems using Singular Value Decomposition (SVD). This tool provides a complete solution for building recommendation systems that factorize user-item rating matrices into lower-dimensional latent factor spaces.

## Project Title and Description

The SVD Matrix Factorization Recommendation System provides an implementation of matrix factorization using Singular Value Decomposition to predict user-item ratings and generate personalized recommendations. It decomposes the user-item rating matrix into user and item factor matrices, capturing latent features that explain user preferences and item characteristics.

This tool solves the problem of providing accurate recommendations by learning low-dimensional representations of users and items. It helps businesses and applications suggest relevant items to users by discovering hidden patterns in user-item interactions through dimensionality reduction.

**Target Audience**: Data scientists, machine learning engineers, software developers building recommendation systems, e-commerce platforms, content platforms, and anyone needing matrix factorization-based recommendation capabilities.

## Features

### SVD Matrix Factorization
- **Truncated SVD**: Efficient SVD implementation for sparse matrices
- **Latent Factor Learning**: Discovers hidden features in user-item interactions
- **Configurable Components**: Adjustable number of latent factors
- **Multiple Algorithms**: Support for ARPACK and randomized SVD algorithms
- **Explained Variance**: Tracks how much variance is captured by factors

### Rating Prediction
- **User-Item Rating Prediction**: Predicts ratings for any user-item pair
- **Bias Adjustment**: Incorporates user and item biases for better predictions
- **Global Mean Fallback**: Handles cold start problems with default ratings

### Recommendations
- **Top-N Recommendations**: Generates personalized item recommendations
- **Ranking by Predicted Ratings**: Sorts items by predicted preference
- **Unrated Items Only**: Recommends only items user hasn't rated

### Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Measures average prediction error
- **Precision and Recall**: Measures recommendation quality (at k)
- **Catalog Coverage**: Measures diversity of recommendations

### Additional Features
- Command-line interface for batch processing
- Configuration via YAML file
- Support for CSV input/output
- Comprehensive logging
- Input validation and error handling
- Model information and statistics
- Factor matrix access

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/svd-matrix-factorization-recommendation
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
python src/main.py --input sample_ratings.csv --user-id 1 --n-recommendations 10
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

svd:
  n_components: 50
  n_iter: 5
  random_state: 42
  algorithm: "arpack"
  tol: 0.0

recommendation:
  default_n_recommendations: 10

evaluation:
  metrics:
    - "rmse"
    - "mae"
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `svd.n_components`: Number of latent factors (default: 50)
- `svd.n_iter`: Number of iterations for SVD (default: 5)
- `svd.random_state`: Random seed for reproducibility (default: 42)
- `svd.algorithm`: SVD algorithm - "arpack" or "randomized" (default: "arpack")
- `svd.tol`: Tolerance for convergence (default: 0.0)
- `recommendation.default_n_recommendations`: Default number of recommendations (default: 10)

## Usage

### Command-Line Interface

#### Generate Recommendations

```bash
python src/main.py --input ratings.csv --user-id 123 \
  --n-recommendations 10 --output recommendations.csv
```

#### Specify Number of Components

```bash
python src/main.py --input ratings.csv --n-components 100 \
  --user-id 123 --n-recommendations 10
```

#### Evaluate Model Performance

```bash
python src/main.py --input train_ratings.csv --test-data test_ratings.csv \
  --evaluation-output evaluation.json
```

### Programmatic Usage

#### Basic Recommendation

```python
import pandas as pd
from src.main import SVDRecommendationSystem

# Load ratings data
ratings = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3],
    'item_id': [1, 2, 1, 2, 1, 2],
    'rating': [5, 4, 4, 5, 3, 4]
})

# Initialize system
system = SVDRecommendationSystem()

# Load data
system.load_data(ratings)

# Fit model
system.fit(n_components=50)

# Generate recommendations
recommendations = system.recommend(user_id=1, n_recommendations=10)

print("Recommendations:")
for item_id, predicted_rating in recommendations:
    print(f"  Item {item_id}: {predicted_rating:.2f}")
```

#### Using SVDRecommender Directly

```python
from src.main import SVDRecommender

# Initialize recommender
recommender = SVDRecommender(
    n_components=50,
    n_iter=5,
    random_state=42
)

# Fit model
recommender.fit(ratings)

# Predict rating
predicted_rating = recommender.predict_rating(user_id=1, item_id=3)

# Get recommendations
recommendations = recommender.recommend_items(user_id=1, n_recommendations=10)

# Get model statistics
explained_variance = recommender.get_explained_variance()
user_factors, item_factors = recommender.get_components()
```

#### Model Evaluation

```python
from src.main import SVDRecommendationSystem

# Load training and test data
train_ratings = pd.read_csv("train_ratings.csv")
test_ratings = pd.read_csv("test_ratings.csv")

# Fit model
system = SVDRecommendationSystem()
system.load_data(train_ratings)
system.fit(n_components=50)

# Evaluate
evaluation = system.evaluate(
    test_data=test_ratings,
    metrics=["rmse", "mae"]
)

print(f"RMSE: {evaluation['rmse']:.4f}")
print(f"MAE: {evaluation['mae']:.4f}")
```

#### Model Information

```python
# Get model information
info = system.get_model_info()

print(f"Number of components: {info['n_components']}")
print(f"Explained variance: {info['explained_variance']:.4f}")
print(f"Number of users: {info['n_users']}")
print(f"Number of items: {info['n_items']}")
print(f"Global mean rating: {info['global_mean']:.2f}")
```

### Common Use Cases

1. **E-commerce Recommendations**: Suggest products to customers
2. **Movie/Content Recommendations**: Recommend movies, shows, or articles
3. **Music Recommendations**: Suggest songs or playlists
4. **Book Recommendations**: Recommend books based on reading history
5. **News Recommendations**: Personalize news feed
6. **Job Recommendations**: Match candidates with job postings
7. **Social Media**: Suggest friends, groups, or content

## Project Structure

```
svd-matrix-factorization-recommendation/
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
  - `SVDRecommender`: Core SVD-based recommender class
  - `MatrixFactorizationEvaluator`: Evaluation metrics
  - `SVDRecommendationSystem`: Main system class with CLI interface
  - `main()`: Command-line interface entry point

- `tests/test_main.py`: Comprehensive unit tests including:
  - SVD recommender tests
  - Evaluation metric tests
  - Integration tests
  - Error handling tests

- `config.yaml`: Configuration file for model parameters

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
- Unit tests for SVD recommender
- Evaluation metric tests
- Integration tests for complete workflows
- Error handling and edge case tests

Current test coverage: >90% of code paths

## Algorithm Details

### Singular Value Decomposition (SVD)

**Mathematical Formulation**:
```
R ≈ U × Σ × V^T
```

Where:
- `R`: User-item rating matrix (m × n)
- `U`: User factor matrix (m × k)
- `Σ`: Singular values (k × k)
- `V`: Item factor matrix (n × k)
- `k`: Number of latent factors (n_components)

**Truncated SVD**:
- Uses only top k singular values
- Reduces dimensionality while preserving most variance
- More efficient than full SVD for large matrices

**Rating Prediction**:
```
r(u,i) = r̄ + b_u + b_i + U_u · V_i^T
```

Where:
- `r(u,i)`: Predicted rating for user u and item i
- `r̄`: Global mean rating
- `b_u`: User bias (mean rating deviation)
- `b_i`: Item bias (mean rating deviation)
- `U_u`: User factor vector
- `V_i`: Item factor vector

### Advantages of SVD

1. **Dimensionality Reduction**: Captures essential patterns in lower dimensions
2. **Scalability**: Efficient for large sparse matrices
3. **Generalization**: Learns latent features that generalize well
4. **Interpretability**: Factor matrices can reveal user/item characteristics
5. **Cold Start Handling**: Can predict for new users/items using global mean

### Limitations

1. **Cold Start Problem**: New users/items with no ratings are challenging
2. **Sparsity**: Requires sufficient ratings for good performance
3. **Non-linearity**: Assumes linear relationships in latent space
4. **Interpretability**: Latent factors are not directly interpretable
5. **Computational Cost**: Can be expensive for very large matrices

## Choosing Number of Components

### Factors to Consider

1. **Data Size**: More components for larger datasets
2. **Sparsity**: Fewer components for very sparse data
3. **Computational Resources**: Balance accuracy vs. speed
4. **Explained Variance**: Aim for 80-90% explained variance
5. **Overfitting**: Too many components can overfit

### Guidelines

- **Small datasets (< 1000 users/items)**: 10-20 components
- **Medium datasets (1000-10000)**: 20-50 components
- **Large datasets (> 10000)**: 50-200 components
- **Very large datasets**: 100-500 components

### Tuning Strategy

1. Start with default (50 components)
2. Evaluate on validation set
3. Increase if explained variance < 80%
4. Decrease if overfitting (large train-test gap)
5. Use cross-validation for optimal value

## Troubleshooting

### Common Issues and Solutions

#### Issue: Low Explained Variance
**Problem**: Too few components or poor data quality

**Solution**:
- Increase `n_components`
- Check data quality and remove outliers
- Ensure sufficient ratings per user/item
- Try different algorithms (randomized vs arpack)

#### Issue: Poor Prediction Accuracy
**Problem**: Model parameters not optimized

**Solution**:
- Tune `n_components` using cross-validation
- Increase `n_iter` for better convergence
- Check for data leakage or preprocessing issues
- Ensure train/test split is appropriate

#### Issue: Memory Error with Large Datasets
**Problem**: Rating matrix too large

**Solution**:
- Use sparse matrix representations
- Reduce `n_components`
- Use randomized algorithm (more memory efficient)
- Process data in batches
- Use dimensionality reduction preprocessing

#### Issue: Cold Start Problem
**Problem**: New users/items have no ratings

**Solution**:
- Use global mean for predictions
- Implement hybrid approach (combine with content-based)
- Collect initial user preferences
- Use demographic information
- Recommend popular items to new users

#### Issue: Slow Training
**Problem**: Large matrix or many components

**Solution**:
- Use randomized algorithm (faster)
- Reduce `n_components`
- Reduce `n_iter` (may reduce accuracy)
- Use sparse matrix operations
- Parallelize computation if possible

### Error Message Explanations

- **"ratings must contain columns"**: Input DataFrame missing required columns
- **"Model must be fitted before prediction"**: Call fit() before predict()
- **"Data must be loaded before fitting"**: Load data before fitting model
- **"n_components reduced to X"**: Auto-adjusted to match matrix dimensions

## Performance Considerations

### Computational Complexity

- **Training**: O(min(mn², m²n)) for full SVD, O(mnk) for truncated SVD
- **Prediction**: O(k) where k is number of components
- **Recommendation**: O(nk) where n is number of items

### Optimization Tips

1. **Use Truncated SVD**: Much faster than full SVD
2. **Choose Right Algorithm**: Randomized for large matrices, ARPACK for accuracy
3. **Sparse Matrices**: Use sparse representations for memory efficiency
4. **Component Selection**: Balance accuracy vs. speed
5. **Caching**: Cache factor matrices for repeated predictions

## Comparison with Other Methods

### vs. Collaborative Filtering

**SVD Advantages**:
- More scalable for large datasets
- Better generalization
- Handles sparsity better
- Faster prediction

**Collaborative Filtering Advantages**:
- More interpretable
- Better for small datasets
- No hyperparameter tuning needed

### vs. Deep Learning

**SVD Advantages**:
- Faster training and prediction
- Less data required
- More interpretable factors
- Lower computational cost

**Deep Learning Advantages**:
- Can capture non-linear patterns
- Better for very large datasets
- Can incorporate side information
- Higher accuracy potential

## Contributing

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Create a feature branch: `git checkout -b feature/new-feature`

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

- [SVD Wikipedia](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- [Matrix Factorization for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- [Netflix Prize](https://www.netflixprize.com/)
- [Scikit-learn TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- [Recommender Systems Handbook](https://www.springer.com/gp/book/9780387858197)
