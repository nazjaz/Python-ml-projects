# Collaborative Filtering Recommendation System

A Python implementation of collaborative filtering recommendation systems using both user-based and item-based approaches with various similarity metrics and evaluation methods. This tool provides a complete solution for building recommendation systems based on user-item interactions.

## Project Title and Description

The Collaborative Filtering Recommendation System provides implementations of two fundamental collaborative filtering approaches: user-based and item-based filtering. It uses similarity metrics to find similar users or items and generates personalized recommendations based on historical interactions.

This tool solves the problem of providing personalized recommendations to users by leveraging patterns in user-item interactions. It helps businesses and applications suggest relevant items to users without requiring explicit item features or user profiles.

**Target Audience**: Data scientists, machine learning engineers, software developers building recommendation systems, e-commerce platforms, content platforms, and anyone needing personalized recommendation capabilities.

## Features

### User-Based Collaborative Filtering
- **User Similarity**: Find users with similar preferences
- **Rating Prediction**: Predict ratings for user-item pairs
- **Top-N Recommendations**: Generate personalized item recommendations
- **Similarity Metrics**: Cosine similarity and Pearson correlation
- **Configurable Neighbors**: Adjustable number of similar users to consider

### Item-Based Collaborative Filtering
- **Item Similarity**: Find items with similar user preferences
- **Rating Prediction**: Predict ratings based on item similarities
- **Top-N Recommendations**: Generate item recommendations
- **Similarity Metrics**: Cosine similarity and adjusted cosine similarity
- **Configurable Neighbors**: Adjustable number of similar items to consider

### Similarity Metrics
- **Cosine Similarity**: Measures angle between rating vectors
- **Pearson Correlation**: Measures linear correlation between ratings
- **Adjusted Cosine Similarity**: Mean-centered cosine similarity for items

### Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Measures average prediction error
- **Precision and Recall**: Measures recommendation quality (at k)

### Additional Features
- Unified interface for both approaches
- Command-line interface for batch processing
- Configuration via YAML file
- Support for CSV input/output
- Comprehensive logging
- Input validation and error handling
- Flexible similarity metric selection
- Top-N recommendation generation

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/collaborative-filtering-recommendation
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
python src/main.py --input sample_ratings.csv --method both --user-id 1
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

user_based:
  similarity_metric: "cosine"
  n_neighbors: 50
  min_common_items: 1

item_based:
  similarity_metric: "cosine"
  n_neighbors: 50
  min_common_users: 1

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
- `user_based.similarity_metric`: Similarity metric - "cosine" or "pearson" (default: "cosine")
- `user_based.n_neighbors`: Number of similar users to consider (default: 50)
- `user_based.min_common_items`: Minimum common items for similarity (default: 1)
- `item_based.similarity_metric`: Similarity metric - "cosine" or "adjusted_cosine" (default: "cosine")
- `item_based.n_neighbors`: Number of similar items to consider (default: 50)
- `item_based.min_common_users`: Minimum common users for similarity (default: 1)
- `recommendation.default_n_recommendations`: Default number of recommendations (default: 10)

## Usage

### Command-Line Interface

#### Generate Recommendations with User-Based Method

```bash
python src/main.py --input ratings.csv --method user_based \
  --user-id 123 --n-recommendations 10 --output recommendations.csv
```

#### Generate Recommendations with Item-Based Method

```bash
python src/main.py --input ratings.csv --method item_based \
  --user-id 123 --n-recommendations 10 --output recommendations.csv
```

#### Use Both Methods

```bash
python src/main.py --input ratings.csv --method both \
  --user-id 123 --n-recommendations 10
```

#### Evaluate Model Performance

```bash
python src/main.py --input train_ratings.csv --method user_based \
  --test-data test_ratings.csv --evaluation-output evaluation.json
```

### Programmatic Usage

#### Basic Recommendation

```python
import pandas as pd
from src.main import CollaborativeFilteringRecommender

# Load ratings data
ratings = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3],
    'item_id': [1, 2, 1, 2, 1, 2],
    'rating': [5, 4, 4, 5, 3, 4]
})

# Initialize recommender
recommender = CollaborativeFilteringRecommender()

# Load data
recommender.load_data(ratings)

# Fit user-based model
recommender.fit_user_based()

# Generate recommendations
recommendations = recommender.recommend(
    user_id=1,
    method="user_based",
    n_recommendations=10
)

print("Recommendations:")
for item_id, predicted_rating in recommendations:
    print(f"  Item {item_id}: {predicted_rating:.2f}")
```

#### Item-Based Recommendation

```python
# Fit item-based model
recommender.fit_item_based()

# Generate recommendations
recommendations = recommender.recommend(
    user_id=1,
    method="item_based",
    n_recommendations=10
)
```

#### Using Individual Classes

```python
from src.main import UserBasedCollaborativeFiltering, ItemBasedCollaborativeFiltering

# User-based filtering
ubcf = UserBasedCollaborativeFiltering(
    similarity_metric="cosine",
    n_neighbors=50
)
ubcf.fit(ratings)

# Predict rating
predicted_rating = ubcf.predict_rating(user_id=1, item_id=3)

# Get recommendations
recommendations = ubcf.recommend_items(user_id=1, n_recommendations=10)

# Item-based filtering
ibcf = ItemBasedCollaborativeFiltering(
    similarity_metric="adjusted_cosine",
    n_neighbors=50
)
ibcf.fit(ratings)

# Predict rating
predicted_rating = ibcf.predict_rating(user_id=1, item_id=3)

# Get recommendations
recommendations = ibcf.recommend_items(user_id=1, n_recommendations=10)
```

#### Model Evaluation

```python
from src.main import CollaborativeFilteringRecommender

# Load training and test data
train_ratings = pd.read_csv("train_ratings.csv")
test_ratings = pd.read_csv("test_ratings.csv")

# Fit model
recommender = CollaborativeFilteringRecommender()
recommender.load_data(train_ratings)
recommender.fit_user_based()

# Evaluate
evaluation = recommender.evaluate(
    test_data=test_ratings,
    method="user_based",
    metrics=["rmse", "mae"]
)

print(f"RMSE: {evaluation['rmse']:.4f}")
print(f"MAE: {evaluation['mae']:.4f}")
```

#### Similarity Calculation

```python
from src.main import SimilarityCalculator
import numpy as np

vec1 = np.array([5, 4, 3, 2, 1])
vec2 = np.array([4, 5, 3, 2, 1])

# Cosine similarity
cosine_sim = SimilarityCalculator.cosine_similarity(vec1, vec2)

# Pearson correlation
pearson_corr = SimilarityCalculator.pearson_correlation(vec1, vec2)

print(f"Cosine Similarity: {cosine_sim:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")
```

### Common Use Cases

1. **E-commerce Recommendations**: Suggest products to customers
2. **Movie/Content Recommendations**: Recommend movies, shows, or articles
3. **Music Recommendations**: Suggest songs or playlists
4. **Book Recommendations**: Recommend books based on reading history
5. **Social Media**: Suggest friends, groups, or content
6. **News Recommendations**: Personalize news feed
7. **Job Recommendations**: Match candidates with job postings

## Project Structure

```
collaborative-filtering-recommendation/
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
  - `SimilarityCalculator`: Similarity metric calculations
  - `UserBasedCollaborativeFiltering`: User-based CF implementation
  - `ItemBasedCollaborativeFiltering`: Item-based CF implementation
  - `RecommendationEvaluator`: Evaluation metrics
  - `CollaborativeFilteringRecommender`: Main recommender class
  - `main()`: Command-line interface entry point

- `tests/test_main.py`: Comprehensive unit tests including:
  - Similarity calculation tests
  - User-based CF tests
  - Item-based CF tests
  - Evaluation tests
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
- Unit tests for similarity calculations
- Unit tests for user-based and item-based filtering
- Evaluation metric tests
- Integration tests for complete workflows
- Error handling and edge case tests

Current test coverage: >90% of code paths

## Algorithm Details

### User-Based Collaborative Filtering

**How it works**:
1. Find users similar to the target user based on rating patterns
2. Use ratings from similar users to predict target user's ratings
3. Recommend items with highest predicted ratings

**Prediction Formula**:
```
r(u,i) = r̄(u) + Σ sim(u,v) * (r(v,i) - r̄(v)) / Σ |sim(u,v)|
```

Where:
- `r(u,i)`: Predicted rating for user u and item i
- `r̄(u)`: Mean rating of user u
- `sim(u,v)`: Similarity between users u and v
- `r(v,i)`: Rating of user v for item i

**Advantages**:
- Intuitive and interpretable
- Works well when user preferences are stable
- Good for diverse item catalogs

**Limitations**:
- Computationally expensive for large user bases
- Sparse data problem (cold start)
- User preferences may change over time

### Item-Based Collaborative Filtering

**How it works**:
1. Find items similar to items the user has rated
2. Use ratings for similar items to predict ratings
3. Recommend items with highest predicted ratings

**Prediction Formula**:
```
r(u,i) = r̄(i) + Σ sim(i,j) * (r(u,j) - r̄(j)) / Σ |sim(i,j)|
```

Where:
- `r(u,i)`: Predicted rating for user u and item i
- `r̄(i)`: Mean rating of item i
- `sim(i,j)`: Similarity between items i and j
- `r(u,j)`: Rating of user u for item j

**Advantages**:
- More stable than user-based (items change less than users)
- Faster for large user bases
- Better for sparse user data
- Items have more ratings than users typically

**Limitations**:
- May miss serendipitous recommendations
- Less effective for new items (cold start)
- Requires sufficient item ratings

### Similarity Metrics

**Cosine Similarity**:
- Measures angle between rating vectors
- Range: [-1, 1]
- Good for high-dimensional sparse data
- Formula: `cos(θ) = (A · B) / (||A|| ||B||)`

**Pearson Correlation**:
- Measures linear correlation
- Range: [-1, 1]
- Accounts for user/item rating bias
- Formula: `r = Σ(xi - x̄)(yi - ȳ) / √(Σ(xi - x̄)² Σ(yi - ȳ)²)`

**Adjusted Cosine Similarity**:
- Mean-centered cosine similarity
- Used for item-based filtering
- Accounts for user rating bias
- Better than standard cosine for items

## Choosing Between User-Based and Item-Based

### Use User-Based When:
- User base is smaller than item catalog
- User preferences are stable
- You need interpretable recommendations
- Items are diverse and hard to compare

### Use Item-Based When:
- Item catalog is smaller than user base
- Items are relatively stable
- You need faster recommendations
- User data is sparse

## Troubleshooting

### Common Issues and Solutions

#### Issue: No Recommendations Generated
**Problem**: User has no similar users/items or insufficient data

**Solution**:
- Reduce `min_common_items` or `min_common_users`
- Increase `n_neighbors`
- Check data sparsity
- Use default ratings (mean ratings)

#### Issue: Poor Prediction Accuracy
**Problem**: Model parameters not optimized

**Solution**:
- Try different similarity metrics
- Adjust `n_neighbors` parameter
- Increase minimum common items/users
- Check data quality and remove outliers
- Use cross-validation to tune parameters

#### Issue: Cold Start Problem
**Problem**: New users or items have no ratings

**Solution**:
- Use hybrid approaches (combine with content-based)
- Use default/popular item recommendations
- Collect initial user preferences
- Use demographic information

#### Issue: Memory Error with Large Datasets
**Problem**: Ratings matrix too large

**Solution**:
- Use sparse matrix representations
- Process data in batches
- Reduce number of neighbors
- Use item-based (more efficient for large user bases)

#### Issue: Slow Recommendation Generation
**Problem**: Computing similarities for all users/items

**Solution**:
- Use item-based (generally faster)
- Reduce `n_neighbors`
- Cache similarity matrices
- Use approximate nearest neighbors

### Error Message Explanations

- **"ratings must contain columns"**: Input DataFrame missing required columns
- **"Model must be fitted before prediction"**: Call fit() before predict()
- **"Data must be loaded before fitting"**: Load data before fitting model
- **"Unknown method"**: Invalid method name (use "user_based" or "item_based")

## Performance Considerations

### Computational Complexity

- **User-Based CF**: O(n_users² * n_items) for similarity, O(n_items) per prediction
- **Item-Based CF**: O(n_items² * n_users) for similarity, O(n_items) per prediction

### Optimization Tips

1. **For large user bases**: Use item-based CF
2. **For large item catalogs**: Use user-based CF
3. **For sparse data**: Use item-based with adjusted cosine
4. **For speed**: Reduce `n_neighbors`, cache similarities
5. **For accuracy**: Increase `n_neighbors`, use better similarity metrics

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

- [Collaborative Filtering Wikipedia](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Item-Based Collaborative Filtering Paper](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)
- [Recommender Systems Handbook](https://www.springer.com/gp/book/9780387858197)
- [Netflix Prize](https://www.netflixprize.com/)
