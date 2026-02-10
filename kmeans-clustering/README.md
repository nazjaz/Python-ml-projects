# K-Means Clustering

A Python implementation of k-means clustering algorithm from scratch with elbow method for optimal cluster number selection. This is the twenty-seventh project in the ML learning series, focusing on understanding unsupervised learning and clustering algorithms.

## Project Title and Description

The K-Means Clustering tool provides a complete implementation of the k-means clustering algorithm from scratch, including k-means++ initialization and the elbow method for finding the optimal number of clusters. It helps users understand how unsupervised learning works and how to select the optimal number of clusters.

This tool solves the problem of learning clustering fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates centroid-based clustering, convergence detection, and hyperparameter optimization using the elbow method.

**Target Audience**: Beginners learning machine learning, students studying unsupervised learning algorithms, and anyone who needs to understand k-means clustering and cluster number selection from scratch.

## Features

- K-means clustering implementation from scratch
- Two initialization methods:
  - Random initialization
  - K-means++ initialization (smarter initialization)
- Elbow method for optimal cluster number selection
- Convergence detection
- Inertia calculation (within-cluster sum of squares)
- Cluster visualization (for 2D data)
- Elbow curve visualization
- Support for multiple features
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
cd /path/to/Python-ml-projects/kmeans-clustering
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
python src/main.py --input sample.csv --n-clusters 3
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  n_clusters: 3
  max_iterations: 300
  tolerance: 1e-4
  init: "random"
  random_state: null
  k_range: [2, 3, 4, 5, 6, 7, 8, 9, 10]
  n_runs: 1
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.n_clusters`: Number of clusters (default: 3)
- `model.max_iterations`: Maximum number of iterations (default: 300)
- `model.tolerance`: Convergence tolerance (default: 1e-4)
- `model.init`: Initialization method. Options: "random", "k-means++" (default: "random")
- `model.random_state`: Random seed for reproducibility (default: null)
- `model.k_range`: List of k values to test during elbow method (default: [2, 3, 4, 5, 6, 7, 8, 9, 10])
- `model.n_runs`: Number of runs per k value for averaging (default: 1)

## Usage

### Basic Usage

```python
from src.main import KMeans
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])

# Initialize and fit model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_
print(f"Cluster labels: {labels}")

# Get centroids
centroids = kmeans.centroids
print(f"Centroids: {centroids}")

# Predict cluster for new points
X_new = np.array([[1.5, 2.5], [8.5, 9.5]])
predictions = kmeans.predict(X_new)
print(f"Predictions: {predictions}")
```

### With K-Means++ Initialization

```python
from src.main import KMeans
import numpy as np

X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])

# Use k-means++ initialization
kmeans = KMeans(n_clusters=2, init="k-means++", random_state=42)
kmeans.fit(X)

print(f"Inertia: {kmeans.inertia:.4f}")
print(f"Iterations: {kmeans.n_iterations}")
```

### Elbow Method for Optimal k

```python
from src.main import ElbowMethod
import numpy as np

X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [15, 16], [16, 17]])

# Find optimal k using elbow method
elbow = ElbowMethod(k_range=[2, 3, 4, 5, 6], random_state=42)
results = elbow.fit(X)

print(f"Optimal k: {results['optimal_k']}")
print(f"\nResults:")
for k, k_result in sorted(results["results"].items()):
    print(f"k={k}: inertia={k_result['inertia']:.4f}")

# Plot elbow curve
elbow.plot_elbow()
```

### With Pandas DataFrame

```python
from src.main import KMeans
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]].values

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

df["cluster"] = kmeans.labels_
```

### Command-Line Usage

Basic clustering:

```bash
python src/main.py --input data.csv --n-clusters 3
```

With k-means++ initialization:

```bash
python src/main.py --input data.csv --n-clusters 3 --init k-means++
```

Use elbow method to find optimal k:

```bash
python src/main.py --input data.csv --elbow
```

Elbow method with custom k range:

```bash
python src/main.py --input data.csv --elbow --k-range "2:10"
```

Plot clusters and elbow curve:

```bash
python src/main.py --input data.csv --elbow --plot
```

Save cluster assignments:

```bash
python src/main.py --input data.csv --n-clusters 3 --output clusters.csv
```

### Complete Example

```python
from src.main import KMeans, ElbowMethod
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with 3 clusters
np.random.seed(42)
cluster1 = np.random.randn(50, 2) + [2, 2]
cluster2 = np.random.randn(50, 2) + [8, 8]
cluster3 = np.random.randn(50, 2) + [5, 2]
X = np.vstack([cluster1, cluster2, cluster3])

# Find optimal k using elbow method
print("Finding optimal k...")
elbow = ElbowMethod(k_range=list(range(2, 11)), random_state=42)
results = elbow.fit(X)

print(f"Optimal k: {results['optimal_k']}")
elbow.plot_elbow()

# Cluster with optimal k
optimal_k = results['optimal_k'] if results['optimal_k'] else 3
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42)
kmeans.fit(X)

print(f"\nClustering Results:")
print(f"Inertia: {kmeans.inertia:.4f}")
print(f"Iterations: {kmeans.n_iterations}")

# Visualize clusters
plt.figure(figsize=(10, 8))
for i in range(optimal_k):
    cluster_points = X[kmeans.labels == i]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"Cluster {i}",
        alpha=0.6,
    )
plt.scatter(
    kmeans.centroids[:, 0],
    kmeans.centroids[:, 1],
    c="black",
    marker="x",
    s=200,
    linewidths=3,
    label="Centroids",
)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.title(f"K-Means Clustering (k={optimal_k})", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Project Structure

```
kmeans-clustering/
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

- `src/main.py`: Core implementation with `KMeans` and `ElbowMethod` classes
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
- K-means initialization
- Fitting and clustering
- Prediction
- Convergence
- Inertia calculation
- K-means++ initialization
- Elbow method
- Error handling
- Different input types

## Understanding K-Means Clustering

### K-Means Algorithm

K-means is an iterative clustering algorithm that partitions data into k clusters:

1. **Initialize**: Choose k initial centroids (randomly or using k-means++)
2. **Assign**: Assign each point to the nearest centroid
3. **Update**: Recalculate centroids as the mean of assigned points
4. **Repeat**: Steps 2-3 until convergence

**Convergence**: Algorithm stops when centroids no longer move significantly (within tolerance).

### K-Means++ Initialization

K-means++ is a smarter initialization method:

1. Choose first centroid randomly
2. For each subsequent centroid:
   - Calculate distance from each point to nearest existing centroid
   - Choose point with probability proportional to distance squared

**Benefits**: Better initial centroids, faster convergence, often better results.

### Elbow Method

The elbow method helps find the optimal number of clusters:

1. Run k-means for different k values
2. Calculate inertia (within-cluster sum of squares) for each k
3. Plot k vs. inertia
4. Find the "elbow" point where inertia decreases sharply then levels off

**Elbow Point**: The k value where adding more clusters doesn't significantly reduce inertia.

### Inertia (Within-Cluster Sum of Squares)

Inertia measures how tightly clusters are formed:

```
Inertia = Σ Σ ||x - μᵢ||²
```

Where:
- `x` is a point in cluster i
- `μᵢ` is the centroid of cluster i

**Lower inertia** = tighter clusters (better clustering).

## Troubleshooting

### Common Issues

**Issue**: Clusters not well separated

**Solution**: 
- Try k-means++ initialization
- Increase number of clusters
- Check if data has natural clusters
- Consider feature scaling

**Issue**: Different results each run

**Solution**: 
- Set random_state for reproducibility
- Use k-means++ initialization (more stable)
- Run multiple times and average

**Issue**: Elbow method not finding clear elbow

**Solution**: 
- Try wider k range
- Check if data has natural clusters
- Consider other metrics (silhouette score)
- May need domain knowledge

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: n_clusters cannot be greater than number of samples`: Reduce n_clusters
- `ValueError: n_clusters must be at least 1`: Use n_clusters >= 1

## Best Practices

1. **Use k-means++ initialization**: Better than random initialization
2. **Use elbow method**: Find optimal k value
3. **Scale features**: K-means is sensitive to feature scale
4. **Run multiple times**: K-means can converge to local minima
5. **Check cluster sizes**: Avoid very small or very large clusters
6. **Visualize results**: Especially for 2D/3D data

## Real-World Applications

- Customer segmentation
- Image compression
- Document clustering
- Anomaly detection
- Market research
- Pattern recognition
- Educational purposes for learning clustering algorithms

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
