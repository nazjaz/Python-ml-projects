# Hierarchical Clustering

A Python implementation of hierarchical clustering algorithm from scratch with single, complete, and average linkage methods and dendrogram visualization. This is the twenty-eighth project in the ML learning series, focusing on understanding agglomerative hierarchical clustering algorithms.

## Project Title and Description

The Hierarchical Clustering tool provides a complete implementation of agglomerative hierarchical clustering from scratch, including three linkage methods (single, complete, average) and dendrogram visualization. It helps users understand how hierarchical clustering builds a tree-like structure of clusters and how different linkage methods affect the clustering results.

This tool solves the problem of learning hierarchical clustering fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates bottom-up clustering, distance calculations, and dendrogram construction.

**Target Audience**: Beginners learning machine learning, students studying unsupervised learning algorithms, and anyone who needs to understand hierarchical clustering and linkage methods from scratch.

## Features

- Hierarchical clustering implementation from scratch
- Three linkage methods:
  - **Single linkage**: Minimum distance between clusters
  - **Complete linkage**: Maximum distance between clusters
  - **Average linkage**: Average distance between clusters
- Dendrogram visualization
- Cluster label assignment (when n_clusters is specified)
- Support for multiple features
- Euclidean distance metric
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
cd /path/to/Python-ml-projects/hierarchical-clustering
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
python src/main.py --input sample.csv --linkage average
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  n_clusters: null
  linkage: "average"
  distance_metric: "euclidean"
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.n_clusters`: Number of clusters (null for full dendrogram without cutting, default: null)
- `model.linkage`: Linkage method. Options: "single", "complete", "average" (default: "average")
- `model.distance_metric`: Distance metric (currently only "euclidean" supported, default: "euclidean")

## Usage

### Basic Usage (Full Dendrogram)

```python
from src.main import HierarchicalClustering
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])

# Initialize and fit model (no n_clusters = full dendrogram)
model = HierarchicalClustering(linkage="average")
model.fit(X)

# Plot dendrogram
model.plot_dendrogram()
```

### With Cluster Assignment

```python
from src.main import HierarchicalClustering
import numpy as np

X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])

# Fit with n_clusters to get cluster labels
model = HierarchicalClustering(n_clusters=2, linkage="average")
model.fit(X)

print(f"Cluster labels: {model.labels}")
print(f"Number of merges: {len(model.linkage_matrix)}")
```

### Different Linkage Methods

```python
from src.main import HierarchicalClustering
import numpy as np

X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])

# Single linkage (minimum distance)
single_model = HierarchicalClustering(n_clusters=2, linkage="single")
single_model.fit(X)

# Complete linkage (maximum distance)
complete_model = HierarchicalClustering(n_clusters=2, linkage="complete")
complete_model.fit(X)

# Average linkage (average distance)
average_model = HierarchicalClustering(n_clusters=2, linkage="average")
average_model.fit(X)
```

### Fit and Predict

```python
from src.main import HierarchicalClustering
import numpy as np

X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])

model = HierarchicalClustering(n_clusters=2, linkage="average")
labels = model.fit_predict(X)

print(f"Labels: {labels}")
```

### With Pandas DataFrame

```python
from src.main import HierarchicalClustering
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]].values

model = HierarchicalClustering(n_clusters=3, linkage="average")
model.fit(X)

df["cluster"] = model.labels
```

### Command-Line Usage

Full dendrogram (no cluster cutting):

```bash
python src/main.py --input data.csv --linkage average --plot
```

With cluster assignment:

```bash
python src/main.py --input data.csv --n-clusters 3 --linkage average
```

Single linkage:

```bash
python src/main.py --input data.csv --n-clusters 3 --linkage single
```

Complete linkage:

```bash
python src/main.py --input data.csv --n-clusters 3 --linkage complete
```

Save dendrogram:

```bash
python src/main.py --input data.csv --linkage average --save-plot dendrogram.png
```

Save cluster assignments:

```bash
python src/main.py --input data.csv --n-clusters 3 --output clusters.csv
```

### Complete Example

```python
from src.main import HierarchicalClustering
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with 3 clusters
np.random.seed(42)
cluster1 = np.random.randn(20, 2) + [2, 2]
cluster2 = np.random.randn(20, 2) + [8, 8]
cluster3 = np.random.randn(20, 2) + [5, 2]
X = np.vstack([cluster1, cluster2, cluster3])

# Compare different linkage methods
linkages = ["single", "complete", "average"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, linkage in enumerate(linkages):
    model = HierarchicalClustering(n_clusters=3, linkage=linkage)
    model.fit(X)
    
    # Plot clusters
    ax = axes[idx]
    for i in range(3):
        cluster_points = X[model.labels == i]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {i}",
            alpha=0.6,
        )
    ax.set_title(f"{linkage.capitalize()} Linkage", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot dendrogram for average linkage
model = HierarchicalClustering(linkage="average")
model.fit(X)
model.plot_dendrogram()
```

## Project Structure

```
hierarchical-clustering/
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

- `src/main.py`: Core implementation with `HierarchicalClustering` class
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
- Fitting with different linkage methods
- Cluster label assignment
- Dendrogram plotting
- Error handling
- Different input types

## Understanding Hierarchical Clustering

### Hierarchical Clustering Algorithm

Hierarchical clustering builds a tree-like structure (dendrogram) by iteratively merging clusters:

1. **Start**: Each point is its own cluster
2. **Find**: Find two closest clusters
3. **Merge**: Merge them into a new cluster
4. **Repeat**: Steps 2-3 until one cluster remains

**Result**: A dendrogram showing the complete hierarchy of cluster merges.

### Linkage Methods

**Single Linkage (Minimum):**
```
d(C₁, C₂) = min{d(x, y) : x ∈ C₁, y ∈ C₂}
```
- Uses minimum distance between clusters
- Tends to create elongated clusters (chaining effect)
- Sensitive to outliers

**Complete Linkage (Maximum):**
```
d(C₁, C₂) = max{d(x, y) : x ∈ C₁, y ∈ C₂}
```
- Uses maximum distance between clusters
- Tends to create compact, spherical clusters
- Less sensitive to outliers

**Average Linkage:**
```
d(C₁, C₂) = (1/|C₁||C₂|) Σ d(x, y) for x ∈ C₁, y ∈ C₂
```
- Uses average distance between clusters
- Balanced approach
- Good general-purpose choice

### Dendrogram

A dendrogram is a tree diagram showing the hierarchical relationship between clusters:

- **X-axis**: Sample indices or cluster sizes
- **Y-axis**: Distance at which clusters merge
- **Height**: Distance between merged clusters
- **Cutting**: Horizontal line at desired distance gives cluster assignment

## Troubleshooting

### Common Issues

**Issue**: Dendrogram too complex for large datasets

**Solution**: 
- Use smaller sample size
- Consider truncating dendrogram
- Use n_clusters to get specific number of clusters

**Issue**: Different linkage methods give different results

**Solution**: 
- This is expected behavior
- Single linkage: elongated clusters
- Complete linkage: compact clusters
- Average linkage: balanced approach
- Choose based on data characteristics

**Issue**: Slow for large datasets

**Solution**: 
- Hierarchical clustering is O(n³) complexity
- Consider sampling for very large datasets
- Use approximate methods for large data

### Error Messages

- `ValueError: Need at least 2 samples for clustering`: Provide at least 2 samples
- `ValueError: n_clusters must be set for fit_predict`: Set n_clusters before calling fit_predict
- `ValueError: Unknown linkage method`: Use "single", "complete", or "average"

## Best Practices

1. **Choose appropriate linkage**: Average is often a good default
2. **Use single linkage for elongated clusters**: When clusters are chain-like
3. **Use complete linkage for compact clusters**: When clusters are spherical
4. **Visualize dendrogram**: Helps understand cluster structure
5. **Consider computational cost**: O(n³) complexity for large datasets
6. **Use n_clusters for specific number**: Cut dendrogram at desired level

## Real-World Applications

- Gene expression analysis
- Document clustering
- Image segmentation
- Taxonomy construction
- Social network analysis
- Market segmentation
- Educational purposes for learning hierarchical clustering

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
