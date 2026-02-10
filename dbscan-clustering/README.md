# DBSCAN Clustering

A Python implementation of DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm from scratch for density-based clustering with noise detection. This is the twenty-ninth project in the ML learning series, focusing on understanding density-based clustering algorithms.

## Project Title and Description

The DBSCAN Clustering tool provides a complete implementation of DBSCAN from scratch, including density-based clustering, noise detection, and core/border point classification. It helps users understand how density-based clustering works and how it can find clusters of arbitrary shape while identifying outliers.

This tool solves the problem of learning density-based clustering fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates density-based clustering, noise detection, and how DBSCAN handles clusters of arbitrary shapes.

**Target Audience**: Beginners learning machine learning, students studying unsupervised learning algorithms, and anyone who needs to understand DBSCAN and density-based clustering from scratch.

## Features

- DBSCAN clustering implementation from scratch
- Density-based clustering (eps and min_samples parameters)
- Noise detection (outlier identification)
- Core point classification
- Border point classification
- Cluster visualization with noise points highlighted
- Support for clusters of arbitrary shape
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
cd /path/to/Python-ml-projects/dbscan-clustering
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
python src/main.py --input sample.csv --eps 0.5 --min-samples 5
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  eps: 0.5
  min_samples: 5
  distance_metric: "euclidean"
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.eps`: Maximum distance between two samples for one to be considered in the neighborhood of the other (default: 0.5)
- `model.min_samples`: Minimum number of samples in a neighborhood for a point to be considered a core point (default: 5)
- `model.distance_metric`: Distance metric (currently only "euclidean" supported, default: "euclidean")

## Usage

### Basic Usage

```python
from src.main import DBSCAN
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])

# Initialize and fit model
dbscan = DBSCAN(eps=2.0, min_samples=2)
dbscan.fit(X)

# Get cluster labels (-1 for noise)
labels = dbscan.labels
print(f"Cluster labels: {labels}")

# Get core samples
core_samples = dbscan.get_core_samples()
print(f"Core samples: {core_samples}")

# Get noise samples
noise_samples = dbscan.get_noise_samples()
print(f"Noise samples: {noise_samples}")
```

### With Cluster Information

```python
from src.main import DBSCAN
import numpy as np

X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])

dbscan = DBSCAN(eps=2.0, min_samples=2)
dbscan.fit(X)

# Get cluster information
info = dbscan.get_cluster_info()
print(f"Number of clusters: {info['n_clusters']}")
print(f"Number of noise points: {info['n_noise']}")
print(f"Number of core points: {info['n_core_samples']}")
print(f"Cluster sizes: {info['cluster_sizes']}")
```

### Fit and Predict

```python
from src.main import DBSCAN
import numpy as np

X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])

dbscan = DBSCAN(eps=2.0, min_samples=2)
labels = dbscan.fit_predict(X)

print(f"Labels: {labels}")
```

### With Pandas DataFrame

```python
from src.main import DBSCAN
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]].values

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

df["cluster"] = dbscan.labels
df["is_noise"] = dbscan.labels == -1
```

### Command-Line Usage

Basic clustering:

```bash
python src/main.py --input data.csv --eps 0.5 --min-samples 5
```

Plot clusters:

```bash
python src/main.py --input data.csv --eps 0.5 --min-samples 5 --plot
```

Save cluster assignments:

```bash
python src/main.py --input data.csv --eps 0.5 --min-samples 5 --output clusters.csv
```

Save cluster plot:

```bash
python src/main.py --input data.csv --eps 0.5 --min-samples 5 --save-plot clusters.png
```

### Complete Example

```python
from src.main import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with clusters and noise
np.random.seed(42)

# Cluster 1
cluster1 = np.random.randn(50, 2) + [2, 2]

# Cluster 2
cluster2 = np.random.randn(50, 2) + [8, 8]

# Noise points
noise = np.random.randn(10, 2) * 3 + [5, 5]

X = np.vstack([cluster1, cluster2, noise])

# Apply DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan.fit(X)

# Get information
info = dbscan.get_cluster_info()
print(f"Number of clusters: {info['n_clusters']}")
print(f"Number of noise points: {info['n_noise']}")
print(f"Number of core points: {info['n_core_samples']}")

# Visualize
dbscan.plot_clusters()

# Compare different eps values
for eps in [0.5, 1.0, 1.5, 2.0]:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan.fit(X)
    info = dbscan.get_cluster_info()
    print(f"eps={eps}: {info['n_clusters']} clusters, {info['n_noise']} noise")
```

## Project Structure

```
dbscan-clustering/
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

- `src/main.py`: Core implementation with `DBSCAN` class
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
- Fitting and clustering
- Noise detection
- Core sample identification
- Cluster information
- Error handling
- Different input types

## Understanding DBSCAN

### DBSCAN Algorithm

DBSCAN is a density-based clustering algorithm that groups points based on density:

1. **Core Point**: Point with at least `min_samples` neighbors within `eps` distance
2. **Border Point**: Point that is neighbor of a core point but doesn't have enough neighbors
3. **Noise Point**: Point that is neither core nor border (outlier)
4. **Cluster**: Group of core points and their border points

**Algorithm Steps:**
1. For each unvisited point:
   - If it has enough neighbors (core point):
     - Create new cluster
     - Expand cluster by adding density-reachable points
   - Else:
     - Mark as noise (if not reachable from any core point)

### Key Parameters

**eps (ε):**
- Maximum distance for two points to be considered neighbors
- Controls the "neighborhood" size
- Too small: Many noise points, many small clusters
- Too large: Few clusters, may merge distinct clusters

**min_samples:**
- Minimum number of points in a neighborhood for a core point
- Controls density requirement
- Too small: Many core points, may create too many clusters
- Too large: Few core points, may miss clusters

### Point Types

**Core Point:**
- Has at least `min_samples` neighbors within `eps` distance
- Forms the backbone of clusters
- Can expand clusters

**Border Point:**
- Neighbor of a core point
- Doesn't have enough neighbors to be core
- Belongs to a cluster but doesn't expand it

**Noise Point:**
- Not a core point
- Not a neighbor of any core point
- Labeled as -1 (outlier)

### Advantages

- Can find clusters of arbitrary shape
- Automatically determines number of clusters
- Robust to outliers (noise detection)
- No need to specify number of clusters
- Handles non-spherical clusters well

### Limitations

- Sensitive to eps and min_samples parameters
- Struggles with varying density clusters
- Can be slow for large datasets (O(n²) in worst case)
- Requires careful parameter tuning

## Troubleshooting

### Common Issues

**Issue**: All points marked as noise

**Solution**: 
- Increase eps value
- Decrease min_samples value
- Check data scale (may need feature scaling)

**Issue**: Everything in one cluster

**Solution**: 
- Decrease eps value
- Increase min_samples value
- Check data distribution

**Issue**: Too many small clusters

**Solution**: 
- Increase eps value
- Decrease min_samples value
- Check for data quality issues

### Error Messages

- `ValueError: eps must be greater than 0`: Set eps > 0
- `ValueError: min_samples must be at least 1`: Set min_samples >= 1
- `ValueError: Model must be fitted before...`: Call `fit()` first

## Best Practices

1. **Scale features**: DBSCAN is sensitive to feature scale
2. **Tune eps carefully**: Use k-distance graph to find good eps
3. **Start with min_samples = 2*dimensions**: Good default for min_samples
4. **Visualize results**: Especially for 2D data
5. **Check noise points**: Verify they are actually outliers
6. **Try different parameters**: DBSCAN is sensitive to parameter choice

## Real-World Applications

- Anomaly detection
- Image segmentation
- Customer segmentation
- Network analysis
- Geographic data clustering
- Outlier detection
- Educational purposes for learning density-based clustering

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
