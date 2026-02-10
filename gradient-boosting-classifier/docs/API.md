# Gradient Boosting Classifier API Documentation

## GradientBoostingClassifier Class

### `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1.0, random_state=None)`

Gradient Boosting Classifier with learning rate and optimization.

#### Parameters

- `n_estimators` (int): Number of boosting stages (trees) (default: 100).
- `learning_rate` (float): Learning rate (shrinkage) (default: 0.1).
- `max_depth` (int): Maximum depth of trees (default: 3).
- `min_samples_split` (int): Minimum samples required to split node (default: 2).
- `min_samples_leaf` (int): Minimum samples required at leaf node (default: 1).
- `subsample` (float): Fraction of samples to use for each tree, 0-1 (default: 1.0).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `estimators_` (list): List of fitted decision tree regressors.
- `init_score_` (float): Initial prediction (log odds).
- `n_features_` (int): Number of features.
- `classes_` (ndarray): Unique class labels.
- `feature_importances_` (ndarray): Feature importance values.
- `feature_names_` (list, optional): Names of features.

### Methods

#### `fit(X, y)`

Fit the gradient boosting classifier.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).
- `y` (array-like): Target labels of shape (n_samples,) (binary: 0, 1).

**Returns:**
- `self`: Returns self for method chaining.

**Raises:**
- `ValueError`: If inputs are invalid.

**Example:**
```python
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gb.fit(X, y)
```

#### `predict(X)`

Predict class labels.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Predicted class labels of shape (n_samples,).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
predictions = gb.predict(X)
```

#### `predict_proba(X)`

Predict class probabilities.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Class probabilities of shape (n_samples, 2).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
probabilities = gb.predict_proba(X)
```

#### `score(X, y)`

Calculate classification accuracy.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True labels.

**Returns:**
- `float`: Classification accuracy (0-1).

**Example:**
```python
accuracy = gb.score(X, y)
```

#### `get_feature_importances()`

Get feature importance as dictionary.

**Returns:**
- `dict`: Dictionary mapping feature names to importance values.

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
importances = gb.get_feature_importances()
```

#### `plot_feature_importance(save_path=None, show=True, top_n=None)`

Plot feature importance.

**Parameters:**
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot (default: True).
- `top_n` (int, optional): Number of top features to show (default: all).

**Example:**
```python
gb.plot_feature_importance(save_path="importance.png")
```

## Usage Examples

### Basic Gradient Boosting

```python
from src.main import GradientBoostingClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

gb = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1)
gb.fit(X, y)
predictions = gb.predict(X)
```

### With Learning Rate Optimization

```python
# Low learning rate
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.01)
gb.fit(X, y)

# High learning rate
gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5)
gb.fit(X, y)
```

### With Depth Optimization

```python
# Shallow trees
gb = GradientBoostingClassifier(n_estimators=100, max_depth=2)
gb.fit(X, y)

# Deeper trees
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5)
gb.fit(X, y)
```

### With Tree Count Optimization

```python
# Few trees
gb = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1)
gb.fit(X, y)

# Many trees
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1)
gb.fit(X, y)
```

### With Subsampling

```python
gb = GradientBoostingClassifier(n_estimators=100, subsample=0.8)
gb.fit(X, y)
```

### Feature Importance

```python
gb = GradientBoostingClassifier(n_estimators=100)
gb.feature_names_ = ["feature1", "feature2"]
gb.fit(X, y)

importances = gb.get_feature_importances()
gb.plot_feature_importance()
```
