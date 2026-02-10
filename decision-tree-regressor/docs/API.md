# Decision Tree Regressor API Documentation

## DecisionTreeRegressor Class

### `DecisionTreeRegressor(criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0, ccp_alpha=0.0, random_state=None)`

Decision Tree Regressor with pruning and feature importance.

#### Parameters

- `criterion` (str): Splitting criterion. Options: "mse", "mae" (default: "mse").
- `max_depth` (int, optional): Maximum depth of tree. If None, nodes expanded until pure (default: None).
- `min_samples_split` (int): Minimum samples required to split node (default: 2).
- `min_samples_leaf` (int): Minimum samples required at leaf node (default: 1).
- `min_impurity_decrease` (float): Minimum impurity decrease for split (default: 0.0).
- `ccp_alpha` (float): Complexity parameter for cost-complexity pruning (default: 0.0).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `root` (TreeNode): Root node of decision tree.
- `n_features_` (int): Number of features.
- `feature_names_` (list, optional): Names of features.
- `feature_importances_` (ndarray): Feature importance values.

### Methods

#### `fit(X, y)`

Fit the decision tree regressor.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).
- `y` (array-like): Target values of shape (n_samples,).

**Returns:**
- `self`: Returns self for method chaining.

**Raises:**
- `ValueError`: If inputs are invalid.

**Example:**
```python
tree = DecisionTreeRegressor(criterion="mse")
tree.fit(X, y)
```

#### `predict(X)`

Predict target values.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Predicted values of shape (n_samples,).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
predictions = tree.predict(X)
```

#### `score(X, y)`

Calculate R-squared score.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True target values.

**Returns:**
- `float`: R-squared score.

**Example:**
```python
r2 = tree.score(X, y)
```

#### `mse(X, y)`

Calculate mean squared error.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True target values.

**Returns:**
- `float`: Mean squared error.

**Example:**
```python
mse = tree.mse(X, y)
```

#### `mae(X, y)`

Calculate mean absolute error.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True target values.

**Returns:**
- `float`: Mean absolute error.

**Example:**
```python
mae = tree.mae(X, y)
```

#### `get_feature_importances()`

Get feature importance as dictionary.

**Returns:**
- `dict`: Dictionary mapping feature names to importance values.

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
importances = tree.get_feature_importances()
```

#### `get_depth()`

Get maximum depth of tree.

**Returns:**
- `int`: Maximum depth.

**Example:**
```python
depth = tree.get_depth()
```

#### `get_n_nodes()`

Get number of nodes in tree.

**Returns:**
- `int`: Number of nodes.

**Example:**
```python
n_nodes = tree.get_n_nodes()
```

#### `print_tree()`

Print tree structure as text.

**Example:**
```python
tree.print_tree()
```

#### `plot_tree(save_path=None, show=True, figsize=(12, 8))`

Plot decision tree visualization.

**Parameters:**
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot (default: True).
- `figsize` (tuple): Figure size (width, height) (default: (12, 8)).

**Example:**
```python
tree.plot_tree(save_path="tree.png")
```

#### `plot_feature_importance(save_path=None, show=True, top_n=None)`

Plot feature importance.

**Parameters:**
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot (default: True).
- `top_n` (int, optional): Number of top features to show (default: all).

**Example:**
```python
tree.plot_feature_importance(save_path="importance.png")
```

## Usage Examples

### Basic Decision Tree Regressor

```python
from src.main import DecisionTreeRegressor
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

tree = DecisionTreeRegressor(criterion="mse")
tree.fit(X, y)
predictions = tree.predict(X)
```

### With Pruning

```python
# Pre-pruning
tree = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
tree.fit(X, y)

# Post-pruning
tree = DecisionTreeRegressor(ccp_alpha=0.1)
tree.fit(X, y)
```

### Feature Importance

```python
tree = DecisionTreeRegressor()
tree.feature_names_ = ["feature1", "feature2"]
tree.fit(X, y)

importances = tree.get_feature_importances()
tree.plot_feature_importance()
```

### Evaluation Metrics

```python
tree = DecisionTreeRegressor()
tree.fit(X, y)

r2 = tree.score(X, y)
mse = tree.mse(X, y)
mae = tree.mae(X, y)
```
