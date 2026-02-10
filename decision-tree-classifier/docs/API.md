# Decision Tree Classifier API Documentation

## DecisionTreeClassifier Class

### `DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0, random_state=None)`

Decision Tree Classifier with information gain and Gini impurity.

#### Parameters

- `criterion` (str): Splitting criterion. Options: "gini", "entropy" (default: "gini").
- `max_depth` (int, optional): Maximum depth of tree. If None, nodes expanded until pure (default: None).
- `min_samples_split` (int): Minimum samples required to split node (default: 2).
- `min_samples_leaf` (int): Minimum samples required at leaf node (default: 1).
- `min_impurity_decrease` (float): Minimum impurity decrease for split (default: 0.0).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `root` (TreeNode): Root node of decision tree.
- `n_features_` (int): Number of features.
- `classes_` (ndarray): Unique class labels.
- `n_classes_` (int): Number of classes.
- `feature_names_` (list, optional): Names of features.

### Methods

#### `fit(X, y)`

Fit the decision tree classifier.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).
- `y` (array-like): Target labels of shape (n_samples,).

**Returns:**
- `self`: Returns self for method chaining.

**Raises:**
- `ValueError`: If inputs are invalid.

**Example:**
```python
tree = DecisionTreeClassifier(criterion="gini")
tree.fit(X, y)
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
predictions = tree.predict(X)
```

#### `predict_proba(X)`

Predict class probabilities.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Class probabilities of shape (n_samples, n_classes).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
probabilities = tree.predict_proba(X)
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
accuracy = tree.score(X, y)
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

## Usage Examples

### Basic Decision Tree

```python
from src.main import DecisionTreeClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

tree = DecisionTreeClassifier(criterion="gini")
tree.fit(X, y)
predictions = tree.predict(X)
```

### With Different Criteria

```python
# Gini impurity
tree_gini = DecisionTreeClassifier(criterion="gini")
tree_gini.fit(X, y)

# Information gain (entropy)
tree_entropy = DecisionTreeClassifier(criterion="entropy")
tree_entropy.fit(X, y)
```

### With Depth Control

```python
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
tree.fit(X, y)

print(f"Depth: {tree.get_depth()}")
print(f"Nodes: {tree.get_n_nodes()}")
```

### Tree Visualization

```python
tree = DecisionTreeClassifier()
tree.fit(X, y)

# Text visualization
tree.print_tree()

# Graphical visualization
tree.plot_tree()
```

### Class Probabilities

```python
tree = DecisionTreeClassifier()
tree.fit(X, y)

probabilities = tree.predict_proba(X)
```

### Multiclass Classification

```python
X = np.array([
    [1, 2], [2, 3], [3, 4],
    [4, 5], [5, 6], [6, 7],
    [7, 8], [8, 9], [9, 10]
])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

tree = DecisionTreeClassifier()
tree.fit(X, y)

predictions = tree.predict(X)
```
