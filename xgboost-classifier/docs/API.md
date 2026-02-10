# XGBoost Classifier API Documentation

## XGBoostClassifier Class

### `XGBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, min_child_weight=1.0, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0, subsample=1.0, early_stopping_rounds=None, random_state=None)`

XGBoost Classifier with tree construction, regularization, and early stopping.

#### Parameters

- `n_estimators` (int): Number of boosting rounds (trees) (default: 100).
- `learning_rate` (float): Learning rate (shrinkage) (default: 0.1).
- `max_depth` (int): Maximum depth of trees (default: 6).
- `min_child_weight` (float): Minimum sum of instance weight in child (default: 1.0).
- `gamma` (float): Minimum loss reduction for split (default: 0.0).
- `reg_lambda` (float): L2 regularization (default: 1.0).
- `reg_alpha` (float): L1 regularization (default: 0.0).
- `subsample` (float): Fraction of samples to use for each tree, 0-1 (default: 1.0).
- `early_stopping_rounds` (int, optional): Early stopping rounds (default: None).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `estimators_` (list): List of fitted XGBoost trees.
- `init_score_` (float): Initial prediction (log odds).
- `n_features_` (int): Number of features.
- `classes_` (ndarray): Unique class labels.
- `feature_importances_` (ndarray): Feature importance values.
- `feature_names_` (list, optional): Names of features.
- `best_iteration_` (int, optional): Best iteration from early stopping.

### Methods

#### `fit(X, y, eval_set=None, verbose=False)`

Fit the XGBoost classifier.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).
- `y` (array-like): Target labels of shape (n_samples,) (binary: 0, 1).
- `eval_set` (tuple, optional): Validation set (X_val, y_val) for early stopping.
- `verbose` (bool): Whether to print progress (default: False).

**Returns:**
- `self`: Returns self for method chaining.

**Raises:**
- `ValueError`: If inputs are invalid.

**Example:**
```python
xgb = XGBoostClassifier(n_estimators=100, early_stopping_rounds=10)
xgb.fit(X_train, y_train, eval_set=(X_val, y_val))
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
predictions = xgb.predict(X)
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
probabilities = xgb.predict_proba(X)
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
accuracy = xgb.score(X, y)
```

#### `get_feature_importances()`

Get feature importance as dictionary.

**Returns:**
- `dict`: Dictionary mapping feature names to importance values.

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
importances = xgb.get_feature_importances()
```

#### `plot_feature_importance(save_path=None, show=True, top_n=None)`

Plot feature importance.

**Parameters:**
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot (default: True).
- `top_n` (int, optional): Number of top features to show (default: all).

**Example:**
```python
xgb.plot_feature_importance(save_path="importance.png")
```

## XGBoostTree Class

### `XGBoostTree(max_depth=6, min_child_weight=1.0, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0, random_state=None)`

XGBoost Tree with regularization and gain-based splitting.

#### Parameters

- `max_depth` (int): Maximum depth of tree (default: 6).
- `min_child_weight` (float): Minimum sum of instance weight in child (default: 1.0).
- `gamma` (float): Minimum loss reduction for split (default: 0.0).
- `reg_lambda` (float): L2 regularization (default: 1.0).
- `reg_alpha` (float): L1 regularization (default: 0.0).
- `random_state` (int, optional): Random seed (default: None).

### Methods

#### `fit(X, gradients, hessians)`

Fit XGBoost tree to gradients and hessians.

**Parameters:**
- `X` (array-like): Feature matrix.
- `gradients` (array-like): Gradient values.
- `hessians` (array-like): Hessian values.

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
tree = XGBoostTree(max_depth=5)
tree.fit(X, gradients, hessians)
```

#### `predict(X)`

Predict target values.

**Parameters:**
- `X` (array-like): Feature matrix.

**Returns:**
- `ndarray`: Predicted values.

**Example:**
```python
predictions = tree.predict(X)
```

## Usage Examples

### Basic XGBoost

```python
from src.main import XGBoostClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

xgb = XGBoostClassifier(n_estimators=50, learning_rate=0.1)
xgb.fit(X, y)
predictions = xgb.predict(X)
```

### With Regularization

```python
# L2 regularization
xgb = XGBoostClassifier(n_estimators=50, reg_lambda=2.0)
xgb.fit(X, y)

# L1 regularization
xgb = XGBoostClassifier(n_estimators=50, reg_alpha=0.5)
xgb.fit(X, y)

# Both
xgb = XGBoostClassifier(n_estimators=50, reg_lambda=1.0, reg_alpha=0.1)
xgb.fit(X, y)
```

### With Early Stopping

```python
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_val = np.array([[1.5, 2.5], [2.5, 3.5]])
y_val = np.array([0, 1])

xgb = XGBoostClassifier(n_estimators=100, early_stopping_rounds=5)
xgb.fit(X_train, y_train, eval_set=(X_val, y_val))
```

### Feature Importance

```python
xgb = XGBoostClassifier(n_estimators=50)
xgb.feature_names_ = ["feature1", "feature2"]
xgb.fit(X, y)

importances = xgb.get_feature_importances()
xgb.plot_feature_importance()
```

### Class Probabilities

```python
xgb = XGBoostClassifier(n_estimators=50)
xgb.fit(X, y)

probabilities = xgb.predict_proba(X)
```
