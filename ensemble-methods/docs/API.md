# Ensemble Methods API Documentation

## VotingClassifier Class

### `VotingClassifier(estimators, voting='hard')`

Voting Classifier with hard and soft voting.

#### Parameters

- `estimators` (list): List of (name, estimator) tuples.
- `voting` (str): "hard" or "soft" voting (default: "hard").

#### Attributes

- `estimators` (list): List of base estimators.
- `estimator_names` (list): List of estimator names.
- `voting` (str): Voting type.
- `classes_` (ndarray): Unique class labels.
- `feature_names_` (list, optional): Names of features.

### Methods

#### `fit(X, y)`

Fit all estimators.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): Target labels.

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
voting = VotingClassifier(estimators=[("dt", dt), ("knn", knn)], voting="hard")
voting.fit(X, y)
```

#### `predict(X)`

Predict using voting.

**Parameters:**
- `X` (array-like): Feature matrix.

**Returns:**
- `ndarray`: Predicted class labels.

**Example:**
```python
predictions = voting.predict(X)
```

#### `predict_proba(X)`

Predict class probabilities.

**Parameters:**
- `X` (array-like): Feature matrix.

**Returns:**
- `ndarray`: Class probabilities.

**Example:**
```python
probabilities = voting.predict_proba(X)
```

#### `score(X, y)`

Calculate classification accuracy.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True labels.

**Returns:**
- `float`: Classification accuracy.

**Example:**
```python
accuracy = voting.score(X, y)
```

## BaggingClassifier Class

### `BaggingClassifier(base_estimator, n_estimators=10, max_samples=1.0, max_features=1.0, random_state=None)`

Bagging Classifier with bootstrap sampling.

#### Parameters

- `base_estimator`: Base estimator to use.
- `n_estimators` (int): Number of estimators (default: 10).
- `max_samples` (float): Fraction of samples for each estimator (default: 1.0).
- `max_features` (float): Fraction of features for each estimator (default: 1.0).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `estimators_` (list): List of fitted estimators.
- `classes_` (ndarray): Unique class labels.
- `feature_names_` (list, optional): Names of features.

### Methods

#### `fit(X, y)`

Fit bagging classifier.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): Target labels.

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
bagging = BaggingClassifier(base_estimator=dt, n_estimators=10)
bagging.fit(X, y)
```

#### `predict(X)`

Predict using majority voting.

**Parameters:**
- `X` (array-like): Feature matrix.

**Returns:**
- `ndarray`: Predicted class labels.

**Example:**
```python
predictions = bagging.predict(X)
```

#### `predict_proba(X)`

Predict class probabilities.

**Parameters:**
- `X` (array-like): Feature matrix.

**Returns:**
- `ndarray`: Class probabilities.

**Example:**
```python
probabilities = bagging.predict_proba(X)
```

#### `score(X, y)`

Calculate classification accuracy.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True labels.

**Returns:**
- `float`: Classification accuracy.

**Example:**
```python
accuracy = bagging.score(X, y)
```

## StackingClassifier Class

### `StackingClassifier(base_estimators, meta_estimator, cv=5, random_state=None)`

Stacking Classifier with meta-learner.

#### Parameters

- `base_estimators` (list): List of (name, estimator) tuples.
- `meta_estimator`: Meta-learner for final prediction.
- `cv` (int): Number of cross-validation folds (default: 5).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `base_estimators` (list): List of base estimators.
- `meta_estimator`: Meta-learner.
- `classes_` (ndarray): Unique class labels.
- `feature_names_` (list, optional): Names of features.

### Methods

#### `fit(X, y)`

Fit stacking classifier.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): Target labels.

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
stacking = StackingClassifier(
    base_estimators=[("dt", dt), ("knn", knn)],
    meta_estimator=meta,
    cv=5
)
stacking.fit(X, y)
```

#### `predict(X)`

Predict using stacking.

**Parameters:**
- `X` (array-like): Feature matrix.

**Returns:**
- `ndarray`: Predicted class labels.

**Example:**
```python
predictions = stacking.predict(X)
```

#### `predict_proba(X)`

Predict class probabilities.

**Parameters:**
- `X` (array-like): Feature matrix.

**Returns:**
- `ndarray`: Class probabilities.

**Example:**
```python
probabilities = stacking.predict_proba(X)
```

#### `score(X, y)`

Calculate classification accuracy.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True labels.

**Returns:**
- `float`: Classification accuracy.

**Example:**
```python
accuracy = stacking.score(X, y)
```

## Base Models

### SimpleDecisionTree

Simple Decision Tree Classifier.

**Parameters:**
- `max_depth` (int): Maximum depth (default: 3).
- `min_samples_split` (int): Minimum samples to split (default: 2).

### SimpleKNN

Simple K-Nearest Neighbors Classifier.

**Parameters:**
- `n_neighbors` (int): Number of neighbors (default: 5).

## Usage Examples

### Voting Classifier

```python
from src.main import VotingClassifier, SimpleDecisionTree, SimpleKNN

dt = SimpleDecisionTree(max_depth=3)
knn = SimpleKNN(n_neighbors=5)

voting = VotingClassifier(
    estimators=[("dt", dt), ("knn", knn)],
    voting="hard"
)
voting.fit(X, y)
predictions = voting.predict(X)
```

### Bagging Classifier

```python
from src.main import BaggingClassifier, SimpleDecisionTree

base_estimator = SimpleDecisionTree(max_depth=3)
bagging = BaggingClassifier(
    base_estimator=base_estimator,
    n_estimators=10,
    max_samples=0.8
)
bagging.fit(X, y)
predictions = bagging.predict(X)
```

### Stacking Classifier

```python
from src.main import StackingClassifier, SimpleDecisionTree, SimpleKNN

dt = SimpleDecisionTree(max_depth=3)
knn = SimpleKNN(n_neighbors=5)
meta = SimpleDecisionTree(max_depth=2)

stacking = StackingClassifier(
    base_estimators=[("dt", dt), ("knn", knn)],
    meta_estimator=meta,
    cv=5
)
stacking.fit(X, y)
predictions = stacking.predict(X)
```
