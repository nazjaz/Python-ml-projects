# Hyperparameter Tuning API Documentation

## GridSearchCV Class

### `GridSearchCV(estimator, param_grid, cv=5, scoring=None, n_jobs=1, verbose=0, random_state=None)`

Grid Search Cross-Validation for hyperparameter tuning.

#### Parameters

- `estimator`: Base estimator to tune.
- `param_grid` (dict): Dictionary of parameter names and lists of values.
- `cv` (int): Number of cross-validation folds (default: 5).
- `scoring` (callable, optional): Scoring function (default: estimator.score).
- `n_jobs` (int): Number of parallel jobs (default: 1, not implemented).
- `verbose` (int): Verbosity level (default: 0).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `best_params_` (dict): Best parameter combination.
- `best_score_` (float): Best cross-validation score.
- `best_estimator_`: Best estimator fitted on all data.
- `cv_results_` (dict): Cross-validation results.

### Methods

#### `fit(X, y)`

Fit grid search.

**Parameters:**
- `X` (ndarray): Feature matrix.
- `y` (ndarray): Target values.

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
gs.fit(X, y)
```

## RandomSearchCV Class

### `RandomSearchCV(estimator, param_distributions, n_iter=10, cv=5, scoring=None, n_jobs=1, verbose=0, random_state=None)`

Random Search Cross-Validation for hyperparameter tuning.

#### Parameters

- `estimator`: Base estimator to tune.
- `param_distributions` (dict): Dictionary of parameter names and lists of values.
- `n_iter` (int): Number of parameter settings sampled (default: 10).
- `cv` (int): Number of cross-validation folds (default: 5).
- `scoring` (callable, optional): Scoring function (default: estimator.score).
- `n_jobs` (int): Number of parallel jobs (default: 1, not implemented).
- `verbose` (int): Verbosity level (default: 0).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `best_params_` (dict): Best parameter combination.
- `best_score_` (float): Best cross-validation score.
- `best_estimator_`: Best estimator fitted on all data.
- `cv_results_` (dict): Cross-validation results.

### Methods

#### `fit(X, y)`

Fit random search.

**Parameters:**
- `X` (ndarray): Feature matrix.
- `y` (ndarray): Target values.

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
rs = RandomSearchCV(estimator=estimator, param_distributions=param_dist, n_iter=20)
rs.fit(X, y)
```

## BayesianOptimization Class

### `BayesianOptimization(estimator, param_space, n_iter=10, cv=5, scoring=None, n_initial=5, verbose=0, random_state=None)`

Bayesian Optimization for hyperparameter tuning.

#### Parameters

- `estimator`: Base estimator to tune.
- `param_space` (dict): Dictionary of parameter names and (min, max) tuples.
- `n_iter` (int): Number of optimization iterations (default: 10).
- `cv` (int): Number of cross-validation folds (default: 5).
- `scoring` (callable, optional): Scoring function (default: estimator.score).
- `n_initial` (int): Number of initial random samples (default: 5).
- `verbose` (int): Verbosity level (default: 0).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `best_params_` (dict): Best parameter combination.
- `best_score_` (float): Best cross-validation score.
- `best_estimator_`: Best estimator fitted on all data.
- `cv_results_` (dict): Cross-validation results.

### Methods

#### `fit(X, y)`

Fit Bayesian optimization.

**Parameters:**
- `X` (ndarray): Feature matrix.
- `y` (ndarray): Target values.

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
bo = BayesianOptimization(estimator=estimator, param_space=param_space, n_iter=20)
bo.fit(X, y)
```

## Usage Examples

### Grid Search

```python
from src.main import GridSearchCV
from src.example_estimator import SimpleClassifier

estimator = SimpleClassifier()
param_grid = {
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5],
}

gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
gs.fit(X, y)

print(f"Best parameters: {gs.best_params_}")
print(f"Best score: {gs.best_score_}")
```

### Random Search

```python
from src.main import RandomSearchCV

param_distributions = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
}

rs = RandomSearchCV(
    estimator=estimator,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
)
rs.fit(X, y)
```

### Bayesian Optimization

```python
from src.main import BayesianOptimization

param_space = {
    "alpha": (0.1, 10.0),
    "learning_rate": (0.001, 0.1),
}

bo = BayesianOptimization(
    estimator=estimator,
    param_space=param_space,
    n_iter=20,
    n_initial=5,
    cv=5,
)
bo.fit(X, y)
```

### Accessing Results

```python
gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
gs.fit(X, y)

# Best parameters and score
print(gs.best_params_)
print(gs.best_score_)

# All CV results
cv_results = gs.cv_results_
for i in range(len(cv_results["mean_test_score"])):
    print(f"Params: {[cv_results[f'param_{k}'][i] for k in param_grid.keys()]}")
    print(f"Score: {cv_results['mean_test_score'][i]:.6f}")

# Best estimator
best_estimator = gs.best_estimator_
predictions = best_estimator.predict(X_test)
```
