"""Hyperparameter Tuning with Grid Search, Random Search, and Bayesian Optimization.

This module provides functionality for hyperparameter tuning using grid search,
random search, and Bayesian optimization methods.
"""

import itertools
import json
import logging
import logging.handlers
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BaseEstimator:
    """Base class for estimators."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEstimator":
        """Fit the estimator."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        raise NotImplementedError

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate score."""
        raise NotImplementedError


class GridSearchCV:
    """Grid Search Cross-Validation for hyperparameter tuning."""

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: Optional[Callable] = None,
        n_jobs: int = 1,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Grid Search.

        Args:
            estimator: Base estimator to tune.
            param_grid: Dictionary of parameter names and lists of values.
            cv: Number of cross-validation folds (default: 5).
            scoring: Scoring function (default: estimator.score).
            n_jobs: Number of parallel jobs (default: 1, not implemented).
            verbose: Verbosity level (default: 0).
            random_state: Random seed (default: None).
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.best_estimator_: Optional[BaseEstimator] = None
        self.cv_results_: Optional[Dict[str, List]] = None

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations.

        Returns:
            List of parameter dictionaries.
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = []

        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def _cross_val_score(
        self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate cross-validation score for parameter set.

        Args:
            X: Feature matrix.
            y: Target values.
            params: Parameter dictionary.

        Returns:
            Tuple of (mean_score, std_score).
        """
        n_samples = len(X)
        fold_size = n_samples // self.cv
        scores = []

        indices = np.arange(n_samples)
        if self.random_state is not None:
            np.random.shuffle(indices)

        for fold in range(self.cv):
            start = fold * fold_size
            end = start + fold_size if fold < self.cv - 1 else n_samples

            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            estimator = self._create_estimator(params)
            estimator.fit(X_train, y_train)

            if self.scoring:
                score = self.scoring(estimator, X_val, y_val)
            else:
                score = estimator.score(X_val, y_val)

            scores.append(score)

        return np.mean(scores), np.std(scores)

    def _create_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        """Create estimator with given parameters.

        Args:
            params: Parameter dictionary.

        Returns:
            Estimator instance.
        """
        estimator = type(self.estimator)()
        for key, value in params.items():
            if hasattr(estimator, key):
                setattr(estimator, key, value)
        return estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GridSearchCV":
        """Fit grid search.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self for method chaining.
        """
        param_combinations = self._generate_param_combinations()

        if self.verbose > 0:
            logger.info(f"Grid search: {len(param_combinations)} parameter combinations")

        best_score = float("-inf")
        best_params = None
        cv_results = defaultdict(list)

        for i, params in enumerate(param_combinations):
            if self.verbose > 0 and (i + 1) % 10 == 0:
                logger.info(f"  Testing combination {i+1}/{len(param_combinations)}")

            mean_score, std_score = self._cross_val_score(X, y, params)

            for key, value in params.items():
                cv_results[f"param_{key}"].append(value)
            cv_results["mean_test_score"].append(mean_score)
            cv_results["std_test_score"].append(std_score)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = dict(cv_results)

        self.best_estimator_ = self._create_estimator(best_params)
        self.best_estimator_.fit(X, y)

        if self.verbose > 0:
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best score: {best_score:.6f}")

        return self


class RandomSearchCV:
    """Random Search Cross-Validation for hyperparameter tuning."""

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: Dict[str, List[Any]],
        n_iter: int = 10,
        cv: int = 5,
        scoring: Optional[Callable] = None,
        n_jobs: int = 1,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Random Search.

        Args:
            estimator: Base estimator to tune.
            param_distributions: Dictionary of parameter names and lists of values.
            n_iter: Number of parameter settings sampled (default: 10).
            cv: Number of cross-validation folds (default: 5).
            scoring: Scoring function (default: estimator.score).
            n_jobs: Number of parallel jobs (default: 1, not implemented).
            verbose: Verbosity level (default: 0).
            random_state: Random seed (default: None).
        """
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.best_estimator_: Optional[BaseEstimator] = None
        self.cv_results_: Optional[Dict[str, List]] = None

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _sample_params(self) -> Dict[str, Any]:
        """Sample random parameter combination.

        Returns:
            Parameter dictionary.
        """
        params = {}
        for key, values in self.param_distributions.items():
            params[key] = random.choice(values)
        return params

    def _cross_val_score(
        self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate cross-validation score for parameter set.

        Args:
            X: Feature matrix.
            y: Target values.
            params: Parameter dictionary.

        Returns:
            Tuple of (mean_score, std_score).
        """
        n_samples = len(X)
        fold_size = n_samples // self.cv
        scores = []

        indices = np.arange(n_samples)
        if self.random_state is not None:
            np.random.shuffle(indices)

        for fold in range(self.cv):
            start = fold * fold_size
            end = start + fold_size if fold < self.cv - 1 else n_samples

            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            estimator = self._create_estimator(params)
            estimator.fit(X_train, y_train)

            if self.scoring:
                score = self.scoring(estimator, X_val, y_val)
            else:
                score = estimator.score(X_val, y_val)

            scores.append(score)

        return np.mean(scores), np.std(scores)

    def _create_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        """Create estimator with given parameters.

        Args:
            params: Parameter dictionary.

        Returns:
            Estimator instance.
        """
        estimator = type(self.estimator)()
        for key, value in params.items():
            if hasattr(estimator, key):
                setattr(estimator, key, value)
        return estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomSearchCV":
        """Fit random search.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self for method chaining.
        """
        if self.verbose > 0:
            logger.info(f"Random search: {self.n_iter} iterations")

        best_score = float("-inf")
        best_params = None
        cv_results = defaultdict(list)

        for i in range(self.n_iter):
            if self.verbose > 0 and (i + 1) % 10 == 0:
                logger.info(f"  Iteration {i+1}/{self.n_iter}")

            params = self._sample_params()
            mean_score, std_score = self._cross_val_score(X, y, params)

            for key, value in params.items():
                cv_results[f"param_{key}"].append(value)
            cv_results["mean_test_score"].append(mean_score)
            cv_results["std_test_score"].append(std_score)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = dict(cv_results)

        self.best_estimator_ = self._create_estimator(best_params)
        self.best_estimator_.fit(X, y)

        if self.verbose > 0:
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best score: {best_score:.6f}")

        return self


class BayesianOptimization:
    """Bayesian Optimization for hyperparameter tuning."""

    def __init__(
        self,
        estimator: BaseEstimator,
        param_space: Dict[str, Tuple[float, float]],
        n_iter: int = 10,
        cv: int = 5,
        scoring: Optional[Callable] = None,
        n_initial: int = 5,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Bayesian Optimization.

        Args:
            estimator: Base estimator to tune.
            param_space: Dictionary of parameter names and (min, max) tuples.
            n_iter: Number of optimization iterations (default: 10).
            cv: Number of cross-validation folds (default: 5).
            scoring: Scoring function (default: estimator.score).
            n_initial: Number of initial random samples (default: 5).
            verbose: Verbosity level (default: 0).
            random_state: Random seed (default: None).
        """
        self.estimator = estimator
        self.param_space = param_space
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_initial = n_initial
        self.verbose = verbose
        self.random_state = random_state

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.best_estimator_: Optional[BaseEstimator] = None
        self.cv_results_: Optional[Dict[str, List]] = None

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _sample_random_params(self) -> Dict[str, Any]:
        """Sample random parameters from space.

        Returns:
            Parameter dictionary.
        """
        params = {}
        for key, (min_val, max_val) in self.param_space.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[key] = random.randint(min_val, max_val)
            else:
                params[key] = random.uniform(min_val, max_val)
        return params

    def _gaussian_process_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple Gaussian Process prediction (simplified).

        Args:
            X_train: Training parameter vectors.
            y_train: Training scores.
            X_test: Test parameter vectors.

        Returns:
            Tuple of (mean, std) predictions.
        """
        if len(X_train) == 0:
            return np.zeros(len(X_test)), np.ones(len(X_test))

        n_train = len(X_train)
        n_test = len(X_test)

        mean = np.zeros(n_test)
        std = np.ones(n_test)

        for i, x_test in enumerate(X_test):
            distances = np.sum((X_train - x_test) ** 2, axis=1)
            weights = np.exp(-distances / (2 * 0.1))
            weights = weights / (np.sum(weights) + 1e-10)

            mean[i] = np.sum(weights * y_train)
            std[i] = np.std(y_train) * 0.5

        return mean, std

    def _acquisition_function(
        self, mean: np.ndarray, std: np.ndarray, best_score: float
    ) -> np.ndarray:
        """Upper Confidence Bound acquisition function.

        Args:
            mean: Mean predictions.
            std: Standard deviation predictions.
            best_score: Best score so far.

        Returns:
            Acquisition values.
        """
        beta = 2.0
        return mean + beta * std

    def _cross_val_score(
        self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]
    ) -> float:
        """Calculate cross-validation score for parameter set.

        Args:
            X: Feature matrix.
            y: Target values.
            params: Parameter dictionary.

        Returns:
            Mean cross-validation score.
        """
        n_samples = len(X)
        fold_size = n_samples // self.cv
        scores = []

        indices = np.arange(n_samples)
        if self.random_state is not None:
            np.random.shuffle(indices)

        for fold in range(self.cv):
            start = fold * fold_size
            end = start + fold_size if fold < self.cv - 1 else n_samples

            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            estimator = self._create_estimator(params)
            estimator.fit(X_train, y_train)

            if self.scoring:
                score = self.scoring(estimator, X_val, y_val)
            else:
                score = estimator.score(X_val, y_val)

            scores.append(score)

        return np.mean(scores)

    def _create_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
        """Create estimator with given parameters.

        Args:
            params: Parameter dictionary.

        Returns:
            Estimator instance.
        """
        estimator = type(self.estimator)()
        for key, value in params.items():
            if hasattr(estimator, key):
                setattr(estimator, key, value)
        return estimator

    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dictionary to vector.

        Args:
            params: Parameter dictionary.

        Returns:
            Parameter vector.
        """
        vector = []
        for key in sorted(self.param_space.keys()):
            min_val, max_val = self.param_space[key]
            value = params[key]
            normalized = (value - min_val) / (max_val - min_val + 1e-10)
            vector.append(normalized)
        return np.array(vector)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianOptimization":
        """Fit Bayesian optimization.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self for method chaining.
        """
        if self.verbose > 0:
            logger.info(f"Bayesian optimization: {self.n_iter} iterations")

        X_samples = []
        y_samples = []
        params_samples = []
        cv_results = defaultdict(list)

        best_score = float("-inf")
        best_params = None

        for i in range(self.n_initial):
            params = self._sample_random_params()
            score = self._cross_val_score(X, y, params)

            params_samples.append(params)
            X_samples.append(self._params_to_vector(params))
            y_samples.append(score)

            for key, value in params.items():
                cv_results[f"param_{key}"].append(value)
            cv_results["mean_test_score"].append(score)
            cv_results["std_test_score"].append(0.0)

            if score > best_score:
                best_score = score
                best_params = params

            if self.verbose > 0:
                logger.info(f"  Initial sample {i+1}/{self.n_initial}: score={score:.6f}")

        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)

        for i in range(self.n_initial, self.n_iter):
            if self.verbose > 0:
                logger.info(f"  Iteration {i+1}/{self.n_iter}")

            n_candidates = 100
            candidates = []
            candidate_params = []

            for _ in range(n_candidates):
                params = self._sample_random_params()
                candidate_params.append(params)
                candidates.append(self._params_to_vector(params))

            candidates = np.array(candidates)

            mean, std = self._gaussian_process_predict(X_samples, y_samples, candidates)
            acquisition = self._acquisition_function(mean, std, best_score)

            best_candidate_idx = np.argmax(acquisition)
            params = candidate_params[best_candidate_idx]

            score = self._cross_val_score(X, y, params)

            params_samples.append(params)
            X_samples = np.vstack([X_samples, candidates[best_candidate_idx:best_candidate_idx+1]])
            y_samples = np.append(y_samples, score)

            for key, value in params.items():
                cv_results[f"param_{key}"].append(value)
            cv_results["mean_test_score"].append(score)
            cv_results["std_test_score"].append(0.0)

            if score > best_score:
                best_score = score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = dict(cv_results)

        self.best_estimator_ = self._create_estimator(best_params)
        self.best_estimator_.fit(X, y)

        if self.verbose > 0:
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best score: {best_score:.6f}")

        return self


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file with data",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of target column",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated list of feature columns (default: all except target)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "random", "bayesian"],
        required=True,
        help="Hyperparameter tuning method",
    )
    parser.add_argument(
        "--param-grid",
        type=str,
        required=True,
        help="Path to JSON file with parameter grid",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=None,
        help="Number of cross-validation folds (default: from config)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="Number of iterations for random/bayesian search (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results CSV",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        tuning_config = config.get("tuning", {})

        cv = args.cv if args.cv is not None else tuning_config.get("cv", 5)
        n_iter = (
            args.n_iter
            if args.n_iter is not None
            else tuning_config.get("n_iter", 10)
        )

        df = pd.read_csv(args.input)
        print(f"\n=== Hyperparameter Tuning ===")
        print(f"Data shape: {df.shape}")
        print(f"Method: {args.method}")
        print(f"CV folds: {cv}")

        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found in data")

        if args.features:
            feature_cols = [col.strip() for col in args.features.split(",")]
            missing = [col for col in feature_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Feature columns not found: {missing}")
        else:
            feature_cols = [col for col in df.columns if col != args.target]

        X = df[feature_cols].values
        y = df[args.target].values

        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")

        with open(args.param_grid, "r") as f:
            param_config = json.load(f)

        estimator_class_name = param_config.get("estimator")
        param_grid = param_config.get("param_grid", {})

        print(f"\nEstimator: {estimator_class_name}")
        print(f"Parameter grid: {list(param_grid.keys())}")

        try:
            from src.example_estimator import SimpleClassifier, SimpleRegressor
            
            if estimator_class_name == "SimpleRegressor":
                estimator = SimpleRegressor()
            else:
                estimator = SimpleClassifier()
        except ImportError:
            from src.example_estimator import SimpleClassifier
            estimator = SimpleClassifier()

        if args.method == "grid":
            searcher = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv,
                verbose=1,
            )
        elif args.method == "random":
            searcher = RandomSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                verbose=1,
            )
        else:
            param_space = {}
            for key, values in param_grid.items():
                if isinstance(values, list):
                    if all(isinstance(v, int) for v in values):
                        param_space[key] = (min(values), max(values))
                    else:
                        param_space[key] = (min(values), max(values))
                else:
                    param_space[key] = values

            searcher = BayesianOptimization(
                estimator=estimator,
                param_space=param_space,
                n_iter=n_iter,
                cv=cv,
                n_initial=5,
                verbose=1,
            )

        print(f"\nStarting hyperparameter tuning...")
        searcher.fit(X, y)

        print(f"\n=== Tuning Results ===")
        print(f"Best parameters: {searcher.best_params_}")
        print(f"Best score: {searcher.best_score_:.6f}")

        if args.output:
            results_df = pd.DataFrame(searcher.cv_results_)
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {e}")
        raise


if __name__ == "__main__":
    main()
