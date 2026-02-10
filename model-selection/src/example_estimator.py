"""Example Estimators for Model Selection.

This module provides simple example estimators for demonstration purposes.
"""

import numpy as np

from src.main import BaseEstimator


class SimpleClassifier(BaseEstimator):
    """Simple Classifier for demonstration."""

    def __init__(self, max_depth: int = 3, min_samples_split: int = 2, learning_rate: float = 0.1) -> None:
        """Initialize Simple Classifier.

        Args:
            max_depth: Maximum depth (default: 3).
            min_samples_split: Minimum samples to split (default: 2).
            learning_rate: Learning rate (default: 0.1).
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.classes_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleClassifier":
        """Fit classifier.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Self for method chaining.
        """
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels.
        """
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before prediction")

        n_samples = len(X)
        predictions = np.random.choice(self.classes_, size=n_samples)
        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate classification accuracy.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Classification accuracy.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


class SimpleRegressor(BaseEstimator):
    """Simple Regressor for demonstration."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 100, learning_rate: float = 0.01) -> None:
        """Initialize Simple Regressor.

        Args:
            alpha: Regularization parameter (default: 1.0).
            max_iter: Maximum iterations (default: 100).
            learning_rate: Learning rate (default: 0.01).
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.coef_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRegressor":
        """Fit regressor.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self for method chaining.
        """
        n_features = X.shape[1]
        self.coef_ = np.random.randn(n_features) * 0.1
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.
        """
        if not self.is_fitted_:
            raise ValueError("Regressor must be fitted before prediction")

        return X.dot(self.coef_)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared score.

        Args:
            X: Feature matrix.
            y: True values.

        Returns:
            R-squared score.
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        return r2
