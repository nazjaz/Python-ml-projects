"""Naive Bayes Classifier.

This module provides functionality to implement naive Bayes classifiers
from scratch with Gaussian, Multinomial, and Bernoulli variants for
different data types.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GaussianNB:
    """Gaussian Naive Bayes classifier for continuous features."""

    def __init__(self, smoothing: float = 1e-9) -> None:
        """Initialize GaussianNB.

        Args:
            smoothing: Smoothing parameter to prevent division by zero.
        """
        self.smoothing = smoothing
        self.classes: Optional[np.ndarray] = None
        self.class_priors: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.variances: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> "GaussianNB":
        """Fit Gaussian Naive Bayes classifier.

        Args:
            X: Feature matrix (continuous values).
            y: Target labels.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) != len(y):
            raise ValueError(
                f"Length mismatch: X has {len(X)} samples, "
                f"y has {len(y)} samples"
            )

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.class_priors[i] = len(X_cls) / len(X)
            self.means[i] = np.mean(X_cls, axis=0)
            self.variances[i] = np.var(X_cls, axis=0) + self.smoothing

        logger.info(
            f"GaussianNB fitted: {n_classes} classes, {n_features} features"
        )

        return self

    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        """Calculate Gaussian probability density function.

        Args:
            x: Feature vector.
            mean: Mean vector.
            var: Variance vector.

        Returns:
            Log probability.
        """
        exponent = -0.5 * np.sum(((x - mean) ** 2) / var)
        normalization = -0.5 * np.sum(np.log(2 * np.pi * var))
        return exponent + normalization

    def predict_proba(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix (shape: [n_samples, n_classes]).

        Raises:
            ValueError: If model not fitted.
        """
        if self.classes is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(X)
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for j in range(n_classes):
                log_likelihood = self._gaussian_pdf(
                    X[i], self.means[j], self.variances[j]
                )
                log_prior = np.log(self.class_priors[j] + self.smoothing)
                log_probs[i, j] = log_likelihood + log_prior

        log_probs = log_probs - np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        return probs

    def predict(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model not fitted.
        """
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        predictions = self.classes[class_indices]
        return predictions

    def score(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> float:
        """Calculate accuracy score.

        Args:
            X: Feature matrix.
            y: True target labels.

        Returns:
            Accuracy score (between 0 and 1).
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        return float(accuracy)


class MultinomialNB:
    """Multinomial Naive Bayes classifier for count data."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize MultinomialNB.

        Args:
            alpha: Smoothing parameter (Laplace smoothing).
        """
        self.alpha = alpha
        self.classes: Optional[np.ndarray] = None
        self.class_priors: Optional[np.ndarray] = None
        self.feature_counts: Optional[np.ndarray] = None
        self.class_counts: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> "MultinomialNB":
        """Fit Multinomial Naive Bayes classifier.

        Args:
            X: Feature matrix (count data, non-negative integers).
            y: Target labels.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) != len(y):
            raise ValueError(
                f"Length mismatch: X has {len(X)} samples, "
                f"y has {len(y)} samples"
            )

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative feature values")

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_priors = np.zeros(n_classes)
        self.feature_counts = np.zeros((n_classes, n_features))
        self.class_counts = np.zeros(n_classes)

        for i, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.class_priors[i] = len(X_cls) / len(X)
            self.feature_counts[i] = np.sum(X_cls, axis=0)
            self.class_counts[i] = np.sum(X_cls)

        logger.info(
            f"MultinomialNB fitted: {n_classes} classes, {n_features} features"
        )

        return self

    def predict_proba(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix (shape: [n_samples, n_classes]).

        Raises:
            ValueError: If model not fitted.
        """
        if self.classes is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(X)
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for j in range(n_classes):
                log_likelihood = np.sum(
                    X[i]
                    * np.log(
                        (self.feature_counts[j] + self.alpha)
                        / (self.class_counts[j] + self.alpha * X.shape[1])
                    )
                )
                log_prior = np.log(self.class_priors[j])
                log_probs[i, j] = log_likelihood + log_prior

        log_probs = log_probs - np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        return probs

    def predict(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model not fitted.
        """
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        predictions = self.classes[class_indices]
        return predictions

    def score(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> float:
        """Calculate accuracy score.

        Args:
            X: Feature matrix.
            y: True target labels.

        Returns:
            Accuracy score (between 0 and 1).
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        return float(accuracy)


class BernoulliNB:
    """Bernoulli Naive Bayes classifier for binary features."""

    def __init__(self, alpha: float = 1.0, binarize: Optional[float] = 0.0) -> None:
        """Initialize BernoulliNB.

        Args:
            alpha: Smoothing parameter (Laplace smoothing).
            binarize: Threshold for binarizing features. If None, assumes
                features are already binary.
        """
        self.alpha = alpha
        self.binarize = binarize
        self.classes: Optional[np.ndarray] = None
        self.class_priors: Optional[np.ndarray] = None
        self.feature_probs: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> "BernoulliNB":
        """Fit Bernoulli Naive Bayes classifier.

        Args:
            X: Feature matrix (binary or continuous to be binarized).
            y: Target labels.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) != len(y):
            raise ValueError(
                f"Length mismatch: X has {len(X)} samples, "
                f"y has {len(y)} samples"
            )

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if self.binarize is not None:
            X = (X > self.binarize).astype(float)

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_priors = np.zeros(n_classes)
        self.feature_probs = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.class_priors[i] = len(X_cls) / len(X)
            self.feature_probs[i] = (
                np.sum(X_cls, axis=0) + self.alpha
            ) / (len(X_cls) + 2 * self.alpha)

        logger.info(
            f"BernoulliNB fitted: {n_classes} classes, {n_features} features"
        )

        return self

    def predict_proba(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix (shape: [n_samples, n_classes]).

        Raises:
            ValueError: If model not fitted.
        """
        if self.classes is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.binarize is not None:
            X = (X > self.binarize).astype(float)

        n_samples = len(X)
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for j in range(n_classes):
                log_likelihood = np.sum(
                    X[i] * np.log(self.feature_probs[j] + 1e-15)
                    + (1 - X[i]) * np.log(1 - self.feature_probs[j] + 1e-15)
                )
                log_prior = np.log(self.class_priors[j])
                log_probs[i, j] = log_likelihood + log_prior

        log_probs = log_probs - np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        return probs

    def predict(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model not fitted.
        """
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        predictions = self.classes[class_indices]
        return predictions

    def score(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> float:
        """Calculate accuracy score.

        Args:
            X: Feature matrix.
            y: True target labels.

        Returns:
            Accuracy score (between 0 and 1).
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        return float(accuracy)


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Naive Bayes Classifier")
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
        "--variant",
        type=str,
        choices=["gaussian", "multinomial", "bernoulli"],
        default=None,
        help="Naive Bayes variant (default: from config or auto-detect)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Smoothing parameter for Multinomial/Bernoulli (default: from config)",
    )
    parser.add_argument(
        "--binarize",
        type=float,
        default=None,
        help="Binarization threshold for Bernoulli (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save model predictions as CSV",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        variant = (
            args.variant
            if args.variant is not None
            else model_config.get("variant", "gaussian")
        )
        alpha = (
            args.alpha
            if args.alpha is not None
            else model_config.get("alpha", 1.0)
        )
        binarize = (
            args.binarize
            if args.binarize is not None
            else model_config.get("binarize", 0.0)
        )

        df = pd.read_csv(args.input)
        print(f"\n=== Naive Bayes Classifier ({variant.capitalize()}) ===")
        print(f"Data shape: {df.shape}")

        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found")

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
        unique_classes = np.unique(y)
        print(f"Classes: {len(unique_classes)} {unique_classes}")

        if variant == "gaussian":
            model = GaussianNB()
        elif variant == "multinomial":
            model = MultinomialNB(alpha=alpha)
        elif variant == "bernoulli":
            model = BernoulliNB(alpha=alpha, binarize=binarize)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        print(f"\nTraining {variant.capitalize()} Naive Bayes...")
        model.fit(X, y)

        print(f"\n=== Model Performance ===")
        print(f"Accuracy: {model.score(X, y):.6f}")

        if args.output:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            output_df = pd.DataFrame({"actual": y, "predicted": predictions})
            for i, cls in enumerate(unique_classes):
                output_df[f"prob_class_{cls}"] = probabilities[:, i]
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


if __name__ == "__main__":
    main()
