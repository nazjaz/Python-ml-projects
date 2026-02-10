"""Polynomial Regression with Degree Selection and Regularization.

This module provides functionality to implement polynomial regression from
scratch with cross-validation for degree selection and regularization support.
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


class PolynomialRegression:
    """Polynomial Regression with degree selection and regularization."""

    def __init__(
        self,
        degree: int = 2,
        regularization: Optional[str] = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ) -> None:
        """Initialize Polynomial Regression.

        Args:
            degree: Polynomial degree (default: 2).
            regularization: Type of regularization. Options: None, "l1", "l2",
                "ridge", "lasso" (default: None).
            alpha: Regularization strength (default: 1.0).
            fit_intercept: Whether to fit intercept term (default: True).
        """
        self.degree = degree
        self.regularization = regularization
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        self.coefficients: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.feature_names: Optional[List[str]] = None

    def _create_polynomial_features(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """Create polynomial features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Tuple of (polynomial features, feature names).
        """
        n_samples, n_features = X.shape
        features_list = []
        feature_names = []

        if self.fit_intercept:
            features_list.append(np.ones(n_samples))
            feature_names.append("intercept")

        for d in range(1, self.degree + 1):
            for i in range(n_features):
                feature = X[:, i] ** d
                features_list.append(feature)
                feature_names.append(f"x{i+1}^{d}")

        X_poly = np.column_stack(features_list)
        return X_poly, feature_names

    def _compute_loss(
        self, X: np.ndarray, y: np.ndarray, coefficients: np.ndarray
    ) -> float:
        """Compute loss function with optional regularization.

        Args:
            X: Feature matrix (with polynomial features).
            y: Target values.
            coefficients: Model coefficients.

        Returns:
            Loss value.
        """
        predictions = X @ coefficients
        mse = np.mean((y - predictions) ** 2)

        if self.regularization is None:
            return mse

        reg_term = 0.0
        if self.regularization in ["l2", "ridge"]:
            reg_term = self.alpha * np.sum(coefficients ** 2)
        elif self.regularization in ["l1", "lasso"]:
            reg_term = self.alpha * np.sum(np.abs(coefficients))

        return mse + reg_term

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit using Ridge regression (L2 regularization).

        Args:
            X: Feature matrix (with polynomial features).
            y: Target values.

        Returns:
            Coefficients.
        """
        n_features = X.shape[1]
        identity = np.eye(n_features)
        if self.fit_intercept:
            identity[0, 0] = 0

        XtX = X.T @ X
        Xty = X.T @ y
        coefficients = np.linalg.solve(XtX + self.alpha * identity, Xty)
        return coefficients

    def _fit_lasso(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit using Lasso regression (L1 regularization) with coordinate descent.

        Args:
            X: Feature matrix (with polynomial features).
            y: Target values.

        Returns:
            Coefficients.
        """
        n_samples, n_features = X.shape
        coefficients = np.zeros(n_features)
        max_iter = 1000
        tol = 1e-4

        X_normalized = X.copy()
        if self.fit_intercept:
            X_normalized[:, 0] = 1.0

        for iteration in range(max_iter):
            coefficients_old = coefficients.copy()

            for j in range(n_features):
                if self.fit_intercept and j == 0:
                    continue

                X_j = X_normalized[:, j]
                residual = y - X_normalized @ coefficients + X_j * coefficients[j]

                numerator = X_j @ residual
                denominator = X_j @ X_j

                if denominator == 0:
                    continue

                if self.regularization == "lasso":
                    soft_threshold = self.alpha / (2 * denominator)
                    coefficients[j] = np.sign(numerator) * max(
                        0, abs(numerator / denominator) - soft_threshold
                    )
                else:
                    coefficients[j] = numerator / denominator

            if np.linalg.norm(coefficients - coefficients_old) < tol:
                break

        return coefficients

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> "PolynomialRegression":
        """Fit polynomial regression model.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        if self.degree < 1:
            raise ValueError("Degree must be at least 1")

        if self.alpha < 0:
            raise ValueError("Alpha must be non-negative")

        X_poly, self.feature_names = self._create_polynomial_features(X)

        if self.regularization in ["l2", "ridge"]:
            self.coefficients = self._fit_ridge(X_poly, y)
        elif self.regularization in ["l1", "lasso"]:
            self.coefficients = self._fit_lasso(X_poly, y)
        else:
            try:
                self.coefficients = np.linalg.lstsq(
                    X_poly, y, rcond=None
                )[0]
            except np.linalg.LinAlgError:
                logger.warning("Singular matrix, using pseudo-inverse")
                self.coefficients = np.linalg.pinv(X_poly) @ y

        if self.fit_intercept:
            self.intercept = self.coefficients[0]
        else:
            self.intercept = 0.0

        logger.info(
            f"Polynomial regression fitted: degree={self.degree}, "
            f"regularization={self.regularization}, alpha={self.alpha}"
        )

        return self

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict target values.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.

        Raises:
            ValueError: If model not fitted.
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_poly, _ = self._create_polynomial_features(X)
        predictions = X_poly @ self.coefficients

        return predictions

    def score(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> float:
        """Calculate R-squared score.

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            R-squared score.
        """
        predictions = self.predict(X)
        y = np.asarray(y, dtype=float)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return r2

    def mse(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> float:
        """Calculate mean squared error.

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            Mean squared error.
        """
        predictions = self.predict(X)
        y = np.asarray(y, dtype=float)
        mse = np.mean((y - predictions) ** 2)
        return mse

    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients as dictionary.

        Returns:
            Dictionary mapping feature names to coefficients.

        Raises:
            ValueError: If model not fitted.
        """
        if self.coefficients is None or self.feature_names is None:
            raise ValueError("Model must be fitted before getting coefficients")

        return dict(zip(self.feature_names, self.coefficients))


def cross_validate_degree(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series],
    degree_range: Tuple[int, int] = (1, 10),
    cv: int = 5,
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    scoring: str = "mse",
    random_state: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """Perform cross-validation to select optimal polynomial degree.

    Args:
        X: Feature matrix.
        y: Target values.
        degree_range: Tuple of (min_degree, max_degree) (default: (1, 10)).
        cv: Number of cross-validation folds (default: 5).
        regularization: Type of regularization (default: None).
        alpha: Regularization strength (default: 1.0).
        scoring: Scoring metric. Options: "mse", "r2" (default: "mse").
        random_state: Random seed for shuffling (default: None).

    Returns:
        Dictionary mapping degrees to scores.

    Raises:
        ValueError: If inputs are invalid.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    min_degree, max_degree = degree_range
    if min_degree < 1 or max_degree < min_degree:
        raise ValueError("Invalid degree range")

    if cv < 2:
        raise ValueError("cv must be at least 2")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_size = n_samples // cv
    results = {}

    for degree in range(min_degree, max_degree + 1):
        scores = []

        for fold in range(cv):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < cv - 1 else n_samples

            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate(
                [indices[:start_idx], indices[end_idx:]]
            )

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            model = PolynomialRegression(
                degree=degree, regularization=regularization, alpha=alpha
            )
            model.fit(X_train, y_train)

            if scoring == "mse":
                score = model.mse(X_val, y_val)
            elif scoring == "r2":
                score = model.score(X_val, y_val)
            else:
                raise ValueError(f"Unknown scoring metric: {scoring}")

            scores.append(score)

        results[degree] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "scores": scores,
        }

        logger.info(
            f"Degree {degree}: {scoring}={results[degree]['mean']:.6f} "
            f"(std={results[degree]['std']:.6f})"
        )

    return results


def select_best_degree(
    cv_results: Dict[int, Dict[str, float]], scoring: str = "mse"
) -> int:
    """Select best degree from cross-validation results.

    Args:
        cv_results: Cross-validation results from cross_validate_degree.
        scoring: Scoring metric used. Options: "mse", "r2" (default: "mse").

    Returns:
        Best degree.
    """
    if scoring == "mse":
        best_degree = min(cv_results.keys(), key=lambda d: cv_results[d]["mean"])
    elif scoring == "r2":
        best_degree = max(cv_results.keys(), key=lambda d: cv_results[d]["mean"])
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")

    return best_degree


def plot_cv_results(
    cv_results: Dict[int, Dict[str, float]],
    scoring: str = "mse",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot cross-validation results.

    Args:
        cv_results: Cross-validation results from cross_validate_degree.
        scoring: Scoring metric used. Options: "mse", "r2" (default: "mse").
        save_path: Optional path to save figure.
        show: Whether to display plot.
    """
    degrees = sorted(cv_results.keys())
    means = [cv_results[d]["mean"] for d in degrees]
    stds = [cv_results[d]["std"] for d in degrees]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        degrees, means, yerr=stds, marker="o", capsize=5, capthick=2, linewidth=2
    )
    ax.set_xlabel("Polynomial Degree", fontsize=12)
    ax.set_ylabel(scoring.upper(), fontsize=12)
    ax.set_title(
        f"Cross-Validation Results: {scoring.upper()} vs Degree",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    best_degree = select_best_degree(cv_results, scoring)
    ax.axvline(
        x=best_degree,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Best degree: {best_degree}",
    )
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"CV results plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_predictions(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series],
    model: PolynomialRegression,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot model predictions.

    Args:
        X: Feature matrix.
        y: True target values.
        model: Fitted polynomial regression model.
        save_path: Optional path to save figure.
        show: Whether to display plot.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.shape[1] > 1:
        logger.warning("Plotting only supported for 1D features")
        return

    X_sorted_idx = np.argsort(X[:, 0])
    X_sorted = X[X_sorted_idx, 0]
    y_sorted = y[X_sorted_idx]

    predictions = model.predict(X_sorted.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_sorted, y_sorted, alpha=0.6, label="Actual", color="blue")
    ax.plot(
        X_sorted,
        predictions,
        label="Predicted",
        color="red",
        linewidth=2,
    )
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Target", fontsize=12)
    ax.set_title(
        f"Polynomial Regression (degree={model.degree})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Predictions plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Polynomial Regression")
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
        "--degree",
        type=int,
        default=None,
        help="Polynomial degree (default: from config or use CV)",
    )
    parser.add_argument(
        "--select-degree",
        action="store_true",
        help="Use cross-validation to select degree",
    )
    parser.add_argument(
        "--degree-range",
        type=str,
        default="1,10",
        help="Degree range for CV (min,max) (default: 1,10)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=None,
        help="Number of CV folds (default: from config)",
    )
    parser.add_argument(
        "--regularization",
        type=str,
        default=None,
        choices=["l1", "l2", "ridge", "lasso"],
        help="Regularization type (default: from config)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Regularization strength (default: from config)",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="mse",
        choices=["mse", "r2"],
        help="Scoring metric for CV (default: mse)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot predictions",
    )
    parser.add_argument(
        "--plot-cv",
        action="store_true",
        help="Plot cross-validation results",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save predictions plot",
    )
    parser.add_argument(
        "--save-cv-plot",
        type=str,
        default=None,
        help="Path to save CV results plot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions as CSV",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Path to CSV file for prediction",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        cv_config = config.get("cross_validation", {})

        degree = (
            args.degree
            if args.degree is not None
            else model_config.get("degree", 2)
        )
        regularization = (
            args.regularization
            if args.regularization is not None
            else model_config.get("regularization")
        )
        alpha = (
            args.alpha
            if args.alpha is not None
            else model_config.get("alpha", 1.0)
        )
        cv = (
            args.cv
            if args.cv is not None
            else cv_config.get("cv", 5)
        )

        df = pd.read_csv(args.input)
        print(f"\n=== Polynomial Regression ===")
        print(f"Data shape: {df.shape}")

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

        if args.select_degree or degree is None:
            print(f"\n=== Cross-Validation for Degree Selection ===")
            degree_range_str = args.degree_range.split(",")
            degree_range = (int(degree_range_str[0]), int(degree_range_str[1]))
            print(f"Degree range: {degree_range[0]} to {degree_range[1]}")
            print(f"CV folds: {cv}")
            print(f"Scoring: {args.scoring}")
            if regularization:
                print(f"Regularization: {regularization}, alpha: {alpha}")

            cv_results = cross_validate_degree(
                X,
                y,
                degree_range=degree_range,
                cv=cv,
                regularization=regularization,
                alpha=alpha,
                scoring=args.scoring,
            )

            best_degree = select_best_degree(cv_results, args.scoring)
            print(f"\nBest degree: {best_degree}")
            print(f"Best {args.scoring}: {cv_results[best_degree]['mean']:.6f}")

            if args.plot_cv or args.save_cv_plot:
                plot_cv_results(
                    cv_results,
                    scoring=args.scoring,
                    save_path=args.save_cv_plot,
                    show=args.plot_cv,
                )

            degree = best_degree

        print(f"\n=== Fitting Model ===")
        print(f"Degree: {degree}")
        if regularization:
            print(f"Regularization: {regularization}, alpha: {alpha}")
        else:
            print("Regularization: None")

        model = PolynomialRegression(
            degree=degree, regularization=regularization, alpha=alpha
        )
        model.fit(X, y)

        print(f"\n=== Model Results ===")
        r2_score = model.score(X, y)
        mse_score = model.mse(X, y)
        print(f"R-squared: {r2_score:.6f}")
        print(f"MSE: {mse_score:.6f}")

        coefficients = model.get_coefficients()
        print(f"\nCoefficients:")
        for name, coef in list(coefficients.items())[:10]:
            print(f"  {name}: {coef:.6f}")
        if len(coefficients) > 10:
            print(f"  ... and {len(coefficients) - 10} more")

        if args.plot or args.save_plot:
            if X.shape[1] == 1:
                plot_predictions(
                    X,
                    y,
                    model,
                    save_path=args.save_plot,
                    show=args.plot,
                )
            else:
                print("\nPlotting only supported for 1D features")

        if args.output:
            predictions = model.predict(X)
            output_df = pd.DataFrame({
                "prediction": predictions,
                args.target: y,
            })
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

        if args.predict:
            df_predict = pd.read_csv(args.predict)
            X_predict = df_predict[feature_cols].values
            predictions = model.predict(X_predict)

            if args.output:
                output_df = pd.DataFrame({
                    "prediction": predictions,
                })
                output_df.to_csv(args.output, index=False)
                print(f"Predictions saved to: {args.output}")
            else:
                print(f"\nPredictions:")
                for i, pred in enumerate(predictions[:10]):
                    print(f"  Sample {i+1}: {pred:.6f}")
                if len(predictions) > 10:
                    print(f"  ... and {len(predictions) - 10} more")

    except Exception as e:
        logger.error(f"Error running polynomial regression: {e}")
        raise


if __name__ == "__main__":
    main()
