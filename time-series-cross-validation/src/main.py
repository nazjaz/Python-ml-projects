"""Time Series Cross-Validation with Walk-Forward and Expanding Window Strategies.

This module provides implementations of time series cross-validation techniques
including walk-forward validation and expanding window strategies for proper
evaluation of time series models while respecting temporal order.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TimeSeriesSplit:
    """Time series cross-validation splitter with walk-forward validation."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None,
    ):
        """Initialize time series splitter.

        Args:
            n_splits: Number of splits (default: 5)
            test_size: Size of test set for each split (default: None, auto-calculate)
            gap: Number of samples to skip between train and test (default: 0)
            max_train_size: Maximum size of training set (default: None, no limit)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train-test splits for time series data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Optional target vector, shape (n_samples,)

        Yields:
            Tuple of (train_indices, test_indices) for each split

        Raises:
            ValueError: If parameters are invalid
        """
        n_samples = len(X)

        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} > n_samples={n_samples}"
            )

        if self.test_size is None:
            test_size = max(1, n_samples // (self.n_splits + 1))
        else:
            test_size = self.test_size

        if test_size * self.n_splits + self.gap * self.n_splits > n_samples:
            raise ValueError(
                f"test_size={test_size} and n_splits={self.n_splits} "
                f"too large for n_samples={n_samples}"
            )

        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * test_size - self.gap
            test_end = test_start + test_size

            if test_start < 0:
                continue

            train_end = test_start - self.gap if self.gap > 0 else test_start

            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class ExpandingWindowSplit:
    """Expanding window cross-validation for time series."""

    def __init__(
        self,
        initial_train_size: int,
        step_size: int = 1,
        n_splits: Optional[int] = None,
        max_train_size: Optional[int] = None,
    ):
        """Initialize expanding window splitter.

        Args:
            initial_train_size: Initial size of training window
            step_size: Number of samples to move forward each split (default: 1)
            n_splits: Number of splits (default: None, use all possible)
            max_train_size: Maximum size of training window (default: None, no limit)
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.n_splits = n_splits
        self.max_train_size = max_train_size

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train-test splits with expanding window.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Optional target vector, shape (n_samples,)

        Yields:
            Tuple of (train_indices, test_indices) for each split

        Raises:
            ValueError: If parameters are invalid
        """
        n_samples = len(X)

        if self.initial_train_size >= n_samples:
            raise ValueError(
                f"initial_train_size={self.initial_train_size} >= n_samples={n_samples}"
            )

        if self.initial_train_size < 1:
            raise ValueError(f"initial_train_size must be >= 1, got {self.initial_train_size}")

        train_start = 0
        train_end = self.initial_train_size
        split_count = 0

        while train_end < n_samples:
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)

            train_indices = np.arange(train_start, train_end)
            test_start = train_end
            test_end = min(n_samples, test_start + self.step_size)

            if test_end > test_start:
                test_indices = np.arange(test_start, test_end)

                yield train_indices, test_indices

                train_end += self.step_size
                split_count += 1

                if self.n_splits is not None and split_count >= self.n_splits:
                    break
            else:
                break


class TimeSeriesCrossValidator:
    """Time series cross-validation with evaluation metrics."""

    def __init__(
        self,
        strategy: str = "walk_forward",
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        initial_train_size: Optional[int] = None,
        step_size: int = 1,
        max_train_size: Optional[int] = None,
    ):
        """Initialize time series cross-validator.

        Args:
            strategy: Validation strategy - "walk_forward" or "expanding_window"
                (default: "walk_forward")
            n_splits: Number of splits for walk-forward (default: 5)
            test_size: Size of test set (default: None, auto-calculate)
            gap: Gap between train and test (default: 0)
            initial_train_size: Initial training size for expanding window (default: None)
            step_size: Step size for expanding window (default: 1)
            max_train_size: Maximum training size (default: None)
        """
        self.strategy = strategy
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.max_train_size = max_train_size
        self.splitter = None

    def _get_splitter(self, n_samples: int) -> Union[TimeSeriesSplit, ExpandingWindowSplit]:
        """Get appropriate splitter based on strategy.

        Args:
            n_samples: Number of samples in dataset

        Returns:
            Splitter instance

        Raises:
            ValueError: If strategy is invalid
        """
        if self.strategy == "walk_forward":
            return TimeSeriesSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                gap=self.gap,
                max_train_size=self.max_train_size,
            )
        elif self.strategy == "expanding_window":
            if self.initial_train_size is None:
                initial_size = max(1, n_samples // 4)
            else:
                initial_size = self.initial_train_size

            return ExpandingWindowSplit(
                initial_train_size=initial_size,
                step_size=self.step_size,
                n_splits=self.n_splits,
                max_train_size=self.max_train_size,
            )
        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. Use 'walk_forward' or 'expanding_window'"
            )

    def cross_validate(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        scoring: Optional[str] = None,
        return_train_score: bool = False,
    ) -> Dict:
        """Perform time series cross-validation.

        Args:
            estimator: Scikit-learn compatible estimator
            X: Feature matrix, shape (n_samples, n_features)
            y: Target vector, shape (n_samples,)
            scoring: Scoring metric - "mse", "mae", "r2" (default: None, auto-select)
            return_train_score: Whether to return training scores (default: False)

        Returns:
            Dictionary with cross-validation results

        Raises:
            ValueError: If input data is invalid
        """
        X = np.array(X)
        y = np.array(y).ravel()

        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X.shape[0]={X.shape[0]}, len(y)={len(y)}"
            )

        if scoring is None:
            if hasattr(estimator, "predict_proba"):
                scoring = "mse"
            else:
                scoring = "mse"

        splitter = self._get_splitter(len(X))

        test_scores = []
        train_scores = [] if return_train_score else None
        split_info = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            estimator_fold = estimator.__class__(**estimator.get_params())
            estimator_fold.fit(X_train, y_train)

            y_test_pred = estimator_fold.predict(X_test)
            test_score = self._calculate_score(y_test, y_test_pred, scoring)
            test_scores.append(test_score)

            if return_train_score:
                y_train_pred = estimator_fold.predict(X_train)
                train_score = self._calculate_score(y_train, y_train_pred, scoring)
                train_scores.append(train_score)

            split_info.append(
                {
                    "fold": fold + 1,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "train_start": int(train_idx[0]) if len(train_idx) > 0 else None,
                    "train_end": int(train_idx[-1]) + 1 if len(train_idx) > 0 else None,
                    "test_start": int(test_idx[0]) if len(test_idx) > 0 else None,
                    "test_end": int(test_idx[-1]) + 1 if len(test_idx) > 0 else None,
                }
            )

            logger.info(
                f"Fold {fold + 1}: train_size={len(train_idx)}, "
                f"test_size={len(test_idx)}, test_score={test_score:.4f}"
            )

        results = {
            "strategy": self.strategy,
            "n_splits": len(test_scores),
            "test_scores": test_scores,
            "test_mean": float(np.mean(test_scores)),
            "test_std": float(np.std(test_scores)),
            "split_info": split_info,
        }

        if return_train_score:
            results["train_scores"] = train_scores
            results["train_mean"] = float(np.mean(train_scores))
            results["train_std"] = float(np.std(train_scores))

        return results

    def _calculate_score(
        self, y_true: np.ndarray, y_pred: np.ndarray, scoring: str
    ) -> float:
        """Calculate score for given metric.

        Args:
            y_true: True values
            y_pred: Predicted values
            scoring: Metric name

        Returns:
            Score value

        Raises:
            ValueError: If scoring metric is invalid
        """
        if scoring == "mse":
            return float(mean_squared_error(y_true, y_pred))
        elif scoring == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif scoring == "mae":
            return float(mean_absolute_error(y_true, y_pred))
        elif scoring == "r2":
            return float(r2_score(y_true, y_pred))
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")


class TimeSeriesValidator:
    """Main time series validation class."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize time series validator.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.validator = None

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "logs/app.log")

        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(handler)

    def validate(
        self,
        estimator: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        strategy: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """Perform time series cross-validation.

        Args:
            estimator: Scikit-learn compatible estimator
            X: Feature matrix
            y: Target vector
            strategy: Validation strategy (default: from config)
            **kwargs: Additional arguments for TimeSeriesCrossValidator

        Returns:
            Dictionary with validation results
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.array(X)

        if isinstance(y, pd.Series):
            y = y.values
        else:
            y = np.array(y)

        cv_config = self.config.get("cross_validation", {})
        strategy = strategy or cv_config.get("strategy", "walk_forward")
        n_splits = kwargs.get("n_splits", cv_config.get("n_splits", 5))
        test_size = kwargs.get("test_size", cv_config.get("test_size", None))
        gap = kwargs.get("gap", cv_config.get("gap", 0))
        initial_train_size = kwargs.get(
            "initial_train_size", cv_config.get("initial_train_size", None)
        )
        step_size = kwargs.get("step_size", cv_config.get("step_size", 1))
        max_train_size = kwargs.get(
            "max_train_size", cv_config.get("max_train_size", None)
        )
        scoring = kwargs.get("scoring", cv_config.get("scoring", None))
        return_train_score = kwargs.get(
            "return_train_score", cv_config.get("return_train_score", False)
        )

        self.validator = TimeSeriesCrossValidator(
            strategy=strategy,
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            initial_train_size=initial_train_size,
            step_size=step_size,
            max_train_size=max_train_size,
        )

        results = self.validator.cross_validate(
            estimator, X, y, scoring=scoring, return_train_score=return_train_score
        )

        return results


def main():
    """Main entry point for time series cross-validator."""
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
        description="Time series cross-validation with walk-forward and expanding window"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        required=True,
        help="Column name for target variable",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to pickled model file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["walk_forward", "expanding_window"],
        default="walk_forward",
        help="Validation strategy (default: walk_forward)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        help="Number of splits (default: from config)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        help="Size of test set for each split",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=0,
        help="Gap between train and test sets (default: 0)",
    )
    parser.add_argument(
        "--initial-train-size",
        type=int,
        help="Initial training size for expanding window",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="Step size for expanding window (default: 1)",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        choices=["mse", "rmse", "mae", "r2"],
        help="Scoring metric (default: mse)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output JSON file for results",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    validator = TimeSeriesValidator(
        config_path=Path(args.config) if args.config else None
    )

    df = pd.read_csv(args.input)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in dataset")

    feature_cols = [col for col in df.columns if col != args.target_col]
    X = df[feature_cols].values
    y = df[args.target_col].values

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    results = validator.validate(
        model,
        X,
        y,
        strategy=args.strategy,
        n_splits=args.n_splits,
        test_size=args.test_size,
        gap=args.gap,
        initial_train_size=args.initial_train_size,
        step_size=args.step_size,
        scoring=args.scoring,
    )

    print("\nTime Series Cross-Validation Results:")
    print("=" * 50)
    print(f"Strategy: {results['strategy']}")
    print(f"Number of splits: {results['n_splits']}")
    print(f"\nTest Scores:")
    for i, score in enumerate(results["test_scores"]):
        print(f"  Fold {i + 1}: {score:.4f}")
    print(f"\nMean Test Score: {results['test_mean']:.4f} ± {results['test_std']:.4f}")

    if "train_scores" in results:
        print(f"\nTrain Scores:")
        for i, score in enumerate(results["train_scores"]):
            print(f"  Fold {i + 1}: {score:.4f}")
        print(
            f"\nMean Train Score: {results['train_mean']:.4f} ± {results['train_std']:.4f}"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
