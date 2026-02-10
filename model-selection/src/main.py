"""Model Selection with Nested Cross-Validation and Learning Curves.

This module provides functionality for model selection using nested cross-validation
and learning curves for bias-variance analysis.
"""

import json
import logging
import logging.handlers
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
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


class NestedCrossValidation:
    """Nested Cross-Validation for model selection."""

    def __init__(
        self,
        estimators: Dict[str, BaseEstimator],
        param_grids: Dict[str, Dict[str, List[Any]]],
        outer_cv: int = 5,
        inner_cv: int = 5,
        scoring: Optional[Callable] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Nested Cross-Validation.

        Args:
            estimators: Dictionary of estimator names and instances.
            param_grids: Dictionary of estimator names and parameter grids.
            outer_cv: Number of outer CV folds (default: 5).
            inner_cv: Number of inner CV folds (default: 5).
            scoring: Scoring function (default: estimator.score).
            verbose: Verbosity level (default: 0).
            random_state: Random seed (default: None).
        """
        self.estimators = estimators
        self.param_grids = param_grids
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.verbose = verbose
        self.random_state = random_state

        self.best_estimator_name_: Optional[str] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.cv_results_: Optional[Dict[str, List]] = None

        if random_state is not None:
            np.random.seed(random_state)

    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations.

        Args:
            param_grid: Parameter grid dictionary.

        Returns:
            List of parameter dictionaries.
        """
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = []

        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def _inner_cv_search(
        self, X: np.ndarray, y: np.ndarray, estimator: BaseEstimator, param_grid: Dict[str, List[Any]]
    ) -> Tuple[Dict[str, Any], float]:
        """Perform inner CV for hyperparameter tuning.

        Args:
            X: Feature matrix.
            y: Target values.
            estimator: Base estimator.
            param_grid: Parameter grid.

        Returns:
            Tuple of (best_params, best_score).
        """
        param_combinations = self._generate_param_combinations(param_grid)
        n_samples = len(X)
        fold_size = n_samples // self.inner_cv

        best_score = float("-inf")
        best_params = None

        indices = np.arange(n_samples)
        if self.random_state is not None:
            np.random.shuffle(indices)

        for params in param_combinations:
            scores = []

            for fold in range(self.inner_cv):
                start = fold * fold_size
                end = start + fold_size if fold < self.inner_cv - 1 else n_samples

                val_indices = indices[start:end]
                train_indices = np.concatenate([indices[:start], indices[end:]])

                X_train, X_val = X[train_indices], X[val_indices]
                y_train, y_val = y[train_indices], y[val_indices]

                est = self._create_estimator(estimator, params)
                est.fit(X_train, y_train)

                if self.scoring:
                    score = self.scoring(est, X_val, y_val)
                else:
                    score = est.score(X_val, y_val)

                scores.append(score)

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        return best_params, best_score

    def _create_estimator(self, estimator: BaseEstimator, params: Dict[str, Any]) -> BaseEstimator:
        """Create estimator with given parameters.

        Args:
            estimator: Base estimator.
            params: Parameter dictionary.

        Returns:
            Estimator instance.
        """
        est = type(estimator)()
        for key, value in params.items():
            if hasattr(est, key):
                setattr(est, key, value)
        return est

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NestedCrossValidation":
        """Fit nested cross-validation.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self for method chaining.
        """
        n_samples = len(X)
        fold_size = n_samples // self.outer_cv

        estimator_scores = defaultdict(list)
        cv_results = defaultdict(list)

        indices = np.arange(n_samples)
        if self.random_state is not None:
            np.random.shuffle(indices)

        for outer_fold in range(self.outer_cv):
            if self.verbose > 0:
                logger.info(f"Outer fold {outer_fold + 1}/{self.outer_cv}")

            start = outer_fold * fold_size
            end = start + fold_size if outer_fold < self.outer_cv - 1 else n_samples

            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            for est_name, estimator in self.estimators.items():
                if self.verbose > 0:
                    logger.info(f"  Testing estimator: {est_name}")

                param_grid = self.param_grids.get(est_name, {})

                best_params, best_inner_score = self._inner_cv_search(
                    X_train, y_train, estimator, param_grid
                )

                best_estimator = self._create_estimator(estimator, best_params)
                best_estimator.fit(X_train, y_train)

                if self.scoring:
                    test_score = self.scoring(best_estimator, X_test, y_test)
                else:
                    test_score = best_estimator.score(X_test, y_test)

                estimator_scores[est_name].append(test_score)

                cv_results["estimator"].append(est_name)
                cv_results["outer_fold"].append(outer_fold)
                cv_results["test_score"].append(test_score)
                cv_results["inner_score"].append(best_inner_score)

                for key, value in best_params.items():
                    cv_results[f"param_{key}"].append(value)

        best_estimator_name = None
        best_mean_score = float("-inf")

        for est_name, scores in estimator_scores.items():
            mean_score = np.mean(scores)
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_estimator_name = est_name

        self.best_estimator_name_ = best_estimator_name
        self.best_score_ = best_mean_score
        self.cv_results_ = dict(cv_results)

        if self.verbose > 0:
            logger.info(f"Best estimator: {best_estimator_name}")
            logger.info(f"Best score: {best_mean_score:.6f}")

        return self


class LearningCurves:
    """Learning Curves for bias-variance analysis."""

    def __init__(
        self,
        estimator: BaseEstimator,
        train_sizes: Optional[np.ndarray] = None,
        cv: int = 5,
        scoring: Optional[Callable] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Learning Curves.

        Args:
            estimator: Base estimator.
            train_sizes: Training set sizes (default: auto).
            cv: Number of cross-validation folds (default: 5).
            scoring: Scoring function (default: estimator.score).
            verbose: Verbosity level (default: 0).
            random_state: Random seed (default: None).
        """
        self.estimator = estimator
        self.train_sizes = train_sizes
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.random_state = random_state

        self.train_scores_: Optional[np.ndarray] = None
        self.val_scores_: Optional[np.ndarray] = None
        self.train_sizes_: Optional[np.ndarray] = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LearningCurves":
        """Fit learning curves.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self for method chaining.
        """
        n_samples = len(X)

        if self.train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes = (train_sizes * n_samples).astype(int)
            train_sizes = np.unique(train_sizes)
            train_sizes = train_sizes[train_sizes > 0]
        else:
            train_sizes = self.train_sizes

        train_scores_list = []
        val_scores_list = []
        train_sizes_list = []

        indices = np.arange(n_samples)
        if self.random_state is not None:
            np.random.shuffle(indices)

        for train_size in train_sizes:
            if self.verbose > 0:
                logger.info(f"Training size: {train_size}/{n_samples}")

            train_scores_fold = []
            val_scores_fold = []

            for fold in range(self.cv):
                fold_indices = np.random.permutation(n_samples)
                train_indices = fold_indices[:train_size]
                val_indices = fold_indices[train_size:]

                X_train, X_val = X[train_indices], X[val_indices]
                y_train, y_val = y[train_indices], y[val_indices]

                estimator = type(self.estimator)()
                for attr in dir(self.estimator):
                    if not attr.startswith("_") and hasattr(self.estimator, attr):
                        try:
                            setattr(estimator, attr, getattr(self.estimator, attr))
                        except:
                            pass

                estimator.fit(X_train, y_train)

                if self.scoring:
                    train_score = self.scoring(estimator, X_train, y_train)
                    val_score = self.scoring(estimator, X_val, y_val)
                else:
                    train_score = estimator.score(X_train, y_train)
                    val_score = estimator.score(X_val, y_val)

                train_scores_fold.append(train_score)
                val_scores_fold.append(val_score)

            train_scores_list.append(train_scores_fold)
            val_scores_list.append(val_scores_fold)
            train_sizes_list.append(train_size)

        self.train_sizes_ = np.array(train_sizes_list)
        self.train_scores_ = np.array(train_scores_list)
        self.val_scores_ = np.array(val_scores_list)

        return self

    def plot_learning_curves(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """Plot learning curves.

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
            title: Plot title.
        """
        if self.train_scores_ is None:
            logger.warning("Learning curves must be fitted before plotting")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        train_mean = np.mean(self.train_scores_, axis=1)
        train_std = np.std(self.train_scores_, axis=1)
        val_mean = np.mean(self.val_scores_, axis=1)
        val_std = np.std(self.val_scores_, axis=1)

        ax.plot(self.train_sizes_, train_mean, "o-", color="blue", label="Training Score")
        ax.fill_between(
            self.train_sizes_,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            color="blue",
        )

        ax.plot(self.train_sizes_, val_mean, "o-", color="red", label="Validation Score")
        ax.fill_between(
            self.train_sizes_,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.2,
            color="red",
        )

        ax.set_xlabel("Training Set Size", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(title or "Learning Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Learning curves plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def get_bias_variance_analysis(self) -> Dict[str, float]:
        """Get bias-variance analysis.

        Returns:
            Dictionary with bias and variance estimates.
        """
        if self.train_scores_ is None:
            raise ValueError("Learning curves must be fitted before analysis")

        train_mean = np.mean(self.train_scores_, axis=1)
        val_mean = np.mean(self.val_scores_, axis=1)

        final_train_score = train_mean[-1]
        final_val_score = val_mean[-1]

        gap = final_train_score - final_val_score

        if gap > 0.1:
            diagnosis = "High Variance (Overfitting)"
        elif gap < -0.1:
            diagnosis = "High Bias (Underfitting)"
        else:
            diagnosis = "Balanced"

        return {
            "final_train_score": float(final_train_score),
            "final_val_score": float(final_val_score),
            "gap": float(gap),
            "diagnosis": diagnosis,
        }


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Model Selection with Nested CV and Learning Curves")
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
        "--nested-cv",
        action="store_true",
        help="Perform nested cross-validation",
    )
    parser.add_argument(
        "--learning-curves",
        action="store_true",
        help="Generate learning curves",
    )
    parser.add_argument(
        "--estimators-config",
        type=str,
        default=None,
        help="Path to JSON file with estimators configuration",
    )
    parser.add_argument(
        "--outer-cv",
        type=int,
        default=None,
        help="Number of outer CV folds (default: from config)",
    )
    parser.add_argument(
        "--inner-cv",
        type=int,
        default=None,
        help="Number of inner CV folds (default: from config)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=None,
        help="Number of CV folds for learning curves (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results CSV",
    )
    parser.add_argument(
        "--plot-learning-curves",
        action="store_true",
        help="Plot learning curves",
    )
    parser.add_argument(
        "--save-learning-curves",
        type=str,
        default=None,
        help="Path to save learning curves plot",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model_selection", {})

        df = pd.read_csv(args.input)
        print(f"\n=== Model Selection ===")
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

        if args.nested_cv:
            outer_cv = (
                args.outer_cv
                if args.outer_cv is not None
                else model_config.get("outer_cv", 5)
            )
            inner_cv = (
                args.inner_cv
                if args.inner_cv is not None
                else model_config.get("inner_cv", 5)
            )

            if args.estimators_config:
                with open(args.estimators_config, "r") as f:
                    estimators_config = json.load(f)
            else:
                raise ValueError("--estimators-config required for nested CV")

            from src.example_estimator import SimpleClassifier, SimpleRegressor

            estimators = {}
            param_grids = {}

            for est_config in estimators_config.get("estimators", []):
                est_name = est_config["name"]
                est_type = est_config["type"]

                if est_type == "SimpleClassifier":
                    estimators[est_name] = SimpleClassifier()
                elif est_type == "SimpleRegressor":
                    estimators[est_name] = SimpleRegressor()
                else:
                    raise ValueError(f"Unknown estimator type: {est_type}")

                param_grids[est_name] = est_config.get("param_grid", {})

            print(f"\n=== Nested Cross-Validation ===")
            print(f"Outer CV: {outer_cv}")
            print(f"Inner CV: {inner_cv}")
            print(f"Estimators: {list(estimators.keys())}")

            nested_cv = NestedCrossValidation(
                estimators=estimators,
                param_grids=param_grids,
                outer_cv=outer_cv,
                inner_cv=inner_cv,
                verbose=1,
            )
            nested_cv.fit(X, y)

            print(f"\n=== Nested CV Results ===")
            print(f"Best estimator: {nested_cv.best_estimator_name_}")
            print(f"Best score: {nested_cv.best_score_:.6f}")

            if args.output:
                results_df = pd.DataFrame(nested_cv.cv_results_)
                results_df.to_csv(args.output, index=False)
                print(f"\nResults saved to: {args.output}")

        if args.learning_curves:
            cv = (
                args.cv if args.cv is not None else model_config.get("cv", 5)
            )

            from src.example_estimator import SimpleClassifier

            estimator = SimpleClassifier()

            print(f"\n=== Learning Curves ===")
            print(f"CV folds: {cv}")

            lc = LearningCurves(estimator=estimator, cv=cv, verbose=1)
            lc.fit(X, y)

            analysis = lc.get_bias_variance_analysis()
            print(f"\n=== Bias-Variance Analysis ===")
            print(f"Final training score: {analysis['final_train_score']:.6f}")
            print(f"Final validation score: {analysis['final_val_score']:.6f}")
            print(f"Gap: {analysis['gap']:.6f}")
            print(f"Diagnosis: {analysis['diagnosis']}")

            if args.plot_learning_curves or args.save_learning_curves:
                lc.plot_learning_curves(
                    save_path=args.save_learning_curves, show=args.plot_learning_curves
                )

    except Exception as e:
        logger.error(f"Error in model selection: {e}")
        raise


if __name__ == "__main__":
    main()
