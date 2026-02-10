"""Feature Selection using Recursive Feature Elimination and Mutual Information.

This module provides implementations of feature selection techniques including
Recursive Feature Elimination (RFE) and Mutual Information scoring for selecting
the most relevant features from datasets.
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MutualInformationSelector:
    """Feature selection using Mutual Information scoring."""

    def __init__(
        self,
        n_features: Optional[int] = None,
        score_threshold: Optional[float] = None,
        discrete_features: Union[bool, List[int]] = "auto",
        random_state: Optional[int] = None,
    ):
        """Initialize Mutual Information selector.

        Args:
            n_features: Number of top features to select (default: None, all)
            score_threshold: Minimum MI score threshold (default: None)
            discrete_features: Whether features are discrete (default: "auto")
            random_state: Random seed for reproducibility (default: None)
        """
        self.n_features = n_features
        self.score_threshold = score_threshold
        self.discrete_features = discrete_features
        self.random_state = random_state
        self.scores_ = None
        self.selected_features_ = None
        self.feature_names_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        task_type: str = "classification",
    ) -> "MutualInformationSelector":
        """Fit Mutual Information selector to data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target vector, shape (n_samples,)
            feature_names: Optional list of feature names (default: None)
            task_type: Task type - "classification" or "regression" (default: "classification")

        Returns:
            Self for method chaining

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

        if task_type not in ["classification", "regression"]:
            raise ValueError(f"task_type must be 'classification' or 'regression'")

        if task_type == "classification":
            mi_scores = mutual_info_classif(
                X,
                y,
                discrete_features=self.discrete_features,
                random_state=self.random_state,
            )
        else:
            mi_scores = mutual_info_regression(
                X,
                y,
                discrete_features=self.discrete_features,
                random_state=self.random_state,
            )

        self.scores_ = mi_scores
        self.feature_names_ = (
            feature_names if feature_names else [f"feature_{i}" for i in range(X.shape[1])]
        )

        if self.n_features is not None:
            top_indices = np.argsort(mi_scores)[-self.n_features:][::-1]
            self.selected_features_ = sorted(top_indices.tolist())
        elif self.score_threshold is not None:
            self.selected_features_ = [
                i for i, score in enumerate(mi_scores) if score >= self.score_threshold
            ]
        else:
            self.selected_features_ = list(range(X.shape[1]))

        logger.info(
            f"Mutual Information selection: {len(self.selected_features_)} features selected "
            f"out of {X.shape[1]}"
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to selected features.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Transformed feature matrix with selected features

        Raises:
            ValueError: If selector is not fitted
        """
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted before transform")

        X = np.array(X)
        return X[:, self.selected_features_]

    def get_feature_scores(self) -> Dict[str, float]:
        """Get Mutual Information scores for all features.

        Returns:
            Dictionary mapping feature names to MI scores

        Raises:
            ValueError: If selector is not fitted
        """
        if self.scores_ is None:
            raise ValueError("Selector must be fitted before getting scores")

        return {
            name: float(score)
            for name, score in zip(self.feature_names_, self.scores_)
        }

    def get_selected_features(self) -> List[str]:
        """Get names of selected features.

        Returns:
            List of selected feature names

        Raises:
            ValueError: If selector is not fitted
        """
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted before getting selected features")

        return [self.feature_names_[i] for i in self.selected_features_]


class RecursiveFeatureElimination:
    """Recursive Feature Elimination (RFE) for feature selection."""

    def __init__(
        self,
        estimator=None,
        n_features_to_select: Optional[int] = None,
        step: int = 1,
        verbose: int = 0,
        importance_getter: Optional[str] = "auto",
    ):
        """Initialize RFE selector.

        Args:
            estimator: Base estimator (default: None, uses RandomForest)
            n_features_to_select: Number of features to select (default: None)
            step: Number of features to remove per iteration (default: 1)
            verbose: Verbosity level (default: 0)
            importance_getter: Method to get feature importance (default: "auto")
        """
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.importance_getter = importance_getter
        self.rfe_ = None
        self.feature_names_ = None
        self.selected_features_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        task_type: str = "classification",
    ) -> "RecursiveFeatureElimination":
        """Fit RFE selector to data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target vector, shape (n_samples,)
            feature_names: Optional list of feature names (default: None)
            task_type: Task type - "classification" or "regression" (default: "classification")

        Returns:
            Self for method chaining

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

        if task_type not in ["classification", "regression"]:
            raise ValueError(f"task_type must be 'classification' or 'regression'")

        if self.estimator is None:
            if task_type == "classification":
                self.estimator = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            else:
                self.estimator = RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                )

        self.feature_names_ = (
            feature_names if feature_names else [f"feature_{i}" for i in range(X.shape[1])]
        )

        if self.n_features_to_select is None:
            self.n_features_to_select = max(1, X.shape[1] // 2)

        self.rfe_ = RFE(
            estimator=self.estimator,
            n_features_to_select=self.n_features_to_select,
            step=self.step,
            verbose=self.verbose,
            importance_getter=self.importance_getter,
        )

        self.rfe_.fit(X, y)

        self.selected_features_ = np.where(self.rfe_.support_)[0].tolist()

        logger.info(
            f"RFE selection: {len(self.selected_features_)} features selected "
            f"out of {X.shape[1]}"
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to selected features.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Transformed feature matrix with selected features

        Raises:
            ValueError: If selector is not fitted
        """
        if self.rfe_ is None:
            raise ValueError("Selector must be fitted before transform")

        X = np.array(X)
        return self.rfe_.transform(X)

    def get_feature_ranking(self) -> Dict[str, int]:
        """Get feature ranking from RFE.

        Returns:
            Dictionary mapping feature names to ranks (lower is better)

        Raises:
            ValueError: If selector is not fitted
        """
        if self.rfe_ is None:
            raise ValueError("Selector must be fitted before getting ranking")

        ranking = self.rfe_.ranking_
        return {
            name: int(rank) for name, rank in zip(self.feature_names_, ranking)
        }

    def get_selected_features(self) -> List[str]:
        """Get names of selected features.

        Returns:
            List of selected feature names

        Raises:
            ValueError: If selector is not fitted
        """
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted before getting selected features")

        return [self.feature_names_[i] for i in self.selected_features_]


class FeatureSelector:
    """Main feature selection class combining RFE and Mutual Information."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize feature selector.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.mi_selector = None
        self.rfe_selector = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.task_type = None

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

    def load_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        task_type: str = "classification",
    ) -> "FeatureSelector":
        """Load data for feature selection.

        Args:
            X: Feature matrix
            y: Target vector
            task_type: Task type - "classification" or "regression" (default: "classification")

        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            X = np.array(X)

        if isinstance(y, pd.Series):
            y = y.values
        else:
            y = np.array(y)

        self.X = X
        self.y = y
        self.task_type = task_type

        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        logger.info(
            f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features, "
            f"task type: {task_type}"
        )

        return self

    def fit_mutual_information(
        self, n_features: Optional[int] = None, **kwargs
    ) -> "FeatureSelector":
        """Fit Mutual Information selector.

        Args:
            n_features: Number of features to select (default: from config)
            **kwargs: Additional arguments for MutualInformationSelector

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is not loaded
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded before fitting")

        mi_config = self.config.get("mutual_information", {})
        n_features = n_features or mi_config.get("n_features", None)
        score_threshold = kwargs.get(
            "score_threshold", mi_config.get("score_threshold", None)
        )

        self.mi_selector = MutualInformationSelector(
            n_features=n_features,
            score_threshold=score_threshold,
            random_state=mi_config.get("random_state", None),
            **{k: v for k, v in kwargs.items() if k != "score_threshold"},
        )
        self.mi_selector.fit(
            self.X, self.y, feature_names=self.feature_names, task_type=self.task_type
        )

        return self

    def fit_rfe(
        self, n_features_to_select: Optional[int] = None, **kwargs
    ) -> "FeatureSelector":
        """Fit RFE selector.

        Args:
            n_features_to_select: Number of features to select (default: from config)
            **kwargs: Additional arguments for RecursiveFeatureElimination

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is not loaded
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded before fitting")

        rfe_config = self.config.get("rfe", {})
        n_features_to_select = n_features_to_select or rfe_config.get(
            "n_features_to_select", None
        )
        step = kwargs.get("step", rfe_config.get("step", 1))

        self.rfe_selector = RecursiveFeatureElimination(
            n_features_to_select=n_features_to_select,
            step=step,
            verbose=rfe_config.get("verbose", 0),
            **{k: v for k, v in kwargs.items() if k not in ["step"]},
        )
        self.rfe_selector.fit(
            self.X, self.y, feature_names=self.feature_names, task_type=self.task_type
        )

        return self

    def fit_all(self) -> "FeatureSelector":
        """Fit both selectors.

        Returns:
            Self for method chaining
        """
        self.fit_mutual_information()
        self.fit_rfe()

        return self

    def get_selected_features(
        self, method: str = "mutual_information"
    ) -> List[str]:
        """Get selected features from specified method.

        Args:
            method: Selection method - "mutual_information" or "rfe" (default: "mutual_information")

        Returns:
            List of selected feature names

        Raises:
            ValueError: If method is invalid or not fitted
        """
        if method == "mutual_information":
            if self.mi_selector is None:
                raise ValueError("Mutual Information selector must be fitted first")
            return self.mi_selector.get_selected_features()
        elif method == "rfe":
            if self.rfe_selector is None:
                raise ValueError("RFE selector must be fitted first")
            return self.rfe_selector.get_selected_features()
        else:
            raise ValueError(f"Unknown method: {method}")

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame], method: str = "mutual_information"
    ) -> np.ndarray:
        """Transform data using selected features.

        Args:
            X: Feature matrix
            method: Selection method - "mutual_information" or "rfe" (default: "mutual_information")

        Returns:
            Transformed feature matrix

        Raises:
            ValueError: If method is invalid or not fitted
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if method == "mutual_information":
            if self.mi_selector is None:
                raise ValueError("Mutual Information selector must be fitted first")
            return self.mi_selector.transform(X)
        elif method == "rfe":
            if self.rfe_selector is None:
                raise ValueError("RFE selector must be fitted first")
            return self.rfe_selector.transform(X)
        else:
            raise ValueError(f"Unknown method: {method}")

    def evaluate_selection(
        self,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        method: str = "mutual_information",
        metric: Optional[str] = None,
    ) -> Dict:
        """Evaluate feature selection performance.

        Args:
            X_test: Test feature matrix (default: None, uses training data)
            y_test: Test target vector (default: None, uses training data)
            method: Selection method (default: "mutual_information")
            metric: Evaluation metric (default: None, auto-selects)

        Returns:
            Dictionary with evaluation results
        """
        if X_test is None:
            X_test = self.X
            y_test = self.y
        else:
            X_test = np.array(X_test)
            y_test = np.array(y_test)

        X_selected = self.transform(X_test, method=method)

        if self.task_type == "classification":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_selected, y_test)
            y_pred = model.predict(X_selected)

            if metric is None or metric == "accuracy":
                score = accuracy_score(y_test, y_pred)
                return {"accuracy": float(score), "n_features": X_selected.shape[1]}
        else:
            from sklearn.linear_model import Ridge

            model = Ridge(random_state=42)
            model.fit(X_selected, y_test)
            y_pred = model.predict(X_selected)

            if metric is None or metric == "r2":
                score = r2_score(y_test, y_pred)
                return {"r2_score": float(score), "n_features": X_selected.shape[1]}
            elif metric == "rmse":
                score = np.sqrt(mean_squared_error(y_test, y_pred))
                return {"rmse": float(score), "n_features": X_selected.shape[1]}

        return {}


def main():
    """Main entry point for feature selector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Feature selection using RFE and Mutual Information"
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
        "--task-type",
        type=str,
        choices=["classification", "regression"],
        default="classification",
        help="Task type: classification or regression (default: classification)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mutual_information", "rfe", "both"],
        default="both",
        help="Feature selection method (default: both)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        help="Number of features to select",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output CSV file with selected features",
    )
    parser.add_argument(
        "--scores-output",
        type=str,
        help="Path to output JSON file with feature scores",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    selector = FeatureSelector(
        config_path=Path(args.config) if args.config else None
    )

    df = pd.read_csv(args.input)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in dataset")

    feature_cols = [col for col in df.columns if col != args.target_col]
    X = df[feature_cols]
    y = df[args.target_col]

    selector.load_data(X, y, task_type=args.task_type)

    if args.method in ["mutual_information", "both"]:
        selector.fit_mutual_information(n_features=args.n_features)
    if args.method in ["rfe", "both"]:
        selector.fit_rfe(n_features_to_select=args.n_features)

    results = {}

    if args.method in ["mutual_information", "both"]:
        mi_features = selector.get_selected_features("mutual_information")
        mi_scores = selector.mi_selector.get_feature_scores()
        results["mutual_information"] = {
            "selected_features": mi_features,
            "n_selected": len(mi_features),
            "scores": mi_scores,
        }
        print("\nMutual Information Selection:")
        print("=" * 50)
        print(f"Selected {len(mi_features)} features:")
        for feature in mi_features:
            print(f"  {feature}: {mi_scores[feature]:.4f}")

    if args.method in ["rfe", "both"]:
        rfe_features = selector.get_selected_features("rfe")
        rfe_ranking = selector.rfe_selector.get_feature_ranking()
        results["rfe"] = {
            "selected_features": rfe_features,
            "n_selected": len(rfe_features),
            "ranking": rfe_ranking,
        }
        print("\nRFE Selection:")
        print("=" * 50)
        print(f"Selected {len(rfe_features)} features:")
        for feature in rfe_features:
            print(f"  {feature}: rank {rfe_ranking[feature]}")

    if args.output:
        if args.method == "mutual_information" or (
            args.method == "both" and "mutual_information" in results
        ):
            selected_features = results["mutual_information"]["selected_features"]
        else:
            selected_features = results["rfe"]["selected_features"]

        output_df = df[selected_features + [args.target_col]]
        output_df.to_csv(args.output, index=False)
        logger.info(f"Selected features saved to {args.output}")

    if args.scores_output:
        with open(args.scores_output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Feature scores saved to {args.scores_output}")


if __name__ == "__main__":
    main()
