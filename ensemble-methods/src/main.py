"""Ensemble Methods: Voting, Bagging, and Stacking.

This module provides functionality to implement ensemble methods from scratch
including voting (hard and soft), bagging, and stacking with multiple base models.
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


class BaseEstimator:
    """Base class for estimators."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEstimator":
        """Fit the estimator."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        raise NotImplementedError


class SimpleDecisionTree(BaseEstimator):
    """Simple Decision Tree Classifier."""

    def __init__(self, max_depth: int = 3, min_samples_split: int = 2) -> None:
        """Initialize Decision Tree.

        Args:
            max_depth: Maximum depth.
            min_samples_split: Minimum samples to split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.classes_ = None

    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0.0
        unique, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1.0 - np.sum(proportions ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Find best split."""
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_gini = self._gini_impurity(y[left_mask])
                right_gini = self._gini_impurity(y[right_mask])

                weighted_gini = (
                    np.sum(left_mask) / len(y) * left_gini
                    + np.sum(right_mask) / len(y) * right_gini
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> Dict:
        """Build decision tree recursively."""
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            unique, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "class": unique[np.argmax(counts)]}

        if len(np.unique(y)) == 1:
            return {"leaf": True, "class": y[0]}

        feature, threshold = self._best_split(X, y)

        if feature is None:
            unique, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "class": unique[np.argmax(counts)]}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleDecisionTree":
        """Fit decision tree."""
        self.classes_ = np.unique(y)
        self.tree = self._build_tree(X, y)
        return self

    def _predict_sample(self, x: np.ndarray, node: Dict) -> int:
        """Predict single sample."""
        if node["leaf"]:
            return node["class"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_sample(x, node["left"])
        else:
            return self._predict_sample(x, node["right"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (simplified)."""
        predictions = self.predict(X)
        proba = np.zeros((len(X), len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            proba[:, i] = (predictions == cls).astype(float)
        return proba


class SimpleKNN(BaseEstimator):
    """Simple K-Nearest Neighbors Classifier."""

    def __init__(self, n_neighbors: int = 5) -> None:
        """Initialize KNN.

        Args:
            n_neighbors: Number of neighbors.
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleKNN":
        """Fit KNN."""
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[: self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            unique, counts = np.unique(nearest_labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        proba = np.zeros((len(X), len(self.classes_)))
        for i, x in enumerate(X):
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[: self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            for j, cls in enumerate(self.classes_):
                proba[i, j] = np.mean(nearest_labels == cls)
        return proba


class VotingClassifier:
    """Voting Classifier with hard and soft voting."""

    def __init__(
        self, estimators: List[Tuple[str, BaseEstimator]], voting: str = "hard"
    ) -> None:
        """Initialize Voting Classifier.

        Args:
            estimators: List of (name, estimator) tuples.
            voting: "hard" or "soft" voting (default: "hard").
        """
        self.estimators = [est[1] for est in estimators]
        self.estimator_names = [est[0] for est in estimators]
        self.voting = voting
        self.classes_ = None
        self.feature_names_ = None

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> "VotingClassifier":
        """Fit all estimators.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Self for method chaining.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.classes_ = np.unique(y)

        for estimator in self.estimators:
            estimator.fit(X, y)

        logger.info(f"Voting classifier fitted with {len(self.estimators)} estimators")
        return self

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict using voting.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.voting == "hard":
            predictions = np.array([est.predict(X) for est in self.estimators])
            final_predictions = []
            for i in range(len(X)):
                votes = predictions[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                final_predictions.append(unique[np.argmax(counts)])
            return np.array(final_predictions)
        else:
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Class probabilities.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.voting == "hard":
            predictions = np.array([est.predict(X) for est in self.estimators])
            proba = np.zeros((len(X), len(self.classes_)))
            for i in range(len(X)):
                votes = predictions[:, i]
                for j, cls in enumerate(self.classes_):
                    proba[i, j] = np.mean(votes == cls)
            return proba
        else:
            proba_list = [est.predict_proba(X) for est in self.estimators]
            avg_proba = np.mean(proba_list, axis=0)
            return avg_proba

    def score(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> float:
        """Calculate classification accuracy.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Classification accuracy.
        """
        predictions = self.predict(X)
        y = np.asarray(y, dtype=int)
        return np.mean(predictions == y)


class BaggingClassifier:
    """Bagging Classifier with bootstrap sampling."""

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Bagging Classifier.

        Args:
            base_estimator: Base estimator to use.
            n_estimators: Number of estimators (default: 10).
            max_samples: Fraction of samples for each estimator (default: 1.0).
            max_features: Fraction of features for each estimator (default: 1.0).
            random_state: Random seed (default: None).
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

        self.estimators_ = []
        self.classes_ = None
        self.feature_names_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> "BaggingClassifier":
        """Fit bagging classifier.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Self for method chaining.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        self.estimators_ = []

        for i in range(self.n_estimators):
            if self.random_state is not None:
                np.random.seed(self.random_state + i)

            n_samples_bootstrap = int(self.max_samples * n_samples)
            sample_indices = np.random.choice(
                n_samples, n_samples_bootstrap, replace=True
            )

            n_features_bootstrap = int(self.max_features * n_features)
            feature_indices = np.random.choice(
                n_features, n_features_bootstrap, replace=False
            )

            X_bootstrap = X[np.ix_(sample_indices, feature_indices)]
            y_bootstrap = y[sample_indices]

            estimator = type(self.base_estimator)(**self._get_estimator_params())
            estimator.fit(X_bootstrap, y_bootstrap)
            estimator.feature_indices_ = feature_indices

            self.estimators_.append(estimator)

        logger.info(
            f"Bagging classifier fitted with {len(self.estimators_)} estimators"
        )
        return self

    def _get_estimator_params(self) -> Dict:
        """Get parameters for new estimator."""
        if isinstance(self.base_estimator, SimpleDecisionTree):
            return {
                "max_depth": self.base_estimator.max_depth,
                "min_samples_split": self.base_estimator.min_samples_split,
            }
        elif isinstance(self.base_estimator, SimpleKNN):
            return {"n_neighbors": self.base_estimator.n_neighbors}
        return {}

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict using majority voting.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = []
        for estimator in self.estimators_:
            X_subset = X[:, estimator.feature_indices_]
            pred = estimator.predict(X_subset)
            predictions.append(pred)

        predictions = np.array(predictions)
        final_predictions = []
        for i in range(len(X)):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])

        return np.array(final_predictions)

    def predict_proba(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Class probabilities.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        proba_list = []
        for estimator in self.estimators_:
            X_subset = X[:, estimator.feature_indices_]
            proba = estimator.predict_proba(X_subset)
            proba_list.append(proba)

        avg_proba = np.mean(proba_list, axis=0)
        return avg_proba

    def score(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> float:
        """Calculate classification accuracy.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Classification accuracy.
        """
        predictions = self.predict(X)
        y = np.asarray(y, dtype=int)
        return np.mean(predictions == y)


class StackingClassifier:
    """Stacking Classifier with meta-learner."""

    def __init__(
        self,
        base_estimators: List[Tuple[str, BaseEstimator]],
        meta_estimator: BaseEstimator,
        cv: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Stacking Classifier.

        Args:
            base_estimators: List of (name, estimator) tuples.
            meta_estimator: Meta-learner for final prediction.
            cv: Number of cross-validation folds (default: 5).
            random_state: Random seed (default: None).
        """
        self.base_estimators = [est[1] for est in base_estimators]
        self.estimator_names = [est[0] for est in base_estimators]
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.random_state = random_state

        self.classes_ = None
        self.feature_names_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _cross_val_predict(
        self, X: np.ndarray, y: np.ndarray, estimator: BaseEstimator
    ) -> np.ndarray:
        """Get cross-validation predictions for stacking.

        Args:
            X: Feature matrix.
            y: Target labels.
            estimator: Base estimator.

        Returns:
            Cross-validation predictions.
        """
        n_samples = len(X)
        fold_size = n_samples // self.cv
        predictions = np.zeros(n_samples)

        indices = np.arange(n_samples)
        if self.random_state is not None:
            np.random.shuffle(indices)

        for fold in range(self.cv):
            start = fold * fold_size
            end = start + fold_size if fold < self.cv - 1 else n_samples

            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]

            estimator_fold = type(estimator)(**self._get_estimator_params(estimator))
            estimator_fold.fit(X_train_fold, y_train_fold)

            fold_predictions = estimator_fold.predict(X_val_fold)
            predictions[val_indices] = fold_predictions

        return predictions

    def _get_estimator_params(self, estimator: BaseEstimator) -> Dict:
        """Get parameters for new estimator."""
        if isinstance(estimator, SimpleDecisionTree):
            return {
                "max_depth": estimator.max_depth,
                "min_samples_split": estimator.min_samples_split,
            }
        elif isinstance(estimator, SimpleKNN):
            return {"n_neighbors": estimator.n_neighbors}
        return {}

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> "StackingClassifier":
        """Fit stacking classifier.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Self for method chaining.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.classes_ = np.unique(y)

        meta_features = []
        for estimator in self.base_estimators:
            cv_predictions = self._cross_val_predict(X, y, estimator)
            meta_features.append(cv_predictions)

        meta_X = np.column_stack(meta_features)

        self.meta_estimator.fit(meta_X, y)

        for estimator in self.base_estimators:
            estimator.fit(X, y)

        logger.info(
            f"Stacking classifier fitted with {len(self.base_estimators)} base estimators"
        )
        return self

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict using stacking.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        meta_features = []
        for estimator in self.base_estimators:
            predictions = estimator.predict(X)
            meta_features.append(predictions)

        meta_X = np.column_stack(meta_features)
        return self.meta_estimator.predict(meta_X)

    def predict_proba(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Class probabilities.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        meta_features = []
        for estimator in self.base_estimators:
            predictions = estimator.predict(X)
            meta_features.append(predictions)

        meta_X = np.column_stack(meta_features)
        return self.meta_estimator.predict_proba(meta_X)

    def score(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> float:
        """Calculate classification accuracy.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Classification accuracy.
        """
        predictions = self.predict(X)
        y = np.asarray(y, dtype=int)
        return np.mean(predictions == y)


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble Methods")
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
        choices=["voting", "bagging", "stacking"],
        required=True,
        help="Ensemble method to use",
    )
    parser.add_argument(
        "--voting",
        type=str,
        choices=["hard", "soft"],
        default="hard",
        help="Voting type for voting classifier (default: hard)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Number of estimators for bagging (default: from config)",
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
    parser.add_argument(
        "--output-predictions",
        type=str,
        default=None,
        help="Path to save predictions as CSV",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})

        df = pd.read_csv(args.input)
        print(f"\n=== Ensemble Methods ===")
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
        print(f"Classes: {len(np.unique(y))}")
        print(f"Method: {args.method}")

        if args.method == "voting":
            dt = SimpleDecisionTree(max_depth=3)
            knn = SimpleKNN(n_neighbors=5)
            dt2 = SimpleDecisionTree(max_depth=5)

            ensemble = VotingClassifier(
                estimators=[
                    ("dt1", dt),
                    ("knn", knn),
                    ("dt2", dt2),
                ],
                voting=args.voting,
            )
            ensemble.feature_names_ = feature_cols
            ensemble.fit(X, y)

            print(f"\n=== Voting Classifier Results ===")
            print(f"Voting type: {args.voting}")
            print(f"Number of base estimators: {len(ensemble.estimators)}")

        elif args.method == "bagging":
            n_estimators = (
                args.n_estimators
                if args.n_estimators is not None
                else model_config.get("n_estimators", 10)
            )

            base_estimator = SimpleDecisionTree(max_depth=3)
            ensemble = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                max_samples=0.8,
                max_features=0.8,
            )
            ensemble.feature_names_ = feature_cols
            ensemble.fit(X, y)

            print(f"\n=== Bagging Classifier Results ===")
            print(f"Number of estimators: {len(ensemble.estimators_)}")
            print(f"Max samples: {ensemble.max_samples}")
            print(f"Max features: {ensemble.max_features}")

        elif args.method == "stacking":
            dt = SimpleDecisionTree(max_depth=3)
            knn = SimpleKNN(n_neighbors=5)
            meta_estimator = SimpleDecisionTree(max_depth=2)

            ensemble = StackingClassifier(
                base_estimators=[
                    ("dt", dt),
                    ("knn", knn),
                ],
                meta_estimator=meta_estimator,
                cv=5,
            )
            ensemble.feature_names_ = feature_cols
            ensemble.fit(X, y)

            print(f"\n=== Stacking Classifier Results ===")
            print(f"Number of base estimators: {len(ensemble.base_estimators)}")
            print(f"Cross-validation folds: {ensemble.cv}")

        accuracy = ensemble.score(X, y)
        print(f"\nTraining accuracy: {accuracy:.6f}")

        if args.output:
            predictions = ensemble.predict(X)
            proba = ensemble.predict_proba(X)
            output_df = pd.DataFrame({
                "prediction": predictions,
                args.target: y,
            })
            for i, cls in enumerate(ensemble.classes_):
                output_df[f"prob_class_{cls}"] = proba[:, i]
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

        if args.predict:
            df_predict = pd.read_csv(args.predict)
            X_predict = df_predict[feature_cols].values
            predictions = ensemble.predict(X_predict)
            proba = ensemble.predict_proba(X_predict)

            if args.output_predictions:
                output_df = pd.DataFrame({
                    "prediction": predictions,
                })
                for i, cls in enumerate(ensemble.classes_):
                    output_df[f"prob_class_{cls}"] = proba[:, i]
                output_df.to_csv(args.output_predictions, index=False)
                print(f"Predictions saved to: {args.output_predictions}")
            else:
                print(f"\nPredictions:")
                for i, pred in enumerate(predictions[:10]):
                    print(f"  Sample {i+1}: {pred}")
                if len(predictions) > 10:
                    print(f"  ... and {len(predictions) - 10} more")

    except Exception as e:
        logger.error(f"Error running ensemble method: {e}")
        raise


if __name__ == "__main__":
    main()
