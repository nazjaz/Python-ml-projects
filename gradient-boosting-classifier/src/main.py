"""Gradient Boosting Classifier with Learning Rate and Optimization.

This module provides functionality to implement Gradient Boosting Classifier
from scratch with learning rate, depth, and tree count optimization.
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


class TreeNode:
    """Node in decision tree."""

    def __init__(self) -> None:
        """Initialize tree node."""
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None
        self.value: Optional[float] = None
        self.is_leaf: bool = False
        self.samples: int = 0


class DecisionTreeRegressor:
    """Simple Decision Tree Regressor for gradient boosting."""

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Decision Tree Regressor.

        Args:
            max_depth: Maximum depth of tree.
            min_samples_split: Minimum samples required to split node.
            min_samples_leaf: Minimum samples required at leaf node.
            random_state: Random seed.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.root: Optional[TreeNode] = None
        self.n_features_: Optional[int] = None

        if random_state is not None:
            np.random.seed(random_state)

    def _mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error.

        Args:
            y: Target values.

        Returns:
            MSE value.
        """
        if len(y) == 0:
            return 0.0
        mean = np.mean(y)
        mse = np.mean((y - mean) ** 2)
        return mse

    def _variance_reduction(
        self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
    ) -> float:
        """Calculate variance reduction.

        Args:
            y_parent: Parent node target values.
            y_left: Left child target values.
            y_right: Right child target values.

        Returns:
            Variance reduction.
        """
        parent_mse = self._mse(y_parent)
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        if n == 0:
            return 0.0

        left_mse = self._mse(y_left) if n_left > 0 else 0.0
        right_mse = self._mse(y_right) if n_right > 0 else 0.0

        weighted_mse = (n_left / n) * left_mse + (n_right / n) * right_mse
        variance_reduction = parent_mse - weighted_mse

        return variance_reduction

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find best split for node.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Tuple of (best_feature_index, best_threshold, best_reduction).
        """
        n_samples, n_features = X.shape
        best_reduction = 0.0
        best_feature = None
        best_threshold = None

        if n_samples < self.min_samples_split:
            return None, None, 0.0

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                reduction = self._variance_reduction(y, y_left, y_right)

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_reduction

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> TreeNode:
        """Build decision tree recursively.

        Args:
            X: Feature matrix.
            y: Target values.
            depth: Current depth.

        Returns:
            Root node of subtree.
        """
        node = TreeNode()
        node.samples = len(y)

        if len(y) == 0:
            node.is_leaf = True
            node.value = 0.0
            return node

        node.value = np.mean(y)

        if len(y) == 1:
            node.is_leaf = True
            return node

        if depth >= self.max_depth:
            node.is_leaf = True
            return node

        if len(y) < self.min_samples_split:
            node.is_leaf = True
            return node

        best_feature, best_threshold, best_reduction = self._best_split(X, y)

        if best_feature is None or best_reduction <= 0:
            node.is_leaf = True
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        """Fit decision tree regressor.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self for method chaining.
        """
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y)
        return self

    def _predict_sample(self, x: np.ndarray, node: Optional[TreeNode] = None) -> float:
        """Predict single sample by traversing tree.

        Args:
            x: Single feature vector.
            node: Current node (default: root).

        Returns:
            Predicted value.
        """
        if node is None:
            node = self.root

        if node.is_leaf:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.
        """
        predictions = np.array([self._predict_sample(x) for x in X])
        return predictions


class GradientBoostingClassifier:
    """Gradient Boosting Classifier with learning rate and optimization."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Gradient Boosting Classifier.

        Args:
            n_estimators: Number of boosting stages (trees) (default: 100).
            learning_rate: Learning rate (shrinkage) (default: 0.1).
            max_depth: Maximum depth of trees (default: 3).
            min_samples_split: Minimum samples required to split node (default: 2).
            min_samples_leaf: Minimum samples required at leaf node (default: 1).
            subsample: Fraction of samples to use for each tree (default: 1.0).
            random_state: Random seed (default: None).
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

        self.estimators_: List[DecisionTreeRegressor] = []
        self.init_score_: Optional[float] = None
        self.n_features_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

        if random_state is not None:
            np.random.seed(random_state)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid function.

        Args:
            x: Input values.

        Returns:
            Sigmoid values.
        """
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _log_loss_gradient(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate negative gradient of log loss.

        Args:
            y_true: True binary labels (0 or 1).
            y_pred: Predicted probabilities.

        Returns:
            Negative gradient (residuals).
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        residuals = y_true - y_pred
        return residuals

    def _initial_prediction(self, y: np.ndarray) -> float:
        """Calculate initial prediction (log odds).

        Args:
            y: Target labels.

        Returns:
            Initial log odds.
        """
        if len(y) == 0:
            return 0.0

        positive_ratio = np.mean(y)
        if positive_ratio == 0:
            return -10.0
        elif positive_ratio == 1:
            return 10.0
        else:
            return np.log(positive_ratio / (1 - positive_ratio))

    def _subsample_indices(self, n_samples: int) -> np.ndarray:
        """Get subsample indices.

        Args:
            n_samples: Total number of samples.

        Returns:
            Indices for subsample.
        """
        if self.subsample >= 1.0:
            return np.arange(n_samples)

        n_sub = int(self.subsample * n_samples)
        return np.random.choice(n_samples, n_sub, replace=False)

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> "GradientBoostingClassifier":
        """Fit gradient boosting classifier.

        Args:
            X: Feature matrix.
            y: Target labels (binary: 0, 1).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("Gradient boosting currently supports binary classification only")

        self.n_features_ = X.shape[1]
        self.classes_ = unique_classes

        y_binary = (y == unique_classes[1]).astype(int)

        self.init_score_ = self._initial_prediction(y_binary)
        predictions = np.full(len(y_binary), self.init_score_)

        self.estimators_ = []

        for i in range(self.n_estimators):
            if self.random_state is not None:
                tree_seed = self.random_state + i
            else:
                tree_seed = None

            probabilities = self._sigmoid(predictions)
            residuals = self._log_loss_gradient(y_binary, probabilities)

            indices = self._subsample_indices(len(X))
            X_sub = X[indices]
            residuals_sub = residuals[indices]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed,
            )
            tree.fit(X_sub, residuals_sub)

            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions

            self.estimators_.append(tree)

        self._calculate_feature_importance()

        logger.info(
            f"Gradient boosting fitted: n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, max_depth={self.max_depth}"
        )

        return self

    def _calculate_feature_importance(self) -> None:
        """Calculate feature importance based on variance reduction."""
        if not self.estimators_:
            return

        importances = np.zeros(self.n_features_)

        for tree in self.estimators_:
            if tree.root is not None:
                tree_importances = self._tree_importance(tree.root)
                importances += tree_importances

        importances /= len(self.estimators_)

        total = np.sum(importances)
        if total > 0:
            importances = importances / total

        self.feature_importances_ = importances

    def _tree_importance(self, node: Optional[TreeNode]) -> np.ndarray:
        """Calculate feature importance for a single tree.

        Args:
            node: Current node.

        Returns:
            Feature importance array.
        """
        importances = np.zeros(self.n_features_)

        if node is None or node.is_leaf:
            return importances

        if node.feature_index is not None:
            importances[node.feature_index] += node.samples

        if node.left is not None:
            importances += self._tree_importance(node.left)

        if node.right is not None:
            importances += self._tree_importance(node.right)

        return importances

    def predict_proba(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Class probabilities.

        Raises:
            ValueError: If model not fitted.
        """
        if not self.estimators_:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = np.full(len(X), self.init_score_)

        for tree in self.estimators_:
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions

        probabilities = self._sigmoid(predictions)
        proba = np.column_stack([1 - probabilities, probabilities])

        return proba

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model not fitted.
        """
        proba = self.predict_proba(X)
        predictions = np.where(proba[:, 1] >= 0.5, self.classes_[1], self.classes_[0])
        return predictions

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
        accuracy = np.mean(predictions == y)
        return accuracy

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importance as dictionary.

        Returns:
            Dictionary mapping feature names to importance values.

        Raises:
            ValueError: If model not fitted.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted before getting feature importance")

        if self.feature_names_ is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features_)]
        else:
            feature_names = self.feature_names_

        return dict(zip(feature_names, self.feature_importances_))

    def plot_feature_importance(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        top_n: Optional[int] = None,
    ) -> None:
        """Plot feature importance.

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
            top_n: Number of top features to show (default: all).
        """
        if self.feature_importances_ is None:
            logger.warning("Model must be fitted before plotting feature importance")
            return

        if self.feature_names_ is None:
            feature_names = [f"Feature {i}" for i in range(self.n_features_)]
        else:
            feature_names = self.feature_names_

        importances = self.feature_importances_
        indices = np.argsort(importances)[::-1]

        if top_n is not None:
            indices = indices[:top_n]

        sorted_importances = importances[indices]
        sorted_names = [feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(sorted_names)), sorted_importances)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(
            f"Gradient Boosting Feature Importance (n_estimators={self.n_estimators})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Gradient Boosting Classifier")
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
        "--n-estimators",
        type=int,
        default=None,
        help="Number of trees (default: from config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: from config)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth (default: from config)",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=None,
        help="Minimum samples to split (default: from config)",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=None,
        help="Minimum samples at leaf (default: from config)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="Subsample fraction (default: from config)",
    )
    parser.add_argument(
        "--plot-importance",
        action="store_true",
        help="Plot feature importance",
    )
    parser.add_argument(
        "--save-importance-plot",
        type=str,
        default=None,
        help="Path to save feature importance plot",
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

        n_estimators = (
            args.n_estimators
            if args.n_estimators is not None
            else model_config.get("n_estimators", 100)
        )
        learning_rate = (
            args.learning_rate
            if args.learning_rate is not None
            else model_config.get("learning_rate", 0.1)
        )
        max_depth = (
            args.max_depth
            if args.max_depth is not None
            else model_config.get("max_depth", 3)
        )
        min_samples_split = (
            args.min_samples_split
            if args.min_samples_split is not None
            else model_config.get("min_samples_split", 2)
        )
        min_samples_leaf = (
            args.min_samples_leaf
            if args.min_samples_leaf is not None
            else model_config.get("min_samples_leaf", 1)
        )
        subsample = (
            args.subsample
            if args.subsample is not None
            else model_config.get("subsample", 1.0)
        )

        df = pd.read_csv(args.input)
        print(f"\n=== Gradient Boosting Classifier ===")
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

        print(f"\nFitting Gradient Boosting...")
        print(f"Number of trees: {n_estimators}")
        print(f"Learning rate: {learning_rate}")
        print(f"Max depth: {max_depth}")
        print(f"Min samples split: {min_samples_split}")
        print(f"Min samples leaf: {min_samples_leaf}")
        print(f"Subsample: {subsample}")

        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
        )
        gb.feature_names_ = feature_cols
        gb.fit(X, y)

        print(f"\n=== Gradient Boosting Results ===")
        print(f"Number of trees: {len(gb.estimators_)}")
        print(f"Classes: {gb.classes_}")

        accuracy = gb.score(X, y)
        print(f"\nTraining accuracy: {accuracy:.6f}")

        importances = gb.get_feature_importances()
        print(f"\nFeature Importance:")
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_importances[:10]:
            print(f"  {name}: {importance:.6f}")
        if len(sorted_importances) > 10:
            print(f"  ... and {len(sorted_importances) - 10} more")

        if args.plot_importance or args.save_importance_plot:
            gb.plot_feature_importance(
                save_path=args.save_importance_plot, show=args.plot_importance
            )

        if args.output:
            predictions = gb.predict(X)
            proba = gb.predict_proba(X)
            output_df = pd.DataFrame({
                "prediction": predictions,
                args.target: y,
            })
            for i, cls in enumerate(gb.classes_):
                output_df[f"prob_class_{cls}"] = proba[:, i]
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

        if args.predict:
            df_predict = pd.read_csv(args.predict)
            X_predict = df_predict[feature_cols].values
            predictions = gb.predict(X_predict)
            proba = gb.predict_proba(X_predict)

            if args.output_predictions:
                output_df = pd.DataFrame({
                    "prediction": predictions,
                })
                for i, cls in enumerate(gb.classes_):
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
        logger.error(f"Error running gradient boosting: {e}")
        raise


if __name__ == "__main__":
    main()
