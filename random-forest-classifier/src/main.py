"""Random Forest Classifier with Bootstrap Sampling and Feature Importance.

This module provides functionality to implement Random Forest Classifier from
scratch with bootstrap sampling and feature importance calculation.
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
        self.value: Optional[Union[int, float]] = None
        self.is_leaf: bool = False
        self.samples: int = 0
        self.impurity: float = 0.0


class DecisionTreeClassifier:
    """Simple Decision Tree Classifier for use in Random Forest."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Decision Tree Classifier.

        Args:
            max_depth: Maximum depth of tree.
            min_samples_split: Minimum samples required to split node.
            min_samples_leaf: Minimum samples required at leaf node.
            max_features: Maximum features to consider for split.
            random_state: Random seed.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.root: Optional[TreeNode] = None
        self.n_features_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None

        if random_state is not None:
            np.random.seed(random_state)

    def _gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity.

        Args:
            y: Target labels.

        Returns:
            Gini impurity value.
        """
        if len(y) == 0:
            return 0.0

        counts = np.bincount(y)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini

    def _information_gain(
        self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
    ) -> float:
        """Calculate information gain.

        Args:
            y_parent: Parent node labels.
            y_left: Left child labels.
            y_right: Right child labels.

        Returns:
            Information gain.
        """
        parent_impurity = self._gini(y_parent)
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        if n == 0:
            return 0.0

        left_impurity = self._gini(y_left) if n_left > 0 else 0.0
        right_impurity = self._gini(y_right) if n_right > 0 else 0.0

        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        information_gain = parent_impurity - weighted_impurity

        return information_gain

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find best split for node.

        Args:
            X: Feature matrix.
            y: Target labels.
            feature_indices: Indices of features to consider.

        Returns:
            Tuple of (best_feature_index, best_threshold, best_gain).
        """
        n_samples, _ = X.shape
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        if n_samples < self.min_samples_split:
            return None, None, 0.0

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> TreeNode:
        """Build decision tree recursively.

        Args:
            X: Feature matrix.
            y: Target labels.
            depth: Current depth.

        Returns:
            Root node of subtree.
        """
        node = TreeNode()
        node.samples = len(y)
        node.impurity = self._gini(y)

        if len(y) == 0:
            node.is_leaf = True
            node.value = 0
            return node

        most_common = np.bincount(y).argmax()
        node.value = most_common

        if len(np.unique(y)) == 1:
            node.is_leaf = True
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            return node

        if len(y) < self.min_samples_split:
            node.is_leaf = True
            return node

        if self.max_features is not None:
            n_features_to_consider = min(self.max_features, self.n_features_)
            feature_indices = np.random.choice(
                self.n_features_, n_features_to_consider, replace=False
            )
        else:
            feature_indices = np.arange(self.n_features_)

        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)

        if best_feature is None or best_gain <= 0:
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

    def _calculate_feature_importance(self, node: Optional[TreeNode] = None) -> np.ndarray:
        """Calculate feature importance based on information gain.

        Args:
            node: Current node (default: root).

        Returns:
            Feature importance array.
        """
        if node is None:
            node = self.root

        importances = np.zeros(self.n_features_)

        if node.is_leaf:
            return importances

        if node.feature_index is not None:
            importances[node.feature_index] += node.samples * node.impurity

        if node.left is not None:
            importances += self._calculate_feature_importance(node.left)

        if node.right is not None:
            importances += self._calculate_feature_importance(node.right)

        return importances

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Fit decision tree classifier.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Self for method chaining.
        """
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y)
        self.feature_importances_ = self._calculate_feature_importance()
        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total
        return self

    def _predict_sample(self, x: np.ndarray, node: Optional[TreeNode] = None) -> int:
        """Predict single sample by traversing tree.

        Args:
            x: Single feature vector.
            node: Current node (default: root).

        Returns:
            Predicted class.
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
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.
        """
        predictions = np.array([self._predict_sample(x) for x in X])
        return predictions


class RandomForestClassifier:
    """Random Forest Classifier with bootstrap sampling and feature importance."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, str, float]] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Random Forest Classifier.

        Args:
            n_estimators: Number of trees in forest (default: 100).
            max_depth: Maximum depth of trees (default: None).
            min_samples_split: Minimum samples required to split node (default: 2).
            min_samples_leaf: Minimum samples required at leaf node (default: 1).
            max_features: Number of features to consider for split. Options:
                - int: exact number
                - "sqrt": sqrt(n_features)
                - "log2": log2(n_features)
                - float: fraction of features
                (default: "sqrt").
            bootstrap: Whether to use bootstrap sampling (default: True).
            random_state: Random seed (default: None).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_: List[DecisionTreeClassifier] = []
        self.n_features_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

        if random_state is not None:
            np.random.seed(random_state)

    def _bootstrap_sample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create bootstrap sample.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Tuple of (bootstrap_X, bootstrap_y).
        """
        n_samples = len(X)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _get_max_features(self, n_features: int) -> int:
        """Get number of features to consider for split.

        Args:
            n_features: Total number of features.

        Returns:
            Number of features to consider.
        """
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features)) if n_features > 1 else 1
        else:
            return n_features

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> "RandomForestClassifier":
        """Fit random forest classifier.

        Args:
            X: Feature matrix.
            y: Target labels.

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

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)

        max_features = self._get_max_features(self.n_features_)

        self.estimators_ = []

        for i in range(self.n_estimators):
            if self.random_state is not None:
                tree_seed = self.random_state + i
            else:
                tree_seed = None

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=tree_seed,
            )

            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

        self._calculate_feature_importance()

        logger.info(
            f"Random forest fitted: n_estimators={self.n_estimators}, "
            f"n_features={self.n_features_}, bootstrap={self.bootstrap}"
        )

        return self

    def _calculate_feature_importance(self) -> None:
        """Calculate feature importance by averaging across trees."""
        if not self.estimators_:
            return

        importances = np.zeros(self.n_features_)

        for tree in self.estimators_:
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        importances /= len(self.estimators_)

        total = np.sum(importances)
        if total > 0:
            importances = importances / total

        self.feature_importances_ = importances

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class labels using majority voting.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model not fitted.
        """
        if not self.estimators_:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        majority_votes = np.array(
            [
                np.bincount(predictions[:, i]).argmax()
                for i in range(predictions.shape[1])
            ]
        )

        return majority_votes

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

        n_samples = len(X)
        n_classes = len(self.classes_)

        predictions = np.array([tree.predict(X) for tree in self.estimators_])

        proba = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            votes = predictions[:, i]
            for j, cls in enumerate(self.classes_):
                proba[i, j] = np.sum(votes == cls) / len(self.estimators_)

        return proba

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
            f"Random Forest Feature Importance (n_estimators={self.n_estimators})",
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

    parser = argparse.ArgumentParser(description="Random Forest Classifier")
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
        "--max-features",
        type=str,
        default=None,
        help="Max features for split (default: from config)",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap sampling",
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
        max_depth = (
            args.max_depth
            if args.max_depth is not None
            else model_config.get("max_depth")
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
        max_features = (
            args.max_features
            if args.max_features is not None
            else model_config.get("max_features", "sqrt")
        )
        bootstrap = not args.no_bootstrap if args.no_bootstrap else model_config.get("bootstrap", True)

        if isinstance(max_features, str) and max_features.isdigit():
            max_features = int(max_features)
        elif isinstance(max_features, str) and max_features.replace(".", "").isdigit():
            max_features = float(max_features)

        df = pd.read_csv(args.input)
        print(f"\n=== Random Forest Classifier ===")
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

        print(f"\nFitting Random Forest...")
        print(f"Number of trees: {n_estimators}")
        print(f"Bootstrap sampling: {bootstrap}")
        if max_depth:
            print(f"Max depth: {max_depth}")
        else:
            print("Max depth: None (unlimited)")
        print(f"Min samples split: {min_samples_split}")
        print(f"Min samples leaf: {min_samples_leaf}")
        print(f"Max features: {max_features}")

        forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
        )
        forest.feature_names_ = feature_cols
        forest.fit(X, y)

        print(f"\n=== Random Forest Results ===")
        print(f"Number of trees: {len(forest.estimators_)}")
        print(f"Classes: {forest.classes_}")

        accuracy = forest.score(X, y)
        print(f"\nTraining accuracy: {accuracy:.6f}")

        importances = forest.get_feature_importances()
        print(f"\nFeature Importance:")
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_importances[:10]:
            print(f"  {name}: {importance:.6f}")
        if len(sorted_importances) > 10:
            print(f"  ... and {len(sorted_importances) - 10} more")

        if args.plot_importance or args.save_importance_plot:
            forest.plot_feature_importance(
                save_path=args.save_importance_plot, show=args.plot_importance
            )

        if args.output:
            predictions = forest.predict(X)
            proba = forest.predict_proba(X)
            output_df = pd.DataFrame({
                "prediction": predictions,
                args.target: y,
            })
            for i, cls in enumerate(forest.classes_):
                output_df[f"prob_class_{cls}"] = proba[:, i]
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

        if args.predict:
            df_predict = pd.read_csv(args.predict)
            X_predict = df_predict[feature_cols].values
            predictions = forest.predict(X_predict)
            proba = forest.predict_proba(X_predict)

            if args.output_predictions:
                output_df = pd.DataFrame({
                    "prediction": predictions,
                })
                for i, cls in enumerate(forest.classes_):
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
        logger.error(f"Error running random forest: {e}")
        raise


if __name__ == "__main__":
    main()
