"""Decision Tree Classifier with Information Gain and Gini Impurity.

This module provides functionality to implement Decision Tree from scratch
with information gain, Gini impurity, and tree visualization.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    """Decision Tree Classifier with information gain and Gini impurity."""

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Decision Tree Classifier.

        Args:
            criterion: Splitting criterion. Options: "gini", "entropy" (default: "gini").
            max_depth: Maximum depth of tree. If None, nodes expanded until pure (default: None).
            min_samples_split: Minimum samples required to split node (default: 2).
            min_samples_leaf: Minimum samples required at leaf node (default: 1).
            min_impurity_decrease: Minimum impurity decrease for split (default: 0.0).
            random_state: Random seed (default: None).
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

        self.root: Optional[TreeNode] = None
        self.n_features_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None

        if random_state is not None:
            np.random.seed(random_state)

    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy.

        Args:
            y: Target labels.

        Returns:
            Entropy value.
        """
        if len(y) == 0:
            return 0.0

        counts = np.bincount(y)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

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
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini

    def _impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion.

        Args:
            y: Target labels.

        Returns:
            Impurity value.
        """
        if self.criterion == "entropy":
            return self._entropy(y)
        elif self.criterion == "gini":
            return self._gini(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

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
        parent_impurity = self._impurity(y_parent)
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        if n == 0:
            return 0.0

        left_impurity = self._impurity(y_left) if n_left > 0 else 0.0
        right_impurity = self._impurity(y_right) if n_right > 0 else 0.0

        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        information_gain = parent_impurity - weighted_impurity

        return information_gain

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find best split for node.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Tuple of (best_feature_index, best_threshold, best_gain).
        """
        n_samples, n_features = X.shape
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        if n_samples < self.min_samples_split:
            return None, None, 0.0

        current_impurity = self._impurity(y)

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

                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain and gain >= self.min_impurity_decrease:
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
        node.impurity = self._impurity(y)

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

        best_feature, best_threshold, best_gain = self._best_split(X, y)

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

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> "DecisionTreeClassifier":
        """Fit decision tree classifier.

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

        if self.criterion not in ["gini", "entropy"]:
            raise ValueError(f"Unknown criterion: {self.criterion}")

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.root = self._build_tree(X, y)

        logger.info(
            f"Decision tree fitted: criterion={self.criterion}, "
            f"max_depth={self.max_depth}, n_classes={self.n_classes_}"
        )

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

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model not fitted.
        """
        if self.root is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = np.array([self._predict_sample(x) for x in X])
        return predictions

    def predict_proba(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Class probabilities.

        Raises:
            ValueError: If model not fitted.
        """
        if self.root is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = self.predict(X)
        proba = np.zeros((len(X), self.n_classes_))

        for i, pred in enumerate(predictions):
            class_idx = np.where(self.classes_ == pred)[0][0]
            proba[i, class_idx] = 1.0

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

    def _get_tree_depth(self, node: Optional[TreeNode] = None) -> int:
        """Get maximum depth of tree.

        Args:
            node: Current node (default: root).

        Returns:
            Maximum depth.
        """
        if node is None:
            node = self.root

        if node.is_leaf:
            return 0

        left_depth = self._get_tree_depth(node.left)
        right_depth = self._get_tree_depth(node.right)

        return 1 + max(left_depth, right_depth)

    def get_depth(self) -> int:
        """Get maximum depth of tree.

        Returns:
            Maximum depth.
        """
        return self._get_tree_depth()

    def _get_n_nodes(self, node: Optional[TreeNode] = None) -> int:
        """Get number of nodes in tree.

        Args:
            node: Current node (default: root).

        Returns:
            Number of nodes.
        """
        if node is None:
            node = self.root

        if node.is_leaf:
            return 1

        return 1 + self._get_n_nodes(node.left) + self._get_n_nodes(node.right)

    def get_n_nodes(self) -> int:
        """Get number of nodes in tree.

        Returns:
            Number of nodes.
        """
        return self._get_n_nodes()

    def _tree_to_text(
        self, node: Optional[TreeNode] = None, depth: int = 0, prefix: str = ""
    ) -> List[str]:
        """Convert tree to text representation.

        Args:
            node: Current node (default: root).
            depth: Current depth.
            prefix: Prefix for current line.

        Returns:
            List of text lines.
        """
        if node is None:
            node = self.root

        lines = []

        if node.is_leaf:
            lines.append(f"{prefix}Leaf: class={node.value}, samples={node.samples}")
            return lines

        feature_name = (
            f"feature_{node.feature_index}"
            if self.feature_names_ is None
            else self.feature_names_[node.feature_index]
        )
        lines.append(
            f"{prefix}{feature_name} <= {node.threshold:.4f} "
            f"[samples={node.samples}, impurity={node.impurity:.4f}]"
        )

        lines.extend(
            self._tree_to_text(
                node.left, depth + 1, prefix + "  |-- "
            )
        )
        lines.extend(
            self._tree_to_text(
                node.right, depth + 1, prefix + "  |-- "
            )
        )

        return lines

    def print_tree(self) -> None:
        """Print tree structure."""
        if self.root is None:
            print("Tree not fitted")
            return

        lines = self._tree_to_text()
        for line in lines:
            print(line)

    def plot_tree(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """Plot decision tree visualization.

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
            figsize: Figure size (width, height).
        """
        if self.root is None:
            logger.warning("Tree not fitted")
            return

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        self._plot_node(ax, self.root, 0.5, 1.0, 0.4, 0.1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Decision Tree (criterion={self.criterion}, depth={self.get_depth()})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Tree plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def _plot_node(
        self,
        ax: plt.Axes,
        node: TreeNode,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        """Plot tree node recursively.

        Args:
            ax: Matplotlib axes.
            node: Current node.
            x: X position.
            y: Y position.
            width: Width of subtree.
            height: Height of subtree.
        """
        if node.is_leaf:
            text = f"Class: {node.value}\nSamples: {node.samples}"
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                fontsize=9,
            )
            return

        feature_name = (
            f"X[{node.feature_index}]"
            if self.feature_names_ is None
            else self.feature_names_[node.feature_index]
        )
        text = f"{feature_name}\n<= {node.threshold:.2f}\nSamples: {node.samples}\nImpurity: {node.impurity:.3f}"

        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            fontsize=9,
        )

        if node.left is not None:
            left_x = x - width / 2
            left_y = y - height
            ax.plot([x, left_x], [y, left_y], "k-", linewidth=1)
            self._plot_node(ax, node.left, left_x, left_y, width / 2, height)

        if node.right is not None:
            right_x = x + width / 2
            right_y = y - height
            ax.plot([x, right_x], [y, right_y], "k-", linewidth=1)
            self._plot_node(ax, node.right, right_x, right_y, width / 2, height)


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Decision Tree Classifier")
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
        "--criterion",
        type=str,
        default=None,
        choices=["gini", "entropy"],
        help="Splitting criterion (default: from config)",
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
        "--print-tree",
        action="store_true",
        help="Print tree structure",
    )
    parser.add_argument(
        "--plot-tree",
        action="store_true",
        help="Plot tree visualization",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save tree plot",
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

        criterion = (
            args.criterion
            if args.criterion is not None
            else model_config.get("criterion", "gini")
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

        df = pd.read_csv(args.input)
        print(f"\n=== Decision Tree Classifier ===")
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

        print(f"\nFitting Decision Tree...")
        print(f"Criterion: {criterion}")
        if max_depth:
            print(f"Max depth: {max_depth}")
        else:
            print("Max depth: None (unlimited)")
        print(f"Min samples split: {min_samples_split}")
        print(f"Min samples leaf: {min_samples_leaf}")

        tree = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        tree.feature_names_ = feature_cols
        tree.fit(X, y)

        print(f"\n=== Decision Tree Results ===")
        print(f"Tree depth: {tree.get_depth()}")
        print(f"Number of nodes: {tree.get_n_nodes()}")
        print(f"Classes: {tree.classes_}")

        accuracy = tree.score(X, y)
        print(f"\nTraining accuracy: {accuracy:.6f}")

        if args.print_tree:
            print(f"\n=== Tree Structure ===")
            tree.print_tree()

        if args.plot_tree or args.save_plot:
            tree.plot_tree(save_path=args.save_plot, show=args.plot_tree)

        if args.output:
            predictions = tree.predict(X)
            proba = tree.predict_proba(X)
            output_df = pd.DataFrame({
                "prediction": predictions,
                args.target: y,
            })
            for i, cls in enumerate(tree.classes_):
                output_df[f"prob_class_{cls}"] = proba[:, i]
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

        if args.predict:
            df_predict = pd.read_csv(args.predict)
            X_predict = df_predict[feature_cols].values
            predictions = tree.predict(X_predict)
            proba = tree.predict_proba(X_predict)

            if args.output_predictions:
                output_df = pd.DataFrame({
                    "prediction": predictions,
                })
                for i, cls in enumerate(tree.classes_):
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
        logger.error(f"Error running decision tree: {e}")
        raise


if __name__ == "__main__":
    main()
