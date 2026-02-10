"""Decision Tree Regressor with Pruning and Feature Importance.

This module provides functionality to implement Decision Tree Regressor from
scratch with pruning techniques and feature importance analysis.
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
        self.impurity: float = 0.0
        self.mse: float = 0.0


class DecisionTreeRegressor:
    """Decision Tree Regressor with pruning and feature importance."""

    def __init__(
        self,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Decision Tree Regressor.

        Args:
            criterion: Splitting criterion. Options: "mse", "mae" (default: "mse").
            max_depth: Maximum depth of tree. If None, nodes expanded until pure (default: None).
            min_samples_split: Minimum samples required to split node (default: 2).
            min_samples_leaf: Minimum samples required at leaf node (default: 1).
            min_impurity_decrease: Minimum impurity decrease for split (default: 0.0).
            ccp_alpha: Complexity parameter for cost-complexity pruning (default: 0.0).
            random_state: Random seed (default: None).
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

        self.root: Optional[TreeNode] = None
        self.n_features_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None

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

    def _mae(self, y: np.ndarray) -> float:
        """Calculate mean absolute error.

        Args:
            y: Target values.

        Returns:
            MAE value.
        """
        if len(y) == 0:
            return 0.0
        median = np.median(y)
        mae = np.mean(np.abs(y - median))
        return mae

    def _impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion.

        Args:
            y: Target values.

        Returns:
            Impurity value.
        """
        if self.criterion == "mse":
            return self._mse(y)
        elif self.criterion == "mae":
            return self._mae(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _variance_reduction(
        self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
    ) -> float:
        """Calculate variance reduction (for MSE) or similar for MAE.

        Args:
            y_parent: Parent node target values.
            y_left: Left child target values.
            y_right: Right child target values.

        Returns:
            Variance reduction.
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
        variance_reduction = parent_impurity - weighted_impurity

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

                if reduction > best_reduction and reduction >= self.min_impurity_decrease:
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
        node.impurity = self._impurity(y)
        node.mse = self._mse(y)

        if len(y) == 0:
            node.is_leaf = True
            node.value = 0.0
            return node

        node.value = np.mean(y)

        if len(y) == 1:
            node.is_leaf = True
            return node

        if self.max_depth is not None and depth >= self.max_depth:
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

    def _calculate_feature_importance(self, node: Optional[TreeNode] = None) -> np.ndarray:
        """Calculate feature importance based on variance reduction.

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

    def _normalize_feature_importance(self) -> None:
        """Normalize feature importance values."""
        if self.feature_importances_ is None:
            return

        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total
        else:
            self.feature_importances_ = np.zeros(self.n_features_)

    def _get_tree_complexity(self, node: Optional[TreeNode] = None) -> int:
        """Get number of leaf nodes (tree complexity).

        Args:
            node: Current node (default: root).

        Returns:
            Number of leaf nodes.
        """
        if node is None:
            node = self.root

        if node.is_leaf:
            return 1

        return self._get_tree_complexity(node.left) + self._get_tree_complexity(node.right)

    def _get_subtree_mse(self, node: Optional[TreeNode] = None) -> float:
        """Calculate total MSE of subtree.

        Args:
            node: Current node (default: root).

        Returns:
            Total MSE.
        """
        if node is None:
            node = self.root

        if node.is_leaf:
            return node.mse * node.samples

        return (
            self._get_subtree_mse(node.left) + self._get_subtree_mse(node.right)
        )

    def _prune_node(self, node: TreeNode) -> bool:
        """Prune node using cost-complexity pruning.

        Args:
            node: Node to potentially prune.

        Returns:
            True if node was pruned.
        """
        if node.is_leaf:
            return False

        if node.left is None or node.right is None:
            return False

        if node.left.is_leaf and node.right.is_leaf:
            subtree_mse = self._get_subtree_mse(node)
            leaf_mse = node.mse * node.samples
            complexity = 2

            if leaf_mse + self.ccp_alpha * 1 < subtree_mse + self.ccp_alpha * complexity:
                node.is_leaf = True
                node.left = None
                node.right = None
                return True

        pruned_left = self._prune_node(node.left)
        pruned_right = self._prune_node(node.right)

        if pruned_left or pruned_right:
            if node.left.is_leaf and node.right.is_leaf:
                subtree_mse = self._get_subtree_mse(node)
                leaf_mse = node.mse * node.samples
                complexity = 2

                if leaf_mse + self.ccp_alpha * 1 < subtree_mse + self.ccp_alpha * complexity:
                    node.is_leaf = True
                    node.left = None
                    node.right = None
                    return True

        return False

    def _post_prune(self) -> None:
        """Apply post-pruning using cost-complexity pruning."""
        if self.ccp_alpha <= 0:
            return

        while True:
            if not self._prune_node(self.root):
                break

        logger.info(f"Post-pruning applied with ccp_alpha={self.ccp_alpha}")

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> "DecisionTreeRegressor":
        """Fit decision tree regressor.

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

        if self.criterion not in ["mse", "mae"]:
            raise ValueError(f"Unknown criterion: {self.criterion}")

        self.n_features_ = X.shape[1]

        self.root = self._build_tree(X, y)

        if self.ccp_alpha > 0:
            self._post_prune()

        self.feature_importances_ = self._calculate_feature_importance()
        self._normalize_feature_importance()

        logger.info(
            f"Decision tree regressor fitted: criterion={self.criterion}, "
            f"max_depth={self.max_depth}, ccp_alpha={self.ccp_alpha}"
        )

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

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict target values.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.

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

    def mae(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> float:
        """Calculate mean absolute error.

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            Mean absolute error.
        """
        predictions = self.predict(X)
        y = np.asarray(y, dtype=float)
        mae = np.mean(np.abs(y - predictions))
        return mae

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
            lines.append(
                f"{prefix}Leaf: value={node.value:.4f}, samples={node.samples}, mse={node.mse:.4f}"
            )
            return lines

        feature_name = (
            f"feature_{node.feature_index}"
            if self.feature_names_ is None
            else self.feature_names_[node.feature_index]
        )
        lines.append(
            f"{prefix}{feature_name} <= {node.threshold:.4f} "
            f"[samples={node.samples}, mse={node.mse:.4f}]"
        )

        lines.extend(
            self._tree_to_text(node.left, depth + 1, prefix + "  |-- ")
        )
        lines.extend(
            self._tree_to_text(node.right, depth + 1, prefix + "  |-- ")
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
            f"Decision Tree Regressor (criterion={self.criterion}, depth={self.get_depth()})",
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
            text = f"Value: {node.value:.2f}\nSamples: {node.samples}\nMSE: {node.mse:.3f}"
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
        text = f"{feature_name}\n<= {node.threshold:.2f}\nSamples: {node.samples}\nMSE: {node.mse:.3f}"

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
        ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
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

    parser = argparse.ArgumentParser(description="Decision Tree Regressor")
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
        choices=["mse", "mae"],
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
        "--ccp-alpha",
        type=float,
        default=None,
        help="Complexity parameter for pruning (default: from config)",
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
        "--plot-importance",
        action="store_true",
        help="Plot feature importance",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save tree plot",
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

        criterion = (
            args.criterion
            if args.criterion is not None
            else model_config.get("criterion", "mse")
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
        ccp_alpha = (
            args.ccp_alpha
            if args.ccp_alpha is not None
            else model_config.get("ccp_alpha", 0.0)
        )

        df = pd.read_csv(args.input)
        print(f"\n=== Decision Tree Regressor ===")
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

        print(f"\nFitting Decision Tree Regressor...")
        print(f"Criterion: {criterion}")
        if max_depth:
            print(f"Max depth: {max_depth}")
        else:
            print("Max depth: None (unlimited)")
        print(f"Min samples split: {min_samples_split}")
        print(f"Min samples leaf: {min_samples_leaf}")
        if ccp_alpha > 0:
            print(f"CCP alpha (pruning): {ccp_alpha}")

        tree = DecisionTreeRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
        )
        tree.feature_names_ = feature_cols
        tree.fit(X, y)

        print(f"\n=== Decision Tree Results ===")
        print(f"Tree depth: {tree.get_depth()}")
        print(f"Number of nodes: {tree.get_n_nodes()}")
        if ccp_alpha > 0:
            print(f"Post-pruning applied with ccp_alpha={ccp_alpha}")

        r2 = tree.score(X, y)
        mse_score = tree.mse(X, y)
        mae_score = tree.mae(X, y)
        print(f"\nR-squared: {r2:.6f}")
        print(f"MSE: {mse_score:.6f}")
        print(f"MAE: {mae_score:.6f}")

        importances = tree.get_feature_importances()
        print(f"\nFeature Importance:")
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_importances[:10]:
            print(f"  {name}: {importance:.6f}")
        if len(sorted_importances) > 10:
            print(f"  ... and {len(sorted_importances) - 10} more")

        if args.print_tree:
            print(f"\n=== Tree Structure ===")
            tree.print_tree()

        if args.plot_tree or args.save_plot:
            tree.plot_tree(save_path=args.save_plot, show=args.plot_tree)

        if args.plot_importance or args.save_importance_plot:
            tree.plot_feature_importance(
                save_path=args.save_importance_plot, show=args.plot_importance
            )

        if args.output:
            predictions = tree.predict(X)
            output_df = pd.DataFrame({
                "prediction": predictions,
                args.target: y,
            })
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

        if args.predict:
            df_predict = pd.read_csv(args.predict)
            X_predict = df_predict[feature_cols].values
            predictions = tree.predict(X_predict)

            if args.output_predictions:
                output_df = pd.DataFrame({
                    "prediction": predictions,
                })
                output_df.to_csv(args.output_predictions, index=False)
                print(f"Predictions saved to: {args.output_predictions}")
            else:
                print(f"\nPredictions:")
                for i, pred in enumerate(predictions[:10]):
                    print(f"  Sample {i+1}: {pred:.6f}")
                if len(predictions) > 10:
                    print(f"  ... and {len(predictions) - 10} more")

    except Exception as e:
        logger.error(f"Error running decision tree regressor: {e}")
        raise


if __name__ == "__main__":
    main()
