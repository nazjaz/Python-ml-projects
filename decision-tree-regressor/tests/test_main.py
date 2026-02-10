"""Unit tests for Decision Tree Regressor implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.main import DecisionTreeRegressor


class TestDecisionTreeRegressor:
    """Test Decision Tree Regressor functionality."""

    def test_initialization(self):
        """Test model initialization."""
        tree = DecisionTreeRegressor(criterion="mse")
        assert tree.criterion == "mse"
        assert tree.max_depth is None
        assert tree.root is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        tree = DecisionTreeRegressor(
            criterion="mae",
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            ccp_alpha=0.1,
            random_state=42,
        )
        assert tree.criterion == "mae"
        assert tree.max_depth == 5
        assert tree.min_samples_split == 5
        assert tree.min_samples_leaf == 2
        assert tree.ccp_alpha == 0.1

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor(criterion="mse")
        tree.fit(X, y)

        assert tree.root is not None
        assert tree.n_features_ == 2
        assert tree.feature_importances_ is not None

    def test_fit_invalid_criterion(self):
        """Test that invalid criterion raises error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([1.5, 2.5, 3.5])
        tree = DecisionTreeRegressor(criterion="invalid")
        with pytest.raises(ValueError, match="Unknown criterion"):
            tree.fit(X, y)

    def test_fit_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([1.5, 2.5])
        tree = DecisionTreeRegressor()
        with pytest.raises(ValueError, match="same length"):
            tree.fit(X, y)

    def test_mse(self):
        """Test MSE calculation."""
        tree = DecisionTreeRegressor(criterion="mse")
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mse = tree._mse(y)
        assert mse >= 0

    def test_mae(self):
        """Test MAE calculation."""
        tree = DecisionTreeRegressor(criterion="mae")
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mae = tree._mae(y)
        assert mae >= 0

    def test_impurity_mse(self):
        """Test impurity calculation with MSE."""
        tree = DecisionTreeRegressor(criterion="mse")
        y = np.array([1.0, 2.0, 3.0, 4.0])
        impurity = tree._impurity(y)
        assert impurity >= 0

    def test_impurity_mae(self):
        """Test impurity calculation with MAE."""
        tree = DecisionTreeRegressor(criterion="mae")
        y = np.array([1.0, 2.0, 3.0, 4.0])
        impurity = tree._impurity(y)
        assert impurity >= 0

    def test_variance_reduction(self):
        """Test variance reduction calculation."""
        tree = DecisionTreeRegressor(criterion="mse")
        y_parent = np.array([1.0, 2.0, 3.0, 4.0])
        y_left = np.array([1.0, 2.0])
        y_right = np.array([3.0, 4.0])
        reduction = tree._variance_reduction(y_parent, y_left, y_right)
        assert reduction >= 0

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        tree = DecisionTreeRegressor()
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            tree.predict(X)

    def test_predict(self):
        """Test prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor()
        tree.fit(X, y)

        predictions = tree.predict(X)
        assert len(predictions) == len(X)
        assert all(isinstance(p, (int, float)) for p in predictions)

    def test_score(self):
        """Test R-squared score."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor()
        tree.fit(X, y)

        score = tree.score(X, y)
        assert -np.inf < score <= 1.0

    def test_mse_metric(self):
        """Test MSE metric."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor()
        tree.fit(X, y)

        mse = tree.mse(X, y)
        assert mse >= 0

    def test_mae_metric(self):
        """Test MAE metric."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor()
        tree.fit(X, y)

        mae = tree.mae(X, y)
        assert mae >= 0

    def test_get_feature_importances(self):
        """Test getting feature importance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor()
        tree.fit(X, y)

        importances = tree.get_feature_importances()
        assert isinstance(importances, dict)
        assert len(importances) == 2

    def test_get_feature_importances_before_fit(self):
        """Test that getting feature importance before fitting raises error."""
        tree = DecisionTreeRegressor()
        with pytest.raises(ValueError, match="must be fitted"):
            tree.get_feature_importances()

    def test_get_depth(self):
        """Test getting tree depth."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor()
        tree.fit(X, y)

        depth = tree.get_depth()
        assert depth >= 0

    def test_get_n_nodes(self):
        """Test getting number of nodes."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor()
        tree.fit(X, y)

        n_nodes = tree.get_n_nodes()
        assert n_nodes > 0

    def test_max_depth(self):
        """Test max_depth parameter."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor(max_depth=2)
        tree.fit(X, y)

        depth = tree.get_depth()
        assert depth <= 2

    def test_min_samples_split(self):
        """Test min_samples_split parameter."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor(min_samples_split=5)
        tree.fit(X, y)

        assert tree.root is not None

    def test_min_samples_leaf(self):
        """Test min_samples_leaf parameter."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor(min_samples_leaf=2)
        tree.fit(X, y)

        assert tree.root is not None

    def test_ccp_alpha_pruning(self):
        """Test cost-complexity pruning."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
        tree = DecisionTreeRegressor(ccp_alpha=0.1)
        tree.fit(X, y)

        assert tree.root is not None

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [2, 3, 4, 5, 6, 7],
            "target": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        })
        X = df[["feature1", "feature2"]]
        y = df["target"]

        tree = DecisionTreeRegressor()
        tree.fit(X, y)

        predictions = tree.predict(X)
        assert len(predictions) == len(df)

    def test_with_list_input(self):
        """Test with list input."""
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        y = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

        tree = DecisionTreeRegressor()
        tree.fit(X, y)

        predictions = tree.predict(X)
        assert len(predictions) == len(X)

    def test_criterion_mse_vs_mae(self):
        """Test both MSE and MAE criteria."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

        tree_mse = DecisionTreeRegressor(criterion="mse")
        tree_mse.fit(X, y)

        tree_mae = DecisionTreeRegressor(criterion="mae")
        tree_mae.fit(X, y)

        assert tree_mse.root is not None
        assert tree_mae.root is not None

    def test_print_tree(self):
        """Test printing tree structure."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor()
        tree.fit(X, y)

        tree.print_tree()

    def test_plot_tree(self):
        """Test plotting tree."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor()
        tree.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "tree.png"
            tree.plot_tree(save_path=str(save_path), show=False)
            assert save_path.exists()

    def test_plot_feature_importance(self):
        """Test plotting feature importance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        tree = DecisionTreeRegressor()
        tree.feature_names_ = ["feature1", "feature2"]
        tree.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "importance.png"
            tree.plot_feature_importance(save_path=str(save_path), show=False)
            assert save_path.exists()
