"""Unit tests for Decision Tree Classifier implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.main import DecisionTreeClassifier


class TestDecisionTreeClassifier:
    """Test Decision Tree Classifier functionality."""

    def test_initialization(self):
        """Test model initialization."""
        tree = DecisionTreeClassifier(criterion="gini")
        assert tree.criterion == "gini"
        assert tree.max_depth is None
        assert tree.root is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        tree = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )
        assert tree.criterion == "entropy"
        assert tree.max_depth == 5
        assert tree.min_samples_split == 5
        assert tree.min_samples_leaf == 2

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier(criterion="gini")
        tree.fit(X, y)

        assert tree.root is not None
        assert tree.n_features_ == 2
        assert tree.n_classes_ == 2

    def test_fit_invalid_criterion(self):
        """Test that invalid criterion raises error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 1])
        tree = DecisionTreeClassifier(criterion="invalid")
        with pytest.raises(ValueError, match="Unknown criterion"):
            tree.fit(X, y)

    def test_fit_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1])
        tree = DecisionTreeClassifier()
        with pytest.raises(ValueError, match="same length"):
            tree.fit(X, y)

    def test_entropy(self):
        """Test entropy calculation."""
        tree = DecisionTreeClassifier(criterion="entropy")
        y = np.array([0, 0, 1, 1])
        entropy = tree._entropy(y)
        assert entropy > 0
        assert entropy <= 1.0

    def test_gini(self):
        """Test Gini impurity calculation."""
        tree = DecisionTreeClassifier(criterion="gini")
        y = np.array([0, 0, 1, 1])
        gini = tree._gini(y)
        assert gini >= 0
        assert gini <= 1.0

    def test_impurity_gini(self):
        """Test impurity calculation with Gini."""
        tree = DecisionTreeClassifier(criterion="gini")
        y = np.array([0, 0, 1, 1])
        impurity = tree._impurity(y)
        assert impurity >= 0

    def test_impurity_entropy(self):
        """Test impurity calculation with entropy."""
        tree = DecisionTreeClassifier(criterion="entropy")
        y = np.array([0, 0, 1, 1])
        impurity = tree._impurity(y)
        assert impurity >= 0

    def test_information_gain(self):
        """Test information gain calculation."""
        tree = DecisionTreeClassifier(criterion="gini")
        y_parent = np.array([0, 0, 1, 1])
        y_left = np.array([0, 0])
        y_right = np.array([1, 1])
        gain = tree._information_gain(y_parent, y_left, y_right)
        assert gain >= 0

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        tree = DecisionTreeClassifier()
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            tree.predict(X)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        predictions = tree.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in tree.classes_ for pred in predictions)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        proba = tree.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == tree.n_classes_
        assert np.allclose(np.sum(proba, axis=1), 1.0)

    def test_score(self):
        """Test classification accuracy."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        accuracy = tree.score(X, y)
        assert 0 <= accuracy <= 1

    def test_get_depth(self):
        """Test getting tree depth."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        depth = tree.get_depth()
        assert depth >= 0

    def test_get_n_nodes(self):
        """Test getting number of nodes."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        n_nodes = tree.get_n_nodes()
        assert n_nodes > 0

    def test_max_depth(self):
        """Test max_depth parameter."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)

        depth = tree.get_depth()
        assert depth <= 2

    def test_min_samples_split(self):
        """Test min_samples_split parameter."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier(min_samples_split=5)
        tree.fit(X, y)

        assert tree.root is not None

    def test_min_samples_leaf(self):
        """Test min_samples_leaf parameter."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier(min_samples_leaf=2)
        tree.fit(X, y)

        assert tree.root is not None

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [2, 3, 4, 5, 6, 7],
            "target": [0, 0, 0, 1, 1, 1]
        })
        X = df[["feature1", "feature2"]]
        y = df["target"]

        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        predictions = tree.predict(X)
        assert len(predictions) == len(df)

    def test_with_list_input(self):
        """Test with list input."""
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        y = [0, 0, 0, 1, 1, 1]

        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        predictions = tree.predict(X)
        assert len(predictions) == len(X)

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X = np.array([
            [1, 2], [2, 3], [3, 4],
            [4, 5], [5, 6], [6, 7],
            [7, 8], [8, 9], [9, 10]
        ])
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        assert tree.n_classes_ == 3
        predictions = tree.predict(X)
        assert len(predictions) == len(X)

    def test_print_tree(self):
        """Test printing tree structure."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        tree.print_tree()

    def test_plot_tree(self):
        """Test plotting tree."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "tree.png"
            tree.plot_tree(save_path=str(save_path), show=False)
            assert save_path.exists()

    def test_criterion_gini_vs_entropy(self):
        """Test both Gini and entropy criteria."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])

        tree_gini = DecisionTreeClassifier(criterion="gini")
        tree_gini.fit(X, y)

        tree_entropy = DecisionTreeClassifier(criterion="entropy")
        tree_entropy.fit(X, y)

        assert tree_gini.root is not None
        assert tree_entropy.root is not None
