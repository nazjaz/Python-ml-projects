"""Tests for feature selection module."""

import numpy as np
import pandas as pd
import pytest

from src.main import (
    FeatureSelector,
    MutualInformationSelector,
    RecursiveFeatureElimination,
)


class TestMutualInformationSelector:
    """Test cases for Mutual Information selector."""

    def test_initialization(self):
        """Test selector initialization."""
        selector = MutualInformationSelector(n_features=10, random_state=42)
        assert selector.n_features == 10
        assert selector.random_state == 42

    def test_fit_classification(self):
        """Test fitting for classification."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = MutualInformationSelector(n_features=10, random_state=42)
        selector.fit(X, y, task_type="classification")

        assert selector.scores_ is not None
        assert len(selector.selected_features_) == 10

    def test_fit_regression(self):
        """Test fitting for regression."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        selector = MutualInformationSelector(n_features=10, random_state=42)
        selector.fit(X, y, task_type="regression")

        assert selector.scores_ is not None
        assert len(selector.selected_features_) == 10

    def test_fit_shape_mismatch(self):
        """Test fitting with shape mismatch."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 50)

        selector = MutualInformationSelector()
        with pytest.raises(ValueError, match="X and y must have same number"):
            selector.fit(X, y)

    def test_fit_invalid_task_type(self):
        """Test fitting with invalid task type."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = MutualInformationSelector()
        with pytest.raises(ValueError, match="task_type must be"):
            selector.fit(X, y, task_type="invalid")

    def test_transform(self):
        """Test data transformation."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = MutualInformationSelector(n_features=10, random_state=42)
        selector.fit(X, y, task_type="classification")

        X_transformed = selector.transform(X)
        assert X_transformed.shape == (100, 10)

    def test_transform_not_fitted(self):
        """Test transformation without fitting."""
        selector = MutualInformationSelector()
        X = np.random.randn(100, 20)

        with pytest.raises(ValueError, match="Selector must be fitted"):
            selector.transform(X)

    def test_get_feature_scores(self):
        """Test getting feature scores."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = MutualInformationSelector(random_state=42)
        selector.fit(X, y, task_type="classification")

        scores = selector.get_feature_scores()
        assert len(scores) == 20
        assert all(isinstance(v, float) for v in scores.values())

    def test_get_selected_features(self):
        """Test getting selected features."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(20)]

        selector = MutualInformationSelector(n_features=10, random_state=42)
        selector.fit(X, y, feature_names=feature_names, task_type="classification")

        selected = selector.get_selected_features()
        assert len(selected) == 10
        assert all(f in feature_names for f in selected)


class TestRecursiveFeatureElimination:
    """Test cases for RFE selector."""

    def test_initialization(self):
        """Test selector initialization."""
        selector = RecursiveFeatureElimination(n_features_to_select=10, step=2)
        assert selector.n_features_to_select == 10
        assert selector.step == 2

    def test_fit_classification(self):
        """Test fitting for classification."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = RecursiveFeatureElimination(n_features_to_select=10)
        selector.fit(X, y, task_type="classification")

        assert selector.rfe_ is not None
        assert len(selector.selected_features_) == 10

    def test_fit_regression(self):
        """Test fitting for regression."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        selector = RecursiveFeatureElimination(n_features_to_select=10)
        selector.fit(X, y, task_type="regression")

        assert selector.rfe_ is not None
        assert len(selector.selected_features_) == 10

    def test_fit_shape_mismatch(self):
        """Test fitting with shape mismatch."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 50)

        selector = RecursiveFeatureElimination()
        with pytest.raises(ValueError, match="X and y must have same number"):
            selector.fit(X, y)

    def test_transform(self):
        """Test data transformation."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = RecursiveFeatureElimination(n_features_to_select=10)
        selector.fit(X, y, task_type="classification")

        X_transformed = selector.transform(X)
        assert X_transformed.shape == (100, 10)

    def test_transform_not_fitted(self):
        """Test transformation without fitting."""
        selector = RecursiveFeatureElimination()
        X = np.random.randn(100, 20)

        with pytest.raises(ValueError, match="Selector must be fitted"):
            selector.transform(X)

    def test_get_feature_ranking(self):
        """Test getting feature ranking."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(20)]

        selector = RecursiveFeatureElimination(n_features_to_select=10)
        selector.fit(X, y, feature_names=feature_names, task_type="classification")

        ranking = selector.get_feature_ranking()
        assert len(ranking) == 20
        assert all(isinstance(v, int) for v in ranking.values())

    def test_get_selected_features(self):
        """Test getting selected features."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(20)]

        selector = RecursiveFeatureElimination(n_features_to_select=10)
        selector.fit(X, y, feature_names=feature_names, task_type="classification")

        selected = selector.get_selected_features()
        assert len(selected) == 10
        assert all(f in feature_names for f in selected)


class TestFeatureSelector:
    """Test cases for main feature selector class."""

    def test_initialization(self):
        """Test selector initialization."""
        selector = FeatureSelector()
        assert selector.mi_selector is None
        assert selector.rfe_selector is None

    def test_load_data(self):
        """Test data loading."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = FeatureSelector()
        selector.load_data(X, y, task_type="classification")

        assert selector.X is not None
        assert selector.y is not None
        assert selector.task_type == "classification"

    def test_load_data_dataframe(self):
        """Test loading data from DataFrame."""
        df = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        X = df[["feature_1", "feature_2"]]
        y = df["target"]

        selector = FeatureSelector()
        selector.load_data(X, y, task_type="classification")

        assert selector.feature_names == ["feature_1", "feature_2"]

    def test_fit_mutual_information(self):
        """Test fitting Mutual Information selector."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = FeatureSelector()
        selector.load_data(X, y, task_type="classification")
        selector.fit_mutual_information(n_features=10)

        assert selector.mi_selector is not None

    def test_fit_mutual_information_no_data(self):
        """Test fitting without loading data."""
        selector = FeatureSelector()
        with pytest.raises(ValueError, match="Data must be loaded"):
            selector.fit_mutual_information()

    def test_fit_rfe(self):
        """Test fitting RFE selector."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = FeatureSelector()
        selector.load_data(X, y, task_type="classification")
        selector.fit_rfe(n_features_to_select=10)

        assert selector.rfe_selector is not None

    def test_fit_rfe_no_data(self):
        """Test fitting without loading data."""
        selector = FeatureSelector()
        with pytest.raises(ValueError, match="Data must be loaded"):
            selector.fit_rfe()

    def test_fit_all(self):
        """Test fitting both selectors."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = FeatureSelector()
        selector.load_data(X, y, task_type="classification")
        selector.fit_all()

        assert selector.mi_selector is not None
        assert selector.rfe_selector is not None

    def test_get_selected_features(self):
        """Test getting selected features."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = FeatureSelector()
        selector.load_data(X, y, task_type="classification")
        selector.fit_mutual_information(n_features=10)

        features = selector.get_selected_features("mutual_information")
        assert isinstance(features, list)
        assert len(features) == 10

    def test_get_selected_features_invalid_method(self):
        """Test getting features with invalid method."""
        selector = FeatureSelector()
        with pytest.raises(ValueError, match="Unknown method"):
            selector.get_selected_features("invalid")

    def test_transform(self):
        """Test data transformation."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = FeatureSelector()
        selector.load_data(X, y, task_type="classification")
        selector.fit_mutual_information(n_features=10)

        X_transformed = selector.transform(X, method="mutual_information")
        assert X_transformed.shape == (100, 10)

    def test_evaluate_selection(self):
        """Test evaluation of feature selection."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = FeatureSelector()
        selector.load_data(X, y, task_type="classification")
        selector.fit_mutual_information(n_features=10)

        evaluation = selector.evaluate_selection(method="mutual_information")
        assert "accuracy" in evaluation or "n_features" in evaluation


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(self):
        """Test complete feature selection workflow."""
        X = np.random.randn(200, 30)
        y = np.random.randint(0, 2, 200)

        selector = FeatureSelector()
        selector.load_data(X, y, task_type="classification")
        selector.fit_all()

        mi_features = selector.get_selected_features("mutual_information")
        rfe_features = selector.get_selected_features("rfe")

        X_mi = selector.transform(X, method="mutual_information")
        X_rfe = selector.transform(X, method="rfe")

        assert len(mi_features) > 0
        assert len(rfe_features) > 0
        assert X_mi.shape[1] == len(mi_features)
        assert X_rfe.shape[1] == len(rfe_features)

    def test_different_n_features(self):
        """Test with different numbers of features."""
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector1 = FeatureSelector()
        selector1.load_data(X, y, task_type="classification")
        selector1.fit_mutual_information(n_features=5)

        selector2 = FeatureSelector()
        selector2.load_data(X, y, task_type="classification")
        selector2.fit_mutual_information(n_features=15)

        features1 = selector1.get_selected_features("mutual_information")
        features2 = selector2.get_selected_features("mutual_information")

        assert len(features1) == 5
        assert len(features2) == 15

    def test_regression_task(self):
        """Test feature selection for regression."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        selector = FeatureSelector()
        selector.load_data(X, y, task_type="regression")
        selector.fit_all()

        mi_features = selector.get_selected_features("mutual_information")
        rfe_features = selector.get_selected_features("rfe")

        assert len(mi_features) > 0
        assert len(rfe_features) > 0
