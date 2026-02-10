"""Tests for model interpretability module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from src.main import (
    ModelInterpreter,
    PermutationImportanceCalculator,
    PartialDependencePlotter,
)

try:
    from src.main import SHAPExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class TestPermutationImportanceCalculator:
    """Test cases for permutation importance calculator."""

    def test_initialization(self):
        """Test calculator initialization."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        calculator = PermutationImportanceCalculator(
            model, n_repeats=5, random_state=42
        )
        assert calculator.n_repeats == 5
        assert calculator.random_state == 42

    def test_calculate_classification(self):
        """Test permutation importance for classification."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        calculator = PermutationImportanceCalculator(model, n_repeats=3)
        importance = calculator.calculate(X, y)

        assert len(importance) == 5
        assert all("importance_mean" in v for v in importance.values())
        assert all("importance_std" in v for v in importance.values())

    def test_calculate_regression(self):
        """Test permutation importance for regression."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        calculator = PermutationImportanceCalculator(model, n_repeats=3)
        importance = calculator.calculate(X, y)

        assert len(importance) == 5

    def test_calculate_shape_mismatch(self):
        """Test calculation with shape mismatch."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 50)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:50], y)

        calculator = PermutationImportanceCalculator(model)
        with pytest.raises(ValueError, match="X and y must have same number"):
            calculator.calculate(X, y)

    def test_get_feature_importance(self):
        """Test getting feature importance."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        calculator = PermutationImportanceCalculator(model, n_repeats=3)
        calculator.calculate(X, y)

        importance = calculator.get_feature_importance()
        assert len(importance) == 5
        assert all(isinstance(v, float) for v in importance.values())

    def test_get_feature_importance_not_calculated(self):
        """Test getting importance without calculation."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        calculator = PermutationImportanceCalculator(model)

        with pytest.raises(ValueError, match="Permutation importance must be calculated"):
            calculator.get_feature_importance()

    def test_plot_importance(self):
        """Test plotting importance."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        calculator = PermutationImportanceCalculator(model, n_repeats=3)
        calculator.calculate(X, y)

        calculator.plot_importance(top_n=5)


class TestPartialDependencePlotter:
    """Test cases for partial dependence plotter."""

    def test_initialization(self):
        """Test plotter initialization."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        plotter = PartialDependencePlotter(model)
        assert plotter.model == model

    def test_plot_single_feature(self):
        """Test plotting single feature."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        plotter = PartialDependencePlotter(model)
        plotter.plot(X, features=0)

    def test_plot_multiple_features(self):
        """Test plotting multiple features."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        plotter = PartialDependencePlotter(model)
        plotter.plot(X, features=[0, 1])

    def test_plot_with_feature_names(self):
        """Test plotting with feature names."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        plotter = PartialDependencePlotter(model, feature_names=feature_names)
        plotter.plot(X, features="feature_0", feature_names=feature_names)


class TestModelInterpreter:
    """Test cases for main model interpreter class."""

    def test_initialization(self):
        """Test interpreter initialization."""
        interpreter = ModelInterpreter()
        assert interpreter.model is None
        assert interpreter.X is None

    def test_load_model_and_data(self):
        """Test loading model and data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        interpreter = ModelInterpreter()
        interpreter.load_model_and_data(model, X, y)

        assert interpreter.model == model
        assert interpreter.X is not None
        assert interpreter.y is not None

    def test_load_model_and_data_dataframe(self):
        """Test loading with DataFrame."""
        df = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        X = df[["feature_1", "feature_2"]]
        y = df["target"]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        interpreter = ModelInterpreter()
        interpreter.load_model_and_data(model, X, y)

        assert interpreter.feature_names == ["feature_1", "feature_2"]

    def test_calculate_permutation_importance(self):
        """Test calculating permutation importance."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        interpreter = ModelInterpreter()
        interpreter.load_model_and_data(model, X, y)

        importance = interpreter.calculate_permutation_importance()
        assert len(importance) == 5

    def test_calculate_permutation_importance_no_data(self):
        """Test calculation without loading data."""
        interpreter = ModelInterpreter()
        with pytest.raises(ValueError, match="Model, X, and y must be loaded"):
            interpreter.calculate_permutation_importance()

    def test_plot_partial_dependence(self):
        """Test plotting partial dependence."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        interpreter = ModelInterpreter()
        interpreter.load_model_and_data(model, X, y)

        interpreter.plot_partial_dependence(features=0)

    def test_plot_partial_dependence_no_data(self):
        """Test plotting without loading data."""
        interpreter = ModelInterpreter()
        with pytest.raises(ValueError, match="Model and data must be loaded"):
            interpreter.plot_partial_dependence(features=0)

    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not available")
    def test_calculate_shap(self):
        """Test calculating SHAP values."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        interpreter = ModelInterpreter()
        interpreter.load_model_and_data(model, X, y)

        importance = interpreter.calculate_shap()
        assert len(importance) == 5

    def test_calculate_shap_no_data(self):
        """Test SHAP calculation without loading data."""
        interpreter = ModelInterpreter()
        with pytest.raises(ValueError, match="Model and data must be loaded"):
            interpreter.calculate_shap()

    def test_generate_all_interpretations(self):
        """Test generating all interpretations."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        interpreter = ModelInterpreter()
        interpreter.load_model_and_data(model, X, y)

        results = interpreter.generate_all_interpretations()

        assert "permutation_importance" in results
        if SHAP_AVAILABLE:
            assert "shap" in results


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow_classification(self):
        """Test complete workflow for classification."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        interpreter = ModelInterpreter()
        interpreter.load_model_and_data(model, X, y)

        perm_importance = interpreter.calculate_permutation_importance()
        interpreter.plot_partial_dependence(features=[0, 1])

        assert perm_importance is not None

    def test_complete_workflow_regression(self):
        """Test complete workflow for regression."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randn(200)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        interpreter = ModelInterpreter()
        interpreter.load_model_and_data(model, X, y)

        perm_importance = interpreter.calculate_permutation_importance()
        interpreter.plot_partial_dependence(features=0)

        assert perm_importance is not None
