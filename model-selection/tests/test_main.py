"""Unit tests for Model Selection implementation."""

import numpy as np
import pytest

from src.example_estimator import SimpleClassifier, SimpleRegressor
from src.main import LearningCurves, NestedCrossValidation


class TestNestedCrossValidation:
    """Test Nested Cross-Validation functionality."""

    def test_initialization(self):
        """Test nested CV initialization."""
        estimators = {"est1": SimpleClassifier()}
        param_grids = {"est1": {"max_depth": [3, 5]}}
        nested_cv = NestedCrossValidation(
            estimators=estimators, param_grids=param_grids, outer_cv=3, inner_cv=3
        )
        assert nested_cv.outer_cv == 3
        assert nested_cv.inner_cv == 3

    def test_generate_param_combinations(self):
        """Test parameter combination generation."""
        estimators = {"est1": SimpleClassifier()}
        param_grids = {"est1": {"max_depth": [3, 5], "min_samples_split": [2]}}
        nested_cv = NestedCrossValidation(
            estimators=estimators, param_grids=param_grids
        )
        combinations = nested_cv._generate_param_combinations(param_grids["est1"])
        assert len(combinations) == 2

    def test_fit(self):
        """Test nested CV fitting."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        estimators = {
            "classifier1": SimpleClassifier(),
            "classifier2": SimpleClassifier(),
        }
        param_grids = {
            "classifier1": {"max_depth": [3, 5]},
            "classifier2": {"max_depth": [3, 5]},
        }

        nested_cv = NestedCrossValidation(
            estimators=estimators,
            param_grids=param_grids,
            outer_cv=3,
            inner_cv=3,
            verbose=0,
        )
        nested_cv.fit(X, y)

        assert nested_cv.best_estimator_name_ is not None
        assert nested_cv.best_score_ is not None
        assert nested_cv.cv_results_ is not None

    def test_cv_results(self):
        """Test CV results structure."""
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)

        estimators = {"est1": SimpleClassifier()}
        param_grids = {"est1": {"max_depth": [3]}}

        nested_cv = NestedCrossValidation(
            estimators=estimators,
            param_grids=param_grids,
            outer_cv=3,
            inner_cv=3,
            verbose=0,
        )
        nested_cv.fit(X, y)

        assert "estimator" in nested_cv.cv_results_
        assert "test_score" in nested_cv.cv_results_


class TestLearningCurves:
    """Test Learning Curves functionality."""

    def test_initialization(self):
        """Test learning curves initialization."""
        estimator = SimpleClassifier()
        lc = LearningCurves(estimator=estimator, cv=5)
        assert lc.cv == 5

    def test_fit(self):
        """Test learning curves fitting."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        estimator = SimpleClassifier()
        lc = LearningCurves(estimator=estimator, cv=3, verbose=0)
        lc.fit(X, y)

        assert lc.train_scores_ is not None
        assert lc.val_scores_ is not None
        assert lc.train_sizes_ is not None

    def test_get_bias_variance_analysis(self):
        """Test bias-variance analysis."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        estimator = SimpleClassifier()
        lc = LearningCurves(estimator=estimator, cv=3, verbose=0)
        lc.fit(X, y)

        analysis = lc.get_bias_variance_analysis()
        assert "final_train_score" in analysis
        assert "final_val_score" in analysis
        assert "gap" in analysis
        assert "diagnosis" in analysis

    def test_get_bias_variance_analysis_before_fit(self):
        """Test error when getting analysis before fitting."""
        estimator = SimpleClassifier()
        lc = LearningCurves(estimator=estimator)
        with pytest.raises(ValueError, match="must be fitted"):
            lc.get_bias_variance_analysis()

    def test_plot_learning_curves(self):
        """Test plotting learning curves."""
        import tempfile
        from pathlib import Path

        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        estimator = SimpleClassifier()
        lc = LearningCurves(estimator=estimator, cv=3, verbose=0)
        lc.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "learning_curves.png"
            lc.plot_learning_curves(save_path=str(save_path), show=False)
            assert save_path.exists()
