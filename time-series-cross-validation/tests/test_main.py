"""Unit tests for time series cross-validation module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.main import (
    ExpandingWindowSplit,
    TimeSeriesCrossValidator,
    TimeSeriesSplit,
    TimeSeriesValidator,
)


class TestTimeSeriesSplit:
    """Test cases for TimeSeriesSplit."""

    def test_basic_split(self):
        """Test basic time series split."""
        X = np.random.randn(100, 5)
        splitter = TimeSeriesSplit(n_splits=5)
        splits = list(splitter.split(X))

        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert train_idx[-1] < test_idx[0]

    def test_custom_test_size(self):
        """Test split with custom test size."""
        X = np.random.randn(100, 5)
        splitter = TimeSeriesSplit(n_splits=3, test_size=10)
        splits = list(splitter.split(X))

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(test_idx) == 10

    def test_with_gap(self):
        """Test split with gap between train and test."""
        X = np.random.randn(100, 5)
        splitter = TimeSeriesSplit(n_splits=3, test_size=10, gap=5)
        splits = list(splitter.split(X))

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            if len(train_idx) > 0:
                assert train_idx[-1] < test_idx[0] - 5

    def test_max_train_size(self):
        """Test split with maximum train size."""
        X = np.random.randn(100, 5)
        splitter = TimeSeriesSplit(n_splits=3, test_size=10, max_train_size=20)
        splits = list(splitter.split(X))

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) <= 20

    def test_too_many_splits(self):
        """Test error when n_splits exceeds n_samples."""
        X = np.random.randn(10, 5)
        splitter = TimeSeriesSplit(n_splits=20)

        with pytest.raises(ValueError, match="Cannot have n_splits"):
            list(splitter.split(X))

    def test_invalid_parameters(self):
        """Test error with invalid parameters."""
        X = np.random.randn(100, 5)
        splitter = TimeSeriesSplit(n_splits=10, test_size=20)

        with pytest.raises(ValueError, match="too large for n_samples"):
            list(splitter.split(X))


class TestExpandingWindowSplit:
    """Test cases for ExpandingWindowSplit."""

    def test_basic_expanding_window(self):
        """Test basic expanding window split."""
        X = np.random.randn(100, 5)
        splitter = ExpandingWindowSplit(initial_train_size=20, step_size=10)
        splits = list(splitter.split(X))

        assert len(splits) > 0
        prev_train_size = 0
        for train_idx, test_idx in splits:
            assert len(train_idx) >= prev_train_size
            assert len(test_idx) == 10
            prev_train_size = len(train_idx)

    def test_step_size_one(self):
        """Test expanding window with step size 1."""
        X = np.random.randn(50, 5)
        splitter = ExpandingWindowSplit(initial_train_size=10, step_size=1)
        splits = list(splitter.split(X))

        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(test_idx) == 1

    def test_n_splits_limit(self):
        """Test expanding window with n_splits limit."""
        X = np.random.randn(100, 5)
        splitter = ExpandingWindowSplit(
            initial_train_size=20, step_size=10, n_splits=3
        )
        splits = list(splitter.split(X))

        assert len(splits) == 3

    def test_max_train_size(self):
        """Test expanding window with max train size."""
        X = np.random.randn(100, 5)
        splitter = ExpandingWindowSplit(
            initial_train_size=20, step_size=10, max_train_size=30
        )
        splits = list(splitter.split(X))

        for train_idx, test_idx in splits:
            assert len(train_idx) <= 30

    def test_invalid_initial_size(self):
        """Test error with invalid initial train size."""
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="initial_train_size.*>= n_samples"):
            splitter = ExpandingWindowSplit(initial_train_size=20)
            list(splitter.split(X))

        with pytest.raises(ValueError, match="initial_train_size must be >= 1"):
            splitter = ExpandingWindowSplit(initial_train_size=0)
            list(splitter.split(X))


class TestTimeSeriesCrossValidator:
    """Test cases for TimeSeriesCrossValidator."""

    def test_walk_forward_validation(self):
        """Test walk-forward cross-validation."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = LinearRegression()

        validator = TimeSeriesCrossValidator(
            strategy="walk_forward", n_splits=5, test_size=10
        )
        results = validator.cross_validate(model, X, y, scoring="mse")

        assert results["strategy"] == "walk_forward"
        assert results["n_splits"] == 5
        assert len(results["test_scores"]) == 5
        assert "test_mean" in results
        assert "test_std" in results

    def test_expanding_window_validation(self):
        """Test expanding window cross-validation."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = LinearRegression()

        validator = TimeSeriesCrossValidator(
            strategy="expanding_window", initial_train_size=20, step_size=10
        )
        results = validator.cross_validate(model, X, y, scoring="mse")

        assert results["strategy"] == "expanding_window"
        assert len(results["test_scores"]) > 0
        assert "test_mean" in results

    def test_different_scoring_metrics(self):
        """Test different scoring metrics."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = LinearRegression()

        validator = TimeSeriesCrossValidator(strategy="walk_forward", n_splits=3)

        for metric in ["mse", "rmse", "mae", "r2"]:
            results = validator.cross_validate(model, X, y, scoring=metric)
            assert len(results["test_scores"]) == 3
            assert all(isinstance(score, float) for score in results["test_scores"])

    def test_return_train_score(self):
        """Test returning training scores."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = LinearRegression()

        validator = TimeSeriesCrossValidator(strategy="walk_forward", n_splits=3)
        results = validator.cross_validate(
            model, X, y, scoring="mse", return_train_score=True
        )

        assert "train_scores" in results
        assert len(results["train_scores"]) == 3
        assert "train_mean" in results
        assert "train_std" in results

    def test_invalid_strategy(self):
        """Test error with invalid strategy."""
        validator = TimeSeriesCrossValidator(strategy="invalid")

        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = LinearRegression()

        with pytest.raises(ValueError, match="Unknown strategy"):
            validator.cross_validate(model, X, y)

    def test_invalid_scoring(self):
        """Test error with invalid scoring metric."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = LinearRegression()

        validator = TimeSeriesCrossValidator(strategy="walk_forward", n_splits=3)

        with pytest.raises(ValueError, match="Unknown scoring metric"):
            validator.cross_validate(model, X, y, scoring="invalid")

    def test_mismatched_shapes(self):
        """Test error with mismatched X and y shapes."""
        X = np.random.randn(100, 5)
        y = np.random.randn(50)
        model = LinearRegression()

        validator = TimeSeriesCrossValidator(strategy="walk_forward", n_splits=3)

        with pytest.raises(ValueError, match="must have same number of samples"):
            validator.cross_validate(model, X, y)

    def test_split_info(self):
        """Test that split info is included in results."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = LinearRegression()

        validator = TimeSeriesCrossValidator(strategy="walk_forward", n_splits=3)
        results = validator.cross_validate(model, X, y)

        assert "split_info" in results
        assert len(results["split_info"]) == 3
        for info in results["split_info"]:
            assert "fold" in info
            assert "train_size" in info
            assert "test_size" in info


class TestTimeSeriesValidator:
    """Test cases for TimeSeriesValidator."""

    def test_initialization_with_config(self):
        """Test validator initialization with config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "logging": {"level": "DEBUG", "file": "logs/test.log"},
                "cross_validation": {
                    "strategy": "expanding_window",
                    "n_splits": 3,
                    "initial_train_size": 20,
                },
            }
            import yaml

            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            validator = TimeSeriesValidator(config_path=config_path)
            assert validator.config["cross_validation"]["strategy"] == "expanding_window"
        finally:
            config_path.unlink()

    def test_initialization_without_config(self):
        """Test validator initialization without config file."""
        validator = TimeSeriesValidator()
        assert isinstance(validator.config, dict)

    def test_validate_walk_forward(self):
        """Test validation with walk-forward strategy."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = LinearRegression()

        validator = TimeSeriesValidator()
        results = validator.validate(
            model, X, y, strategy="walk_forward", n_splits=3
        )

        assert results["strategy"] == "walk_forward"
        assert len(results["test_scores"]) == 3

    def test_validate_expanding_window(self):
        """Test validation with expanding window strategy."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = LinearRegression()

        validator = TimeSeriesValidator()
        results = validator.validate(
            model,
            X,
            y,
            strategy="expanding_window",
            initial_train_size=20,
            step_size=10,
        )

        assert results["strategy"] == "expanding_window"
        assert len(results["test_scores"]) > 0

    def test_validate_with_dataframe(self):
        """Test validation with pandas DataFrame/Series."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "target": np.random.randn(100),
            }
        )
        X = df[["feature1", "feature2"]]
        y = df["target"]
        model = LinearRegression()

        validator = TimeSeriesValidator()
        results = validator.validate(model, X, y, strategy="walk_forward", n_splits=3)

        assert results["strategy"] == "walk_forward"
        assert len(results["test_scores"]) == 3

    def test_validate_with_different_models(self):
        """Test validation with different model types."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        validator = TimeSeriesValidator()

        for model in [LinearRegression(), DecisionTreeRegressor()]:
            results = validator.validate(
                model, X, y, strategy="walk_forward", n_splits=3
            )
            assert len(results["test_scores"]) == 3


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_walk_forward(self):
        """Test end-to-end walk-forward validation."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(200) * 0.1

        model = LinearRegression()
        validator = TimeSeriesCrossValidator(
            strategy="walk_forward", n_splits=5, test_size=20
        )

        results = validator.cross_validate(
            model, X, y, scoring="mse", return_train_score=True
        )

        assert results["n_splits"] == 5
        assert len(results["test_scores"]) == 5
        assert len(results["train_scores"]) == 5
        assert all(score >= 0 for score in results["test_scores"])

    def test_end_to_end_expanding_window(self):
        """Test end-to-end expanding window validation."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(200) * 0.1

        model = LinearRegression()
        validator = TimeSeriesCrossValidator(
            strategy="expanding_window", initial_train_size=30, step_size=15
        )

        results = validator.cross_validate(
            model, X, y, scoring="r2", return_train_score=True
        )

        assert len(results["test_scores"]) > 0
        assert len(results["train_scores"]) > 0
        assert all(-1 <= score <= 1 for score in results["test_scores"])

    def test_temporal_order_preservation(self):
        """Test that temporal order is preserved in splits."""
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)

        splitter = TimeSeriesSplit(n_splits=3, test_size=10)
        splits = list(splitter.split(X))

        for train_idx, test_idx in splits:
            assert train_idx[-1] < test_idx[0]
            assert np.all(np.diff(train_idx) == 1)
            assert np.all(np.diff(test_idx) == 1)
