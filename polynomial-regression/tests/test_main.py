"""Unit tests for Polynomial Regression implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.main import (
    PolynomialRegression,
    cross_validate_degree,
    select_best_degree,
)


class TestPolynomialRegression:
    """Test Polynomial Regression functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = PolynomialRegression(degree=2)
        assert model.degree == 2
        assert model.regularization is None
        assert model.alpha == 1.0
        assert model.fit_intercept is True
        assert model.coefficients is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        model = PolynomialRegression(
            degree=3, regularization="l2", alpha=0.5, fit_intercept=False
        )
        assert model.degree == 3
        assert model.regularization == "l2"
        assert model.alpha == 0.5
        assert model.fit_intercept is False

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = PolynomialRegression(degree=1)
        model.fit(X, y)

        assert model.coefficients is not None
        assert model.feature_names is not None

    def test_fit_invalid_degree(self):
        """Test that invalid degree raises error."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        model = PolynomialRegression(degree=0)
        with pytest.raises(ValueError, match="at least 1"):
            model.fit(X, y)

    def test_fit_invalid_alpha(self):
        """Test that negative alpha raises error."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        model = PolynomialRegression(degree=2, alpha=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            model.fit(X, y)

    def test_fit_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise error."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4])
        model = PolynomialRegression(degree=2)
        with pytest.raises(ValueError, match="same length"):
            model.fit(X, y)

    def test_fit_with_l2_regularization(self):
        """Test fitting with L2 regularization."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = PolynomialRegression(degree=2, regularization="l2", alpha=0.1)
        model.fit(X, y)

        assert model.coefficients is not None

    def test_fit_with_ridge_regularization(self):
        """Test fitting with Ridge regularization."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = PolynomialRegression(degree=2, regularization="ridge", alpha=0.1)
        model.fit(X, y)

        assert model.coefficients is not None

    def test_fit_with_l1_regularization(self):
        """Test fitting with L1 regularization."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = PolynomialRegression(degree=2, regularization="l1", alpha=0.1)
        model.fit(X, y)

        assert model.coefficients is not None

    def test_fit_with_lasso_regularization(self):
        """Test fitting with Lasso regularization."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = PolynomialRegression(degree=2, regularization="lasso", alpha=0.1)
        model.fit(X, y)

        assert model.coefficients is not None

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        model = PolynomialRegression(degree=2)
        X = np.array([[1], [2], [3]])
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)

    def test_predict(self):
        """Test prediction."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = PolynomialRegression(degree=1)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_score(self):
        """Test R-squared score."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = PolynomialRegression(degree=1)
        model.fit(X, y)

        score = model.score(X, y)
        assert 0 <= score <= 1

    def test_mse(self):
        """Test mean squared error."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = PolynomialRegression(degree=1)
        model.fit(X, y)

        mse = model.mse(X, y)
        assert mse >= 0

    def test_get_coefficients(self):
        """Test getting coefficients."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = PolynomialRegression(degree=2)
        model.fit(X, y)

        coefficients = model.get_coefficients()
        assert isinstance(coefficients, dict)
        assert len(coefficients) > 0

    def test_get_coefficients_before_fit(self):
        """Test that getting coefficients before fitting raises error."""
        model = PolynomialRegression(degree=2)
        with pytest.raises(ValueError, match="must be fitted"):
            model.get_coefficients()

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "target": [2, 4, 6, 8, 10]
        })
        X = df[["feature1"]]
        y = df["target"]

        model = PolynomialRegression(degree=2)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(df)

    def test_with_list_input(self):
        """Test with list input."""
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 6, 8, 10]

        model = PolynomialRegression(degree=2)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_multiple_features(self):
        """Test with multiple features."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([3, 5, 7, 9, 11])
        model = PolynomialRegression(degree=2)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_high_degree(self):
        """Test with high polynomial degree."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
        model = PolynomialRegression(degree=5)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)


class TestCrossValidation:
    """Test cross-validation functionality."""

    def test_cross_validate_degree(self):
        """Test cross-validation for degree selection."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

        cv_results = cross_validate_degree(
            X, y, degree_range=(1, 5), cv=3, scoring="mse"
        )

        assert isinstance(cv_results, dict)
        assert len(cv_results) > 0
        for degree, results in cv_results.items():
            assert "mean" in results
            assert "std" in results
            assert "scores" in results

    def test_cross_validate_degree_with_regularization(self):
        """Test cross-validation with regularization."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

        cv_results = cross_validate_degree(
            X,
            y,
            degree_range=(1, 5),
            cv=3,
            regularization="l2",
            alpha=0.1,
            scoring="mse",
        )

        assert isinstance(cv_results, dict)
        assert len(cv_results) > 0

    def test_cross_validate_degree_invalid_range(self):
        """Test that invalid degree range raises error."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        with pytest.raises(ValueError, match="Invalid degree range"):
            cross_validate_degree(X, y, degree_range=(5, 1), cv=3)

    def test_cross_validate_degree_invalid_cv(self):
        """Test that invalid CV folds raises error."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        with pytest.raises(ValueError, match="at least 2"):
            cross_validate_degree(X, y, degree_range=(1, 3), cv=1)

    def test_select_best_degree_mse(self):
        """Test selecting best degree for MSE."""
        cv_results = {
            1: {"mean": 10.0, "std": 1.0, "scores": [9, 10, 11]},
            2: {"mean": 5.0, "std": 1.0, "scores": [4, 5, 6]},
            3: {"mean": 8.0, "std": 1.0, "scores": [7, 8, 9]},
        }

        best_degree = select_best_degree(cv_results, scoring="mse")
        assert best_degree == 2

    def test_select_best_degree_r2(self):
        """Test selecting best degree for R2."""
        cv_results = {
            1: {"mean": 0.5, "std": 0.1, "scores": [0.4, 0.5, 0.6]},
            2: {"mean": 0.9, "std": 0.1, "scores": [0.8, 0.9, 1.0]},
            3: {"mean": 0.7, "std": 0.1, "scores": [0.6, 0.7, 0.8]},
        }

        best_degree = select_best_degree(cv_results, scoring="r2")
        assert best_degree == 2
