"""Unit tests for LDA implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import LDA


class TestLDA:
    """Test LDA functionality."""

    def test_initialization(self):
        """Test model initialization."""
        lda = LDA(n_components=2)
        assert lda.n_components == 2
        assert lda.solver == "eigen"
        assert lda.shrinkage is None
        assert lda.scalings is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        lda = LDA(n_components=3, solver="svd", shrinkage=0.5)
        assert lda.n_components == 3
        assert lda.solver == "svd"
        assert lda.shrinkage == 0.5

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(n_components=1)
        lda.fit(X, y)

        assert lda.scalings is not None
        assert lda.classes_ is not None
        assert lda.means_ is not None
        assert lda.priors_ is not None
        assert lda.xbar_ is not None

    def test_fit_insufficient_classes(self):
        """Test that insufficient classes raise error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 0])
        lda = LDA()
        with pytest.raises(ValueError, match="at least 2 classes"):
            lda.fit(X, y)

    def test_fit_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1])
        lda = LDA()
        with pytest.raises(ValueError, match="same length"):
            lda.fit(X, y)

    def test_fit_with_n_components(self):
        """Test fitting with specified n_components."""
        X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(n_components=1)
        lda.fit(X, y)

        assert lda.n_components_ == 1
        assert lda.scalings.shape[1] == 1

    def test_fit_auto_n_components(self):
        """Test fitting with auto n_components."""
        X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(n_components=None)
        lda.fit(X, y)

        assert lda.n_components_ == min(3, 2 - 1)  # min(n_features, n_classes - 1)

    def test_fit_with_svd_solver(self):
        """Test fitting with SVD solver."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(solver="svd", n_components=1)
        lda.fit(X, y)

        assert lda.scalings is not None
        assert lda.n_components_ == 1

    def test_fit_with_shrinkage(self):
        """Test fitting with shrinkage."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(solver="eigen", shrinkage=0.5, n_components=1)
        lda.fit(X, y)

        assert lda.scalings is not None

    def test_fit_invalid_shrinkage(self):
        """Test that invalid shrinkage raises error."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(solver="eigen", shrinkage=1.5, n_components=1)
        with pytest.raises(ValueError, match="between 0 and 1"):
            lda.fit(X, y)

    def test_fit_invalid_solver(self):
        """Test that invalid solver raises error."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(solver="invalid", n_components=1)
        with pytest.raises(ValueError, match="Unknown solver"):
            lda.fit(X, y)

    def test_transform_before_fit(self):
        """Test that transformation before fitting raises error."""
        lda = LDA(n_components=2)
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            lda.transform(X)

    def test_transform(self):
        """Test data transformation."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(n_components=1)
        lda.fit(X, y)

        X_transformed = lda.transform(X)
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] == 1

    def test_fit_transform(self):
        """Test fit_transform method."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(n_components=1)
        X_transformed = lda.fit_transform(X, y)

        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] == 1

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        lda = LDA(n_components=2)
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            lda.predict(X)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(n_components=1)
        lda.fit(X, y)

        predictions = lda.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in lda.classes_ for pred in predictions)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(n_components=1)
        lda.fit(X, y)

        proba = lda.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == len(lda.classes_)
        assert np.allclose(np.sum(proba, axis=1), 1.0)

    def test_score(self):
        """Test classification accuracy."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(n_components=1)
        lda.fit(X, y)

        accuracy = lda.score(X, y)
        assert 0 <= accuracy <= 1

    def test_get_explained_variance_ratio(self):
        """Test getting explained variance ratio."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(n_components=1)
        lda.fit(X, y)

        variance_ratio = lda.get_explained_variance_ratio()
        assert len(variance_ratio) == 1
        assert variance_ratio[0] > 0

    def test_get_explained_variance_ratio_before_fit(self):
        """Test that getting variance ratio before fitting raises error."""
        lda = LDA(n_components=2)
        with pytest.raises(ValueError, match="must be fitted"):
            lda.get_explained_variance_ratio()

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [2, 3, 4, 5, 6, 7],
            "target": [0, 0, 0, 1, 1, 1]
        })
        X = df[["feature1", "feature2"]]
        y = df["target"]

        lda = LDA(n_components=1)
        lda.fit(X, y)

        X_transformed = lda.transform(X)
        assert X_transformed.shape[0] == len(df)

    def test_with_list_input(self):
        """Test with list input."""
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        y = [0, 0, 0, 1, 1, 1]

        lda = LDA(n_components=1)
        lda.fit(X, y)

        X_transformed = lda.transform(X)
        assert X_transformed.shape[0] == len(X)

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X = np.array([
            [1, 2], [2, 3], [3, 4],
            [4, 5], [5, 6], [6, 7],
            [7, 8], [8, 9], [9, 10]
        ])
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        lda = LDA(n_components=2)
        lda.fit(X, y)

        assert len(lda.classes_) == 3
        assert lda.n_components_ == min(2, 3 - 1)

        predictions = lda.predict(X)
        assert len(predictions) == len(X)

    def test_plot_components_without_data(self):
        """Test plotting components without data."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(n_components=1)
        lda.fit(X, y)

        lda.plot_components(show=False)

    def test_plot_components_with_data(self):
        """Test plotting components with data."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        lda = LDA(n_components=1)
        lda.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "plot.png"
            lda.plot_components(X=X, y=y, save_path=str(save_path), show=False)
            assert save_path.exists()
