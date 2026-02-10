"""Unit tests for PCA implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import PCA


class TestPCA:
    """Test PCA functionality."""

    def test_initialization(self):
        """Test model initialization."""
        pca = PCA(n_components=2)
        assert pca.n_components == 2
        assert pca.whiten is False
        assert pca.components is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        pca = PCA(n_components=3, whiten=True)
        assert pca.n_components == 3
        assert pca.whiten is True

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        pca.fit(X)

        assert pca.components is not None
        assert pca.explained_variance is not None
        assert pca.explained_variance_ratio is not None
        assert pca.mean is not None

    def test_fit_insufficient_samples(self):
        """Test that insufficient samples raise error."""
        X = np.array([[1, 2]])
        pca = PCA()
        with pytest.raises(ValueError, match="at least 2 samples"):
            pca.fit(X)

    def test_fit_with_n_components_int(self):
        """Test fitting with integer n_components."""
        X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
        pca = PCA(n_components=2)
        pca.fit(X)

        assert pca.n_components_ == 2
        assert pca.components.shape[0] == 2

    def test_fit_with_n_components_float(self):
        """Test fitting with float n_components (variance threshold)."""
        X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
        pca = PCA(n_components=0.95)
        pca.fit(X)

        assert pca.n_components_ >= 1
        cumulative_var = np.sum(pca.explained_variance_ratio)
        assert cumulative_var >= 0.95

    def test_fit_n_components_float_invalid(self):
        """Test that invalid float n_components raises error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        pca = PCA(n_components=1.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            pca.fit(X)

    def test_transform_before_fit(self):
        """Test that transformation before fitting raises error."""
        pca = PCA(n_components=2)
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            pca.transform(X)

    def test_transform(self):
        """Test data transformation."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        pca.fit(X)

        X_transformed = pca.transform(X)
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] == 2

    def test_fit_transform(self):
        """Test fit_transform method."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)

        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] == 2

    def test_inverse_transform(self):
        """Test inverse transformation."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        pca.fit(X)

        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        assert X_reconstructed.shape == X.shape

    def test_inverse_transform_before_fit(self):
        """Test that inverse transformation before fitting raises error."""
        pca = PCA(n_components=2)
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            pca.inverse_transform(X)

    def test_get_explained_variance(self):
        """Test getting explained variance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        pca.fit(X)

        explained_variance = pca.get_explained_variance()
        assert len(explained_variance) == 2
        assert np.all(explained_variance >= 0)

    def test_get_explained_variance_ratio(self):
        """Test getting explained variance ratio."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        pca.fit(X)

        explained_variance_ratio = pca.get_explained_variance_ratio()
        assert len(explained_variance_ratio) == 2
        assert np.all(explained_variance_ratio >= 0)
        assert np.all(explained_variance_ratio <= 1)

    def test_get_cumulative_variance(self):
        """Test getting cumulative variance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        pca.fit(X)

        cumulative_variance = pca.get_cumulative_variance()
        assert len(cumulative_variance) == 2
        assert cumulative_variance[-1] <= 1.0
        assert np.all(np.diff(cumulative_variance) >= 0)

    def test_plot_explained_variance_no_exception(self):
        """Test that plot doesn't raise exceptions."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        pca.fit(X)
        pca.plot_explained_variance(show=False)

    def test_plot_explained_variance_cumulative(self):
        """Test plotting cumulative variance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        pca.fit(X)
        pca.plot_explained_variance(show=False, cumulative=True)

    def test_plot_explained_variance_save(self):
        """Test saving explained variance plot."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        pca.fit(X)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name
            try:
                pca.plot_explained_variance(save_path=save_path, show=False)
                assert Path(save_path).exists()
            finally:
                Path(save_path).unlink()

    def test_plot_components_no_exception(self):
        """Test that component plot doesn't raise exceptions."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        pca.fit(X)
        pca.plot_components(show=False)

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [2, 3, 4, 5, 6]}
        )
        pca = PCA(n_components=2)
        pca.fit(df.values)

        X_transformed = pca.transform(df.values)
        assert len(X_transformed) == len(df)

    def test_variance_ratio_sums_to_one(self):
        """Test that variance ratios sum appropriately."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=None)
        pca.fit(X)

        total_variance = np.sum(pca.explained_variance_ratio)
        assert abs(total_variance - 1.0) < 1e-10

    def test_whiten(self):
        """Test whitening option."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2, whiten=True)
        pca.fit(X)

        assert pca.components is not None
        X_transformed = pca.transform(X)
        assert X_transformed.shape[1] == 2

    def test_dimensionality_reduction(self):
        """Test dimensionality reduction."""
        X = np.random.randn(100, 10)
        pca = PCA(n_components=3)
        pca.fit(X)

        X_transformed = pca.transform(X)
        assert X_transformed.shape[1] == 3
        assert X_transformed.shape[0] == len(X)

    def test_reconstruction_quality(self):
        """Test reconstruction quality."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        pca = PCA(n_components=2)
        pca.fit(X)

        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        mse = np.mean((X - X_reconstructed) ** 2)
        assert mse < 1e-10
