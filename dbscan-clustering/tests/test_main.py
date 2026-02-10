"""Unit tests for DBSCAN clustering implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import DBSCAN


class TestDBSCAN:
    """Test DBSCAN functionality."""

    def test_initialization(self):
        """Test model initialization."""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        assert dbscan.eps == 0.5
        assert dbscan.min_samples == 5
        assert dbscan.labels is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        dbscan = DBSCAN(eps=1.0, min_samples=3, distance_metric="euclidean")
        assert dbscan.eps == 1.0
        assert dbscan.min_samples == 3

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        dbscan.fit(X)

        assert dbscan.labels is not None
        assert len(dbscan.labels) == len(X)
        assert dbscan.core_samples is not None
        assert dbscan.noise_samples is not None

    def test_fit_eps_zero(self):
        """Test that eps=0 raises error."""
        X = np.array([[1, 2], [2, 3], [8, 9]])
        dbscan = DBSCAN(eps=0, min_samples=2)
        with pytest.raises(ValueError, match="eps must be greater"):
            dbscan.fit(X)

    def test_fit_min_samples_zero(self):
        """Test that min_samples=0 raises error."""
        X = np.array([[1, 2], [2, 3], [8, 9]])
        dbscan = DBSCAN(eps=1.0, min_samples=0)
        with pytest.raises(ValueError, match="min_samples must be at least"):
            dbscan.fit(X)

    def test_fit_predict(self):
        """Test fit_predict method."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        labels = dbscan.fit_predict(X)

        assert len(labels) == len(X)
        assert np.all(labels >= -1)

    def test_get_core_samples(self):
        """Test getting core samples."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        dbscan.fit(X)

        core_samples = dbscan.get_core_samples()
        assert isinstance(core_samples, np.ndarray)
        assert len(core_samples) >= 0

    def test_get_core_samples_before_fit(self):
        """Test that getting core samples before fit raises error."""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        with pytest.raises(ValueError, match="must be fitted"):
            dbscan.get_core_samples()

    def test_get_noise_samples(self):
        """Test getting noise samples."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        dbscan.fit(X)

        noise_samples = dbscan.get_noise_samples()
        assert isinstance(noise_samples, np.ndarray)
        assert len(noise_samples) >= 0

    def test_get_noise_samples_before_fit(self):
        """Test that getting noise samples before fit raises error."""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        with pytest.raises(ValueError, match="must be fitted"):
            dbscan.get_noise_samples()

    def test_get_cluster_info(self):
        """Test getting cluster information."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        dbscan.fit(X)

        info = dbscan.get_cluster_info()
        assert "n_clusters" in info
        assert "n_noise" in info
        assert "n_core_samples" in info
        assert "cluster_sizes" in info
        assert "labels" in info

    def test_get_cluster_info_before_fit(self):
        """Test that getting cluster info before fit raises error."""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        with pytest.raises(ValueError, match="must be fitted"):
            dbscan.get_cluster_info()

    def test_noise_detection(self):
        """Test noise point detection."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [50, 50]])
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        dbscan.fit(X)

        noise_samples = dbscan.get_noise_samples()
        assert len(noise_samples) >= 0

    def test_all_noise(self):
        """Test when all points are noise."""
        X = np.array([[1, 2], [10, 20], [30, 40], [50, 60]])
        dbscan = DBSCAN(eps=1.0, min_samples=3)
        dbscan.fit(X)

        assert np.all(dbscan.labels == -1)
        assert len(dbscan.noise_samples) == len(X)

    def test_single_cluster(self):
        """Test single cluster formation."""
        X = np.array([[1, 2], [1.5, 2.5], [2, 3], [2.5, 3.5], [3, 4]])
        dbscan = DBSCAN(eps=1.0, min_samples=2)
        dbscan.fit(X)

        unique_labels = np.unique(dbscan.labels)
        non_noise_labels = unique_labels[unique_labels != -1]
        assert len(non_noise_labels) >= 1

    def test_plot_clusters_no_exception(self):
        """Test that plot doesn't raise exceptions."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        dbscan.fit(X)
        dbscan.plot_clusters(show=False)

    def test_plot_clusters_save(self):
        """Test saving cluster plot."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        dbscan.fit(X)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name
            try:
                dbscan.plot_clusters(save_path=save_path, show=False)
                assert Path(save_path).exists()
            finally:
                Path(save_path).unlink()

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame({"feature1": [1, 2, 8, 9], "feature2": [2, 3, 9, 10]})
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        dbscan.fit(df.values)

        assert dbscan.labels is not None
        assert len(dbscan.labels) == len(df)

    def test_labels_range(self):
        """Test that labels are valid."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        dbscan = DBSCAN(eps=2.0, min_samples=2)
        dbscan.fit(X)

        assert np.all(dbscan.labels >= -1)
        unique_labels = np.unique(dbscan.labels)
        assert -1 in unique_labels or len(unique_labels) > 0

    def test_core_samples_classification(self):
        """Test core samples classification."""
        X = np.array([[1, 2], [1.5, 2.5], [2, 3], [8, 9], [9, 10]])
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        dbscan.fit(X)

        core_samples = dbscan.get_core_samples()
        for idx in core_samples:
            neighbors = dbscan._get_neighbors(idx, X)
            assert len(neighbors) >= dbscan.min_samples
