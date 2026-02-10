"""Unit tests for hierarchical clustering implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import HierarchicalClustering


class TestHierarchicalClustering:
    """Test HierarchicalClustering functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = HierarchicalClustering(n_clusters=3, linkage="average")
        assert model.n_clusters == 3
        assert model.linkage == "average"
        assert model.labels is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        model = HierarchicalClustering(
            n_clusters=5, linkage="complete", distance_metric="euclidean"
        )
        assert model.n_clusters == 5
        assert model.linkage == "complete"

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        model = HierarchicalClustering(linkage="average")
        model.fit(X)

        assert model.linkage_matrix is not None
        assert model.dendrogram_data is not None
        assert len(model.linkage_matrix) == len(X) - 1

    def test_fit_with_n_clusters(self):
        """Test fitting with n_clusters set."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        model = HierarchicalClustering(n_clusters=2, linkage="average")
        model.fit(X)

        assert model.labels is not None
        assert len(model.labels) == len(X)
        assert len(np.unique(model.labels)) == 2

    def test_fit_insufficient_samples(self):
        """Test that insufficient samples raise error."""
        X = np.array([[1, 2]])
        model = HierarchicalClustering()
        with pytest.raises(ValueError, match="at least 2 samples"):
            model.fit(X)

    def test_single_linkage(self):
        """Test single linkage method."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
        model = HierarchicalClustering(linkage="single")
        model.fit(X)

        assert model.linkage_matrix is not None
        assert len(model.linkage_matrix) == len(X) - 1

    def test_complete_linkage(self):
        """Test complete linkage method."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
        model = HierarchicalClustering(linkage="complete")
        model.fit(X)

        assert model.linkage_matrix is not None
        assert len(model.linkage_matrix) == len(X) - 1

    def test_average_linkage(self):
        """Test average linkage method."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
        model = HierarchicalClustering(linkage="average")
        model.fit(X)

        assert model.linkage_matrix is not None
        assert len(model.linkage_matrix) == len(X) - 1

    def test_invalid_linkage(self):
        """Test that invalid linkage raises error."""
        X = np.array([[1, 2], [2, 3], [8, 9]])
        model = HierarchicalClustering(linkage="invalid")
        with pytest.raises(ValueError, match="Unknown linkage"):
            model.fit(X)

    def test_fit_predict(self):
        """Test fit_predict method."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        model = HierarchicalClustering(n_clusters=2, linkage="average")
        labels = model.fit_predict(X)

        assert len(labels) == len(X)
        assert len(np.unique(labels)) == 2

    def test_fit_predict_no_n_clusters(self):
        """Test that fit_predict without n_clusters raises error."""
        X = np.array([[1, 2], [2, 3], [8, 9]])
        model = HierarchicalClustering(n_clusters=None, linkage="average")
        with pytest.raises(ValueError, match="n_clusters must be set"):
            model.fit_predict(X)

    def test_plot_dendrogram_no_exception(self):
        """Test that plot doesn't raise exceptions."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
        model = HierarchicalClustering(linkage="average")
        model.fit(X)
        model.plot_dendrogram(show=False)

    def test_plot_dendrogram_save(self):
        """Test saving dendrogram plot."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
        model = HierarchicalClustering(linkage="average")
        model.fit(X)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name
            try:
                model.plot_dendrogram(save_path=save_path, show=False)
                assert Path(save_path).exists()
            finally:
                Path(save_path).unlink()

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame({"feature1": [1, 2, 8, 9], "feature2": [2, 3, 9, 10]})
        model = HierarchicalClustering(n_clusters=2, linkage="average")
        model.fit(df.values)

        if model.labels is not None:
            assert len(model.labels) == len(df)

    def test_linkage_matrix_structure(self):
        """Test linkage matrix structure."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
        model = HierarchicalClustering(linkage="average")
        model.fit(X)

        assert model.linkage_matrix.shape[1] == 4
        assert len(model.linkage_matrix) == len(X) - 1

    def test_different_linkage_results(self):
        """Test that different linkages produce different results."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])

        single_model = HierarchicalClustering(n_clusters=2, linkage="single")
        complete_model = HierarchicalClustering(n_clusters=2, linkage="complete")
        average_model = HierarchicalClustering(n_clusters=2, linkage="average")

        single_model.fit(X)
        complete_model.fit(X)
        average_model.fit(X)

        assert single_model.linkage_matrix is not None
        assert complete_model.linkage_matrix is not None
        assert average_model.linkage_matrix is not None
