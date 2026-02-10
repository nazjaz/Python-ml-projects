"""Unit tests for k-means clustering implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import KMeans, ElbowMethod


class TestKMeans:
    """Test KMeans functionality."""

    def test_initialization(self):
        """Test model initialization."""
        kmeans = KMeans(n_clusters=3)
        assert kmeans.n_clusters == 3
        assert kmeans.max_iterations == 300
        assert kmeans.centroids is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        kmeans = KMeans(
            n_clusters=5,
            max_iterations=100,
            tolerance=1e-5,
            init="k-means++",
            random_state=42,
        )
        assert kmeans.n_clusters == 5
        assert kmeans.max_iterations == 100
        assert kmeans.init == "k-means++"

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X)

        assert kmeans.centroids is not None
        assert kmeans.labels is not None
        assert kmeans.inertia is not None
        assert kmeans.n_iterations > 0

    def test_fit_invalid_n_clusters(self):
        """Test that invalid n_clusters raises error."""
        X = np.array([[1], [2], [3]])
        kmeans = KMeans(n_clusters=5)
        with pytest.raises(ValueError, match="cannot be greater"):
            kmeans.fit(X)

    def test_fit_n_clusters_zero(self):
        """Test that n_clusters=0 raises error."""
        X = np.array([[1], [2], [3]])
        kmeans = KMeans(n_clusters=0)
        with pytest.raises(ValueError, match="must be at least 1"):
            kmeans.fit(X)

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        kmeans = KMeans(n_clusters=2)
        X = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="must be fitted"):
            kmeans.predict(X)

    def test_predict(self):
        """Test cluster prediction."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X)

        X_test = np.array([[1.5, 2.5], [8.5, 9.5]])
        predictions = kmeans.predict(X_test)
        assert len(predictions) == len(X_test)
        assert np.all(predictions >= 0)
        assert np.all(predictions < kmeans.n_clusters)

    def test_fit_predict(self):
        """Test fit_predict method."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X)

        assert len(labels) == len(X)
        assert np.all(labels >= 0)
        assert np.all(labels < kmeans.n_clusters)

    def test_random_initialization(self):
        """Test random initialization."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        kmeans = KMeans(n_clusters=2, init="random", random_state=42)
        kmeans.fit(X)

        assert kmeans.centroids is not None
        assert kmeans.centroids.shape == (2, 2)

    def test_kmeans_plusplus_initialization(self):
        """Test k-means++ initialization."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        kmeans = KMeans(n_clusters=2, init="k-means++", random_state=42)
        kmeans.fit(X)

        assert kmeans.centroids is not None
        assert kmeans.centroids.shape == (2, 2)

    def test_invalid_initialization(self):
        """Test that invalid initialization raises error."""
        X = np.array([[1, 2], [2, 3], [8, 9]])
        kmeans = KMeans(n_clusters=2, init="invalid")
        with pytest.raises(ValueError, match="Unknown initialization"):
            kmeans.fit(X)

    def test_convergence(self):
        """Test that model converges."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        kmeans = KMeans(
            n_clusters=2, max_iterations=300, tolerance=1e-4, random_state=42
        )
        kmeans.fit(X)

        assert kmeans.n_iterations <= kmeans.max_iterations

    def test_inertia_calculation(self):
        """Test inertia calculation."""
        X = np.array([[1, 2], [2, 3], [8, 9], [9, 10], [5, 5]])
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X)

        assert kmeans.inertia >= 0
        assert isinstance(kmeans.inertia, float)

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame({"feature1": [1, 2, 8, 9, 5], "feature2": [2, 3, 9, 10, 5]})
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(df.values)

        predictions = kmeans.predict(df.values)
        assert len(predictions) == len(df)

    def test_single_cluster(self):
        """Test with single cluster."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        kmeans = KMeans(n_clusters=1, random_state=42)
        kmeans.fit(X)

        assert kmeans.centroids.shape == (1, 2)
        assert np.all(kmeans.labels == 0)

    def test_all_same_cluster(self):
        """Test that all points can be in same cluster."""
        X = np.array([[1, 1], [1, 1], [1, 1]])
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X)

        assert kmeans.centroids is not None


class TestElbowMethod:
    """Test ElbowMethod functionality."""

    def test_initialization(self):
        """Test optimizer initialization."""
        elbow = ElbowMethod(k_range=[2, 3, 4, 5])
        assert elbow.k_range == [2, 3, 4, 5]
        assert elbow.n_runs == 1

    def test_fit(self):
        """Test elbow method fitting."""
        X = np.array(
            [
                [1, 2],
                [2, 3],
                [8, 9],
                [9, 10],
                [15, 16],
                [16, 17],
                [5, 5],
                [6, 6],
            ]
        )
        elbow = ElbowMethod(k_range=[2, 3, 4], random_state=42)
        results = elbow.fit(X)

        assert "k_range" in results
        assert "inertias" in results
        assert "results" in results
        assert len(results["inertias"]) == len(elbow.k_range)

    def test_fit_results_structure(self):
        """Test elbow method results structure."""
        X = np.array(
            [
                [1, 2],
                [2, 3],
                [8, 9],
                [9, 10],
                [15, 16],
                [16, 17],
            ]
        )
        elbow = ElbowMethod(k_range=[2, 3, 4], random_state=42)
        results = elbow.fit(X)

        for k, k_result in results["results"].items():
            assert "inertia" in k_result
            assert "std" in k_result
            assert "runs" in k_result
            assert k_result["inertia"] >= 0

    def test_plot_elbow_no_exception(self):
        """Test that plot doesn't raise exceptions."""
        X = np.array(
            [
                [1, 2],
                [2, 3],
                [8, 9],
                [9, 10],
                [15, 16],
                [16, 17],
            ]
        )
        elbow = ElbowMethod(k_range=[2, 3, 4], random_state=42)
        elbow.fit(X)
        elbow.plot_elbow(show=False)

    def test_plot_elbow_save(self):
        """Test saving elbow plot."""
        X = np.array(
            [
                [1, 2],
                [2, 3],
                [8, 9],
                [9, 10],
                [15, 16],
                [16, 17],
            ]
        )
        elbow = ElbowMethod(k_range=[2, 3, 4], random_state=42)
        elbow.fit(X)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name
            try:
                elbow.plot_elbow(save_path=save_path, show=False)
                assert Path(save_path).exists()
            finally:
                Path(save_path).unlink()

    def test_multiple_runs(self):
        """Test elbow method with multiple runs."""
        X = np.array(
            [
                [1, 2],
                [2, 3],
                [8, 9],
                [9, 10],
                [15, 16],
                [16, 17],
            ]
        )
        elbow = ElbowMethod(k_range=[2, 3], n_runs=3, random_state=42)
        results = elbow.fit(X)

        for k, k_result in results["results"].items():
            assert len(k_result["runs"]) == 3

    def test_find_elbow(self):
        """Test elbow detection."""
        X = np.array(
            [
                [1, 2],
                [2, 3],
                [8, 9],
                [9, 10],
                [15, 16],
                [16, 17],
            ]
        )
        elbow = ElbowMethod(k_range=[2, 3, 4, 5], random_state=42)
        results = elbow.fit(X)

        assert "optimal_k" in results
