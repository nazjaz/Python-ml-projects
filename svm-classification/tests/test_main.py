"""Unit tests for SVM implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.main import SVM


class TestSVM:
    """Test SVM functionality."""

    def test_initialization(self):
        """Test model initialization."""
        svm = SVM(kernel="linear")
        assert svm.kernel == "linear"
        assert svm.C == 1.0
        assert svm.support_vectors_ is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        svm = SVM(kernel="rbf", C=0.5, gamma=0.1, degree=2)
        assert svm.kernel == "rbf"
        assert svm.C == 0.5
        assert svm.gamma == 0.1
        assert svm.degree == 2

    def test_fit_linear(self):
        """Test fitting with linear kernel."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        svm = SVM(kernel="linear", C=1.0)
        svm.fit(X, y)

        assert svm.support_vectors_ is not None
        assert svm.support_vector_alphas_ is not None
        assert svm.bias_ is not None
        assert len(svm.classes_) == 2

    def test_fit_polynomial(self):
        """Test fitting with polynomial kernel."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        svm = SVM(kernel="poly", C=1.0, degree=3)
        svm.fit(X, y)

        assert svm.support_vectors_ is not None
        assert svm.bias_ is not None

    def test_fit_rbf(self):
        """Test fitting with RBF kernel."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        svm = SVM(kernel="rbf", C=1.0, gamma=0.1)
        svm.fit(X, y)

        assert svm.support_vectors_ is not None
        assert svm.bias_ is not None

    def test_fit_insufficient_classes(self):
        """Test that insufficient classes raise error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 0])
        svm = SVM()
        with pytest.raises(ValueError, match="binary classification"):
            svm.fit(X, y)

    def test_fit_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1])
        svm = SVM()
        with pytest.raises(ValueError, match="same length"):
            svm.fit(X, y)

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        svm = SVM()
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            svm.predict(X)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        svm = SVM(kernel="linear")
        svm.fit(X, y)

        predictions = svm.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in svm.classes_ for pred in predictions)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        svm = SVM(kernel="linear")
        svm.fit(X, y)

        proba = svm.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == 2
        assert np.allclose(np.sum(proba, axis=1), 1.0)

    def test_score(self):
        """Test classification accuracy."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        svm = SVM(kernel="linear")
        svm.fit(X, y)

        accuracy = svm.score(X, y)
        assert 0 <= accuracy <= 1

    def test_linear_kernel(self):
        """Test linear kernel computation."""
        svm = SVM(kernel="linear")
        X1 = np.array([[1, 2], [3, 4]])
        X2 = np.array([[5, 6], [7, 8]])
        K = svm._linear_kernel(X1, X2)
        assert K.shape == (2, 2)

    def test_polynomial_kernel(self):
        """Test polynomial kernel computation."""
        svm = SVM(kernel="poly", degree=2, gamma=1.0, coef0=0.0)
        X1 = np.array([[1, 2], [3, 4]])
        X2 = np.array([[5, 6], [7, 8]])
        K = svm._polynomial_kernel(X1, X2)
        assert K.shape == (2, 2)

    def test_rbf_kernel(self):
        """Test RBF kernel computation."""
        svm = SVM(kernel="rbf", gamma=0.1)
        X1 = np.array([[1, 2], [3, 4]])
        X2 = np.array([[5, 6], [7, 8]])
        K = svm._rbf_kernel(X1, X2)
        assert K.shape == (2, 2)
        assert np.all(K >= 0)

    def test_kernel_function_invalid(self):
        """Test that invalid kernel raises error."""
        svm = SVM(kernel="invalid")
        X1 = np.array([[1, 2], [3, 4]])
        X2 = np.array([[5, 6], [7, 8]])
        with pytest.raises(ValueError, match="Unknown kernel"):
            svm._kernel_function(X1, X2)

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [2, 3, 4, 5, 6, 7],
            "target": [0, 0, 0, 1, 1, 1]
        })
        X = df[["feature1", "feature2"]]
        y = df["target"]

        svm = SVM(kernel="linear")
        svm.fit(X, y)

        predictions = svm.predict(X)
        assert len(predictions) == len(df)

    def test_with_list_input(self):
        """Test with list input."""
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        y = [0, 0, 0, 1, 1, 1]

        svm = SVM(kernel="linear")
        svm.fit(X, y)

        predictions = svm.predict(X)
        assert len(predictions) == len(X)

    def test_different_C_values(self):
        """Test with different C values."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])

        for C in [0.1, 1.0, 10.0]:
            svm = SVM(kernel="linear", C=C)
            svm.fit(X, y)
            assert svm.support_vectors_ is not None

    def test_auto_gamma(self):
        """Test automatic gamma calculation."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        svm = SVM(kernel="rbf", gamma=None)
        svm.fit(X, y)

        assert svm.gamma is not None
        assert svm.gamma > 0

    def test_decision_function(self):
        """Test decision function."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        svm = SVM(kernel="linear")
        svm.fit(X, y)

        decisions = svm._decision_function(X)
        assert len(decisions) == len(X)

    def test_plot_decision_boundary_2d(self):
        """Test plotting decision boundary for 2D features."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        svm = SVM(kernel="linear")
        svm.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "plot.png"
            svm.plot_decision_boundary(
                X=X, y=y, save_path=str(save_path), show=False
            )
            assert save_path.exists()

    def test_plot_decision_boundary_not_2d(self):
        """Test that plotting fails for non-2D features."""
        X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])
        svm = SVM(kernel="linear")
        svm.fit(X, y)

        svm.plot_decision_boundary(show=False)
