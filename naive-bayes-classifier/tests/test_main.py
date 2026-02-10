"""Unit tests for naive Bayes classifier implementation."""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import GaussianNB, MultinomialNB, BernoulliNB


class TestGaussianNB:
    """Test GaussianNB functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = GaussianNB()
        assert model.smoothing == 1e-9
        assert model.classes is None

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0, 0, 1, 1, 1])
        model = GaussianNB()
        model.fit(X, y)

        assert model.classes is not None
        assert model.class_priors is not None
        assert model.means is not None
        assert model.variances is not None

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        model = GaussianNB()
        X = np.array([[1.0], [2.0]])
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0, 0, 1, 1, 1])
        model = GaussianNB()
        model.fit(X, y)

        X_test = np.array([[1.5], [4.5]])
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert np.all(np.isin(predictions, y))

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0, 0, 1, 1, 1])
        model = GaussianNB()
        model.fit(X, y)

        X_test = np.array([[1.5], [4.5]])
        probabilities = model.predict_proba(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert probabilities.shape[1] == len(model.classes)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)

    def test_score(self):
        """Test accuracy score calculation."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0, 0, 1, 1, 1])
        model = GaussianNB()
        model.fit(X, y)

        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_multiclass(self):
        """Test multiclass classification."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = GaussianNB()
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_multiple_features(self):
        """Test with multiple features."""
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
        y = np.array([0, 0, 1, 1, 1])
        model = GaussianNB()
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)


class TestMultinomialNB:
    """Test MultinomialNB functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = MultinomialNB(alpha=0.5)
        assert model.alpha == 0.5
        assert model.classes is None

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 1, 1, 1])
        model = MultinomialNB()
        model.fit(X, y)

        assert model.classes is not None
        assert model.class_priors is not None
        assert model.feature_counts is not None

    def test_fit_negative_values(self):
        """Test that negative values raise error."""
        X = np.array([[-1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1, 1])
        model = MultinomialNB()
        with pytest.raises(ValueError, match="non-negative"):
            model.fit(X, y)

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        model = MultinomialNB()
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 1, 1, 1])
        model = MultinomialNB()
        model.fit(X, y)

        X_test = np.array([[1, 2], [4, 5]])
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 1, 1, 1])
        model = MultinomialNB()
        model.fit(X, y)

        X_test = np.array([[1, 2], [4, 5]])
        probabilities = model.predict_proba(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)

    def test_score(self):
        """Test accuracy score calculation."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 1, 1, 1])
        model = MultinomialNB()
        model.fit(X, y)

        score = model.score(X, y)
        assert 0.0 <= score <= 1.0


class TestBernoulliNB:
    """Test BernoulliNB functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = BernoulliNB(alpha=0.5, binarize=0.5)
        assert model.alpha == 0.5
        assert model.binarize == 0.5
        assert model.classes is None

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 1]])
        y = np.array([0, 0, 1, 1, 1])
        model = BernoulliNB()
        model.fit(X, y)

        assert model.classes is not None
        assert model.class_priors is not None
        assert model.feature_probs is not None

    def test_fit_with_binarize(self):
        """Test fitting with binarization."""
        X = np.array([[0.3, 0.7], [0.6, 0.4], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
        y = np.array([0, 0, 1, 1, 1])
        model = BernoulliNB(binarize=0.5)
        model.fit(X, y)

        assert model.classes is not None

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        model = BernoulliNB()
        X = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 1]])
        y = np.array([0, 0, 1, 1, 1])
        model = BernoulliNB()
        model.fit(X, y)

        X_test = np.array([[0, 1], [1, 1]])
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 1]])
        y = np.array([0, 0, 1, 1, 1])
        model = BernoulliNB()
        model.fit(X, y)

        X_test = np.array([[0, 1], [1, 1]])
        probabilities = model.predict_proba(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)

    def test_score(self):
        """Test accuracy score calculation."""
        X = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 1]])
        y = np.array([0, 0, 1, 1, 1])
        model = BernoulliNB()
        model.fit(X, y)

        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "feature1": [0, 1, 1, 0, 1],
                "feature2": [1, 0, 1, 0, 1],
                "target": [0, 0, 1, 1, 1],
            }
        )
        X = df[["feature1", "feature2"]]
        y = df["target"]
        model = BernoulliNB()
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(df)


class TestCommonFunctionality:
    """Test common functionality across all variants."""

    def test_fit_empty_data(self):
        """Test that fitting with empty data raises error."""
        for ModelClass in [GaussianNB, MultinomialNB, BernoulliNB]:
            model = ModelClass()
            X = np.array([]).reshape(0, 1)
            y = np.array([])
            with pytest.raises(ValueError, match="cannot be empty"):
                model.fit(X, y)

    def test_fit_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        for ModelClass in [GaussianNB, MultinomialNB, BernoulliNB]:
            model = ModelClass()
            X = np.array([[1], [2], [3]])
            y = np.array([0, 1])
            with pytest.raises(ValueError, match="Length mismatch"):
                model.fit(X, y)

    def test_predict_proba_sums_to_one(self):
        """Test that predicted probabilities sum to one."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0, 0, 1, 1, 1])

        models = [
            GaussianNB(),
            MultinomialNB(),
            BernoulliNB(binarize=2.5),
        ]

        for model in models:
            if isinstance(model, MultinomialNB):
                X_fit = np.abs(X).astype(int)
            else:
                X_fit = X

            model.fit(X_fit, y)
            probabilities = model.predict_proba(X_fit)
            assert np.allclose(np.sum(probabilities, axis=1), 1.0)
