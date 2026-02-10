"""Unit tests for Ensemble Methods implementation."""

import numpy as np
import pandas as pd
import pytest

from src.main import (
    BaggingClassifier,
    SimpleDecisionTree,
    SimpleKNN,
    StackingClassifier,
    VotingClassifier,
)


class TestSimpleDecisionTree:
    """Test Simple Decision Tree functionality."""

    def test_initialization(self):
        """Test tree initialization."""
        dt = SimpleDecisionTree(max_depth=3)
        assert dt.max_depth == 3
        assert dt.tree is None

    def test_fit(self):
        """Test fitting tree."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        dt = SimpleDecisionTree(max_depth=2)
        dt.fit(X, y)

        assert dt.tree is not None
        assert dt.classes_ is not None

    def test_predict(self):
        """Test prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        dt = SimpleDecisionTree(max_depth=2)
        dt.fit(X, y)

        predictions = dt.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in dt.classes_ for pred in predictions)


class TestSimpleKNN:
    """Test Simple KNN functionality."""

    def test_initialization(self):
        """Test KNN initialization."""
        knn = SimpleKNN(n_neighbors=5)
        assert knn.n_neighbors == 5

    def test_fit(self):
        """Test fitting KNN."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        knn = SimpleKNN(n_neighbors=3)
        knn.fit(X, y)

        assert knn.X_train is not None
        assert knn.y_train is not None

    def test_predict(self):
        """Test prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        knn = SimpleKNN(n_neighbors=3)
        knn.fit(X, y)

        predictions = knn.predict(X)
        assert len(predictions) == len(X)


class TestVotingClassifier:
    """Test Voting Classifier functionality."""

    def test_initialization(self):
        """Test voting classifier initialization."""
        dt = SimpleDecisionTree(max_depth=3)
        knn = SimpleKNN(n_neighbors=5)
        voting = VotingClassifier(
            estimators=[("dt", dt), ("knn", knn)], voting="hard"
        )
        assert len(voting.estimators) == 2
        assert voting.voting == "hard"

    def test_fit(self):
        """Test fitting voting classifier."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        dt = SimpleDecisionTree(max_depth=3)
        knn = SimpleKNN(n_neighbors=5)
        voting = VotingClassifier(
            estimators=[("dt", dt), ("knn", knn)], voting="hard"
        )
        voting.fit(X, y)

        assert voting.classes_ is not None

    def test_predict_hard(self):
        """Test hard voting prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        dt = SimpleDecisionTree(max_depth=3)
        knn = SimpleKNN(n_neighbors=5)
        voting = VotingClassifier(
            estimators=[("dt", dt), ("knn", knn)], voting="hard"
        )
        voting.fit(X, y)

        predictions = voting.predict(X)
        assert len(predictions) == len(X)

    def test_predict_soft(self):
        """Test soft voting prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        dt = SimpleDecisionTree(max_depth=3)
        knn = SimpleKNN(n_neighbors=5)
        voting = VotingClassifier(
            estimators=[("dt", dt), ("knn", knn)], voting="soft"
        )
        voting.fit(X, y)

        predictions = voting.predict(X)
        assert len(predictions) == len(X)

    def test_score(self):
        """Test scoring."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        dt = SimpleDecisionTree(max_depth=3)
        knn = SimpleKNN(n_neighbors=5)
        voting = VotingClassifier(
            estimators=[("dt", dt), ("knn", knn)], voting="hard"
        )
        voting.fit(X, y)

        accuracy = voting.score(X, y)
        assert 0 <= accuracy <= 1


class TestBaggingClassifier:
    """Test Bagging Classifier functionality."""

    def test_initialization(self):
        """Test bagging classifier initialization."""
        base_estimator = SimpleDecisionTree(max_depth=3)
        bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=5)
        assert bagging.n_estimators == 5
        assert len(bagging.estimators_) == 0

    def test_fit(self):
        """Test fitting bagging classifier."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        base_estimator = SimpleDecisionTree(max_depth=3)
        bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=5)
        bagging.fit(X, y)

        assert len(bagging.estimators_) == 5
        assert bagging.classes_ is not None

    def test_predict(self):
        """Test prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        base_estimator = SimpleDecisionTree(max_depth=3)
        bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=5)
        bagging.fit(X, y)

        predictions = bagging.predict(X)
        assert len(predictions) == len(X)

    def test_score(self):
        """Test scoring."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        base_estimator = SimpleDecisionTree(max_depth=3)
        bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=5)
        bagging.fit(X, y)

        accuracy = bagging.score(X, y)
        assert 0 <= accuracy <= 1


class TestStackingClassifier:
    """Test Stacking Classifier functionality."""

    def test_initialization(self):
        """Test stacking classifier initialization."""
        dt = SimpleDecisionTree(max_depth=3)
        knn = SimpleKNN(n_neighbors=5)
        meta = SimpleDecisionTree(max_depth=2)
        stacking = StackingClassifier(
            base_estimators=[("dt", dt), ("knn", knn)],
            meta_estimator=meta,
            cv=3,
        )
        assert len(stacking.base_estimators) == 2
        assert stacking.cv == 3

    def test_fit(self):
        """Test fitting stacking classifier."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        dt = SimpleDecisionTree(max_depth=3)
        knn = SimpleKNN(n_neighbors=3)
        meta = SimpleDecisionTree(max_depth=2)
        stacking = StackingClassifier(
            base_estimators=[("dt", dt), ("knn", knn)],
            meta_estimator=meta,
            cv=3,
        )
        stacking.fit(X, y)

        assert stacking.classes_ is not None

    def test_predict(self):
        """Test prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        dt = SimpleDecisionTree(max_depth=3)
        knn = SimpleKNN(n_neighbors=3)
        meta = SimpleDecisionTree(max_depth=2)
        stacking = StackingClassifier(
            base_estimators=[("dt", dt), ("knn", knn)],
            meta_estimator=meta,
            cv=3,
        )
        stacking.fit(X, y)

        predictions = stacking.predict(X)
        assert len(predictions) == len(X)

    def test_score(self):
        """Test scoring."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        dt = SimpleDecisionTree(max_depth=3)
        knn = SimpleKNN(n_neighbors=3)
        meta = SimpleDecisionTree(max_depth=2)
        stacking = StackingClassifier(
            base_estimators=[("dt", dt), ("knn", knn)],
            meta_estimator=meta,
            cv=3,
        )
        stacking.fit(X, y)

        accuracy = stacking.score(X, y)
        assert 0 <= accuracy <= 1
