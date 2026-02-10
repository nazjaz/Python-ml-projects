"""Unit tests for Hyperparameter Tuning implementation."""

import numpy as np
import pytest

from src.example_estimator import SimpleClassifier, SimpleRegressor
from src.main import BayesianOptimization, GridSearchCV, RandomSearchCV


class TestGridSearchCV:
    """Test Grid Search functionality."""

    def test_initialization(self):
        """Test grid search initialization."""
        estimator = SimpleClassifier()
        param_grid = {"max_depth": [3, 5], "min_samples_split": [2, 5]}
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3)
        assert gs.cv == 3
        assert len(gs._generate_param_combinations()) == 4

    def test_generate_param_combinations(self):
        """Test parameter combination generation."""
        estimator = SimpleClassifier()
        param_grid = {"max_depth": [3, 5], "min_samples_split": [2]}
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid)
        combinations = gs._generate_param_combinations()
        assert len(combinations) == 2

    def test_fit(self):
        """Test grid search fitting."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        estimator = SimpleClassifier()
        param_grid = {"max_depth": [3, 5], "min_samples_split": [2]}
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, verbose=0)
        gs.fit(X, y)

        assert gs.best_params_ is not None
        assert gs.best_score_ is not None
        assert gs.best_estimator_ is not None
        assert gs.cv_results_ is not None

    def test_cv_results(self):
        """Test cross-validation results."""
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)

        estimator = SimpleClassifier()
        param_grid = {"max_depth": [3]}
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, verbose=0)
        gs.fit(X, y)

        assert "mean_test_score" in gs.cv_results_
        assert "std_test_score" in gs.cv_results_


class TestRandomSearchCV:
    """Test Random Search functionality."""

    def test_initialization(self):
        """Test random search initialization."""
        estimator = SimpleClassifier()
        param_distributions = {"max_depth": [3, 5, 7], "min_samples_split": [2, 5, 10]}
        rs = RandomSearchCV(
            estimator=estimator, param_distributions=param_distributions, n_iter=5
        )
        assert rs.n_iter == 5

    def test_sample_params(self):
        """Test parameter sampling."""
        estimator = SimpleClassifier()
        param_distributions = {"max_depth": [3, 5], "min_samples_split": [2]}
        rs = RandomSearchCV(
            estimator=estimator, param_distributions=param_distributions, n_iter=5
        )
        params = rs._sample_params()
        assert "max_depth" in params
        assert "min_samples_split" in params

    def test_fit(self):
        """Test random search fitting."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        estimator = SimpleClassifier()
        param_distributions = {"max_depth": [3, 5, 7], "min_samples_split": [2, 5]}
        rs = RandomSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=5,
            cv=3,
            verbose=0,
        )
        rs.fit(X, y)

        assert rs.best_params_ is not None
        assert rs.best_score_ is not None
        assert rs.best_estimator_ is not None


class TestBayesianOptimization:
    """Test Bayesian Optimization functionality."""

    def test_initialization(self):
        """Test Bayesian optimization initialization."""
        estimator = SimpleRegressor()
        param_space = {"alpha": (0.1, 10.0), "learning_rate": (0.001, 0.1)}
        bo = BayesianOptimization(
            estimator=estimator, param_space=param_space, n_iter=5
        )
        assert bo.n_iter == 5
        assert bo.n_initial == 5

    def test_sample_random_params(self):
        """Test random parameter sampling."""
        estimator = SimpleRegressor()
        param_space = {"alpha": (0.1, 10.0), "learning_rate": (0.001, 0.1)}
        bo = BayesianOptimization(
            estimator=estimator, param_space=param_space, n_iter=5
        )
        params = bo._sample_random_params()
        assert "alpha" in params
        assert "learning_rate" in params
        assert 0.1 <= params["alpha"] <= 10.0

    def test_params_to_vector(self):
        """Test parameter to vector conversion."""
        estimator = SimpleRegressor()
        param_space = {"alpha": (0.1, 10.0), "learning_rate": (0.001, 0.1)}
        bo = BayesianOptimization(
            estimator=estimator, param_space=param_space, n_iter=5
        )
        params = {"alpha": 5.0, "learning_rate": 0.05}
        vector = bo._params_to_vector(params)
        assert len(vector) == 2

    def test_acquisition_function(self):
        """Test acquisition function."""
        estimator = SimpleRegressor()
        param_space = {"alpha": (0.1, 10.0)}
        bo = BayesianOptimization(
            estimator=estimator, param_space=param_space, n_iter=5
        )
        mean = np.array([0.5, 0.6, 0.4])
        std = np.array([0.1, 0.2, 0.15])
        best_score = 0.5

        acquisition = bo._acquisition_function(mean, std, best_score)
        assert len(acquisition) == 3
        assert np.all(acquisition >= 0)

    def test_fit(self):
        """Test Bayesian optimization fitting."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        estimator = SimpleRegressor()
        param_space = {"alpha": (0.1, 10.0), "learning_rate": (0.001, 0.1)}
        bo = BayesianOptimization(
            estimator=estimator,
            param_space=param_space,
            n_iter=5,
            n_initial=3,
            cv=3,
            verbose=0,
        )
        bo.fit(X, y)

        assert bo.best_params_ is not None
        assert bo.best_score_ is not None
        assert bo.best_estimator_ is not None
