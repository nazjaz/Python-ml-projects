"""Tests for SVD matrix factorization recommendation module."""

import numpy as np
import pandas as pd
import pytest

from src.main import (
    MatrixFactorizationEvaluator,
    SVDRecommendationSystem,
    SVDRecommender,
)


class TestSVDRecommender:
    """Test cases for SVD recommender."""

    def test_initialization(self):
        """Test recommender initialization."""
        recommender = SVDRecommender(n_components=50, n_iter=5, random_state=42)
        assert recommender.n_components == 50
        assert recommender.n_iter == 5
        assert recommender.random_state == 42

    def test_fit(self):
        """Test model fitting."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
                "item_id": [1, 2, 1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4, 4, 3],
            }
        )

        recommender = SVDRecommender(n_components=2, random_state=42)
        recommender.fit(ratings)

        assert recommender.svd_model is not None
        assert recommender.user_factors is not None
        assert recommender.item_factors is not None
        assert len(recommender.user_ids) == 4
        assert len(recommender.item_ids) == 2

    def test_fit_invalid_data(self):
        """Test fitting with invalid data."""
        ratings = pd.DataFrame({"user_id": [1, 2], "item_id": [1, 2]})

        recommender = SVDRecommender()
        with pytest.raises(ValueError, match="ratings must contain columns"):
            recommender.fit(ratings)

    def test_fit_auto_adjust_components(self):
        """Test automatic adjustment of n_components."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 2],
                "rating": [5, 4, 4, 5],
            }
        )

        recommender = SVDRecommender(n_components=100, random_state=42)
        recommender.fit(ratings)

        assert recommender.n_components <= min(ratings.pivot_table(
            index="user_id", columns="item_id", values="rating"
        ).shape)

    def test_predict_rating(self):
        """Test rating prediction."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        recommender = SVDRecommender(n_components=2, random_state=42)
        recommender.fit(ratings)

        prediction = recommender.predict_rating(1, 3)
        assert isinstance(prediction, (int, float))
        assert 0 <= prediction <= 5

    def test_predict_rating_not_fitted(self):
        """Test prediction without fitting."""
        recommender = SVDRecommender()
        with pytest.raises(ValueError, match="Model must be fitted"):
            recommender.predict_rating(1, 1)

    def test_predict_rating_new_user(self):
        """Test prediction for new user."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 2],
                "rating": [5, 4, 4, 5],
            }
        )

        recommender = SVDRecommender(n_components=2, random_state=42)
        recommender.fit(ratings)

        prediction = recommender.predict_rating(999, 1)
        assert isinstance(prediction, (int, float))
        assert prediction == recommender.global_mean

    def test_recommend_items(self):
        """Test item recommendation."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        recommender = SVDRecommender(n_components=2, random_state=42)
        recommender.fit(ratings)

        recommendations = recommender.recommend_items(1, n_recommendations=5)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

    def test_recommend_items_not_fitted(self):
        """Test recommendation without fitting."""
        recommender = SVDRecommender()
        with pytest.raises(ValueError, match="Model must be fitted"):
            recommender.recommend_items(1)

    def test_get_explained_variance(self):
        """Test getting explained variance."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        recommender = SVDRecommender(n_components=2, random_state=42)
        recommender.fit(ratings)

        variance = recommender.get_explained_variance()
        assert 0 <= variance <= 1

    def test_get_components(self):
        """Test getting factor matrices."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        recommender = SVDRecommender(n_components=2, random_state=42)
        recommender.fit(ratings)

        user_factors, item_factors = recommender.get_components()
        assert user_factors.shape[0] == len(recommender.user_ids)
        assert item_factors.shape[0] == len(recommender.item_ids)
        assert user_factors.shape[1] == recommender.n_components
        assert item_factors.shape[1] == recommender.n_components


class TestMatrixFactorizationEvaluator:
    """Test cases for matrix factorization evaluator."""

    def test_calculate_rmse(self):
        """Test RMSE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        rmse = MatrixFactorizationEvaluator.calculate_rmse(y_true, y_pred)
        assert abs(rmse) < 1e-10

    def test_calculate_rmse_with_error(self):
        """Test RMSE calculation with prediction errors."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])

        rmse = MatrixFactorizationEvaluator.calculate_rmse(y_true, y_pred)
        assert rmse > 0
        assert abs(rmse - 1.0) < 1e-10

    def test_calculate_mae(self):
        """Test MAE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        mae = MatrixFactorizationEvaluator.calculate_mae(y_true, y_pred)
        assert abs(mae) < 1e-10

    def test_calculate_mae_with_error(self):
        """Test MAE calculation with prediction errors."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])

        mae = MatrixFactorizationEvaluator.calculate_mae(y_true, y_pred)
        assert mae > 0
        assert abs(mae - 1.0) < 1e-10

    def test_calculate_precision_recall(self):
        """Test precision and recall calculation."""
        recommendations = [(1, 4.5), (2, 4.0), (3, 3.5), (4, 3.0), (5, 2.5)]
        relevant_items = {1, 2, 3, 6, 7}

        precision, recall = MatrixFactorizationEvaluator.calculate_precision_recall(
            recommendations, relevant_items, k=3
        )

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert precision == 1.0
        assert abs(recall - 0.6) < 1e-10

    def test_calculate_coverage(self):
        """Test catalog coverage calculation."""
        recommendations_all_users = {
            1: [(1, 4.5), (2, 4.0)],
            2: [(2, 4.5), (3, 4.0)],
            3: [(3, 4.5), (4, 4.0)],
        }
        all_items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

        coverage = MatrixFactorizationEvaluator.calculate_coverage(
            recommendations_all_users, all_items
        )

        assert 0 <= coverage <= 1
        assert coverage == 0.4


class TestSVDRecommendationSystem:
    """Test cases for main recommendation system class."""

    def test_initialization(self):
        """Test system initialization."""
        system = SVDRecommendationSystem()
        assert system.model is None
        assert system.ratings is None

    def test_load_data(self):
        """Test data loading."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 2],
                "rating": [5, 4, 4, 5],
            }
        )

        system = SVDRecommendationSystem()
        system.load_data(ratings)

        assert system.ratings is not None
        assert len(system.ratings) == 4

    def test_fit(self):
        """Test model fitting."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        system = SVDRecommendationSystem()
        system.load_data(ratings)
        system.fit(n_components=2)

        assert system.model is not None
        assert system.model.svd_model is not None

    def test_fit_no_data(self):
        """Test fitting without loading data."""
        system = SVDRecommendationSystem()
        with pytest.raises(ValueError, match="Data must be loaded"):
            system.fit()

    def test_predict_rating(self):
        """Test rating prediction."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        system = SVDRecommendationSystem()
        system.load_data(ratings)
        system.fit(n_components=2)

        prediction = system.predict_rating(1, 3)
        assert isinstance(prediction, (int, float))
        assert 0 <= prediction <= 5

    def test_predict_rating_not_fitted(self):
        """Test prediction without fitting."""
        system = SVDRecommendationSystem()
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1],
                "item_id": [1, 2],
                "rating": [5, 4],
            }
        )
        system.load_data(ratings)

        with pytest.raises(ValueError, match="Model must be fitted"):
            system.predict_rating(1, 1)

    def test_recommend(self):
        """Test recommendation generation."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        system = SVDRecommendationSystem()
        system.load_data(ratings)
        system.fit(n_components=2)

        recommendations = system.recommend(1, n_recommendations=5)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

    def test_recommend_not_fitted(self):
        """Test recommendation without fitting."""
        system = SVDRecommendationSystem()
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1],
                "item_id": [1, 2],
                "rating": [5, 4],
            }
        )
        system.load_data(ratings)

        with pytest.raises(ValueError, match="Model must be fitted"):
            system.recommend(1)

    def test_evaluate(self):
        """Test model evaluation."""
        train_ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        test_ratings = pd.DataFrame(
            {
                "user_id": [1, 2],
                "item_id": [3, 3],
                "rating": [4, 3],
            }
        )

        system = SVDRecommendationSystem()
        system.load_data(train_ratings)
        system.fit(n_components=2)

        evaluation = system.evaluate(test_ratings)
        assert "rmse" in evaluation or "mae" in evaluation

    def test_evaluate_not_fitted(self):
        """Test evaluation without fitting."""
        system = SVDRecommendationSystem()
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1],
                "item_id": [1, 2],
                "rating": [5, 4],
            }
        )
        system.load_data(ratings)

        test_ratings = pd.DataFrame(
            {
                "user_id": [1],
                "item_id": [3],
                "rating": [4],
            }
        )

        with pytest.raises(ValueError, match="Model must be fitted"):
            system.evaluate(test_ratings)

    def test_get_model_info(self):
        """Test getting model information."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        system = SVDRecommendationSystem()
        system.load_data(ratings)
        system.fit(n_components=2)

        info = system.get_model_info()
        assert "n_components" in info
        assert "explained_variance" in info
        assert "n_users" in info
        assert "n_items" in info
        assert "global_mean" in info

    def test_get_model_info_not_fitted(self):
        """Test getting model info without fitting."""
        system = SVDRecommendationSystem()
        with pytest.raises(ValueError, match="Model must be fitted"):
            system.get_model_info()


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(self):
        """Test complete recommendation workflow."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "item_id": [1, 2, 3, 1, 2, 4, 2, 3, 4],
                "rating": [5, 4, 3, 4, 5, 4, 3, 4, 5],
            }
        )

        system = SVDRecommendationSystem()
        system.load_data(ratings)
        system.fit(n_components=2)

        recommendations = system.recommend(1, n_recommendations=5)
        prediction = system.predict_rating(1, 4)
        info = system.get_model_info()

        assert isinstance(recommendations, list)
        assert isinstance(prediction, (int, float))
        assert info is not None

    def test_different_n_components(self):
        """Test with different numbers of components."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
                "item_id": [1, 2, 1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4, 4, 3],
            }
        )

        system1 = SVDRecommendationSystem()
        system1.load_data(ratings)
        system1.fit(n_components=2)

        system2 = SVDRecommendationSystem()
        system2.load_data(ratings)
        system2.fit(n_components=4)

        info1 = system1.get_model_info()
        info2 = system2.get_model_info()

        assert info1["n_components"] == 2
        assert info2["n_components"] == 4
        assert info2["explained_variance"] >= info1["explained_variance"]
