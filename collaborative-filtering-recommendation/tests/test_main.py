"""Tests for collaborative filtering recommendation module."""

import numpy as np
import pandas as pd
import pytest

from src.main import (
    CollaborativeFilteringRecommender,
    ItemBasedCollaborativeFiltering,
    RecommendationEvaluator,
    SimilarityCalculator,
    UserBasedCollaborativeFiltering,
)


class TestSimilarityCalculator:
    """Test cases for similarity calculation."""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([1, 2, 3])

        similarity = SimilarityCalculator.cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-10

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors."""
        vec1 = np.array([1, 0])
        vec2 = np.array([0, 1])

        similarity = SimilarityCalculator.cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-10

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([0, 0, 0])

        similarity = SimilarityCalculator.cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_pearson_correlation(self):
        """Test Pearson correlation calculation."""
        vec1 = np.array([1, 2, 3, 4, 5])
        vec2 = np.array([2, 4, 6, 8, 10])

        correlation = SimilarityCalculator.pearson_correlation(vec1, vec2)
        assert abs(correlation - 1.0) < 1e-10

    def test_pearson_correlation_negative(self):
        """Test Pearson correlation for negative correlation."""
        vec1 = np.array([1, 2, 3, 4, 5])
        vec2 = np.array([5, 4, 3, 2, 1])

        correlation = SimilarityCalculator.pearson_correlation(vec1, vec2)
        assert abs(correlation - (-1.0)) < 1e-10

    def test_adjusted_cosine_similarity(self):
        """Test adjusted cosine similarity."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([1, 2, 3])
        mean1 = 2.0
        mean2 = 2.0

        similarity = SimilarityCalculator.adjusted_cosine_similarity(
            vec1, vec2, mean1, mean2
        )
        assert abs(similarity - 1.0) < 1e-10


class TestUserBasedCollaborativeFiltering:
    """Test cases for user-based collaborative filtering."""

    def test_initialization(self):
        """Test detector initialization."""
        ubcf = UserBasedCollaborativeFiltering(
            similarity_metric="cosine", n_neighbors=50
        )
        assert ubcf.similarity_metric == "cosine"
        assert ubcf.n_neighbors == 50

    def test_fit(self):
        """Test model fitting."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        ubcf = UserBasedCollaborativeFiltering()
        ubcf.fit(ratings)

        assert ubcf.ratings_matrix is not None
        assert len(ubcf.user_ids) == 3
        assert len(ubcf.item_ids) == 2

    def test_fit_invalid_data(self):
        """Test fitting with invalid data."""
        ratings = pd.DataFrame({"user_id": [1, 2], "item_id": [1, 2]})

        ubcf = UserBasedCollaborativeFiltering()
        with pytest.raises(ValueError, match="ratings must contain columns"):
            ubcf.fit(ratings)

    def test_predict_rating(self):
        """Test rating prediction."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        ubcf = UserBasedCollaborativeFiltering()
        ubcf.fit(ratings)

        prediction = ubcf.predict_rating(1, 3)
        assert isinstance(prediction, (int, float))
        assert 0 <= prediction <= 5

    def test_predict_rating_not_fitted(self):
        """Test prediction without fitting."""
        ubcf = UserBasedCollaborativeFiltering()
        with pytest.raises(ValueError, match="Model must be fitted"):
            ubcf.predict_rating(1, 1)

    def test_recommend_items(self):
        """Test item recommendation."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        ubcf = UserBasedCollaborativeFiltering()
        ubcf.fit(ratings)

        recommendations = ubcf.recommend_items(1, n_recommendations=5)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

    def test_recommend_items_not_fitted(self):
        """Test recommendation without fitting."""
        ubcf = UserBasedCollaborativeFiltering()
        with pytest.raises(ValueError, match="Model must be fitted"):
            ubcf.recommend_items(1)


class TestItemBasedCollaborativeFiltering:
    """Test cases for item-based collaborative filtering."""

    def test_initialization(self):
        """Test detector initialization."""
        ibcf = ItemBasedCollaborativeFiltering(
            similarity_metric="cosine", n_neighbors=50
        )
        assert ibcf.similarity_metric == "cosine"
        assert ibcf.n_neighbors == 50

    def test_fit(self):
        """Test model fitting."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        ibcf = ItemBasedCollaborativeFiltering()
        ibcf.fit(ratings)

        assert ibcf.ratings_matrix is not None
        assert len(ibcf.user_ids) == 3
        assert len(ibcf.item_ids) == 2

    def test_fit_invalid_data(self):
        """Test fitting with invalid data."""
        ratings = pd.DataFrame({"user_id": [1, 2], "item_id": [1, 2]})

        ibcf = ItemBasedCollaborativeFiltering()
        with pytest.raises(ValueError, match="ratings must contain columns"):
            ibcf.fit(ratings)

    def test_predict_rating(self):
        """Test rating prediction."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        ibcf = ItemBasedCollaborativeFiltering()
        ibcf.fit(ratings)

        prediction = ibcf.predict_rating(1, 3)
        assert isinstance(prediction, (int, float))
        assert 0 <= prediction <= 5

    def test_predict_rating_not_fitted(self):
        """Test prediction without fitting."""
        ibcf = ItemBasedCollaborativeFiltering()
        with pytest.raises(ValueError, match="Model must be fitted"):
            ibcf.predict_rating(1, 1)

    def test_recommend_items(self):
        """Test item recommendation."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        ibcf = ItemBasedCollaborativeFiltering()
        ibcf.fit(ratings)

        recommendations = ibcf.recommend_items(1, n_recommendations=5)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

    def test_recommend_items_not_fitted(self):
        """Test recommendation without fitting."""
        ibcf = ItemBasedCollaborativeFiltering()
        with pytest.raises(ValueError, match="Model must be fitted"):
            ibcf.recommend_items(1)


class TestRecommendationEvaluator:
    """Test cases for recommendation evaluator."""

    def test_calculate_rmse(self):
        """Test RMSE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        rmse = RecommendationEvaluator.calculate_rmse(y_true, y_pred)
        assert abs(rmse) < 1e-10

    def test_calculate_rmse_with_error(self):
        """Test RMSE calculation with prediction errors."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])

        rmse = RecommendationEvaluator.calculate_rmse(y_true, y_pred)
        assert rmse > 0
        assert abs(rmse - 1.0) < 1e-10

    def test_calculate_mae(self):
        """Test MAE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        mae = RecommendationEvaluator.calculate_mae(y_true, y_pred)
        assert abs(mae) < 1e-10

    def test_calculate_mae_with_error(self):
        """Test MAE calculation with prediction errors."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])

        mae = RecommendationEvaluator.calculate_mae(y_true, y_pred)
        assert mae > 0
        assert abs(mae - 1.0) < 1e-10

    def test_calculate_precision_recall(self):
        """Test precision and recall calculation."""
        recommendations = [(1, 4.5), (2, 4.0), (3, 3.5), (4, 3.0), (5, 2.5)]
        relevant_items = {1, 2, 3, 6, 7}

        precision, recall = RecommendationEvaluator.calculate_precision_recall(
            recommendations, relevant_items, k=3
        )

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert precision == 1.0
        assert abs(recall - 0.6) < 1e-10


class TestCollaborativeFilteringRecommender:
    """Test cases for main recommender class."""

    def test_initialization(self):
        """Test recommender initialization."""
        recommender = CollaborativeFilteringRecommender()
        assert recommender.user_based is None
        assert recommender.item_based is None

    def test_load_data(self):
        """Test data loading."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 2],
                "rating": [5, 4, 4, 5],
            }
        )

        recommender = CollaborativeFilteringRecommender()
        recommender.load_data(ratings)

        assert recommender.ratings is not None
        assert len(recommender.ratings) == 4

    def test_fit_user_based(self):
        """Test fitting user-based model."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 2],
                "rating": [5, 4, 4, 5],
            }
        )

        recommender = CollaborativeFilteringRecommender()
        recommender.load_data(ratings)
        recommender.fit_user_based()

        assert recommender.user_based is not None

    def test_fit_user_based_no_data(self):
        """Test fitting without loading data."""
        recommender = CollaborativeFilteringRecommender()
        with pytest.raises(ValueError, match="Data must be loaded"):
            recommender.fit_user_based()

    def test_fit_item_based(self):
        """Test fitting item-based model."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 2],
                "rating": [5, 4, 4, 5],
            }
        )

        recommender = CollaborativeFilteringRecommender()
        recommender.load_data(ratings)
        recommender.fit_item_based()

        assert recommender.item_based is not None

    def test_fit_all(self):
        """Test fitting both models."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 2],
                "rating": [5, 4, 4, 5],
            }
        )

        recommender = CollaborativeFilteringRecommender()
        recommender.load_data(ratings)
        recommender.fit_all()

        assert recommender.user_based is not None
        assert recommender.item_based is not None

    def test_recommend_user_based(self):
        """Test recommendation with user-based method."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        recommender = CollaborativeFilteringRecommender()
        recommender.load_data(ratings)
        recommender.fit_user_based()

        recommendations = recommender.recommend(1, method="user_based")
        assert isinstance(recommendations, list)

    def test_recommend_item_based(self):
        """Test recommendation with item-based method."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        recommender = CollaborativeFilteringRecommender()
        recommender.load_data(ratings)
        recommender.fit_item_based()

        recommendations = recommender.recommend(1, method="item_based")
        assert isinstance(recommendations, list)

    def test_recommend_invalid_method(self):
        """Test recommendation with invalid method."""
        recommender = CollaborativeFilteringRecommender()
        with pytest.raises(ValueError, match="Unknown method"):
            recommender.recommend(1, method="invalid")

    def test_evaluate(self):
        """Test evaluation."""
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

        recommender = CollaborativeFilteringRecommender()
        recommender.load_data(train_ratings)
        recommender.fit_user_based()

        evaluation = recommender.evaluate(test_ratings, method="user_based")
        assert "rmse" in evaluation or "mae" in evaluation


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

        recommender = CollaborativeFilteringRecommender()
        recommender.load_data(ratings)
        recommender.fit_all()

        ub_recommendations = recommender.recommend(1, method="user_based")
        ib_recommendations = recommender.recommend(1, method="item_based")

        assert isinstance(ub_recommendations, list)
        assert isinstance(ib_recommendations, list)

    def test_different_similarity_metrics(self):
        """Test with different similarity metrics."""
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [1, 2, 1, 2, 1, 2],
                "rating": [5, 4, 4, 5, 3, 4],
            }
        )

        ubcf_cosine = UserBasedCollaborativeFiltering(similarity_metric="cosine")
        ubcf_cosine.fit(ratings)

        ubcf_pearson = UserBasedCollaborativeFiltering(similarity_metric="pearson")
        ubcf_pearson.fit(ratings)

        pred1 = ubcf_cosine.predict_rating(1, 3)
        pred2 = ubcf_pearson.predict_rating(1, 3)

        assert isinstance(pred1, (int, float))
        assert isinstance(pred2, (int, float))
