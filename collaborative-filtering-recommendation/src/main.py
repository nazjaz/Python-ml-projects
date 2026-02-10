"""Collaborative Filtering Recommendation System.

This module provides implementations of collaborative filtering recommendation
systems using both user-based and item-based approaches with various similarity
metrics and evaluation methods.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Calculate similarity between users or items."""

    @staticmethod
    def cosine_similarity(
        vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def pearson_correlation(
        vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """Calculate Pearson correlation coefficient between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Pearson correlation coefficient between -1 and 1
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        common_mask = ~(np.isnan(vec1) | np.isnan(vec2))
        if np.sum(common_mask) < 2:
            return 0.0

        vec1_common = vec1[common_mask]
        vec2_common = vec2[common_mask]

        if np.std(vec1_common) == 0 or np.std(vec2_common) == 0:
            return 0.0

        correlation, _ = pearsonr(vec1_common, vec2_common)

        if np.isnan(correlation):
            return 0.0

        return float(correlation)

    @staticmethod
    def adjusted_cosine_similarity(
        vec1: np.ndarray, vec2: np.ndarray, mean1: float, mean2: float
    ) -> float:
        """Calculate adjusted cosine similarity (mean-centered).

        Args:
            vec1: First vector
            vec2: Second vector
            mean1: Mean of first vector
            mean2: Mean of second vector

        Returns:
            Adjusted cosine similarity score
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        vec1_centered = vec1 - mean1
        vec2_centered = vec2 - mean2

        return SimilarityCalculator.cosine_similarity(vec1_centered, vec2_centered)


class UserBasedCollaborativeFiltering:
    """User-based collaborative filtering recommendation system."""

    def __init__(
        self,
        similarity_metric: str = "cosine",
        n_neighbors: int = 50,
        min_common_items: int = 1,
    ):
        """Initialize user-based collaborative filtering.

        Args:
            similarity_metric: Similarity metric - "cosine" or "pearson"
                (default: "cosine")
            n_neighbors: Number of similar users to consider (default: 50)
            min_common_items: Minimum common items for similarity calculation
                (default: 1)
        """
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.min_common_items = min_common_items
        self.ratings_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.user_similarities = None
        self.user_means = None

    def fit(self, ratings: pd.DataFrame) -> "UserBasedCollaborativeFiltering":
        """Fit the user-based collaborative filtering model.

        Args:
            ratings: DataFrame with columns ['user_id', 'item_id', 'rating']

        Returns:
            Self for method chaining

        Raises:
            ValueError: If input data is invalid
        """
        required_cols = ["user_id", "item_id", "rating"]
        if not all(col in ratings.columns for col in required_cols):
            raise ValueError(
                f"ratings must contain columns: {required_cols}"
            )

        self.user_ids = ratings["user_id"].unique()
        self.item_ids = ratings["item_id"].unique()

        self.ratings_matrix = ratings.pivot_table(
            index="user_id", columns="item_id", values="rating"
        )

        self.user_means = self.ratings_matrix.mean(axis=1)

        logger.info(
            f"Fitted user-based CF with {len(self.user_ids)} users and "
            f"{len(self.item_ids)} items"
        )

        return self

    def _calculate_user_similarity(
        self, user1_id: Union[int, str], user2_id: Union[int, str]
    ) -> float:
        """Calculate similarity between two users.

        Args:
            user1_id: First user ID
            user2_id: Second user ID

        Returns:
            Similarity score
        """
        if user1_id not in self.ratings_matrix.index:
            return 0.0
        if user2_id not in self.ratings_matrix.index:
            return 0.0

        user1_ratings = self.ratings_matrix.loc[user1_id].values
        user2_ratings = self.ratings_matrix.loc[user2_id].values

        common_mask = ~(np.isnan(user1_ratings) | np.isnan(user2_ratings))
        common_count = np.sum(common_mask)

        if common_count < self.min_common_items:
            return 0.0

        user1_common = user1_ratings[common_mask]
        user2_common = user2_ratings[common_mask]

        if self.similarity_metric == "cosine":
            similarity = SimilarityCalculator.cosine_similarity(
                user1_common, user2_common
            )
        else:
            similarity = SimilarityCalculator.pearson_correlation(
                user1_common, user2_common
            )

        return similarity

    def predict_rating(
        self, user_id: Union[int, str], item_id: Union[int, str]
    ) -> float:
        """Predict rating for a user-item pair.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Predicted rating

        Raises:
            ValueError: If model is not fitted
        """
        if self.ratings_matrix is None:
            raise ValueError("Model must be fitted before prediction")

        if user_id not in self.ratings_matrix.index:
            return self.user_means.mean() if len(self.user_means) > 0 else 0.0

        if item_id not in self.ratings_matrix.columns:
            return self.user_means[user_id] if user_id in self.user_means.index else 0.0

        user_ratings = self.ratings_matrix.loc[user_id]
        if not pd.isna(user_ratings[item_id]):
            return float(user_ratings[item_id])

        user_mean = self.user_means[user_id]

        similarities = []
        ratings = []

        for other_user_id in self.ratings_matrix.index:
            if other_user_id == user_id:
                continue

            other_user_ratings = self.ratings_matrix.loc[other_user_id]
            if pd.isna(other_user_ratings[item_id]):
                continue

            similarity = self._calculate_user_similarity(user_id, other_user_id)
            if similarity > 0:
                other_user_mean = self.user_means[other_user_id]
                similarities.append(similarity)
                ratings.append(other_user_ratings[item_id] - other_user_mean)

        if len(similarities) == 0:
            return float(user_mean)

        similarities = np.array(similarities)
        ratings = np.array(ratings)

        if self.n_neighbors > 0 and len(similarities) > self.n_neighbors:
            top_indices = np.argsort(similarities)[-self.n_neighbors:]
            similarities = similarities[top_indices]
            ratings = ratings[top_indices]

        similarity_sum = np.sum(np.abs(similarities))
        if similarity_sum == 0:
            return float(user_mean)

        prediction = user_mean + np.sum(similarities * ratings) / similarity_sum

        return float(np.clip(prediction, 0, 5))

    def recommend_items(
        self, user_id: Union[int, str], n_recommendations: int = 10
    ) -> List[Tuple[Union[int, str], float]]:
        """Recommend top N items for a user.

        Args:
            user_id: User ID
            n_recommendations: Number of recommendations (default: 10)

        Returns:
            List of (item_id, predicted_rating) tuples, sorted by rating

        Raises:
            ValueError: If model is not fitted
        """
        if self.ratings_matrix is None:
            raise ValueError("Model must be fitted before recommendation")

        if user_id not in self.ratings_matrix.index:
            return []

        user_ratings = self.ratings_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings.isna()].index

        predictions = []
        for item_id in unrated_items:
            predicted_rating = self.predict_rating(user_id, item_id)
            predictions.append((item_id, predicted_rating))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_recommendations]


class ItemBasedCollaborativeFiltering:
    """Item-based collaborative filtering recommendation system."""

    def __init__(
        self,
        similarity_metric: str = "cosine",
        n_neighbors: int = 50,
        min_common_users: int = 1,
    ):
        """Initialize item-based collaborative filtering.

        Args:
            similarity_metric: Similarity metric - "cosine" or "adjusted_cosine"
                (default: "cosine")
            n_neighbors: Number of similar items to consider (default: 50)
            min_common_users: Minimum common users for similarity calculation
                (default: 1)
        """
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.min_common_users = min_common_users
        self.ratings_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.item_similarities = None
        self.item_means = None

    def fit(self, ratings: pd.DataFrame) -> "ItemBasedCollaborativeFiltering":
        """Fit the item-based collaborative filtering model.

        Args:
            ratings: DataFrame with columns ['user_id', 'item_id', 'rating']

        Returns:
            Self for method chaining

        Raises:
            ValueError: If input data is invalid
        """
        required_cols = ["user_id", "item_id", "rating"]
        if not all(col in ratings.columns for col in required_cols):
            raise ValueError(
                f"ratings must contain columns: {required_cols}"
            )

        self.user_ids = ratings["user_id"].unique()
        self.item_ids = ratings["item_id"].unique()

        self.ratings_matrix = ratings.pivot_table(
            index="user_id", columns="item_id", values="rating"
        )

        self.item_means = self.ratings_matrix.mean(axis=0)

        logger.info(
            f"Fitted item-based CF with {len(self.user_ids)} users and "
            f"{len(self.item_ids)} items"
        )

        return self

    def _calculate_item_similarity(
        self, item1_id: Union[int, str], item2_id: Union[int, str]
    ) -> float:
        """Calculate similarity between two items.

        Args:
            item1_id: First item ID
            item2_id: Second item ID

        Returns:
            Similarity score
        """
        if item1_id not in self.ratings_matrix.columns:
            return 0.0
        if item2_id not in self.ratings_matrix.columns:
            return 0.0

        item1_ratings = self.ratings_matrix[item1_id].values
        item2_ratings = self.ratings_matrix[item2_id].values

        common_mask = ~(np.isnan(item1_ratings) | np.isnan(item2_ratings))
        common_count = np.sum(common_mask)

        if common_count < self.min_common_users:
            return 0.0

        item1_common = item1_ratings[common_mask]
        item2_common = item2_ratings[common_mask]

        if self.similarity_metric == "cosine":
            similarity = SimilarityCalculator.cosine_similarity(
                item1_common, item2_common
            )
        else:
            item1_mean = self.item_means[item1_id]
            item2_mean = self.item_means[item2_id]
            similarity = SimilarityCalculator.adjusted_cosine_similarity(
                item1_common, item2_common, item1_mean, item2_mean
            )

        return similarity

    def predict_rating(
        self, user_id: Union[int, str], item_id: Union[int, str]
    ) -> float:
        """Predict rating for a user-item pair.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Predicted rating

        Raises:
            ValueError: If model is not fitted
        """
        if self.ratings_matrix is None:
            raise ValueError("Model must be fitted before prediction")

        if user_id not in self.ratings_matrix.index:
            return self.item_means.mean() if len(self.item_means) > 0 else 0.0

        if item_id not in self.ratings_matrix.columns:
            return self.item_means.mean() if len(self.item_means) > 0 else 0.0

        user_ratings = self.ratings_matrix.loc[user_id]
        if not pd.isna(user_ratings[item_id]):
            return float(user_ratings[item_id])

        item_mean = self.item_means[item_id]

        similarities = []
        ratings = []

        for other_item_id in self.ratings_matrix.columns:
            if other_item_id == item_id:
                continue

            if pd.isna(user_ratings[other_item_id]):
                continue

            similarity = self._calculate_item_similarity(item_id, other_item_id)
            if similarity > 0:
                other_item_mean = self.item_means[other_item_id]
                similarities.append(similarity)
                ratings.append(user_ratings[other_item_id] - other_item_mean)

        if len(similarities) == 0:
            return float(item_mean)

        similarities = np.array(similarities)
        ratings = np.array(ratings)

        if self.n_neighbors > 0 and len(similarities) > self.n_neighbors:
            top_indices = np.argsort(similarities)[-self.n_neighbors:]
            similarities = similarities[top_indices]
            ratings = ratings[top_indices]

        similarity_sum = np.sum(np.abs(similarities))
        if similarity_sum == 0:
            return float(item_mean)

        prediction = item_mean + np.sum(similarities * ratings) / similarity_sum

        return float(np.clip(prediction, 0, 5))

    def recommend_items(
        self, user_id: Union[int, str], n_recommendations: int = 10
    ) -> List[Tuple[Union[int, str], float]]:
        """Recommend top N items for a user.

        Args:
            user_id: User ID
            n_recommendations: Number of recommendations (default: 10)

        Returns:
            List of (item_id, predicted_rating) tuples, sorted by rating

        Raises:
            ValueError: If model is not fitted
        """
        if self.ratings_matrix is None:
            raise ValueError("Model must be fitted before recommendation")

        if user_id not in self.ratings_matrix.index:
            return []

        user_ratings = self.ratings_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings.isna()].index

        predictions = []
        for item_id in unrated_items:
            predicted_rating = self.predict_rating(user_id, item_id)
            predictions.append((item_id, predicted_rating))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_recommendations]


class RecommendationEvaluator:
    """Evaluate recommendation system performance."""

    @staticmethod
    def calculate_rmse(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Calculate Root Mean Squared Error.

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            RMSE value
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def calculate_mae(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Calculate Mean Absolute Error.

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            MAE value
        """
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def calculate_precision_recall(
        recommendations: List[Tuple[Union[int, str], float]],
        relevant_items: set,
        k: int = 10,
    ) -> Tuple[float, float]:
        """Calculate precision and recall at k.

        Args:
            recommendations: List of (item_id, rating) tuples
            relevant_items: Set of relevant item IDs
            k: Number of top recommendations to consider (default: 10)

        Returns:
            Tuple of (precision, recall)
        """
        top_k = recommendations[:k]
        recommended_items = {item_id for item_id, _ in top_k}

        if len(recommended_items) == 0:
            return 0.0, 0.0

        relevant_recommended = recommended_items & relevant_items

        precision = len(relevant_recommended) / len(recommended_items)
        recall = (
            len(relevant_recommended) / len(relevant_items)
            if len(relevant_items) > 0
            else 0.0
        )

        return float(precision), float(recall)


class CollaborativeFilteringRecommender:
    """Main recommender class combining user-based and item-based approaches."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize collaborative filtering recommender.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.user_based = None
        self.item_based = None
        self.ratings = None

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "logs/app.log")

        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(handler)

    def load_data(self, ratings: Union[pd.DataFrame, str]) -> "CollaborativeFilteringRecommender":
        """Load ratings data.

        Args:
            ratings: DataFrame or path to CSV file with ratings

        Returns:
            Self for method chaining
        """
        if isinstance(ratings, str):
            self.ratings = pd.read_csv(ratings)
        else:
            self.ratings = ratings.copy()

        logger.info(f"Loaded {len(self.ratings)} ratings")

        return self

    def fit_user_based(
        self, similarity_metric: Optional[str] = None, **kwargs
    ) -> "CollaborativeFilteringRecommender":
        """Fit user-based collaborative filtering model.

        Args:
            similarity_metric: Similarity metric (default: from config)
            **kwargs: Additional arguments for UserBasedCollaborativeFiltering

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is not loaded
        """
        if self.ratings is None:
            raise ValueError("Data must be loaded before fitting")

        ub_config = self.config.get("user_based", {})
        similarity_metric = similarity_metric or ub_config.get("similarity_metric", "cosine")
        n_neighbors = kwargs.get("n_neighbors", ub_config.get("n_neighbors", 50))
        min_common_items = kwargs.get(
            "min_common_items", ub_config.get("min_common_items", 1)
        )

        self.user_based = UserBasedCollaborativeFiltering(
            similarity_metric=similarity_metric,
            n_neighbors=n_neighbors,
            min_common_items=min_common_items,
        )
        self.user_based.fit(self.ratings)

        return self

    def fit_item_based(
        self, similarity_metric: Optional[str] = None, **kwargs
    ) -> "CollaborativeFilteringRecommender":
        """Fit item-based collaborative filtering model.

        Args:
            similarity_metric: Similarity metric (default: from config)
            **kwargs: Additional arguments for ItemBasedCollaborativeFiltering

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is not loaded
        """
        if self.ratings is None:
            raise ValueError("Data must be loaded before fitting")

        ib_config = self.config.get("item_based", {})
        similarity_metric = similarity_metric or ib_config.get("similarity_metric", "cosine")
        n_neighbors = kwargs.get("n_neighbors", ib_config.get("n_neighbors", 50))
        min_common_users = kwargs.get(
            "min_common_users", ib_config.get("min_common_users", 1)
        )

        self.item_based = ItemBasedCollaborativeFiltering(
            similarity_metric=similarity_metric,
            n_neighbors=n_neighbors,
            min_common_users=min_common_users,
        )
        self.item_based.fit(self.ratings)

        return self

    def fit_all(self) -> "CollaborativeFilteringRecommender":
        """Fit both user-based and item-based models.

        Returns:
            Self for method chaining
        """
        self.fit_user_based()
        self.fit_item_based()

        return self

    def recommend(
        self,
        user_id: Union[int, str],
        method: str = "user_based",
        n_recommendations: int = 10,
    ) -> List[Tuple[Union[int, str], float]]:
        """Generate recommendations for a user.

        Args:
            user_id: User ID
            method: Recommendation method - "user_based" or "item_based"
                (default: "user_based")
            n_recommendations: Number of recommendations (default: 10)

        Returns:
            List of (item_id, predicted_rating) tuples

        Raises:
            ValueError: If method is invalid or model not fitted
        """
        if method == "user_based":
            if self.user_based is None:
                raise ValueError("User-based model must be fitted first")
            return self.user_based.recommend_items(user_id, n_recommendations)
        elif method == "item_based":
            if self.item_based is None:
                raise ValueError("Item-based model must be fitted first")
            return self.item_based.recommend_items(user_id, n_recommendations)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'user_based' or 'item_based'")

    def evaluate(
        self,
        test_data: pd.DataFrame,
        method: str = "user_based",
        metrics: Optional[List[str]] = None,
    ) -> Dict:
        """Evaluate recommendation system on test data.

        Args:
            test_data: Test DataFrame with columns ['user_id', 'item_id', 'rating']
            method: Evaluation method - "user_based" or "item_based"
                (default: "user_based")
            metrics: List of metrics to calculate (default: ['rmse', 'mae'])

        Returns:
            Dictionary with evaluation metrics
        """
        if metrics is None:
            metrics = ["rmse", "mae"]

        if method == "user_based":
            model = self.user_based
        elif method == "item_based":
            model = self.item_based
        else:
            raise ValueError(f"Unknown method: {method}")

        if model is None:
            raise ValueError(f"{method} model must be fitted first")

        y_true = []
        y_pred = []

        for _, row in test_data.iterrows():
            user_id = row["user_id"]
            item_id = row["item_id"]
            true_rating = row["rating"]

            try:
                pred_rating = model.predict_rating(user_id, item_id)
                y_true.append(true_rating)
                y_pred.append(pred_rating)
            except Exception as e:
                logger.warning(f"Failed to predict for user {user_id}, item {item_id}: {e}")
                continue

        if len(y_true) == 0:
            return {"error": "No valid predictions"}

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        results = {}

        if "rmse" in metrics:
            results["rmse"] = RecommendationEvaluator.calculate_rmse(y_true, y_pred)

        if "mae" in metrics:
            results["mae"] = RecommendationEvaluator.calculate_mae(y_true, y_pred)

        return results


def main():
    """Main entry point for collaborative filtering recommender."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collaborative filtering recommendation system"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with ratings (user_id, item_id, rating)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["user_based", "item_based", "both"],
        default="both",
        help="Collaborative filtering method (default: both)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="User ID to generate recommendations for",
    )
    parser.add_argument(
        "--n-recommendations",
        type=int,
        default=10,
        help="Number of recommendations (default: 10)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test CSV file for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output CSV file for recommendations",
    )
    parser.add_argument(
        "--evaluation-output",
        type=str,
        help="Path to output JSON file for evaluation metrics",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    recommender = CollaborativeFilteringRecommender(
        config_path=Path(args.config) if args.config else None
    )

    recommender.load_data(args.input)

    if args.method in ["user_based", "both"]:
        recommender.fit_user_based()
    if args.method in ["item_based", "both"]:
        recommender.fit_item_based()

    if args.user_id:
        if args.method == "both":
            print("Generating recommendations with both methods...")
            for method in ["user_based", "item_based"]:
                recommendations = recommender.recommend(
                    args.user_id, method=method, n_recommendations=args.n_recommendations
                )
                print(f"\n{method.replace('_', ' ').title()} Recommendations:")
                for item_id, rating in recommendations:
                    print(f"  Item {item_id}: {rating:.2f}")
        else:
            recommendations = recommender.recommend(
                args.user_id, method=args.method, n_recommendations=args.n_recommendations
            )
            print(f"\nRecommendations for user {args.user_id}:")
            for item_id, rating in recommendations:
                print(f"  Item {item_id}: {rating:.2f}")

        if args.output:
            recommendations = recommender.recommend(
                args.user_id, method=args.method, n_recommendations=args.n_recommendations
            )
            output_df = pd.DataFrame(
                recommendations, columns=["item_id", "predicted_rating"]
            )
            output_df["user_id"] = args.user_id
            output_df = output_df[["user_id", "item_id", "predicted_rating"]]
            output_df.to_csv(args.output, index=False)
            logger.info(f"Recommendations saved to {args.output}")

    if args.test_data:
        test_data = pd.read_csv(args.test_data)
        evaluation_results = {}

        if args.method in ["user_based", "both"]:
            evaluation_results["user_based"] = recommender.evaluate(
                test_data, method="user_based"
            )
        if args.method in ["item_based", "both"]:
            evaluation_results["item_based"] = recommender.evaluate(
                test_data, method="item_based"
            )

        print("\nEvaluation Results:")
        print("=" * 50)
        for method, metrics in evaluation_results.items():
            print(f"\n{method.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")

        if args.evaluation_output:
            with open(args.evaluation_output, "w") as f:
                json.dump(evaluation_results, f, indent=2)
            logger.info(f"Evaluation results saved to {args.evaluation_output}")


if __name__ == "__main__":
    main()
