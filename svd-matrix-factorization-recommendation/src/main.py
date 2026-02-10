"""Matrix Factorization for Recommendation Systems using SVD.

This module provides implementations of matrix factorization for recommendation
systems using Singular Value Decomposition (SVD) to predict user-item ratings
and generate personalized recommendations.
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
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SVDRecommender:
    """SVD-based matrix factorization for recommendation systems."""

    def __init__(
        self,
        n_components: int = 50,
        n_iter: int = 5,
        random_state: Optional[int] = None,
        algorithm: str = "arpack",
        tol: float = 0.0,
    ):
        """Initialize SVD recommender.

        Args:
            n_components: Number of latent factors (default: 50)
            n_iter: Number of iterations for SVD (default: 5)
            random_state: Random seed for reproducibility (default: None)
            algorithm: SVD algorithm - "arpack" or "randomized" (default: "arpack")
            tol: Tolerance for convergence (default: 0.0)
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.algorithm = algorithm
        self.tol = tol
        self.svd_model = None
        self.ratings_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None
        self.user_factors = None
        self.item_factors = None
        self.scaler = StandardScaler()

    def fit(self, ratings: pd.DataFrame) -> "SVDRecommender":
        """Fit SVD model to ratings data.

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
        self.item_means = self.ratings_matrix.mean(axis=0)
        self.global_mean = self.ratings_matrix.values[~np.isnan(self.ratings_matrix.values)].mean()

        ratings_matrix_filled = self.ratings_matrix.fillna(0).values

        if self.n_components > min(ratings_matrix_filled.shape):
            self.n_components = min(ratings_matrix_filled.shape)
            logger.warning(
                f"n_components reduced to {self.n_components} to match matrix dimensions"
            )

        self.svd_model = TruncatedSVD(
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
            algorithm=self.algorithm,
            tol=self.tol,
        )

        self.svd_model.fit(ratings_matrix_filled)

        self.user_factors = self.svd_model.transform(ratings_matrix_filled)
        self.item_factors = self.svd_model.components_.T

        logger.info(
            f"SVD model fitted with {self.n_components} components, "
            f"explained variance: {self.svd_model.explained_variance_ratio_.sum():.4f}"
        )

        return self

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
        if self.svd_model is None:
            raise ValueError("Model must be fitted before prediction")

        if user_id not in self.ratings_matrix.index:
            return float(self.global_mean)

        if item_id not in self.ratings_matrix.columns:
            return float(self.global_mean)

        user_idx = list(self.ratings_matrix.index).index(user_id)
        item_idx = list(self.ratings_matrix.columns).index(item_id)

        user_factor = self.user_factors[user_idx]
        item_factor = self.item_factors[item_idx]

        prediction = np.dot(user_factor, item_factor)

        user_mean = self.user_means[user_id]
        item_mean = self.item_means[item_id]

        prediction = prediction + user_mean + item_mean - self.global_mean

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
        if self.svd_model is None:
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

    def get_explained_variance(self) -> float:
        """Get total explained variance ratio.

        Returns:
            Total explained variance ratio

        Raises:
            ValueError: If model is not fitted
        """
        if self.svd_model is None:
            raise ValueError("Model must be fitted before getting variance")

        return float(self.svd_model.explained_variance_ratio_.sum())

    def get_components(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get user and item factor matrices.

        Returns:
            Tuple of (user_factors, item_factors)

        Raises:
            ValueError: If model is not fitted
        """
        if self.svd_model is None:
            raise ValueError("Model must be fitted before getting components")

        return self.user_factors, self.item_factors


class MatrixFactorizationEvaluator:
    """Evaluate matrix factorization recommendation system performance."""

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

    @staticmethod
    def calculate_coverage(
        recommendations_all_users: Dict[Union[int, str], List[Tuple[Union[int, str], float]]],
        all_items: set,
    ) -> float:
        """Calculate catalog coverage.

        Args:
            recommendations_all_users: Dictionary mapping user_id to recommendations
            all_items: Set of all item IDs

        Returns:
            Coverage ratio (0 to 1)
        """
        recommended_items = set()
        for user_recommendations in recommendations_all_users.values():
            for item_id, _ in user_recommendations:
                recommended_items.add(item_id)

        if len(all_items) == 0:
            return 0.0

        coverage = len(recommended_items) / len(all_items)
        return float(coverage)


class SVDRecommendationSystem:
    """Main recommendation system class using SVD matrix factorization."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize SVD recommendation system.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.model = None
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

    def load_data(self, ratings: Union[pd.DataFrame, str]) -> "SVDRecommendationSystem":
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

    def fit(
        self,
        n_components: Optional[int] = None,
        n_iter: Optional[int] = None,
        **kwargs,
    ) -> "SVDRecommendationSystem":
        """Fit SVD model to data.

        Args:
            n_components: Number of latent factors (default: from config)
            n_iter: Number of iterations (default: from config)
            **kwargs: Additional arguments for SVDRecommender

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is not loaded
        """
        if self.ratings is None:
            raise ValueError("Data must be loaded before fitting")

        svd_config = self.config.get("svd", {})
        n_components = n_components or svd_config.get("n_components", 50)
        n_iter = n_iter or svd_config.get("n_iter", 5)
        random_state = kwargs.get("random_state", svd_config.get("random_state", None))
        algorithm = kwargs.get("algorithm", svd_config.get("algorithm", "arpack"))
        tol = kwargs.get("tol", svd_config.get("tol", 0.0))

        self.model = SVDRecommender(
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state,
            algorithm=algorithm,
            tol=tol,
        )
        self.model.fit(self.ratings)

        return self

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
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_rating(user_id, item_id)

    def recommend(
        self, user_id: Union[int, str], n_recommendations: int = 10
    ) -> List[Tuple[Union[int, str], float]]:
        """Generate recommendations for a user.

        Args:
            user_id: User ID
            n_recommendations: Number of recommendations (default: 10)

        Returns:
            List of (item_id, predicted_rating) tuples

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before recommendation")

        return self.model.recommend_items(user_id, n_recommendations)

    def evaluate(
        self,
        test_data: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> Dict:
        """Evaluate recommendation system on test data.

        Args:
            test_data: Test DataFrame with columns ['user_id', 'item_id', 'rating']
            metrics: List of metrics to calculate (default: ['rmse', 'mae'])

        Returns:
            Dictionary with evaluation metrics
        """
        if metrics is None:
            metrics = ["rmse", "mae"]

        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")

        y_true = []
        y_pred = []

        for _, row in test_data.iterrows():
            user_id = row["user_id"]
            item_id = row["item_id"]
            true_rating = row["rating"]

            try:
                pred_rating = self.model.predict_rating(user_id, item_id)
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
            results["rmse"] = MatrixFactorizationEvaluator.calculate_rmse(y_true, y_pred)

        if "mae" in metrics:
            results["mae"] = MatrixFactorizationEvaluator.calculate_mae(y_true, y_pred)

        return results

    def get_model_info(self) -> Dict:
        """Get model information and statistics.

        Returns:
            Dictionary with model information

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting info")

        return {
            "n_components": self.model.n_components,
            "explained_variance": self.model.get_explained_variance(),
            "n_users": len(self.model.user_ids),
            "n_items": len(self.model.item_ids),
            "global_mean": float(self.model.global_mean),
        }


def main():
    """Main entry point for SVD recommendation system."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SVD-based matrix factorization recommendation system"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with ratings (user_id, item_id, rating)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        help="Number of latent factors (default: from config)",
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

    recommender = SVDRecommendationSystem(
        config_path=Path(args.config) if args.config else None
    )

    recommender.load_data(args.input)

    recommender.fit(n_components=args.n_components)

    model_info = recommender.get_model_info()
    print("\nModel Information:")
    print("=" * 50)
    print(f"Number of components: {model_info['n_components']}")
    print(f"Explained variance: {model_info['explained_variance']:.4f}")
    print(f"Number of users: {model_info['n_users']}")
    print(f"Number of items: {model_info['n_items']}")
    print(f"Global mean rating: {model_info['global_mean']:.2f}")

    if args.user_id:
        recommendations = recommender.recommend(
            args.user_id, n_recommendations=args.n_recommendations
        )
        print(f"\nRecommendations for user {args.user_id}:")
        for item_id, rating in recommendations:
            print(f"  Item {item_id}: {rating:.2f}")

        if args.output:
            output_df = pd.DataFrame(
                recommendations, columns=["item_id", "predicted_rating"]
            )
            output_df["user_id"] = args.user_id
            output_df = output_df[["user_id", "item_id", "predicted_rating"]]
            output_df.to_csv(args.output, index=False)
            logger.info(f"Recommendations saved to {args.output}")

    if args.test_data:
        test_data = pd.read_csv(args.test_data)
        evaluation = recommender.evaluate(test_data)

        print("\nEvaluation Results:")
        print("=" * 50)
        for metric, value in evaluation.items():
            print(f"{metric.upper()}: {value:.4f}")

        if args.evaluation_output:
            with open(args.evaluation_output, "w") as f:
                json.dump(evaluation, f, indent=2)
            logger.info(f"Evaluation results saved to {args.evaluation_output}")


if __name__ == "__main__":
    main()
