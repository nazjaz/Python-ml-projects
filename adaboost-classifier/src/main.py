"""AdaBoost Classifier with Weak Learners and Adaptive Boosting.

This module provides functionality to implement AdaBoost from scratch with
weak learners (decision stumps) and adaptive boosting iterations.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DecisionStump:
    """Decision Stump (weak learner) - single-level decision tree."""

    def __init__(self) -> None:
        """Initialize decision stump."""
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.polarity: int = 1
        self.alpha: float = 0.0

    def _make_prediction(self, X: np.ndarray) -> np.ndarray:
        """Make predictions based on feature threshold.

        Args:
            X: Feature matrix.

        Returns:
            Predictions.
        """
        predictions = np.ones(len(X))
        feature_values = X[:, self.feature_index]

        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1

        return predictions

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray
    ) -> "DecisionStump":
        """Fit decision stump to weighted data.

        Args:
            X: Feature matrix.
            y: Target labels (-1, 1).
            sample_weights: Sample weights.

        Returns:
            Self for method chaining.
        """
        n_samples, n_features = X.shape
        min_error = float("inf")

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1

                    error = np.sum(sample_weights * (predictions != y))

                    if error < min_error:
                        min_error = error
                        self.feature_index = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels (-1, 1).
        """
        return self._make_prediction(X)


class AdaBoostClassifier:
    """AdaBoost Classifier with weak learners and adaptive boosting."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize AdaBoost Classifier.

        Args:
            n_estimators: Number of weak learners (default: 50).
            learning_rate: Learning rate (shrinkage) (default: 1.0).
            random_state: Random seed (default: None).
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.estimators_: List[DecisionStump] = []
        self.estimator_weights_: List[float] = []
        self.n_features_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

        if random_state is not None:
            np.random.seed(random_state)

    def _calculate_alpha(self, error: float) -> float:
        """Calculate learner weight (alpha) based on error.

        Args:
            error: Weighted error rate.

        Returns:
            Learner weight (alpha).
        """
        if error <= 0:
            return 10.0
        elif error >= 1:
            return 0.0
        else:
            return 0.5 * np.log((1 - error) / error) * self.learning_rate

    def _update_weights(
        self,
        sample_weights: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Update sample weights based on misclassifications.

        Args:
            sample_weights: Current sample weights.
            y: True labels.
            predictions: Predictions from weak learner.
            alpha: Learner weight.

        Returns:
            Updated sample weights.
        """
        incorrect = predictions != y
        sample_weights = sample_weights * np.exp(alpha * incorrect)
        sample_weights = sample_weights / np.sum(sample_weights)
        return sample_weights

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> "AdaBoostClassifier":
        """Fit AdaBoost classifier.

        Args:
            X: Feature matrix.
            y: Target labels (binary: 0, 1 or -1, 1).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("AdaBoost currently supports binary classification only")

        self.n_features_ = X.shape[1]
        self.classes_ = unique_classes

        y_binary = y.copy()
        if unique_classes[0] == 0 and unique_classes[1] == 1:
            y_binary = 2 * y_binary - 1

        n_samples = len(X)
        sample_weights = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.estimator_weights_ = []

        for i in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y_binary, sample_weights)

            predictions = stump.predict(X)
            error = np.sum(sample_weights * (predictions != y_binary))

            if error >= 0.5:
                logger.warning(f"Error >= 0.5 at iteration {i}, stopping early")
                break

            if error == 0:
                alpha = 10.0
            else:
                alpha = self._calculate_alpha(error)

            stump.alpha = alpha
            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)

            sample_weights = self._update_weights(
                sample_weights, y_binary, predictions, alpha
            )

        self._calculate_feature_importance()

        logger.info(
            f"AdaBoost fitted: n_estimators={len(self.estimators_)}, "
            f"learning_rate={self.learning_rate}"
        )

        return self

    def _calculate_feature_importance(self) -> None:
        """Calculate feature importance based on learner weights."""
        if not self.estimators_:
            return

        importances = np.zeros(self.n_features_)

        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            if stump.feature_index is not None:
                importances[stump.feature_index] += alpha

        total = np.sum(importances)
        if total > 0:
            importances = importances / total

        self.feature_importances_ = importances

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class labels using weighted voting.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model not fitted.
        """
        if not self.estimators_:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = np.zeros(len(X))

        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            stump_predictions = stump.predict(X)
            predictions += alpha * stump_predictions

        predictions = np.sign(predictions)

        if self.classes_[0] == 0 and self.classes_[1] == 1:
            predictions = (predictions + 1) // 2

        return np.where(predictions >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Class probabilities.

        Raises:
            ValueError: If model not fitted.
        """
        if not self.estimators_:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = np.zeros(len(X))

        for stump, alpha in zip(self.estimators_, self.estimator_weights_):
            stump_predictions = stump.predict(X)
            predictions += alpha * stump_predictions

        predictions = np.clip(predictions, -10, 10)
        probabilities = 1.0 / (1.0 + np.exp(-2 * predictions))

        proba = np.column_stack([1 - probabilities, probabilities])

        return proba

    def score(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> float:
        """Calculate classification accuracy.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Classification accuracy.
        """
        predictions = self.predict(X)
        y = np.asarray(y, dtype=int)
        accuracy = np.mean(predictions == y)
        return accuracy

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importance as dictionary.

        Returns:
            Dictionary mapping feature names to importance values.

        Raises:
            ValueError: If model not fitted.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted before getting feature importance")

        if self.feature_names_ is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features_)]
        else:
            feature_names = self.feature_names_

        return dict(zip(feature_names, self.feature_importances_))

    def plot_feature_importance(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        top_n: Optional[int] = None,
    ) -> None:
        """Plot feature importance.

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
            top_n: Number of top features to show (default: all).
        """
        if self.feature_importances_ is None:
            logger.warning("Model must be fitted before plotting feature importance")
            return

        if self.feature_names_ is None:
            feature_names = [f"Feature {i}" for i in range(self.n_features_)]
        else:
            feature_names = self.feature_names_

        importances = self.feature_importances_
        indices = np.argsort(importances)[::-1]

        if top_n is not None:
            indices = indices[:top_n]

        sorted_importances = importances[indices]
        sorted_names = [feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(sorted_names)), sorted_importances)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(
            f"AdaBoost Feature Importance (n_estimators={len(self.estimators_)})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def get_estimator_errors(self) -> List[float]:
        """Get error rates for each estimator.

        Returns:
            List of error rates.
        """
        if not self.estimators_:
            return []

        errors = []
        for i, (stump, alpha) in enumerate(zip(self.estimators_, self.estimator_weights_)):
            if alpha == 0:
                error = 0.5
            elif alpha >= 10:
                error = 0.0
            else:
                error = 1.0 / (1.0 + np.exp(2 * alpha / self.learning_rate))
            errors.append(error)

        return errors


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="AdaBoost Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file with data",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of target column",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated list of feature columns (default: all except target)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Number of weak learners (default: from config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: from config)",
    )
    parser.add_argument(
        "--plot-importance",
        action="store_true",
        help="Plot feature importance",
    )
    parser.add_argument(
        "--save-importance-plot",
        type=str,
        default=None,
        help="Path to save feature importance plot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions as CSV",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Path to CSV file for prediction",
    )
    parser.add_argument(
        "--output-predictions",
        type=str,
        default=None,
        help="Path to save predictions as CSV",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})

        n_estimators = (
            args.n_estimators
            if args.n_estimators is not None
            else model_config.get("n_estimators", 50)
        )
        learning_rate = (
            args.learning_rate
            if args.learning_rate is not None
            else model_config.get("learning_rate", 1.0)
        )

        df = pd.read_csv(args.input)
        print(f"\n=== AdaBoost Classifier ===")
        print(f"Data shape: {df.shape}")

        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found in data")

        if args.features:
            feature_cols = [col.strip() for col in args.features.split(",")]
            missing = [col for col in feature_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Feature columns not found: {missing}")
        else:
            feature_cols = [col for col in df.columns if col != args.target]

        X = df[feature_cols].values
        y = df[args.target].values

        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Classes: {len(np.unique(y))}")

        print(f"\nFitting AdaBoost...")
        print(f"Number of weak learners: {n_estimators}")
        print(f"Learning rate: {learning_rate}")

        adaboost = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
        adaboost.feature_names_ = feature_cols
        adaboost.fit(X, y)

        print(f"\n=== AdaBoost Results ===")
        print(f"Number of weak learners fitted: {len(adaboost.estimators_)}")
        print(f"Classes: {adaboost.classes_}")

        errors = adaboost.get_estimator_errors()
        if errors:
            print(f"\nError rates by iteration:")
            for i, error in enumerate(errors[:10]):
                print(f"  Iteration {i+1}: {error:.6f}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

        accuracy = adaboost.score(X, y)
        print(f"\nTraining accuracy: {accuracy:.6f}")

        importances = adaboost.get_feature_importances()
        print(f"\nFeature Importance:")
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_importances[:10]:
            print(f"  {name}: {importance:.6f}")
        if len(sorted_importances) > 10:
            print(f"  ... and {len(sorted_importances) - 10} more")

        if args.plot_importance or args.save_importance_plot:
            adaboost.plot_feature_importance(
                save_path=args.save_importance_plot, show=args.plot_importance
            )

        if args.output:
            predictions = adaboost.predict(X)
            proba = adaboost.predict_proba(X)
            output_df = pd.DataFrame({
                "prediction": predictions,
                args.target: y,
            })
            for i, cls in enumerate(adaboost.classes_):
                output_df[f"prob_class_{cls}"] = proba[:, i]
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

        if args.predict:
            df_predict = pd.read_csv(args.predict)
            X_predict = df_predict[feature_cols].values
            predictions = adaboost.predict(X_predict)
            proba = adaboost.predict_proba(X_predict)

            if args.output_predictions:
                output_df = pd.DataFrame({
                    "prediction": predictions,
                })
                for i, cls in enumerate(adaboost.classes_):
                    output_df[f"prob_class_{cls}"] = proba[:, i]
                output_df.to_csv(args.output_predictions, index=False)
                print(f"Predictions saved to: {args.output_predictions}")
            else:
                print(f"\nPredictions:")
                for i, pred in enumerate(predictions[:10]):
                    print(f"  Sample {i+1}: {pred}")
                if len(predictions) > 10:
                    print(f"  ... and {len(predictions) - 10} more")

    except Exception as e:
        logger.error(f"Error running AdaBoost: {e}")
        raise


if __name__ == "__main__":
    main()
