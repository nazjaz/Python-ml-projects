"""Anomaly Detection using Isolation Forest, One-Class SVM, and Local Outlier Factor.

This module provides implementations of three popular anomaly detection algorithms:
Isolation Forest, One-Class SVM, and Local Outlier Factor (LOF) for identifying
outliers and anomalies in datasets.
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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Isolation Forest anomaly detection.

    Isolation Forest isolates observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float, str] = "auto",
        contamination: float = 0.1,
        max_features: Union[int, float] = 1.0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        """Initialize Isolation Forest detector.

        Args:
            n_estimators: Number of base estimators (default: 100)
            max_samples: Number of samples to draw for training each estimator
                (default: "auto")
            contamination: Expected proportion of outliers (default: 0.1)
            max_features: Number of features to draw for each estimator
                (default: 1.0)
            random_state: Random seed for reproducibility (default: None)
            n_jobs: Number of parallel jobs (default: None)
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Fit Isolation Forest model to data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If input data is invalid
        """
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if len(X) < 2:
            raise ValueError("X must have at least 2 samples")

        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.model.fit(X_scaled)

        logger.info(
            f"Isolation Forest fitted with {self.n_estimators} estimators, "
            f"contamination={self.contamination}"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies in data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Array of predictions: 1 for normal, -1 for anomaly

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        return predictions

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Anomaly scores (lower values indicate more anomalous)

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before computing scores")

        X = np.array(X)
        X_scaled = self.scaler.transform(X)

        scores = self.model.decision_function(X_scaled)

        return scores


class OneClassSVMDetector:
    """One-Class SVM anomaly detection.

    One-Class SVM learns a decision function for novelty detection: classifying
    new data as similar or different to the training set.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.1,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        shrinking: bool = True,
        tol: float = 1e-3,
        cache_size: float = 200,
    ):
        """Initialize One-Class SVM detector.

        Args:
            kernel: Kernel type - "linear", "poly", "rbf", "sigmoid"
                (default: "rbf")
            nu: Upper bound on fraction of training errors and lower bound
                on fraction of support vectors (default: 0.1)
            gamma: Kernel coefficient - "scale", "auto", or float (default: "scale")
            degree: Degree for polynomial kernel (default: 3)
            coef0: Independent term in kernel function (default: 0.0)
            shrinking: Whether to use shrinking heuristic (default: True)
            tol: Tolerance for stopping criterion (default: 1e-3)
            cache_size: Cache size in MB (default: 200)
        """
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        """Fit One-Class SVM model to data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If input data is invalid
        """
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if len(X) < 2:
            raise ValueError("X must have at least 2 samples")

        X_scaled = self.scaler.fit_transform(X)

        self.model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            shrinking=self.shrinking,
            tol=self.tol,
            cache_size=self.cache_size,
        )

        self.model.fit(X_scaled)

        logger.info(
            f"One-Class SVM fitted with kernel={self.kernel}, nu={self.nu}, "
            f"gamma={self.gamma}"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies in data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Array of predictions: 1 for normal, -1 for anomaly

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        return predictions

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Anomaly scores (lower values indicate more anomalous)

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before computing scores")

        X = np.array(X)
        X_scaled = self.scaler.transform(X)

        scores = self.model.decision_function(X_scaled)

        return scores


class LocalOutlierFactorDetector:
    """Local Outlier Factor (LOF) anomaly detection.

    LOF computes a score reflecting the degree of abnormality of observations.
    It measures the local density deviation of a given data point with respect
    to its neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        metric_params: Optional[Dict] = None,
        contamination: float = 0.1,
        novelty: bool = False,
        n_jobs: Optional[int] = None,
    ):
        """Initialize Local Outlier Factor detector.

        Args:
            n_neighbors: Number of neighbors to use (default: 20)
            algorithm: Algorithm for nearest neighbors - "auto", "ball_tree",
                "kd_tree", "brute" (default: "auto")
            leaf_size: Leaf size for tree-based algorithms (default: 30)
            metric: Distance metric (default: "minkowski")
            p: Power parameter for Minkowski metric (default: 2)
            metric_params: Additional parameters for metric (default: None)
            contamination: Expected proportion of outliers (default: 0.1)
            novelty: Whether to use novelty detection mode (default: False)
            n_jobs: Number of parallel jobs (default: None)
        """
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.novelty = novelty
        self.n_jobs = n_jobs
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "LocalOutlierFactorDetector":
        """Fit LOF model to data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If input data is invalid
        """
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if len(X) < self.n_neighbors + 1:
            raise ValueError(
                f"X must have at least {self.n_neighbors + 1} samples for LOF"
            )

        X_scaled = self.scaler.fit_transform(X)

        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            contamination=self.contamination,
            novelty=self.novelty,
            n_jobs=self.n_jobs,
        )

        self.model.fit(X_scaled)

        logger.info(
            f"LOF fitted with n_neighbors={self.n_neighbors}, "
            f"contamination={self.contamination}"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies in data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Array of predictions: 1 for normal, -1 for anomaly

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        return predictions

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (negative LOF scores).

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Anomaly scores (lower values indicate more anomalous)

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before computing scores")

        X = np.array(X)
        X_scaled = self.scaler.transform(X)

        scores = -self.model.score_samples(X_scaled)

        return scores


class AnomalyDetector:
    """Main anomaly detection class combining all algorithms."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize anomaly detector.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.detectors = {}
        self.results = {}

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

    def fit_isolation_forest(
        self, X: np.ndarray, **kwargs
    ) -> "AnomalyDetector":
        """Fit Isolation Forest detector.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            **kwargs: Additional arguments for IsolationForestDetector

        Returns:
            Self for method chaining
        """
        if_config = self.config.get("isolation_forest", {})
        if_config.update(kwargs)

        detector = IsolationForestDetector(**if_config)
        detector.fit(X)

        self.detectors["isolation_forest"] = detector

        return self

    def fit_one_class_svm(
        self, X: np.ndarray, **kwargs
    ) -> "AnomalyDetector":
        """Fit One-Class SVM detector.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            **kwargs: Additional arguments for OneClassSVMDetector

        Returns:
            Self for method chaining
        """
        svm_config = self.config.get("one_class_svm", {})
        svm_config.update(kwargs)

        detector = OneClassSVMDetector(**svm_config)
        detector.fit(X)

        self.detectors["one_class_svm"] = detector

        return self

    def fit_local_outlier_factor(
        self, X: np.ndarray, **kwargs
    ) -> "AnomalyDetector":
        """Fit Local Outlier Factor detector.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            **kwargs: Additional arguments for LocalOutlierFactorDetector

        Returns:
            Self for method chaining
        """
        lof_config = self.config.get("local_outlier_factor", {})
        lof_config.update(kwargs)

        detector = LocalOutlierFactorDetector(**lof_config)
        detector.fit(X)

        self.detectors["local_outlier_factor"] = detector

        return self

    def fit_all(self, X: np.ndarray) -> "AnomalyDetector":
        """Fit all detectors to data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Self for method chaining
        """
        self.fit_isolation_forest(X)
        self.fit_one_class_svm(X)
        self.fit_local_outlier_factor(X)

        return self

    def predict(
        self, X: np.ndarray, method: Optional[str] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Predict anomalies using specified method or all methods.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            method: Detection method - "isolation_forest", "one_class_svm",
                "local_outlier_factor", or None for all (default: None)

        Returns:
            Predictions array or dictionary of predictions by method

        Raises:
            ValueError: If method is invalid or not fitted
        """
        if method is None:
            results = {}
            for name, detector in self.detectors.items():
                results[name] = detector.predict(X)
            return results

        if method not in self.detectors:
            raise ValueError(
                f"Method '{method}' not fitted. Available: {list(self.detectors.keys())}"
            )

        return self.detectors[method].predict(X)

    def get_scores(
        self, X: np.ndarray, method: Optional[str] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Get anomaly scores using specified method or all methods.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            method: Detection method or None for all (default: None)

        Returns:
            Scores array or dictionary of scores by method

        Raises:
            ValueError: If method is invalid or not fitted
        """
        if method is None:
            results = {}
            for name, detector in self.detectors.items():
                results[name] = detector.decision_function(X)
            return results

        if method not in self.detectors:
            raise ValueError(
                f"Method '{method}' not fitted. Available: {list(self.detectors.keys())}"
            )

        return self.detectors[method].decision_function(X)

    def evaluate(
        self, X: np.ndarray, y_true: Optional[np.ndarray] = None
    ) -> Dict:
        """Evaluate anomaly detection performance.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y_true: True labels (1 for normal, -1 for anomaly) (default: None)

        Returns:
            Dictionary with evaluation metrics for each method
        """
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        results = {}

        predictions = self.predict(X)

        for method, y_pred in predictions.items():
            method_results = {
                "n_anomalies": int(np.sum(y_pred == -1)),
                "n_normal": int(np.sum(y_pred == 1)),
                "anomaly_rate": float(np.sum(y_pred == -1) / len(y_pred)),
            }

            if y_true is not None:
                method_results["accuracy"] = float(accuracy_score(y_true, y_pred))
                method_results["precision"] = float(
                    precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
                )
                method_results["recall"] = float(
                    recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
                )
                method_results["f1_score"] = float(
                    f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
                )
                method_results["confusion_matrix"] = confusion_matrix(
                    y_true, y_pred
                ).tolist()
                method_results["classification_report"] = classification_report(
                    y_true, y_pred, output_dict=True
                )

            results[method] = method_results

        return results

    def plot_results(
        self,
        X: np.ndarray,
        predictions: Optional[Dict[str, np.ndarray]] = None,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> None:
        """Plot anomaly detection results (2D visualization).

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            predictions: Predictions dictionary (default: None, will compute)
            save_path: Path to save figure (default: None)
            figsize: Figure size (default: (15, 5))

        Raises:
            ValueError: If X is not 2D or has more than 2 features
        """
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError("X must be 2D array")

        if X.shape[1] > 2:
            logger.warning(
                "X has more than 2 features. Using first 2 features for visualization."
            )
            X_plot = X[:, :2]
        else:
            X_plot = X

        if predictions is None:
            predictions = self.predict(X)

        n_methods = len(predictions)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)

        if n_methods == 1:
            axes = [axes]

        for idx, (method, y_pred) in enumerate(predictions.items()):
            normal_mask = y_pred == 1
            anomaly_mask = y_pred == -1

            axes[idx].scatter(
                X_plot[normal_mask, 0],
                X_plot[normal_mask, 1],
                c="blue",
                label="Normal",
                alpha=0.6,
                s=20,
            )
            axes[idx].scatter(
                X_plot[anomaly_mask, 0],
                X_plot[anomaly_mask, 1],
                c="red",
                label="Anomaly",
                alpha=0.6,
                s=20,
                marker="x",
            )

            axes[idx].set_title(f"{method.replace('_', ' ').title()}")
            axes[idx].set_xlabel("Feature 1")
            axes[idx].set_ylabel("Feature 2")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        plt.close()


def main():
    """Main entry point for anomaly detector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Anomaly detection using Isolation Forest, One-Class SVM, and LOF"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output CSV file with predictions",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["isolation_forest", "one_class_svm", "local_outlier_factor", "all"],
        default="all",
        help="Detection method to use (default: all)",
    )
    parser.add_argument(
        "--true-labels",
        type=str,
        help="Column name for true labels (optional, for evaluation)",
    )
    parser.add_argument(
        "--evaluation-output",
        type=str,
        help="Path to output JSON file for evaluation metrics",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        help="Path to output PNG file for visualization",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    detector = AnomalyDetector(
        config_path=Path(args.config) if args.config else None
    )

    df = pd.read_csv(args.input)

    feature_cols = [
        col
        for col in df.columns
        if col != args.true_labels and df[col].dtype in [np.number]
    ]

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found in CSV")

    X = df[feature_cols].values

    logger.info(f"Loaded data with {len(X)} samples and {len(feature_cols)} features")

    if args.method == "all":
        detector.fit_all(X)
    elif args.method == "isolation_forest":
        detector.fit_isolation_forest(X)
    elif args.method == "one_class_svm":
        detector.fit_one_class_svm(X)
    elif args.method == "local_outlier_factor":
        detector.fit_local_outlier_factor(X)

    predictions = detector.predict(X)

    if args.true_labels and args.true_labels in df.columns:
        y_true = df[args.true_labels].values
        y_true_binary = np.where(y_true == -1, -1, 1)
        evaluation = detector.evaluate(X, y_true_binary)

        if args.evaluation_output:
            with open(args.evaluation_output, "w") as f:
                json.dump(evaluation, f, indent=2)
            logger.info(f"Evaluation results saved to {args.evaluation_output}")

        print("\nEvaluation Results:")
        print("=" * 50)
        for method, metrics in evaluation.items():
            print(f"\n{method.replace('_', ' ').title()}:")
            print(f"  Anomalies detected: {metrics['n_anomalies']}")
            print(f"  Anomaly rate: {metrics['anomaly_rate']:.2%}")
            if "accuracy" in metrics:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1 Score: {metrics['f1_score']:.4f}")
    else:
        print("\nDetection Results:")
        print("=" * 50)
        for method, y_pred in predictions.items():
            n_anomalies = np.sum(y_pred == -1)
            print(
                f"{method.replace('_', ' ').title()}: "
                f"{n_anomalies} anomalies detected ({n_anomalies/len(y_pred):.2%})"
            )

    if args.output:
        if isinstance(predictions, dict):
            output_df = df.copy()
            for method, y_pred in predictions.items():
                output_df[f"{method}_prediction"] = y_pred
                scores = detector.get_scores(X, method=method)
                output_df[f"{method}_score"] = scores
            output_df.to_csv(args.output, index=False)
        else:
            output_df = df.copy()
            output_df["prediction"] = predictions
            scores = detector.get_scores(X, args.method)
            output_df["score"] = scores
            output_df.to_csv(args.output, index=False)

        logger.info(f"Predictions saved to {args.output}")

    if args.plot_output and X.shape[1] >= 2:
        detector.plot_results(X, predictions, save_path=Path(args.plot_output))


if __name__ == "__main__":
    main()
