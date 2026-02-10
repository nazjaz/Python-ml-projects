"""Multi-Label Classification using Binary Relevance, Classifier Chains, and Label Powerset.

This module provides implementations of three multi-label classification methods:
Binary Relevance, Classifier Chains, and Label Powerset for handling problems
where each sample can belong to multiple classes simultaneously.
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
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BinaryRelevance(BaseEstimator, ClassifierMixin):
    """Binary Relevance method for multi-label classification.

    Trains one binary classifier per label independently, treating each
    label as a separate binary classification problem.
    """

    def __init__(
        self,
        base_estimator=None,
        require_dense: Tuple[bool, bool] = (True, True),
    ):
        """Initialize Binary Relevance classifier.

        Args:
            base_estimator: Base binary classifier (default: LogisticRegression)
            require_dense: Whether base estimator requires dense input
                (features, labels) (default: (True, True))
        """
        self.base_estimator = base_estimator or LogisticRegression(
            random_state=42, max_iter=1000
        )
        self.require_dense = require_dense
        self.estimators_ = None
        self.label_names_ = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, label_names: Optional[List[str]] = None
    ) -> "BinaryRelevance":
        """Fit Binary Relevance model.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Label matrix, shape (n_samples, n_labels) or list of label sets
            label_names: Optional list of label names (default: None)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If input data is invalid
        """
        X = np.array(X)

        if isinstance(y, list):
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(y)
            if label_names is None:
                label_names = mlb.classes_.tolist()

        y = np.array(y)

        if y.ndim == 1:
            raise ValueError("y must be 2D array or list of label sets for multi-label")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )

        n_labels = y.shape[1]
        self.estimators_ = []
        self.label_names_ = (
            label_names if label_names else [f"label_{i}" for i in range(n_labels)]
        )

        logger.info(f"Training {n_labels} binary classifiers...")

        for i in range(n_labels):
            estimator = clone(self.base_estimator)
            y_binary = y[:, i]
            estimator.fit(X, y_binary)
            self.estimators_.append(estimator)

        logger.info("Binary Relevance model fitted successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for samples.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Binary label matrix, shape (n_samples, n_labels)

        Raises:
            ValueError: If model is not fitted
        """
        if self.estimators_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        n_samples = X.shape[0]
        n_labels = len(self.estimators_)

        predictions = np.zeros((n_samples, n_labels), dtype=int)

        for i, estimator in enumerate(self.estimators_):
            predictions[:, i] = estimator.predict(X)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict label probabilities.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Probability matrix, shape (n_samples, n_labels)

        Raises:
            ValueError: If model is not fitted or base estimator doesn't support predict_proba
        """
        if self.estimators_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        n_samples = X.shape[0]
        n_labels = len(self.estimators_)

        if not hasattr(self.base_estimator, "predict_proba"):
            raise ValueError("Base estimator must support predict_proba")

        probabilities = np.zeros((n_samples, n_labels))

        for i, estimator in enumerate(self.estimators_):
            proba = estimator.predict_proba(X)
            if proba.shape[1] == 2:
                probabilities[:, i] = proba[:, 1]
            else:
                probabilities[:, i] = proba[:, 0]

        return probabilities


class ClassifierChain(BaseEstimator, ClassifierMixin):
    """Classifier Chain method for multi-label classification.

    Trains binary classifiers in a chain, where each classifier uses
    predictions from previous classifiers as additional features.
    """

    def __init__(
        self,
        base_estimator=None,
        order: Optional[List[int]] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize Classifier Chain.

        Args:
            base_estimator: Base binary classifier (default: LogisticRegression)
            order: Order of labels in chain (default: None, random order)
            random_state: Random seed for chain order (default: None)
        """
        self.base_estimator = base_estimator or LogisticRegression(
            random_state=42, max_iter=1000
        )
        self.order = order
        self.random_state = random_state
        self.estimators_ = None
        self.label_names_ = None
        self.chain_order_ = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, label_names: Optional[List[str]] = None
    ) -> "ClassifierChain":
        """Fit Classifier Chain model.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Label matrix, shape (n_samples, n_labels) or list of label sets
            label_names: Optional list of label names (default: None)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If input data is invalid
        """
        X = np.array(X)

        if isinstance(y, list):
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(y)
            if label_names is None:
                label_names = mlb.classes_.tolist()

        y = np.array(y)

        if y.ndim == 1:
            raise ValueError("y must be 2D array or list of label sets for multi-label")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )

        n_labels = y.shape[1]
        self.label_names_ = (
            label_names if label_names else [f"label_{i}" for i in range(n_labels)]
        )

        if self.order is None:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            self.chain_order_ = np.random.permutation(n_labels).tolist()
        else:
            self.chain_order_ = self.order.copy()

        self.estimators_ = []
        X_chain = X.copy()

        logger.info(f"Training classifier chain with {n_labels} labels...")

        for i, label_idx in enumerate(self.chain_order_):
            estimator = clone(self.base_estimator)
            y_binary = y[:, label_idx]
            estimator.fit(X_chain, y_binary)
            self.estimators_.append(estimator)

            if i < n_labels - 1:
                predictions = estimator.predict(X_chain)
                X_chain = np.hstack([X_chain, predictions.reshape(-1, 1)])

        logger.info("Classifier Chain model fitted successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for samples.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Binary label matrix, shape (n_samples, n_labels)

        Raises:
            ValueError: If model is not fitted
        """
        if self.estimators_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        n_samples = X.shape[0]
        n_labels = len(self.estimators_)

        predictions = np.zeros((n_samples, n_labels), dtype=int)
        X_chain = X.copy()

        for i, label_idx in enumerate(self.chain_order_):
            estimator = self.estimators_[i]
            pred = estimator.predict(X_chain)
            predictions[:, label_idx] = pred

            if i < n_labels - 1:
                X_chain = np.hstack([X_chain, pred.reshape(-1, 1)])

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict label probabilities.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Probability matrix, shape (n_samples, n_labels)

        Raises:
            ValueError: If model is not fitted or base estimator doesn't support predict_proba
        """
        if self.estimators_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        n_samples = X.shape[0]
        n_labels = len(self.estimators_)

        if not hasattr(self.base_estimator, "predict_proba"):
            raise ValueError("Base estimator must support predict_proba")

        probabilities = np.zeros((n_samples, n_labels))
        X_chain = X.copy()

        for i, label_idx in enumerate(self.chain_order_):
            estimator = self.estimators_[i]
            proba = estimator.predict_proba(X_chain)

            if proba.shape[1] == 2:
                probabilities[:, label_idx] = proba[:, 1]
            else:
                probabilities[:, label_idx] = proba[:, 0]

            if i < n_labels - 1:
                pred = estimator.predict(X_chain)
                X_chain = np.hstack([X_chain, pred.reshape(-1, 1)])

        return probabilities


class LabelPowerset(BaseEstimator, ClassifierMixin):
    """Label Powerset method for multi-label classification.

    Transforms multi-label problem into multi-class problem by treating
    each unique label combination as a separate class.
    """

    def __init__(self, base_estimator=None):
        """Initialize Label Powerset classifier.

        Args:
            base_estimator: Base multi-class classifier (default: RandomForestClassifier)
        """
        self.base_estimator = base_estimator or RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.estimator_ = None
        self.label_encoder_ = None
        self.label_names_ = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, label_names: Optional[List[str]] = None
    ) -> "LabelPowerset":
        """Fit Label Powerset model.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Label matrix, shape (n_samples, n_labels) or list of label sets
            label_names: Optional list of label names (default: None)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If input data is invalid
        """
        X = np.array(X)

        if isinstance(y, list):
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(y)
            if label_names is None:
                label_names = mlb.classes_.tolist()

        y = np.array(y)

        if y.ndim == 1:
            raise ValueError("y must be 2D array or list of label sets for multi-label")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )

        self.label_names_ = (
            label_names if label_names else [f"label_{i}" for i in range(y.shape[1])]
        )

        self.label_encoder_ = MultiLabelBinarizer()
        y_encoded = self.label_encoder_.fit_transform(y)

        y_powerset = []
        for row in y_encoded:
            label_set = tuple(row)
            y_powerset.append(label_set)

        unique_labels = list(set(y_powerset))
        self.label_to_class_ = {label: idx for idx, label in enumerate(unique_labels)}
        self.class_to_label_ = {idx: label for label, idx in self.label_to_class_.items()}
        y_classes = np.array([self.label_to_class_[label] for label in y_powerset])

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y_classes)

        logger.info(
            f"Label Powerset model fitted with {len(unique_labels)} unique label combinations"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for samples.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Binary label matrix, shape (n_samples, n_labels)

        Raises:
            ValueError: If model is not fitted
        """
        if self.estimator_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        y_classes = self.estimator_.predict(X)

        n_labels = len(self.label_names_)
        predictions = np.zeros((len(y_classes), n_labels), dtype=int)

        for i, class_idx in enumerate(y_classes):
            if class_idx in self.class_to_label_:
                label_set = self.class_to_label_[class_idx]
                predictions[i] = np.array(label_set)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict label probabilities.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Probability matrix, shape (n_samples, n_labels)

        Raises:
            ValueError: If model is not fitted or base estimator doesn't support predict_proba
        """
        if self.estimator_ is None:
            raise ValueError("Model must be fitted before prediction")

        if not hasattr(self.base_estimator, "predict_proba"):
            raise ValueError("Base estimator must support predict_proba")

        X = np.array(X)
        class_proba = self.estimator_.predict_proba(X)

        n_labels = len(self.label_names_)
        n_samples = X.shape[0]

        probabilities = np.zeros((n_samples, n_labels))

        for i in range(n_samples):
            for class_idx, prob in enumerate(class_proba[i]):
                if class_idx in self.class_to_label_:
                    label_set = self.class_to_label_[class_idx]
                    probabilities[i] += prob * np.array(label_set)

        return probabilities


class MultiLabelEvaluator:
    """Evaluate multi-label classification performance."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> Dict[str, float]:
        """Evaluate multi-label classification performance.

        Args:
            y_true: True labels, shape (n_samples, n_labels)
            y_pred: Predicted labels, shape (n_samples, n_labels)
            average: Averaging strategy - "macro", "micro", "samples" (default: "macro")

        Returns:
            Dictionary with evaluation metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics = {
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, average=average, zero_division=0)
            ),
            "jaccard_score": float(
                jaccard_score(y_true, y_pred, average=average, zero_division=0)
            ),
        }

        return metrics


class MultiLabelClassifier:
    """Main multi-label classification class combining all methods."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize multi-label classifier.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.binary_relevance = None
        self.classifier_chain = None
        self.label_powerset = None
        self.X = None
        self.y = None
        self.label_names = None

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

    def load_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List],
    ) -> "MultiLabelClassifier":
        """Load data for multi-label classification.

        Args:
            X: Feature matrix
            y: Label matrix or list of label sets

        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.array(X)

        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, list):
            pass
        else:
            y = np.array(y)

        self.X = X
        self.y = y

        if isinstance(y, list) and len(y) > 0 and isinstance(y[0], (list, tuple, set)):
            mlb = MultiLabelBinarizer()
            y_binarized = mlb.fit_transform(y)
            self.label_names = mlb.classes_.tolist()
        elif isinstance(y, np.ndarray) and y.ndim == 2:
            self.label_names = [f"label_{i}" for i in range(y.shape[1])]

        logger.info(
            f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features, "
            f"{len(self.label_names) if self.label_names else 'unknown'} labels"
        )

        return self

    def fit_binary_relevance(
        self, base_estimator=None
    ) -> "MultiLabelClassifier":
        """Fit Binary Relevance model.

        Args:
            base_estimator: Base binary classifier (default: from config)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is not loaded
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded before fitting")

        br_config = self.config.get("binary_relevance", {})

        if base_estimator is None:
            estimator_type = br_config.get("estimator", "logistic_regression")
            if estimator_type == "random_forest":
                base_estimator = RandomForestClassifier(
                    n_estimators=br_config.get("n_estimators", 100),
                    random_state=br_config.get("random_state", 42),
                )
            else:
                base_estimator = LogisticRegression(
                    random_state=br_config.get("random_state", 42),
                    max_iter=br_config.get("max_iter", 1000),
                )

        self.binary_relevance = BinaryRelevance(base_estimator=base_estimator)
        self.binary_relevance.fit(self.X, self.y, label_names=self.label_names)

        return self

    def fit_classifier_chain(
        self, base_estimator=None, order: Optional[List[int]] = None
    ) -> "MultiLabelClassifier":
        """Fit Classifier Chain model.

        Args:
            base_estimator: Base binary classifier (default: from config)
            order: Chain order (default: from config or random)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is not loaded
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded before fitting")

        cc_config = self.config.get("classifier_chain", {})

        if base_estimator is None:
            estimator_type = cc_config.get("estimator", "logistic_regression")
            if estimator_type == "random_forest":
                base_estimator = RandomForestClassifier(
                    n_estimators=cc_config.get("n_estimators", 100),
                    random_state=cc_config.get("random_state", 42),
                )
            else:
                base_estimator = LogisticRegression(
                    random_state=cc_config.get("random_state", 42),
                    max_iter=cc_config.get("max_iter", 1000),
                )

        order = order or cc_config.get("order", None)

        self.classifier_chain = ClassifierChain(
            base_estimator=base_estimator,
            order=order,
            random_state=cc_config.get("random_state", None),
        )
        self.classifier_chain.fit(self.X, self.y, label_names=self.label_names)

        return self

    def fit_label_powerset(
        self, base_estimator=None
    ) -> "MultiLabelClassifier":
        """Fit Label Powerset model.

        Args:
            base_estimator: Base multi-class classifier (default: from config)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is not loaded
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded before fitting")

        lp_config = self.config.get("label_powerset", {})

        if base_estimator is None:
            estimator_type = lp_config.get("estimator", "random_forest")
            if estimator_type == "random_forest":
                base_estimator = RandomForestClassifier(
                    n_estimators=lp_config.get("n_estimators", 100),
                    random_state=lp_config.get("random_state", 42),
                )
            else:
                base_estimator = LogisticRegression(
                    random_state=lp_config.get("random_state", 42),
                    max_iter=lp_config.get("max_iter", 1000),
                )

        self.label_powerset = LabelPowerset(base_estimator=base_estimator)
        self.label_powerset.fit(self.X, self.y, label_names=self.label_names)

        return self

    def fit_all(self) -> "MultiLabelClassifier":
        """Fit all models.

        Returns:
            Self for method chaining
        """
        self.fit_binary_relevance()
        self.fit_classifier_chain()
        self.fit_label_powerset()

        return self

    def predict(
        self, X: np.ndarray, method: str = "binary_relevance"
    ) -> np.ndarray:
        """Predict labels using specified method.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            method: Prediction method - "binary_relevance", "classifier_chain",
                or "label_powerset" (default: "binary_relevance")

        Returns:
            Binary label matrix, shape (n_samples, n_labels)

        Raises:
            ValueError: If method is invalid or not fitted
        """
        if method == "binary_relevance":
            if self.binary_relevance is None:
                raise ValueError("Binary Relevance model must be fitted first")
            return self.binary_relevance.predict(X)
        elif method == "classifier_chain":
            if self.classifier_chain is None:
                raise ValueError("Classifier Chain model must be fitted first")
            return self.classifier_chain.predict(X)
        elif method == "label_powerset":
            if self.label_powerset is None:
                raise ValueError("Label Powerset model must be fitted first")
            return self.label_powerset.predict(X)
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'binary_relevance', "
                f"'classifier_chain', or 'label_powerset'"
            )

    def evaluate(
        self,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        method: str = "binary_relevance",
    ) -> Dict:
        """Evaluate model performance.

        Args:
            X_test: Test feature matrix (default: None, uses training data)
            y_test: Test labels (default: None, uses training data)
            method: Evaluation method (default: "binary_relevance")

        Returns:
            Dictionary with evaluation metrics
        """
        if X_test is None:
            X_test = self.X
            y_test = self.y
        else:
            X_test = np.array(X_test)
            if isinstance(y_test, list):
                mlb = MultiLabelBinarizer()
                y_test = mlb.fit_transform(y_test)
            y_test = np.array(y_test)

        y_pred = self.predict(X_test, method=method)

        return MultiLabelEvaluator.evaluate(y_test, y_pred)


def main():
    """Main entry point for multi-label classifier."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-label classification using Binary Relevance, Classifier Chains, and Label Powerset"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--target-cols",
        type=str,
        nargs="+",
        required=True,
        help="Column names for target labels",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["binary_relevance", "classifier_chain", "label_powerset", "all"],
        default="all",
        help="Classification method (default: all)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test CSV file for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output CSV file with predictions",
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

    classifier = MultiLabelClassifier(
        config_path=Path(args.config) if args.config else None
    )

    df = pd.read_csv(args.input)

    feature_cols = [col for col in df.columns if col not in args.target_cols]
    X = df[feature_cols]
    y = df[args.target_cols].values

    classifier.load_data(X, y)

    if args.method in ["binary_relevance", "all"]:
        classifier.fit_binary_relevance()
    if args.method in ["classifier_chain", "all"]:
        classifier.fit_classifier_chain()
    if args.method in ["label_powerset", "all"]:
        classifier.fit_label_powerset()

    if args.test_data:
        test_df = pd.read_csv(args.test_data)
        X_test = test_df[feature_cols]
        y_test = test_df[args.target_cols].values

        results = {}

        if args.method in ["binary_relevance", "all"]:
            results["binary_relevance"] = classifier.evaluate(
                X_test, y_test, method="binary_relevance"
            )
        if args.method in ["classifier_chain", "all"]:
            results["classifier_chain"] = classifier.evaluate(
                X_test, y_test, method="classifier_chain"
            )
        if args.method in ["label_powerset", "all"]:
            results["label_powerset"] = classifier.evaluate(
                X_test, y_test, method="label_powerset"
            )

        print("\nEvaluation Results:")
        print("=" * 50)
        for method, metrics in results.items():
            print(f"\n{method.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        if args.evaluation_output:
            with open(args.evaluation_output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {args.evaluation_output}")

    if args.output:
        if args.method == "all":
            method = "binary_relevance"
        else:
            method = args.method

        predictions = classifier.predict(X, method=method)

        output_df = pd.DataFrame(predictions, columns=args.target_cols)
        output_df = pd.concat([df[feature_cols], output_df], axis=1)
        output_df.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
