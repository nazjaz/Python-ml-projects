"""Classification Metrics Calculator.

This module provides functionality to calculate basic evaluation metrics
for classification tasks including accuracy, precision, recall, and F1-score.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Calculate classification evaluation metrics."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize ClassificationMetrics with configuration.

        Args:
            config_path: Path to configuration YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file.

        Returns:
            Dictionary containing configuration settings.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if not config:
                raise ValueError("Configuration file is empty")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in configuration file: {e}")
            raise

    def _setup_logging(self) -> None:
        """Configure logging based on configuration settings."""
        log_level = self.config.get("logging", {}).get("level", "INFO")
        log_file = self.config.get("logging", {}).get("file", "logs/app.log")
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - " "%(message)s"
        )

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=10485760, backupCount=5
                ),
                logging.StreamHandler(),
            ],
        )

    def _validate_inputs(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
    ) -> tuple:
        """Validate and convert inputs to numpy arrays.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Tuple of (y_true_array, y_pred_array) as numpy arrays.

        Raises:
            ValueError: If inputs are invalid or have mismatched lengths.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true has {len(y_true)} "
                f"samples, y_pred has {len(y_pred)} samples"
            )

        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")

        return y_true, y_pred

    def accuracy(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
    ) -> float:
        """Calculate accuracy score.

        Accuracy is the proportion of correct predictions among
        all predictions.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Accuracy score between 0 and 1.

        Example:
            >>> metrics = ClassificationMetrics()
            >>> y_true = [0, 1, 1, 0, 1]
            >>> y_pred = [0, 1, 0, 0, 1]
            >>> metrics.accuracy(y_true, y_pred)
            0.8
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        accuracy_score = correct / total if total > 0 else 0.0

        logger.debug(
            f"Accuracy: {correct}/{total} = {accuracy_score:.4f}"
        )
        return float(accuracy_score)

    def precision(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
        average: str = "binary",
        pos_label: Union[int, str] = 1,
    ) -> Union[float, Dict[str, float]]:
        """Calculate precision score.

        Precision is the proportion of true positives among all
        positive predictions (TP / (TP + FP)).

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            average: Averaging strategy. Options: 'binary', 'macro',
                'micro', 'weighted', None. Default: 'binary'.
            pos_label: Positive class label for binary classification.
                Default: 1.

        Returns:
            Precision score(s). Float for binary/macro/micro/weighted,
            dict for per-class precision when average=None.

        Example:
            >>> metrics = ClassificationMetrics()
            >>> y_true = [0, 1, 1, 0, 1]
            >>> y_pred = [0, 1, 0, 0, 1]
            >>> metrics.precision(y_true, y_pred)
            1.0
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        if average == "binary":
            tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
            fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
            precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            logger.debug(
                f"Precision (binary): TP={tp}, FP={fp}, "
                f"Precision={precision_score:.4f}"
            )
            return float(precision_score)

        return self._precision_multiclass(y_true, y_pred, average)

    def _precision_multiclass(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str,
    ) -> Union[float, Dict[str, float]]:
        """Calculate precision for multiclass classification.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            average: Averaging strategy.

        Returns:
            Precision score(s) based on averaging strategy.
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        class_precisions = {}

        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            class_precisions[str(cls)] = float(precision)

        if average is None:
            return class_precisions

        if average == "macro":
            result = np.mean(list(class_precisions.values()))
        elif average == "micro":
            total_tp = sum(
                np.sum((y_true == cls) & (y_pred == cls)) for cls in classes
            )
            total_fp = sum(
                np.sum((y_true != cls) & (y_pred == cls)) for cls in classes
            )
            result = (
                total_tp / (total_tp + total_fp)
                if (total_tp + total_fp) > 0
                else 0.0
            )
        elif average == "weighted":
            class_counts = {
                str(cls): np.sum(y_true == cls) for cls in classes
            }
            total_samples = len(y_true)
            weighted_sum = sum(
                class_precisions[str(cls)] * class_counts[str(cls)]
                for cls in classes
            )
            result = weighted_sum / total_samples if total_samples > 0 else 0.0
        else:
            raise ValueError(
                f"Invalid average parameter: {average}. "
                "Must be 'binary', 'macro', 'micro', 'weighted', or None"
            )

        logger.debug(f"Precision ({average}): {result:.4f}")
        return float(result)

    def recall(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
        average: str = "binary",
        pos_label: Union[int, str] = 1,
    ) -> Union[float, Dict[str, float]]:
        """Calculate recall score.

        Recall is the proportion of true positives among all
        actual positives (TP / (TP + FN)).

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            average: Averaging strategy. Options: 'binary', 'macro',
                'micro', 'weighted', None. Default: 'binary'.
            pos_label: Positive class label for binary classification.
                Default: 1.

        Returns:
            Recall score(s). Float for binary/macro/micro/weighted,
            dict for per-class recall when average=None.

        Example:
            >>> metrics = ClassificationMetrics()
            >>> y_true = [0, 1, 1, 0, 1]
            >>> y_pred = [0, 1, 0, 0, 1]
            >>> metrics.recall(y_true, y_pred)
            0.6666666666666666
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        if average == "binary":
            tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
            fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
            recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            logger.debug(
                f"Recall (binary): TP={tp}, FN={fn}, "
                f"Recall={recall_score:.4f}"
            )
            return float(recall_score)

        return self._recall_multiclass(y_true, y_pred, average)

    def _recall_multiclass(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str,
    ) -> Union[float, Dict[str, float]]:
        """Calculate recall for multiclass classification.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            average: Averaging strategy.

        Returns:
            Recall score(s) based on averaging strategy.
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        class_recalls = {}

        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            class_recalls[str(cls)] = float(recall)

        if average is None:
            return class_recalls

        if average == "macro":
            result = np.mean(list(class_recalls.values()))
        elif average == "micro":
            total_tp = sum(
                np.sum((y_true == cls) & (y_pred == cls)) for cls in classes
            )
            total_fn = sum(
                np.sum((y_true == cls) & (y_pred != cls)) for cls in classes
            )
            result = (
                total_tp / (total_tp + total_fn)
                if (total_tp + total_fn) > 0
                else 0.0
            )
        elif average == "weighted":
            class_counts = {
                str(cls): np.sum(y_true == cls) for cls in classes
            }
            total_samples = len(y_true)
            weighted_sum = sum(
                class_recalls[str(cls)] * class_counts[str(cls)]
                for cls in classes
            )
            result = weighted_sum / total_samples if total_samples > 0 else 0.0
        else:
            raise ValueError(
                f"Invalid average parameter: {average}. "
                "Must be 'binary', 'macro', 'micro', 'weighted', or None"
            )

        logger.debug(f"Recall ({average}): {result:.4f}")
        return float(result)

    def f1_score(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
        average: str = "binary",
        pos_label: Union[int, str] = 1,
    ) -> Union[float, Dict[str, float]]:
        """Calculate F1-score.

        F1-score is the harmonic mean of precision and recall:
        F1 = 2 * (precision * recall) / (precision + recall)

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            average: Averaging strategy. Options: 'binary', 'macro',
                'micro', 'weighted', None. Default: 'binary'.
            pos_label: Positive class label for binary classification.
                Default: 1.

        Returns:
            F1-score(s). Float for binary/macro/micro/weighted,
            dict for per-class F1-score when average=None.

        Example:
            >>> metrics = ClassificationMetrics()
            >>> y_true = [0, 1, 1, 0, 1]
            >>> y_pred = [0, 1, 0, 0, 1]
            >>> metrics.f1_score(y_true, y_pred)
            0.8
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        precision_val = self.precision(
            y_true, y_pred, average=average, pos_label=pos_label
        )
        recall_val = self.recall(
            y_true, y_pred, average=average, pos_label=pos_label
        )

        if average is None:
            f1_scores = {}
            for cls in precision_val.keys():
                p = precision_val[cls]
                r = recall_val[cls]
                f1 = (
                    2 * p * r / (p + r)
                    if (p + r) > 0
                    else 0.0
                )
                f1_scores[cls] = float(f1)
            logger.debug(f"F1-score (per-class): {f1_scores}")
            return f1_scores

        f1 = (
            2 * precision_val * recall_val / (precision_val + recall_val)
            if (precision_val + recall_val) > 0
            else 0.0
        )

        logger.debug(f"F1-score ({average}): {f1:.4f}")
        return float(f1)

    def calculate_all_metrics(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
        average: str = "binary",
        pos_label: Union[int, str] = 1,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Calculate all classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            average: Averaging strategy for precision, recall, F1.
                Default: 'binary'.
            pos_label: Positive class label for binary classification.
                Default: 1.

        Returns:
            Dictionary containing all calculated metrics:
            - accuracy: Accuracy score
            - precision: Precision score(s)
            - recall: Recall score(s)
            - f1_score: F1-score(s)

        Example:
            >>> metrics = ClassificationMetrics()
            >>> y_true = [0, 1, 1, 0, 1]
            >>> y_pred = [0, 1, 0, 0, 1]
            >>> results = metrics.calculate_all_metrics(y_true, y_pred)
            >>> print(results['accuracy'])
            0.8
        """
        logger.info("Calculating all classification metrics")
        results = {
            "accuracy": self.accuracy(y_true, y_pred),
            "precision": self.precision(
                y_true, y_pred, average=average, pos_label=pos_label
            ),
            "recall": self.recall(
                y_true, y_pred, average=average, pos_label=pos_label
            ),
            "f1_score": self.f1_score(
                y_true, y_pred, average=average, pos_label=pos_label
            ),
        }

        logger.info("Metrics calculation complete")
        return results

    def confusion_matrix(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
    ) -> Dict[str, Dict[str, int]]:
        """Calculate confusion matrix.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary representation of confusion matrix with
            structure: {true_label: {predicted_label: count}}.

        Example:
            >>> metrics = ClassificationMetrics()
            >>> y_true = [0, 1, 1, 0, 1]
            >>> y_pred = [0, 1, 0, 0, 1]
            >>> cm = metrics.confusion_matrix(y_true, y_pred)
            >>> print(cm['1']['1'])
            2
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))

        matrix = {}
        for true_cls in classes:
            matrix[str(true_cls)] = {}
            for pred_cls in classes:
                count = np.sum(
                    (y_true == true_cls) & (y_pred == pred_cls)
                )
                matrix[str(true_cls)][str(pred_cls)] = int(count)

        logger.debug(f"Confusion matrix calculated for {len(classes)} classes")
        return matrix


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Classification Metrics Calculator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--y-true",
        type=str,
        required=True,
        help="Path to CSV file with true labels or comma-separated values",
    )
    parser.add_argument(
        "--y-pred",
        type=str,
        required=True,
        help="Path to CSV file with predicted labels or comma-separated values",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Column name if input is CSV file (default: first column)",
    )
    parser.add_argument(
        "--average",
        type=str,
        default="binary",
        choices=["binary", "macro", "micro", "weighted", "none"],
        help="Averaging strategy for multiclass metrics",
    )
    parser.add_argument(
        "--pos-label",
        type=str,
        default="1",
        help="Positive class label for binary classification",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON (optional)",
    )

    args = parser.parse_args()

    metrics = ClassificationMetrics(config_path=args.config)

    try:
        # Load true labels
        if args.y_true.endswith(".csv"):
            df_true = pd.read_csv(args.y_true)
            column = args.column if args.column else df_true.columns[0]
            y_true = df_true[column].values
        else:
            y_true = [int(x.strip()) for x in args.y_true.split(",")]

        # Load predicted labels
        if args.y_pred.endswith(".csv"):
            df_pred = pd.read_csv(args.y_pred)
            column = args.column if args.column else df_pred.columns[0]
            y_pred = df_pred[column].values
        else:
            y_pred = [int(x.strip()) for x in args.y_pred.split(",")]

        print("\n=== Classification Metrics ===")
        print(f"True labels: {len(y_true)} samples")
        print(f"Predicted labels: {len(y_pred)} samples")

        average = None if args.average == "none" else args.average
        pos_label = (
            int(args.pos_label)
            if args.pos_label.isdigit()
            else args.pos_label
        )

        results = metrics.calculate_all_metrics(
            y_true, y_pred, average=average, pos_label=pos_label
        )

        print("\n=== Results ===")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")

        confusion = metrics.confusion_matrix(y_true, y_pred)
        print("\n=== Confusion Matrix ===")
        for true_label, pred_dict in confusion.items():
            for pred_label, count in pred_dict.items():
                print(f"True={true_label}, Pred={pred_label}: {count}")

        if args.output:
            import json

            output_data = {
                "metrics": results,
                "confusion_matrix": confusion,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise


if __name__ == "__main__":
    main()
