"""Classification Visualization Tool.

This module provides functionality to create confusion matrices and
classification reports with visualization capabilities.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 8)


class ClassificationVisualizer:
    """Create confusion matrices and classification reports with visualization."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize ClassificationVisualizer with configuration.

        Args:
            config_path: Path to configuration YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_parameters()

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

    def _initialize_parameters(self) -> None:
        """Initialize visualization parameters from configuration."""
        viz_config = self.config.get("visualization", {})
        self.figsize = tuple(viz_config.get("figsize", [10, 8]))
        self.dpi = viz_config.get("dpi", 100)
        self.colormap = viz_config.get("colormap", "Blues")
        self.fontsize = viz_config.get("fontsize", 12)
        self.save_format = viz_config.get("save_format", "png")

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

    def confusion_matrix(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
        labels: Optional[List] = None,
    ) -> np.ndarray:
        """Calculate confusion matrix.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: List of labels to include in matrix. If None,
                all unique labels are used.

        Returns:
            Confusion matrix as numpy array.

        Example:
            >>> viz = ClassificationVisualizer()
            >>> y_true = [0, 1, 1, 0, 1]
            >>> y_pred = [0, 1, 0, 0, 1]
            >>> cm = viz.confusion_matrix(y_true, y_pred)
            >>> print(cm)
            [[2 0]
             [1 2]]
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        if labels is None:
            labels = sorted(list(set(np.concatenate([y_true, y_pred]))))

        n_classes = len(labels)
        cm = np.zeros((n_classes, n_classes), dtype=int)

        label_to_idx = {label: idx for idx, label in enumerate(labels)}

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label in label_to_idx and pred_label in label_to_idx:
                true_idx = label_to_idx[true_label]
                pred_idx = label_to_idx[pred_label]
                cm[true_idx, pred_idx] += 1

        logger.debug(f"Confusion matrix calculated: {n_classes}x{n_classes}")
        return cm

    def classification_report(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
        labels: Optional[List] = None,
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """Generate classification report with precision, recall, F1-score.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: List of labels to include in report. If None,
                all unique labels are used.
            target_names: Optional names for labels. If None, labels
                are used as names.

        Returns:
            Dictionary containing per-class metrics and overall metrics:
            - per_class: Dictionary with metrics for each class
            - accuracy: Overall accuracy
            - macro_avg: Macro-averaged metrics
            - weighted_avg: Weighted-averaged metrics

        Example:
            >>> viz = ClassificationVisualizer()
            >>> y_true = [0, 1, 1, 0, 1]
            >>> y_pred = [0, 1, 0, 0, 1]
            >>> report = viz.classification_report(y_true, y_pred)
            >>> print(report['accuracy'])
            0.8
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        if labels is None:
            labels = sorted(list(set(np.concatenate([y_true, y_pred]))))

        if target_names is None:
            target_names = [str(label) for label in labels]

        cm = self.confusion_matrix(y_true, y_pred, labels=labels)
        n_classes = len(labels)

        per_class = {}
        total_samples = len(y_true)
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for i, (label, name) in enumerate(zip(labels, target_names)):
            tp = int(cm[i, i])
            fp = int(np.sum(cm[:, i]) - tp)
            fn = int(np.sum(cm[i, :]) - tp)
            tn = int(np.sum(cm) - tp - fp - fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            support = int(tp + fn)

            per_class[name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "support": support,
            }

            total_tp += tp
            total_fp += fp
            total_fn += fn

        accuracy = np.sum(np.diag(cm)) / np.sum(cm)

        macro_precision = np.mean([v["precision"] for v in per_class.values()])
        macro_recall = np.mean([v["recall"] for v in per_class.values()])
        macro_f1 = np.mean([v["f1_score"] for v in per_class.values()])

        weighted_precision = sum(
            v["precision"] * v["support"] for v in per_class.values()
        ) / total_samples
        weighted_recall = sum(
            v["recall"] * v["support"] for v in per_class.values()
        ) / total_samples
        weighted_f1 = sum(
            v["f1_score"] * v["support"] for v in per_class.values()
        ) / total_samples

        report = {
            "per_class": per_class,
            "accuracy": float(accuracy),
            "macro_avg": {
                "precision": float(macro_precision),
                "recall": float(macro_recall),
                "f1_score": float(macro_f1),
                "support": total_samples,
            },
            "weighted_avg": {
                "precision": float(weighted_precision),
                "recall": float(weighted_recall),
                "f1_score": float(weighted_f1),
                "support": total_samples,
            },
        }

        logger.info("Classification report generated")
        return report

    def plot_confusion_matrix(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
        labels: Optional[List] = None,
        target_names: Optional[List[str]] = None,
        normalize: bool = False,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot confusion matrix as heatmap.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: List of labels to include. If None, all unique
                labels are used.
            target_names: Optional names for labels. If None, labels
                are used as names.
            normalize: If True, normalize confusion matrix to show
                percentages. Default: False.
            title: Optional title for the plot. Default: auto-generated.
            save_path: Optional path to save the figure. Default: None.
            show: Whether to display the plot. Default: True.

        Example:
            >>> viz = ClassificationVisualizer()
            >>> y_true = [0, 1, 1, 0, 1]
            >>> y_pred = [0, 1, 0, 0, 1]
            >>> viz.plot_confusion_matrix(y_true, y_pred)
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        if labels is None:
            labels = sorted(list(set(np.concatenate([y_true, y_pred]))))

        if target_names is None:
            target_names = [str(label) for label in labels]

        cm = self.confusion_matrix(y_true, y_pred, labels=labels)

        if normalize:
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_to_plot = cm_normalized
            fmt = ".2f"
            label = "Normalized Confusion Matrix"
        else:
            cm_to_plot = cm
            fmt = "d"
            label = "Confusion Matrix"

        plt.figure(figsize=self.figsize, dpi=self.dpi)
        sns.heatmap(
            cm_to_plot,
            annot=True,
            fmt=fmt,
            cmap=self.colormap,
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={"label": "Count" if not normalize else "Proportion"},
        )

        if title is None:
            title = label

        plt.title(title, fontsize=self.fontsize + 2, fontweight="bold")
        plt.ylabel("True Label", fontsize=self.fontsize)
        plt.xlabel("Predicted Label", fontsize=self.fontsize)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_classification_report(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
        labels: Optional[List] = None,
        target_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot classification report as heatmap.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: List of labels to include. If None, all unique
                labels are used.
            target_names: Optional names for labels. If None, labels
                are used as names.
            title: Optional title for the plot. Default: auto-generated.
            save_path: Optional path to save the figure. Default: None.
            show: Whether to display the plot. Default: True.

        Example:
            >>> viz = ClassificationVisualizer()
            >>> y_true = [0, 1, 1, 0, 1]
            >>> y_pred = [0, 1, 0, 0, 1]
            >>> viz.plot_classification_report(y_true, y_pred)
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        if labels is None:
            labels = sorted(list(set(np.concatenate([y_true, y_pred]))))

        if target_names is None:
            target_names = [str(label) for label in labels]

        report = self.classification_report(y_true, y_pred, labels, target_names)

        metrics_data = []
        metric_names = ["precision", "recall", "f1_score"]

        for class_name in target_names:
            row = [report["per_class"][class_name][metric] for metric in metric_names]
            metrics_data.append(row)

        metrics_data.append(
            [
                report["macro_avg"][metric] for metric in metric_names
            ]
        )
        metrics_data.append(
            [
                report["weighted_avg"][metric] for metric in metric_names
            ]
        )

        row_labels = target_names + ["macro avg", "weighted avg"]
        metrics_df = pd.DataFrame(
            metrics_data, index=row_labels, columns=metric_names
        )

        plt.figure(figsize=self.figsize, dpi=self.dpi)
        sns.heatmap(
            metrics_df,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            cbar_kws={"label": "Score"},
            vmin=0,
            vmax=1,
        )

        if title is None:
            title = "Classification Report"

        plt.title(title, fontsize=self.fontsize + 2, fontweight="bold")
        plt.ylabel("Class", fontsize=self.fontsize)
        plt.xlabel("Metric", fontsize=self.fontsize)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Classification report plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def print_classification_report(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
        labels: Optional[List] = None,
        target_names: Optional[List[str]] = None,
    ) -> None:
        """Print formatted classification report to console.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: List of labels to include. If None, all unique
                labels are used.
            target_names: Optional names for labels. If None, labels
                are used as names.
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        if labels is None:
            labels = sorted(list(set(np.concatenate([y_true, y_pred]))))

        if target_names is None:
            target_names = [str(label) for label in labels]

        report = self.classification_report(y_true, y_pred, labels, target_names)

        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)

        print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)

        for class_name in target_names:
            metrics = report["per_class"][class_name]
            print(
                f"{class_name:<15} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1_score']:<12.4f} "
                f"{metrics['support']:<10}"
            )

        print("-" * 60)
        macro = report["macro_avg"]
        print(
            f"{'macro avg':<15} "
            f"{macro['precision']:<12.4f} "
            f"{macro['recall']:<12.4f} "
            f"{macro['f1_score']:<12.4f} "
            f"{macro['support']:<10}"
        )

        weighted = report["weighted_avg"]
        print(
            f"{'weighted avg':<15} "
            f"{weighted['precision']:<12.4f} "
            f"{weighted['recall']:<12.4f} "
            f"{weighted['f1_score']:<12.4f} "
            f"{weighted['support']:<10}"
        )

        print("-" * 60)
        print(f"\nAccuracy: {report['accuracy']:.4f}")
        print("=" * 60 + "\n")

    def save_report(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
        labels: Optional[List] = None,
        target_names: Optional[List[str]] = None,
        output_path: str = "classification_report.json",
    ) -> None:
        """Save classification report to JSON file.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: List of labels to include. If None, all unique
                labels are used.
            target_names: Optional names for labels. If None, labels
                are used as names.
            output_path: Path to save JSON file.
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        if labels is None:
            labels = sorted(list(set(np.concatenate([y_true, y_pred]))))

        if target_names is None:
            target_names = [str(label) for label in labels]

        report = self.classification_report(y_true, y_pred, labels, target_names)
        cm = self.confusion_matrix(y_true, y_pred, labels=labels)

        output_data = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "labels": [str(label) for label in labels],
            "target_names": target_names,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Classification report saved to: {output_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Classification Visualization Tool"
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
        "--labels",
        type=str,
        default=None,
        help="Comma-separated list of labels (default: auto-detect)",
    )
    parser.add_argument(
        "--target-names",
        type=str,
        default=None,
        help="Comma-separated list of target names (default: use labels)",
    )
    parser.add_argument(
        "--plot-cm",
        action="store_true",
        help="Plot confusion matrix",
    )
    parser.add_argument(
        "--plot-report",
        action="store_true",
        help="Plot classification report",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize confusion matrix",
    )
    parser.add_argument(
        "--save-cm",
        type=str,
        default=None,
        help="Path to save confusion matrix plot",
    )
    parser.add_argument(
        "--save-report-plot",
        type=str,
        default=None,
        help="Path to save classification report plot",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Path to save classification report as JSON",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (useful when saving)",
    )

    args = parser.parse_args()

    viz = ClassificationVisualizer(config_path=args.config)

    try:
        # Load true labels
        if args.y_true.endswith(".csv"):
            df_true = pd.read_csv(args.y_true)
            column = args.column if args.column else df_true.columns[0]
            y_true = df_true[column].values
        else:
            y_true = [x.strip() for x in args.y_true.split(",")]

        # Load predicted labels
        if args.y_pred.endswith(".csv"):
            df_pred = pd.read_csv(args.y_pred)
            column = args.column if args.column else df_pred.columns[0]
            y_pred = df_pred[column].values
        else:
            y_pred = [x.strip() for x in args.y_pred.split(",")]

        labels = None
        if args.labels:
            labels = [x.strip() for x in args.labels.split(",")]

        target_names = None
        if args.target_names:
            target_names = [x.strip() for x in args.target_names.split(",")]

        print("\n=== Classification Visualization ===")
        print(f"True labels: {len(y_true)} samples")
        print(f"Predicted labels: {len(y_pred)} samples")

        # Print classification report
        viz.print_classification_report(y_true, y_pred, labels, target_names)

        # Plot confusion matrix
        if args.plot_cm or args.save_cm:
            viz.plot_confusion_matrix(
                y_true,
                y_pred,
                labels=labels,
                target_names=target_names,
                normalize=args.normalize,
                save_path=args.save_cm,
                show=not args.no_show,
            )

        # Plot classification report
        if args.plot_report or args.save_report_plot:
            viz.plot_classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=target_names,
                save_path=args.save_report_plot,
                show=not args.no_show,
            )

        # Save report
        if args.save_report:
            viz.save_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=target_names,
                output_path=args.save_report,
            )

    except Exception as e:
        logger.error(f"Error processing classification data: {e}")
        raise


if __name__ == "__main__":
    main()
