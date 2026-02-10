"""ROC Curve and AUC Calculator.

This module provides functionality to calculate ROC curves and AUC
(Area Under the Curve) for binary classification model evaluation.
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

# Set style for better visualizations
plt.rcParams["figure.figsize"] = (10, 8)


class ROCCalculator:
    """Calculate ROC curve and AUC for binary classification."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize ROCCalculator with configuration.

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
        self.linewidth = viz_config.get("linewidth", 2)
        self.fontsize = viz_config.get("fontsize", 12)
        self.save_format = viz_config.get("save_format", "png")

    def _validate_inputs(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_scores: Union[List, np.ndarray, pd.Series],
    ) -> tuple:
        """Validate and convert inputs to numpy arrays.

        Args:
            y_true: True binary labels (0 or 1).
            y_scores: Predicted scores or probabilities.

        Returns:
            Tuple of (y_true_array, y_scores_array) as numpy arrays.

        Raises:
            ValueError: If inputs are invalid or have mismatched lengths.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_scores = np.asarray(y_scores, dtype=float)

        if len(y_true) != len(y_scores):
            raise ValueError(
                f"Length mismatch: y_true has {len(y_true)} "
                f"samples, y_scores has {len(y_scores)} samples"
            )

        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")

        unique_labels = np.unique(y_true)
        if len(unique_labels) > 2:
            raise ValueError(
                "ROC curve is only for binary classification. "
                f"Found {len(unique_labels)} unique labels."
            )

        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                "y_true must contain only 0 and 1 for binary classification"
            )

        return y_true, y_scores

    def roc_curve(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_scores: Union[List, np.ndarray, pd.Series],
        pos_label: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ROC curve (True Positive Rate vs False Positive Rate).

        Args:
            y_true: True binary labels (0 or 1).
            y_scores: Predicted scores or probabilities.
            pos_label: Label of the positive class. Default: 1.

        Returns:
            Tuple of (fpr, tpr, thresholds):
            - fpr: False Positive Rate array
            - tpr: True Positive Rate array
            - thresholds: Threshold values used

        Example:
            >>> calc = ROCCalculator()
            >>> y_true = [0, 0, 1, 1]
            >>> y_scores = [0.1, 0.4, 0.35, 0.8]
            >>> fpr, tpr, thresholds = calc.roc_curve(y_true, y_scores)
        """
        y_true, y_scores = self._validate_inputs(y_true, y_scores)

        # Convert to binary if needed
        y_true_binary = (y_true == pos_label).astype(int)

        # Sort by scores in descending order
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_scores_sorted = y_scores[desc_score_indices]
        y_true_sorted = y_true_binary[desc_score_indices]

        # Get unique thresholds
        thresholds = np.unique(y_scores_sorted)
        thresholds = np.append(thresholds, thresholds[-1] - 1e-10)
        thresholds = np.insert(thresholds, 0, thresholds[0] + 1e-10)

        n_positives = np.sum(y_true_sorted == 1)
        n_negatives = len(y_true_sorted) - n_positives

        if n_positives == 0:
            logger.warning("No positive samples found")
            return (
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 0.0]),
            )

        if n_negatives == 0:
            logger.warning("No negative samples found")
            return (
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 0.0]),
            )

        tpr = []
        fpr = []

        for threshold in thresholds:
            y_pred = (y_scores_sorted >= threshold).astype(int)

            tp = np.sum((y_true_sorted == 1) & (y_pred == 1))
            fp = np.sum((y_true_sorted == 0) & (y_pred == 1))
            fn = np.sum((y_true_sorted == 1) & (y_pred == 0))
            tn = np.sum((y_true_sorted == 0) & (y_pred == 0))

            tpr_val = tp / n_positives if n_positives > 0 else 0.0
            fpr_val = fp / n_negatives if n_negatives > 0 else 0.0

            tpr.append(tpr_val)
            fpr.append(fpr_val)

        fpr = np.array(fpr)
        tpr = np.array(tpr)

        # Ensure we start at (0, 0) and end at (1, 1)
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)

        logger.debug(f"ROC curve calculated with {len(fpr)} points")
        return fpr, tpr, thresholds

    def auc(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_scores: Union[List, np.ndarray, pd.Series],
        pos_label: int = 1,
    ) -> float:
        """Calculate Area Under the ROC Curve (AUC).

        Uses the trapezoidal rule to compute the area under the ROC curve.
        AUC ranges from 0 to 1, where 1.0 indicates perfect classification
        and 0.5 indicates random performance.

        Args:
            y_true: True binary labels (0 or 1).
            y_scores: Predicted scores or probabilities.
            pos_label: Label of the positive class. Default: 1.

        Returns:
            AUC score (float between 0 and 1).

        Example:
            >>> calc = ROCCalculator()
            >>> y_true = [0, 0, 1, 1]
            >>> y_scores = [0.1, 0.4, 0.35, 0.8]
            >>> auc_score = calc.auc(y_true, y_scores)
            >>> print(auc_score)
            0.75
        """
        fpr, tpr, _ = self.roc_curve(y_true, y_scores, pos_label=pos_label)

        # Calculate AUC using trapezoidal rule
        # Sort by FPR to ensure proper integration
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]

        auc_score = np.trapz(tpr_sorted, fpr_sorted)

        logger.debug(f"AUC: {auc_score:.4f}")
        return float(auc_score)

    def plot_roc_curve(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_scores: Union[List, np.ndarray, pd.Series],
        pos_label: int = 1,
        title: Optional[str] = None,
        label: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot ROC curve.

        Args:
            y_true: True binary labels (0 or 1).
            y_scores: Predicted scores or probabilities.
            pos_label: Label of the positive class. Default: 1.
            title: Optional title for the plot. Default: auto-generated.
            label: Optional label for the curve in legend.
            save_path: Optional path to save the figure. Default: None.
            show: Whether to display the plot. Default: True.

        Example:
            >>> calc = ROCCalculator()
            >>> y_true = [0, 0, 1, 1]
            >>> y_scores = [0.1, 0.4, 0.35, 0.8]
            >>> calc.plot_roc_curve(y_true, y_scores)
        """
        fpr, tpr, _ = self.roc_curve(y_true, y_scores, pos_label=pos_label)
        auc_score = self.auc(y_true, y_scores, pos_label=pos_label)

        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(
            fpr,
            tpr,
            linewidth=self.linewidth,
            label=label if label else f"ROC curve (AUC = {auc_score:.3f})",
        )
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")

        if title is None:
            title = "ROC Curve"

        plt.title(title, fontsize=self.fontsize + 2, fontweight="bold")
        plt.xlabel("False Positive Rate", fontsize=self.fontsize)
        plt.ylabel("True Positive Rate", fontsize=self.fontsize)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right", fontsize=self.fontsize)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"ROC curve saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def calculate_all_metrics(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_scores: Union[List, np.ndarray, pd.Series],
        pos_label: int = 1,
    ) -> Dict[str, Union[float, Dict[str, np.ndarray]]]:
        """Calculate ROC curve and AUC.

        Args:
            y_true: True binary labels (0 or 1).
            y_scores: Predicted scores or probabilities.
            pos_label: Label of the positive class. Default: 1.

        Returns:
            Dictionary containing:
            - auc: AUC score
            - roc_curve: Dictionary with fpr, tpr, thresholds arrays

        Example:
            >>> calc = ROCCalculator()
            >>> y_true = [0, 0, 1, 1]
            >>> y_scores = [0.1, 0.4, 0.35, 0.8]
            >>> results = calc.calculate_all_metrics(y_true, y_scores)
            >>> print(results['auc'])
            0.75
        """
        logger.info("Calculating ROC curve and AUC")
        fpr, tpr, thresholds = self.roc_curve(
            y_true, y_scores, pos_label=pos_label
        )
        auc_score = self.auc(y_true, y_scores, pos_label=pos_label)

        results = {
            "auc": auc_score,
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
        }

        logger.info(f"AUC calculation complete: {auc_score:.4f}")
        return results

    def print_report(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_scores: Union[List, np.ndarray, pd.Series],
        pos_label: int = 1,
    ) -> None:
        """Print formatted ROC and AUC report to console.

        Args:
            y_true: True binary labels (0 or 1).
            y_scores: Predicted scores or probabilities.
            pos_label: Label of the positive class. Default: 1.
        """
        fpr, tpr, thresholds = self.roc_curve(
            y_true, y_scores, pos_label=pos_label
        )
        auc_score = self.auc(y_true, y_scores, pos_label=pos_label)

        print("\n" + "=" * 60)
        print("ROC CURVE AND AUC REPORT")
        print("=" * 60)

        print(f"\nAUC Score: {auc_score:.6f}")

        print(f"\nROC Curve Points: {len(fpr)}")
        print(f"FPR Range: [{np.min(fpr):.4f}, {np.max(fpr):.4f}]")
        print(f"TPR Range: [{np.min(tpr):.4f}, {np.max(tpr):.4f}]")

        print(f"\nSample Points (first 5):")
        print(f"{'FPR':<12} {'TPR':<12} {'Threshold':<12}")
        print("-" * 36)
        for i in range(min(5, len(fpr))):
            threshold = (
                thresholds[i] if i < len(thresholds) else "N/A"
            )
            print(f"{fpr[i]:<12.4f} {tpr[i]:<12.4f} {str(threshold):<12}")

        print(f"\nSample Points (last 5):")
        print(f"{'FPR':<12} {'TPR':<12} {'Threshold':<12}")
        print("-" * 36)
        start_idx = max(0, len(fpr) - 5)
        for i in range(start_idx, len(fpr)):
            threshold = (
                thresholds[i] if i < len(thresholds) else "N/A"
            )
            print(f"{fpr[i]:<12.4f} {tpr[i]:<12.4f} {str(threshold):<12}")

        print("\n" + "=" * 60)

    def save_report(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_scores: Union[List, np.ndarray, pd.Series],
        pos_label: int = 1,
        output_path: str = "roc_auc_report.json",
    ) -> None:
        """Save ROC curve and AUC report to JSON file.

        Args:
            y_true: True binary labels (0 or 1).
            y_scores: Predicted scores or probabilities.
            pos_label: Label of the positive class. Default: 1.
            output_path: Path to save JSON file.
        """
        results = self.calculate_all_metrics(y_true, y_scores, pos_label=pos_label)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"ROC and AUC report saved to: {output_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ROC Curve and AUC Calculator"
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
        "--y-scores",
        type=str,
        required=True,
        help="Path to CSV file with predicted scores or comma-separated values",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Column name if input is CSV file (default: first column)",
    )
    parser.add_argument(
        "--pos-label",
        type=int,
        default=1,
        help="Label of the positive class (default: 1)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot ROC curve",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save ROC curve plot",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Path to save report as JSON",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (useful when saving)",
    )

    args = parser.parse_args()

    calc = ROCCalculator(config_path=args.config)

    try:
        # Load true labels
        if args.y_true.endswith(".csv"):
            df_true = pd.read_csv(args.y_true)
            column = args.column if args.column else df_true.columns[0]
            y_true = df_true[column].values
        else:
            y_true = [int(float(x.strip())) for x in args.y_true.split(",")]

        # Load predicted scores
        if args.y_scores.endswith(".csv"):
            df_scores = pd.read_csv(args.y_scores)
            column = args.column if args.column else df_scores.columns[0]
            y_scores = df_scores[column].values
        else:
            y_scores = [float(x.strip()) for x in args.y_scores.split(",")]

        print("\n=== ROC Curve and AUC Calculator ===")
        print(f"True labels: {len(y_true)} samples")
        print(f"Predicted scores: {len(y_scores)} samples")

        # Print report
        calc.print_report(y_true, y_scores, pos_label=args.pos_label)

        # Plot ROC curve
        if args.plot or args.save_plot:
            calc.plot_roc_curve(
                y_true,
                y_scores,
                pos_label=args.pos_label,
                save_path=args.save_plot,
                show=not args.no_show,
            )

        # Save report
        if args.save_report:
            calc.save_report(
                y_true,
                y_scores,
                pos_label=args.pos_label,
                output_path=args.save_report,
            )

    except Exception as e:
        logger.error(f"Error calculating ROC and AUC: {e}")
        raise


if __name__ == "__main__":
    main()
