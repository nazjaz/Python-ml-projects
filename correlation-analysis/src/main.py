"""Correlation Analysis Tool.

This module provides functionality to calculate correlation matrices and
visualize feature relationships using heatmaps and scatter plots.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


class CorrelationAnalyzer:
    """Analyzes and visualizes correlations between features."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize CorrelationAnalyzer with configuration.

        Args:
            config_path: Path to configuration YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_parameters()
        self.data: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None

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
        """Initialize algorithm parameters from configuration."""
        analysis_config = self.config.get("correlation", {})
        self.method = analysis_config.get("method", "pearson")
        self.figsize = tuple(analysis_config.get("figsize", [10, 8]))
        self.dpi = analysis_config.get("dpi", 100)
        self.cmap = analysis_config.get("colormap", "coolwarm")
        self.output_dir = analysis_config.get("output_dir", "plots")

        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = self.figsize
        plt.rcParams["figure.dpi"] = self.dpi

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def load_data(
        self,
        file_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Load data from file or use provided DataFrame.

        Args:
            file_path: Path to CSV file (optional).
            dataframe: Pandas DataFrame (optional).

        Returns:
            Loaded DataFrame.

        Raises:
            ValueError: If neither file_path nor dataframe provided.
            FileNotFoundError: If file doesn't exist.
        """
        if dataframe is not None:
            self.data = dataframe.copy()
            logger.info(f"Loaded DataFrame with shape {self.data.shape}")
        elif file_path is not None:
            try:
                self.data = pd.read_csv(file_path)
                logger.info(
                    f"Loaded CSV file: {file_path}, "
                    f"shape: {self.data.shape}"
                )
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                raise
        else:
            raise ValueError("Either file_path or dataframe must be provided")

        return self.data

    def get_numeric_columns(self) -> List[str]:
        """Get list of numerical columns in the dataset.

        Returns:
            List of numerical column names.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        numeric_cols = self.data.select_dtypes(include=["number"]).columns.tolist()
        logger.info(f"Found {len(numeric_cols)} numerical columns")
        return numeric_cols

    def calculate_correlation(
        self,
        method: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Calculate correlation matrix for numerical columns.

        Args:
            method: Correlation method (pearson, spearman, kendall).
            columns: List of columns to include (None for all numeric).

        Returns:
            Correlation matrix DataFrame.

        Raises:
            ValueError: If no data loaded or invalid method.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        method = method or self.method
        valid_methods = ["pearson", "spearman", "kendall"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method: {method}. Use one of {valid_methods}"
            )

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for correlation")
            return pd.DataFrame()

        numeric_data = self.data[columns]
        self.correlation_matrix = numeric_data.corr(method=method)

        logger.info(
            f"Calculated {method} correlation matrix: "
            f"{len(self.correlation_matrix)}x{len(self.correlation_matrix)}"
        )

        return self.correlation_matrix

    def get_strong_correlations(
        self,
        threshold: float = 0.7,
        method: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """Get pairs of features with strong correlations.

        Args:
            threshold: Minimum absolute correlation value.
            method: Correlation method (None uses calculated matrix).

        Returns:
            List of tuples (col1, col2, correlation).

        Raises:
            ValueError: If no correlation matrix calculated.
        """
        if self.correlation_matrix is None:
            if method:
                self.calculate_correlation(method=method)
            else:
                self.calculate_correlation()

        strong_corr = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                col1 = self.correlation_matrix.columns[i]
                col2 = self.correlation_matrix.columns[j]
                corr_value = self.correlation_matrix.iloc[i, j]

                if abs(corr_value) >= threshold:
                    strong_corr.append((col1, col2, corr_value))

        strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)

        logger.info(
            f"Found {len(strong_corr)} pairs with |correlation| >= {threshold}"
        )

        return strong_corr

    def plot_heatmap(
        self,
        method: Optional[str] = None,
        columns: Optional[List[str]] = None,
        annot: bool = True,
        fmt: str = ".2f",
        save_path: Optional[str] = None,
    ) -> None:
        """Create correlation heatmap visualization.

        Args:
            method: Correlation method (None uses calculated matrix).
            columns: List of columns to include (None for all numeric).
            annot: Whether to annotate cells with correlation values.
            fmt: Format string for annotations.
            save_path: Path to save plot (None displays plot).

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if self.correlation_matrix is None or method:
            self.calculate_correlation(method=method, columns=columns)

        if self.correlation_matrix is None or self.correlation_matrix.empty:
            logger.warning("No correlation matrix to plot")
            return

        plt.figure(figsize=self.figsize)
        sns.heatmap(
            self.correlation_matrix,
            annot=annot,
            fmt=fmt,
            cmap=self.cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title(f"Correlation Heatmap ({self.method.capitalize()})")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved heatmap to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_scatter(
        self,
        x_col: str,
        y_col: str,
        save_path: Optional[str] = None,
    ) -> None:
        """Create scatter plot for two features.

        Args:
            x_col: Name of x-axis column.
            y_col: Name of y-axis column.
            save_path: Path to save plot (None displays plot).

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if x_col not in self.data.columns:
            raise ValueError(f"Column '{x_col}' not found in data")
        if y_col not in self.data.columns:
            raise ValueError(f"Column '{y_col}' not found in data")

        if not pd.api.types.is_numeric_dtype(self.data[x_col]):
            raise ValueError(f"Column '{x_col}' is not numeric")
        if not pd.api.types.is_numeric_dtype(self.data[y_col]):
            raise ValueError(f"Column '{y_col}' is not numeric")

        correlation = self.data[x_col].corr(self.data[y_col])

        plt.figure(figsize=self.figsize)
        plt.scatter(self.data[x_col], self.data[y_col], alpha=0.6, s=50)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Scatter Plot: {x_col} vs {y_col}\nCorrelation: {correlation:.3f}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved scatter plot to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_scatter_matrix(
        self,
        columns: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Create scatter plot matrix for multiple features.

        Args:
            columns: List of columns to plot (None for all numeric, max 5).
            save_path: Path to save plot (None displays plot).

        Raises:
            ValueError: If no data loaded or too many columns.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if columns is None:
            columns = self.get_numeric_columns()[:5]

        if len(columns) > 5:
            logger.warning("Too many columns, limiting to first 5")
            columns = columns[:5]

        if not columns:
            logger.warning("No numerical columns found")
            return

        numeric_data = self.data[columns]

        fig = plt.figure(figsize=(12, 12))
        pd.plotting.scatter_matrix(
            numeric_data, alpha=0.6, figsize=(12, 12), diagonal="kde"
        )
        plt.suptitle("Scatter Plot Matrix", y=0.995, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved scatter matrix to: {save_path}")
        else:
            plt.show()

        plt.close()

    def analyze_correlations(
        self,
        method: Optional[str] = None,
        threshold: float = 0.7,
    ) -> Dict[str, any]:
        """Analyze correlations and generate summary.

        Args:
            method: Correlation method (None uses config default).
            threshold: Threshold for strong correlations.

        Returns:
            Dictionary with correlation analysis.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        method = method or self.method
        self.calculate_correlation(method=method)

        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return {"message": "No correlation matrix calculated"}

        strong_corr = self.get_strong_correlations(threshold=threshold)

        analysis = {
            "method": method,
            "matrix_size": len(self.correlation_matrix),
            "strong_correlations": strong_corr,
            "strong_correlation_count": len(strong_corr),
            "mean_correlation": float(self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].mean()),
            "max_correlation": float(self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].max()),
            "min_correlation": float(self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].min()),
        }

        logger.info(f"Correlation analysis complete: {len(strong_corr)} strong pairs")
        return analysis


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Correlation Analysis Tool"
    )
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
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=["heatmap", "scatter", "scatter_matrix", "all"],
        default="all",
        help="Type of plot to create",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["pearson", "spearman", "kendall"],
        default=None,
        help="Correlation method (default from config)",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="Specific columns to analyze (optional)",
    )
    parser.add_argument(
        "--x",
        type=str,
        default=None,
        help="X column for scatter plot",
    )
    parser.add_argument(
        "--y",
        type=str,
        default=None,
        help="Y column for scatter plot",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold for strong correlations",
    )

    args = parser.parse_args()

    analyzer = CorrelationAnalyzer(config_path=args.config)

    try:
        analyzer.load_data(file_path=args.input)

        print("\n=== Numerical Columns ===")
        numeric_cols = analyzer.get_numeric_columns()
        print(f"Found {len(numeric_cols)} numerical columns: {numeric_cols}")

        print("\n=== Correlation Analysis ===")
        analysis = analyzer.analyze_correlations(
            method=args.method, threshold=args.threshold
        )
        print(f"Method: {analysis['method']}")
        print(f"Mean correlation: {analysis['mean_correlation']:.3f}")
        print(f"Max correlation: {analysis['max_correlation']:.3f}")
        print(f"Min correlation: {analysis['min_correlation']:.3f}")
        print(
            f"Strong correlations (|r| >= {args.threshold}): "
            f"{analysis['strong_correlation_count']}"
        )

        if analysis["strong_correlations"]:
            print("\nStrong correlation pairs:")
            for col1, col2, corr in analysis["strong_correlations"][:10]:
                print(f"  {col1} - {col2}: {corr:.3f}")

        if args.plot == "heatmap":
            analyzer.plot_heatmap(method=args.method, columns=args.columns)
        elif args.plot == "scatter":
            if args.x and args.y:
                analyzer.plot_scatter(args.x, args.y)
            else:
                print("Error: --x and --y required for scatter plot")
        elif args.plot == "scatter_matrix":
            analyzer.plot_scatter_matrix(columns=args.columns)
        elif args.plot == "all":
            analyzer.plot_heatmap(method=args.method, columns=args.columns)
            if len(numeric_cols) >= 2:
                analyzer.plot_scatter_matrix(columns=args.columns or numeric_cols[:5])

        print("\nVisualization complete!")

    except Exception as e:
        logger.error(f"Error analyzing correlations: {e}")
        raise


if __name__ == "__main__":
    main()
