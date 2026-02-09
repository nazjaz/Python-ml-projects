"""Data Visualization Tool for Exploratory Data Analysis.

This module provides functionality to visualize data distributions using
histograms, box plots, and density plots for exploratory data analysis.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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


class DataVisualizer:
    """Creates visualizations for exploratory data analysis."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize DataVisualizer with configuration.

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
        viz_config = self.config.get("visualization", {})
        self.figsize = tuple(viz_config.get("figsize", [10, 6]))
        self.dpi = viz_config.get("dpi", 100)
        self.style = viz_config.get("style", "whitegrid")
        self.color_palette = viz_config.get("color_palette", "husl")
        self.output_dir = viz_config.get("output_dir", "plots")

        sns.set_style(self.style)
        sns.set_palette(self.color_palette)
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

    def plot_histogram(
        self,
        columns: Optional[List[str]] = None,
        bins: Optional[int] = None,
        kde: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """Create histogram plots for numerical columns.

        Args:
            columns: List of column names to plot (None for all numeric).
            bins: Number of bins for histogram (None for auto).
            kde: Whether to overlay kernel density estimate.
            save_path: Path to save plot (None displays plot).

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for histogram")
            return

        num_cols = len(columns)
        cols = min(3, num_cols)
        rows = (num_cols + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        if num_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if num_cols > 1 else [axes]

        for idx, col in enumerate(columns):
            if col not in self.data.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping histogram"
                )
                continue

            ax = axes[idx] if num_cols > 1 else axes[0]
            self.data[col].hist(bins=bins, ax=ax, alpha=0.7, edgecolor="black")
            if kde:
                ax2 = ax.twinx()
                self.data[col].plot.kde(ax=ax2, alpha=0.5, color="red")
                ax2.set_ylabel("Density", color="red")
            ax.set_title(f"Histogram: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

        for idx in range(num_cols, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved histogram to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_boxplot(
        self,
        columns: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Create box plots for numerical columns.

        Args:
            columns: List of column names to plot (None for all numeric).
            save_path: Path to save plot (None displays plot).

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for box plot")
            return

        num_cols = len(columns)
        cols = min(3, num_cols)
        rows = (num_cols + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        if num_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if num_cols > 1 else [axes]

        for idx, col in enumerate(columns):
            if col not in self.data.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping box plot"
                )
                continue

            ax = axes[idx] if num_cols > 1 else axes[0]
            self.data[[col]].boxplot(ax=ax)
            ax.set_title(f"Box Plot: {col}")
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

        for idx in range(num_cols, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved box plot to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_density(
        self,
        columns: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Create density plots (KDE) for numerical columns.

        Args:
            columns: List of column names to plot (None for all numeric).
            save_path: Path to save plot (None displays plot).

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for density plot")
            return

        num_cols = len(columns)
        cols = min(3, num_cols)
        rows = (num_cols + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        if num_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if num_cols > 1 else [axes]

        for idx, col in enumerate(columns):
            if col not in self.data.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping density plot"
                )
                continue

            ax = axes[idx] if num_cols > 1 else axes[0]
            self.data[col].plot.kde(ax=ax, alpha=0.7)
            ax.set_title(f"Density Plot: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)

        for idx in range(num_cols, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved density plot to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_all_distributions(
        self,
        columns: Optional[List[str]] = None,
        bins: Optional[int] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        """Create all distribution plots (histogram, box plot, density).

        Args:
            columns: List of column names to plot (None for all numeric).
            bins: Number of bins for histogram (None for auto).
            save_dir: Directory to save plots (None uses config default).

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found")
            return

        save_dir = save_dir or self.output_dir

        logger.info(f"Creating all distribution plots for {len(columns)} columns")

        self.plot_histogram(
            columns=columns,
            bins=bins,
            save_path=f"{save_dir}/histograms.png" if save_dir else None,
        )

        self.plot_boxplot(
            columns=columns, save_path=f"{save_dir}/boxplots.png" if save_dir else None
        )

        self.plot_density(
            columns=columns, save_path=f"{save_dir}/density_plots.png" if save_dir else None
        )

        logger.info("Completed all distribution plots")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Data Visualization Tool for EDA"
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
        choices=["histogram", "boxplot", "density", "all"],
        default="all",
        help="Type of plot to create",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="Specific columns to plot (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save plot (optional)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Number of bins for histogram (optional)",
    )

    args = parser.parse_args()

    visualizer = DataVisualizer(config_path=args.config)

    try:
        visualizer.load_data(file_path=args.input)

        print("\n=== Numerical Columns ===")
        numeric_cols = visualizer.get_numeric_columns()
        print(f"Found {len(numeric_cols)} numerical columns: {numeric_cols}")

        if args.plot == "histogram":
            visualizer.plot_histogram(
                columns=args.columns, bins=args.bins, save_path=args.output
            )
        elif args.plot == "boxplot":
            visualizer.plot_boxplot(columns=args.columns, save_path=args.output)
        elif args.plot == "density":
            visualizer.plot_density(columns=args.columns, save_path=args.output)
        elif args.plot == "all":
            visualizer.plot_all_distributions(
                columns=args.columns, bins=args.bins
            )

        print("\nVisualization complete!")

    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        raise


if __name__ == "__main__":
    main()
