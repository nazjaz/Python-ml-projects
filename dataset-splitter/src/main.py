"""Dataset Splitter Tool.

This module provides functionality to split datasets into training, validation,
and test sets with configurable ratios and stratification.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Splits datasets into training, validation, and test sets."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize DatasetSplitter with configuration.

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
        self.target: Optional[pd.Series] = None

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
        split_config = self.config.get("split", {})
        self.train_ratio = split_config.get("train_ratio", 0.7)
        self.val_ratio = split_config.get("val_ratio", 0.15)
        self.test_ratio = split_config.get("test_ratio", 0.15)
        self.random_state = split_config.get("random_state", 42)
        self.shuffle = split_config.get("shuffle", True)
        self.stratify = split_config.get("stratify", True)

        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            logger.warning(
                f"Ratios sum to {total_ratio}, normalizing to 1.0"
            )
            self.train_ratio /= total_ratio
            self.val_ratio /= total_ratio
            self.test_ratio /= total_ratio

    def load_data(
        self,
        file_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Load data from file or use provided DataFrame.

        Args:
            file_path: Path to CSV file (optional).
            dataframe: Pandas DataFrame (optional).
            target_column: Name of target column for stratification (optional).

        Returns:
            Tuple of (features DataFrame, target Series).

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

        if target_column:
            if target_column not in self.data.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            self.target = self.data[target_column]
            self.data = self.data.drop(columns=[target_column])
            logger.info(f"Set target column: {target_column}")
        else:
            self.target = None

        return self.data, self.target

    def split(
        self,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        stratify: Optional[bool] = None,
        random_state: Optional[int] = None,
        shuffle: Optional[bool] = None,
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]:
        """Split dataset into training, validation, and test sets.

        Args:
            train_ratio: Training set ratio (default from config).
            val_ratio: Validation set ratio (default from config).
            test_ratio: Test set ratio (default from config).
            stratify: Whether to stratify by target (default from config).
            random_state: Random seed (default from config).
            shuffle: Whether to shuffle data (default from config).

        Returns:
            Dictionary with keys 'train', 'val', 'test', each containing
            tuple of (features DataFrame, target Series).

        Raises:
            ValueError: If no data loaded or ratios invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        train_ratio = train_ratio if train_ratio is not None else self.train_ratio
        val_ratio = val_ratio if val_ratio is not None else self.val_ratio
        test_ratio = test_ratio if test_ratio is not None else self.test_ratio
        stratify = stratify if stratify is not None else self.stratify
        random_state = random_state if random_state is not None else self.random_state
        shuffle = shuffle if shuffle is not None else self.shuffle

        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            logger.warning(f"Ratios sum to {total_ratio}, normalizing")
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio

        stratify_col = None
        if stratify and self.target is not None:
            stratify_col = self.target
            logger.info("Using stratification based on target column")
        elif stratify and self.target is None:
            logger.warning(
                "Stratification requested but no target column, "
                "proceeding without stratification"
            )
            stratify_col = None

        X = self.data.copy()
        y = self.target.copy() if self.target is not None else None

        if val_ratio > 0:
            test_size = test_ratio
            val_size = val_ratio / (train_ratio + val_ratio)

            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify_col,
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify_col if stratify_col is not None else None,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_ratio,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify_col,
            )
            X_val = pd.DataFrame()
            y_val = None

        splits = {
            "train": (X_train, y_train),
            "val": (X_val, y_val) if val_ratio > 0 else (None, None),
            "test": (X_test, y_test),
        }

        logger.info(
            f"Split complete - Train: {len(X_train)}, "
            f"Val: {len(X_val) if val_ratio > 0 else 0}, "
            f"Test: {len(X_test)}"
        )

        return splits

    def get_split_summary(
        self,
        splits: Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]],
    ) -> Dict[str, any]:
        """Get summary of dataset splits.

        Args:
            splits: Dictionary returned from split() method.

        Returns:
            Dictionary with split summary statistics.
        """
        summary = {}

        for split_name, (X, y) in splits.items():
            if X is None:
                continue

            split_info = {
                "samples": len(X),
                "features": len(X.columns),
                "feature_names": list(X.columns),
            }

            if y is not None:
                if pd.api.types.is_numeric_dtype(y):
                    split_info["target_type"] = "numeric"
                    split_info["target_stats"] = {
                        "mean": float(y.mean()),
                        "std": float(y.std()),
                        "min": float(y.min()),
                        "max": float(y.max()),
                    }
                else:
                    split_info["target_type"] = "categorical"
                    split_info["target_distribution"] = y.value_counts().to_dict()

            summary[split_name] = split_info

        return summary

    def save_splits(
        self,
        splits: Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]],
        output_dir: str = "splits",
    ) -> None:
        """Save splits to CSV files.

        Args:
            splits: Dictionary returned from split() method.
            output_dir: Directory to save split files.

        Raises:
            ValueError: If splits invalid.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for split_name, (X, y) in splits.items():
            if X is None:
                continue

            X_path = Path(output_dir) / f"X_{split_name}.csv"
            X.to_csv(X_path, index=False)
            logger.info(f"Saved {split_name} features to: {X_path}")

            if y is not None:
                y_path = Path(output_dir) / f"y_{split_name}.csv"
                y.to_frame().to_csv(y_path, index=False)
                logger.info(f"Saved {split_name} target to: {y_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Splitter Tool")
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
        "--target",
        type=str,
        default=None,
        help="Name of target column for stratification (optional)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Training set ratio (default from config)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Validation set ratio (default from config)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Test set ratio (default from config)",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratification",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="splits",
        help="Directory to save split files",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed (default from config)",
    )

    args = parser.parse_args()

    splitter = DatasetSplitter(config_path=args.config)

    try:
        splitter.load_data(file_path=args.input, target_column=args.target)

        print("\n=== Dataset Information ===")
        print(f"Shape: {splitter.data.shape}")
        print(f"Features: {len(splitter.data.columns)}")
        if splitter.target is not None:
            print(f"Target: {args.target}")
            print(f"Target type: {splitter.target.dtype}")

        print("\n=== Splitting Dataset ===")
        splits = splitter.split(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify=not args.no_stratify,
            random_state=args.random_state,
        )

        print("\n=== Split Summary ===")
        summary = splitter.get_split_summary(splits)
        for split_name, info in summary.items():
            print(f"\n{split_name.upper()}:")
            print(f"  Samples: {info['samples']}")
            print(f"  Features: {info['features']}")
            if "target_type" in info:
                print(f"  Target type: {info['target_type']}")
                if info["target_type"] == "categorical":
                    print(f"  Classes: {len(info['target_distribution'])}")

        print(f"\n=== Saving Splits ===")
        splitter.save_splits(splits, output_dir=args.output_dir)
        print(f"Splits saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        raise


if __name__ == "__main__":
    main()
