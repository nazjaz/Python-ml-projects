"""Data Augmentation Tool for Numerical Data.

This module provides functionality to perform data augmentation techniques
for numerical data including noise injection and scaling variations.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DataAugmenter:
    """Performs data augmentation on numerical datasets."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize DataAugmenter with configuration.

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
        self.augmentation_history: List[Dict[str, any]] = []

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
        aug_config = self.config.get("augmentation", {})
        self.noise_type = aug_config.get("noise_type", "gaussian")
        self.noise_std = aug_config.get("noise_std", 0.1)
        self.noise_mean = aug_config.get("noise_mean", 0.0)
        self.scaling_type = aug_config.get("scaling_type", "multiplicative")
        self.scaling_factor_min = aug_config.get("scaling_factor_min", 0.9)
        self.scaling_factor_max = aug_config.get("scaling_factor_max", 1.1)
        self.random_state = aug_config.get("random_state", 42)

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

    def inject_noise(
        self,
        columns: Optional[List[str]] = None,
        noise_type: Optional[str] = None,
        noise_std: Optional[float] = None,
        noise_mean: Optional[float] = None,
        random_state: Optional[int] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Inject noise into numerical columns.

        Args:
            columns: List of column names to augment (None for all numeric).
            noise_type: Type of noise (gaussian, uniform, laplace).
            noise_std: Standard deviation for noise (default from config).
            noise_mean: Mean for noise (default from config).
            random_state: Random seed (default from config).
            inplace: Whether to modify in place.

        Returns:
            DataFrame with noise injected.

        Raises:
            ValueError: If no data loaded, columns invalid, or noise type invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        noise_type = noise_type or self.noise_type
        noise_std = noise_std if noise_std is not None else self.noise_std
        noise_mean = noise_mean if noise_mean is not None else self.noise_mean
        random_state = random_state if random_state is not None else self.random_state

        np.random.seed(random_state)

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for noise injection")
            return self.data if inplace else self.data.copy()

        result = self.data if inplace else self.data.copy()

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping noise injection"
                )
                continue

            original_values = result[col].values.copy()

            if noise_type == "gaussian":
                noise = np.random.normal(noise_mean, noise_std, len(result))
            elif noise_type == "uniform":
                noise = np.random.uniform(
                    noise_mean - noise_std * np.sqrt(3),
                    noise_mean + noise_std * np.sqrt(3),
                    len(result),
                )
            elif noise_type == "laplace":
                noise = np.random.laplace(noise_mean, noise_std, len(result))
            else:
                raise ValueError(
                    f"Invalid noise type: {noise_type}. "
                    f"Use 'gaussian', 'uniform', or 'laplace'"
                )

            result[col] = original_values + noise

            logger.info(
                f"Injected {noise_type} noise into '{col}' "
                f"(mean={noise_mean}, std={noise_std})"
            )

        self.augmentation_history.append(
            {
                "method": "noise_injection",
                "noise_type": noise_type,
                "noise_std": noise_std,
                "noise_mean": noise_mean,
                "columns": columns,
            }
        )

        if not inplace:
            return result

    def apply_scaling_variations(
        self,
        columns: Optional[List[str]] = None,
        scaling_type: Optional[str] = None,
        scaling_factor_min: Optional[float] = None,
        scaling_factor_max: Optional[float] = None,
        random_state: Optional[int] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Apply scaling variations to numerical columns.

        Args:
            columns: List of column names to augment (None for all numeric).
            scaling_type: Type of scaling (multiplicative, additive, percentage).
            scaling_factor_min: Minimum scaling factor (default from config).
            scaling_factor_max: Maximum scaling factor (default from config).
            random_state: Random seed (default from config).
            inplace: Whether to modify in place.

        Returns:
            DataFrame with scaling variations applied.

        Raises:
            ValueError: If no data loaded, columns invalid, or scaling type invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        scaling_type = scaling_type or self.scaling_type
        scaling_factor_min = (
            scaling_factor_min
            if scaling_factor_min is not None
            else self.scaling_factor_min
        )
        scaling_factor_max = (
            scaling_factor_max
            if scaling_factor_max is not None
            else self.scaling_factor_max
        )
        random_state = random_state if random_state is not None else self.random_state

        if scaling_factor_min >= scaling_factor_max:
            raise ValueError(
                "scaling_factor_min must be less than scaling_factor_max"
            )

        np.random.seed(random_state)

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for scaling variations")
            return self.data if inplace else self.data.copy()

        result = self.data if inplace else self.data.copy()

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping scaling"
                )
                continue

            original_values = result[col].values.copy()

            if scaling_type == "multiplicative":
                factors = np.random.uniform(
                    scaling_factor_min, scaling_factor_max, len(result)
                )
                result[col] = original_values * factors
            elif scaling_type == "additive":
                factors = np.random.uniform(
                    scaling_factor_min, scaling_factor_max, len(result)
                )
                result[col] = original_values + factors
            elif scaling_type == "percentage":
                factors = np.random.uniform(
                    scaling_factor_min, scaling_factor_max, len(result)
                )
                result[col] = original_values * (1 + factors)
            else:
                raise ValueError(
                    f"Invalid scaling type: {scaling_type}. "
                    f"Use 'multiplicative', 'additive', or 'percentage'"
                )

            logger.info(
                f"Applied {scaling_type} scaling to '{col}' "
                f"(range: [{scaling_factor_min}, {scaling_factor_max}])"
            )

        self.augmentation_history.append(
            {
                "method": "scaling_variations",
                "scaling_type": scaling_type,
                "scaling_factor_min": scaling_factor_min,
                "scaling_factor_max": scaling_factor_max,
                "columns": columns,
            }
        )

        if not inplace:
            return result

    def augment_all(
        self,
        noise_type: Optional[str] = None,
        noise_std: Optional[float] = None,
        scaling_type: Optional[str] = None,
        scaling_factor_min: Optional[float] = None,
        scaling_factor_max: Optional[float] = None,
        columns: Optional[List[str]] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Apply all augmentation techniques.

        Args:
            noise_type: Type of noise (default from config).
            noise_std: Standard deviation for noise (default from config).
            scaling_type: Type of scaling (default from config).
            scaling_factor_min: Minimum scaling factor (default from config).
            scaling_factor_max: Maximum scaling factor (default from config).
            columns: List of columns to augment (None for all numeric).
            inplace: Whether to modify in place.

        Returns:
            DataFrame with all augmentations applied.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if inplace:
            self.inject_noise(
                columns=columns,
                noise_type=noise_type,
                noise_std=noise_std,
                inplace=True,
            )
            self.apply_scaling_variations(
                columns=columns,
                scaling_type=scaling_type,
                scaling_factor_min=scaling_factor_min,
                scaling_factor_max=scaling_factor_max,
                inplace=True,
            )
            result = self.data
        else:
            result = self.inject_noise(
                columns=columns,
                noise_type=noise_type,
                noise_std=noise_std,
                inplace=False,
            )
            temp_augmenter = DataAugmenter()
            temp_augmenter.data = result
            result = temp_augmenter.apply_scaling_variations(
                columns=columns,
                scaling_type=scaling_type,
                scaling_factor_min=scaling_factor_min,
                scaling_factor_max=scaling_factor_max,
                inplace=False,
            )

        logger.info("Applied all augmentation techniques")
        return result

    def get_augmentation_summary(self) -> Dict[str, any]:
        """Get summary of augmentation operations performed.

        Returns:
            Dictionary with augmentation summary.
        """
        return {
            "total_operations": len(self.augmentation_history),
            "operations": self.augmentation_history.copy(),
        }

    def save_augmented_data(self, output_path: str) -> None:
        """Save augmented data to CSV file.

        Args:
            output_path: Path to output CSV file.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.data.to_csv(output_path, index=False)
        logger.info(f"Saved augmented data to: {output_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Data Augmentation Tool")
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
        "--method",
        type=str,
        choices=["noise", "scaling", "all"],
        default="all",
        help="Augmentation method to apply",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="Specific columns to augment (optional)",
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        choices=["gaussian", "uniform", "laplace"],
        default=None,
        help="Type of noise (default from config)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="Noise standard deviation (default from config)",
    )
    parser.add_argument(
        "--scaling-type",
        type=str,
        choices=["multiplicative", "additive", "percentage"],
        default=None,
        help="Type of scaling (default from config)",
    )
    parser.add_argument(
        "--scaling-min",
        type=float,
        default=None,
        help="Minimum scaling factor (default from config)",
    )
    parser.add_argument(
        "--scaling-max",
        type=float,
        default=None,
        help="Maximum scaling factor (default from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (optional)",
    )

    args = parser.parse_args()

    augmenter = DataAugmenter(config_path=args.config)

    try:
        augmenter.load_data(file_path=args.input)

        print("\n=== Numerical Columns ===")
        numeric_cols = augmenter.get_numeric_columns()
        print(f"Found {len(numeric_cols)} numerical columns: {numeric_cols}")

        print("\n=== Applying Augmentation ===")
        if args.method == "noise":
            augmented_data = augmenter.inject_noise(
                columns=args.columns,
                noise_type=args.noise_type,
                noise_std=args.noise_std,
            )
        elif args.method == "scaling":
            augmented_data = augmenter.apply_scaling_variations(
                columns=args.columns,
                scaling_type=args.scaling_type,
                scaling_factor_min=args.scaling_min,
                scaling_factor_max=args.scaling_max,
            )
        elif args.method == "all":
            augmented_data = augmenter.augment_all(
                columns=args.columns,
                noise_type=args.noise_type,
                noise_std=args.noise_std,
                scaling_type=args.scaling_type,
                scaling_factor_min=args.scaling_min,
                scaling_factor_max=args.scaling_max,
            )

        print("\n=== Augmentation Summary ===")
        summary = augmenter.get_augmentation_summary()
        print(f"Total operations: {summary['total_operations']}")
        for op in summary["operations"]:
            print(f"  {op['method']}: {op.get('columns', 'all columns')}")

        if args.output:
            augmenter.save_augmented_data(args.output)
            print(f"\nAugmented data saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error augmenting data: {e}")
        raise


if __name__ == "__main__":
    main()
