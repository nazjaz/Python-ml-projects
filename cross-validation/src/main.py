"""Cross-Validation Tool.

This module provides functionality to perform cross-validation with
k-fold, stratified k-fold, and leave-one-out strategies.
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

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CrossValidator:
    """Perform cross-validation with various strategies."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize CrossValidator with configuration.

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
        """Initialize cross-validation parameters from configuration."""
        cv_config = self.config.get("cross_validation", {})
        self.default_n_splits = cv_config.get("n_splits", 5)
        self.default_random_state = cv_config.get("random_state", 42)
        self.default_shuffle = cv_config.get("shuffle", True)

    def _validate_inputs(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Optional[Union[List, np.ndarray, pd.Series]] = None,
    ) -> tuple:
        """Validate and convert inputs to numpy arrays.

        Args:
            X: Feature data.
            y: Target data (optional).

        Returns:
            Tuple of (X_array, y_array) as numpy arrays.

        Raises:
            ValueError: If inputs are invalid.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            else:
                y = np.asarray(y)

            if len(y) != len(X):
                raise ValueError(
                    f"Length mismatch: X has {len(X)} samples, "
                    f"y has {len(y)} samples"
                )
        else:
            y = None

        return X, y

    def k_fold_split(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Optional[Union[List, np.ndarray, pd.Series]] = None,
        n_splits: Optional[int] = None,
        shuffle: Optional[bool] = None,
        random_state: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Perform k-fold cross-validation split.

        Divides data into k folds and returns train/test indices
        for each fold.

        Args:
            X: Feature data.
            y: Target data (optional, not used for splitting).
            n_splits: Number of folds. Default: from config.
            shuffle: Whether to shuffle data before splitting.
                Default: from config.
            random_state: Random seed for reproducibility.
                Default: from config.

        Returns:
            List of tuples (train_indices, test_indices) for each fold.

        Example:
            >>> cv = CrossValidator()
            >>> X = [[1], [2], [3], [4], [5]]
            >>> splits = cv.k_fold_split(X, n_splits=3)
            >>> len(splits)
            3
        """
        X, _ = self._validate_inputs(X, y)

        n_splits = n_splits if n_splits is not None else self.default_n_splits
        shuffle = shuffle if shuffle is not None else self.default_shuffle
        random_state = (
            random_state
            if random_state is not None
            else self.default_random_state
        )

        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        if n_splits > len(X):
            raise ValueError(
                f"n_splits ({n_splits}) cannot be greater than "
                f"number of samples ({len(X)})"
            )

        indices = np.arange(len(X))

        if shuffle:
            rng = np.random.RandomState(random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(n_splits, len(X) // n_splits, dtype=int)
        fold_sizes[: len(X) % n_splits] += 1

        current = 0
        splits = []

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate(
                [indices[:start], indices[stop:]]
            )
            splits.append((train_indices, test_indices))
            current = stop

        logger.info(f"K-fold split completed: {n_splits} folds")
        return splits

    def stratified_k_fold_split(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
        n_splits: Optional[int] = None,
        shuffle: Optional[bool] = None,
        random_state: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Perform stratified k-fold cross-validation split.

        Divides data into k folds while maintaining class distribution
        in each fold. Requires target labels.

        Args:
            X: Feature data.
            y: Target labels (required for stratification).
            n_splits: Number of folds. Default: from config.
            shuffle: Whether to shuffle data before splitting.
                Default: from config.
            random_state: Random seed for reproducibility.
                Default: from config.

        Returns:
            List of tuples (train_indices, test_indices) for each fold.

        Raises:
            ValueError: If y is None or n_splits > number of samples per class.

        Example:
            >>> cv = CrossValidator()
            >>> X = [[1], [2], [3], [4], [5], [6]]
            >>> y = [0, 0, 1, 1, 0, 1]
            >>> splits = cv.stratified_k_fold_split(X, y, n_splits=3)
            >>> len(splits)
            3
        """
        X, y = self._validate_inputs(X, y)

        if y is None:
            raise ValueError("y is required for stratified k-fold")

        n_splits = n_splits if n_splits is not None else self.default_n_splits
        shuffle = shuffle if shuffle is not None else self.default_shuffle
        random_state = (
            random_state
            if random_state is not None
            else self.default_random_state
        )

        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        if n_splits > len(X):
            raise ValueError(
                f"n_splits ({n_splits}) cannot be greater than "
                f"number of samples ({len(X)})"
            )

        classes, class_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)

        if n_splits > min(np.bincount(class_indices)):
            raise ValueError(
                f"n_splits ({n_splits}) cannot be greater than "
                f"number of samples in smallest class"
            )

        class_indices_list = [
            np.where(class_indices == i)[0] for i in range(n_classes)
        ]

        if shuffle:
            rng = np.random.RandomState(random_state)
            for indices in class_indices_list:
                rng.shuffle(indices)

        splits = []

        for fold_idx in range(n_splits):
            test_indices = []
            train_indices = []

            for class_indices in class_indices_list:
                n_samples = len(class_indices)
                fold_size = n_samples // n_splits
                remainder = n_samples % n_splits

                start = fold_idx * fold_size + min(fold_idx, remainder)
                end = start + fold_size + (1 if fold_idx < remainder else 0)

                test_indices.extend(class_indices[start:end])
                train_indices.extend(
                    np.concatenate(
                        [class_indices[:start], class_indices[end:]]
                    )
                )

            test_indices = np.array(test_indices)
            train_indices = np.array(train_indices)

            if shuffle:
                rng = np.random.RandomState(random_state + fold_idx)
                rng.shuffle(test_indices)
                rng.shuffle(train_indices)

            splits.append((train_indices, test_indices))

        logger.info(f"Stratified k-fold split completed: {n_splits} folds")
        return splits

    def leave_one_out_split(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Optional[Union[List, np.ndarray, pd.Series]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Perform leave-one-out cross-validation split.

        Creates n splits where each split uses one sample as test
        and the rest as training data.

        Args:
            X: Feature data.
            y: Target data (optional, not used for splitting).

        Returns:
            List of tuples (train_indices, test_indices) for each fold.

        Example:
            >>> cv = CrossValidator()
            >>> X = [[1], [2], [3], [4]]
            >>> splits = cv.leave_one_out_split(X)
            >>> len(splits)
            4
        """
        X, _ = self._validate_inputs(X, y)

        n_samples = len(X)
        splits = []

        for i in range(n_samples):
            test_indices = np.array([i])
            train_indices = np.concatenate(
                [np.arange(i), np.arange(i + 1, n_samples)]
            )
            splits.append((train_indices, test_indices))

        logger.info(f"Leave-one-out split completed: {n_samples} folds")
        return splits

    def get_split_summary(
        self,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        y: Optional[Union[List, np.ndarray, pd.Series]] = None,
    ) -> Dict[str, Union[int, List[Dict[str, Union[int, float]]]]]:
        """Get summary statistics for cross-validation splits.

        Args:
            splits: List of (train_indices, test_indices) tuples.
            y: Target labels (optional, for class distribution).

        Returns:
            Dictionary containing summary statistics for each fold.

        Example:
            >>> cv = CrossValidator()
            >>> X = [[1], [2], [3], [4], [5]]
            >>> splits = cv.k_fold_split(X, n_splits=3)
            >>> summary = cv.get_split_summary(splits)
            >>> print(summary['n_folds'])
            3
        """
        summary = {
            "n_folds": len(splits),
            "folds": [],
        }

        for fold_idx, (train_indices, test_indices) in enumerate(splits):
            fold_info = {
                "fold": fold_idx + 1,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "train_ratio": len(train_indices)
                / (len(train_indices) + len(test_indices)),
                "test_ratio": len(test_indices)
                / (len(train_indices) + len(test_indices)),
            }

            if y is not None:
                y_array = np.asarray(y)
                train_labels = y_array[train_indices]
                test_labels = y_array[test_indices]

                train_classes, train_counts = np.unique(
                    train_labels, return_counts=True
                )
                test_classes, test_counts = np.unique(
                    test_labels, return_counts=True
                )

                fold_info["train_class_distribution"] = {
                    str(cls): int(count)
                    for cls, count in zip(train_classes, train_counts)
                }
                fold_info["test_class_distribution"] = {
                    str(cls): int(count)
                    for cls, count in zip(test_classes, test_counts)
                }

            summary["folds"].append(fold_info)

        logger.debug(f"Split summary generated for {len(splits)} folds")
        return summary

    def print_summary(
        self,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        y: Optional[Union[List, np.ndarray, pd.Series]] = None,
        strategy: str = "cross-validation",
    ) -> None:
        """Print formatted summary of cross-validation splits.

        Args:
            splits: List of (train_indices, test_indices) tuples.
            y: Target labels (optional, for class distribution).
            strategy: Name of cross-validation strategy.
        """
        summary = self.get_split_summary(splits, y)

        print("\n" + "=" * 60)
        print(f"{strategy.upper()} SUMMARY")
        print("=" * 60)

        print(f"\nNumber of Folds: {summary['n_folds']}")

        for fold_info in summary["folds"]:
            print(f"\n--- Fold {fold_info['fold']} ---")
            print(f"Train Size: {fold_info['train_size']} ({fold_info['train_ratio']:.2%})")
            print(f"Test Size:  {fold_info['test_size']} ({fold_info['test_ratio']:.2%})")

            if "train_class_distribution" in fold_info:
                print(f"\nTrain Class Distribution:")
                for cls, count in fold_info["train_class_distribution"].items():
                    print(f"  Class {cls}: {count}")
                print(f"\nTest Class Distribution:")
                for cls, count in fold_info["test_class_distribution"].items():
                    print(f"  Class {cls}: {count}")

        print("\n" + "=" * 60)

    def save_splits(
        self,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        y: Optional[Union[List, np.ndarray, pd.Series]] = None,
        output_path: str = "cv_splits.json",
    ) -> None:
        """Save cross-validation splits to JSON file.

        Args:
            splits: List of (train_indices, test_indices) tuples.
            y: Target labels (optional).
            output_path: Path to save JSON file.
        """
        summary = self.get_split_summary(splits, y)

        output_data = {
            "summary": summary,
            "splits": [
                {
                    "train_indices": train_idx.tolist(),
                    "test_indices": test_idx.tolist(),
                }
                for train_idx, test_idx in splits
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Cross-validation splits saved to: {output_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Validation Tool")
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
        default=None,
        help="Name of target column (optional)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["kfold", "stratified", "loo"],
        default="kfold",
        help="Cross-validation strategy",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Number of folds (for k-fold and stratified)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle data before splitting",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle data before splitting",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save splits as JSON",
    )

    args = parser.parse_args()

    cv = CrossValidator(config_path=args.config)

    try:
        df = pd.read_csv(args.input)
        print(f"\n=== Cross-Validation Tool ===")
        print(f"Data shape: {df.shape}")

        if args.target:
            if args.target not in df.columns:
                raise ValueError(f"Target column '{args.target}' not found")
            X = df.drop(columns=[args.target])
            y = df[args.target]
        else:
            X = df
            y = None

        print(f"Features: {X.shape[1]}")
        if y is not None:
            print(f"Target: {args.target}")
            print(f"Classes: {len(y.unique())}")

        shuffle = args.shuffle
        if args.no_shuffle:
            shuffle = False

        if args.strategy == "kfold":
            splits = cv.k_fold_split(
                X,
                y,
                n_splits=args.n_splits,
                shuffle=shuffle,
                random_state=args.random_state,
            )
            strategy_name = "K-Fold Cross-Validation"
        elif args.strategy == "stratified":
            if y is None:
                raise ValueError(
                    "Target column required for stratified k-fold"
                )
            splits = cv.stratified_k_fold_split(
                X,
                y,
                n_splits=args.n_splits,
                shuffle=shuffle,
                random_state=args.random_state,
            )
            strategy_name = "Stratified K-Fold Cross-Validation"
        elif args.strategy == "loo":
            splits = cv.leave_one_out_split(X, y)
            strategy_name = "Leave-One-Out Cross-Validation"

        cv.print_summary(splits, y, strategy=strategy_name)

        if args.save:
            cv.save_splits(splits, y, output_path=args.save)

    except Exception as e:
        logger.error(f"Error performing cross-validation: {e}")
        raise


if __name__ == "__main__":
    main()
