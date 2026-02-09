"""Unit tests for dataset splitter implementation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import DatasetSplitter


class TestDatasetSplitter:
    """Test DatasetSplitter functionality."""

    def create_temp_csv(self, content: str) -> str:
        """Create temporary CSV file for testing.

        Args:
            content: CSV content as string.

        Returns:
            Path to temporary CSV file.
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        temp_file.write(content)
        temp_file.close()
        return temp_file.name

    def create_temp_config(self, config_dict: dict) -> str:
        """Create temporary config file for testing.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Path to temporary config file.
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name

    def test_initialization_with_default_config(self):
        """Test initialization with default config file."""
        splitter = DatasetSplitter()
        assert splitter.train_ratio == 0.7
        assert splitter.val_ratio == 0.15
        assert splitter.test_ratio == 0.15

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "split": {
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "random_state": 123,
                "shuffle": False,
                "stratify": False,
            },
            "logging": {"level": "WARNING", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            splitter = DatasetSplitter(config_path=config_path)
            assert splitter.train_ratio == 0.8
            assert splitter.val_ratio == 0.1
            assert splitter.random_state == 123
        finally:
            Path(config_path).unlink()

    def test_load_data_from_file(self):
        """Test loading data from CSV file."""
        csv_content = "feature1,feature2,label\n1,2,A\n3,4,B\n5,6,A"
        csv_path = self.create_temp_csv(csv_content)

        try:
            splitter = DatasetSplitter()
            X, y = splitter.load_data(file_path=csv_path, target_column="label")
            assert len(X) == 3
            assert len(X.columns) == 2
            assert y is not None
            assert len(y) == 3
        finally:
            Path(csv_path).unlink()

    def test_load_data_from_dataframe(self):
        """Test loading data from pandas DataFrame."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "label": ["A", "B", "A"]}
        )
        splitter = DatasetSplitter()
        X, y = splitter.load_data(dataframe=df, target_column="label")
        assert len(X) == 3

    def test_load_data_no_source(self):
        """Test that loading without source raises error."""
        splitter = DatasetSplitter()
        with pytest.raises(ValueError, match="must be provided"):
            splitter.load_data()

    def test_load_data_no_target(self):
        """Test loading data without target column."""
        csv_content = "feature1,feature2\n1,2\n3,4\n5,6"
        csv_path = self.create_temp_csv(csv_content)

        try:
            splitter = DatasetSplitter()
            X, y = splitter.load_data(file_path=csv_path)
            assert len(X) == 3
            assert y is None
        finally:
            Path(csv_path).unlink()

    def test_load_data_invalid_target(self):
        """Test loading with invalid target column."""
        csv_content = "feature1,feature2\n1,2\n3,4"
        csv_path = self.create_temp_csv(csv_content)

        try:
            splitter = DatasetSplitter()
            with pytest.raises(ValueError, match="not found"):
                splitter.load_data(file_path=csv_path, target_column="invalid")
        finally:
            Path(csv_path).unlink()

    def test_split_basic(self):
        """Test basic dataset splitting."""
        csv_content = "feature1,feature2\n1,2\n3,4\n5,6\n7,8\n9,10"
        csv_path = self.create_temp_csv(csv_content)

        try:
            splitter = DatasetSplitter()
            splitter.load_data(file_path=csv_path)
            splits = splitter.split()

            assert "train" in splits
            assert "val" in splits
            assert "test" in splits
            assert splits["train"][0] is not None
            assert splits["test"][0] is not None
        finally:
            Path(csv_path).unlink()

    def test_split_with_target(self):
        """Test splitting with target column."""
        csv_content = "feature1,feature2,label\n1,2,A\n3,4,B\n5,6,A\n7,8,B\n9,10,A"
        csv_path = self.create_temp_csv(csv_content)

        try:
            splitter = DatasetSplitter()
            splitter.load_data(file_path=csv_path, target_column="label")
            splits = splitter.split()

            X_train, y_train = splits["train"]
            assert X_train is not None
            assert y_train is not None
            assert len(X_train) == len(y_train)
        finally:
            Path(csv_path).unlink()

    def test_split_custom_ratios(self):
        """Test splitting with custom ratios."""
        csv_content = "feature1,feature2\n1,2\n3,4\n5,6\n7,8\n9,10"
        csv_path = self.create_temp_csv(csv_content)

        try:
            splitter = DatasetSplitter()
            splitter.load_data(file_path=csv_path)
            splits = splitter.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

            assert splits["train"][0] is not None
            assert splits["test"][0] is not None
        finally:
            Path(csv_path).unlink()

    def test_split_no_validation(self):
        """Test splitting without validation set."""
        csv_content = "feature1,feature2\n1,2\n3,4\n5,6\n7,8\n9,10"
        csv_path = self.create_temp_csv(csv_content)

        try:
            splitter = DatasetSplitter()
            splitter.load_data(file_path=csv_path)
            splits = splitter.split(train_ratio=0.8, val_ratio=0.0, test_ratio=0.2)

            assert splits["train"][0] is not None
            assert splits["val"][0] is None
            assert splits["test"][0] is not None
        finally:
            Path(csv_path).unlink()

    def test_split_stratification(self):
        """Test splitting with stratification."""
        csv_content = "feature1,feature2,label\n1,2,A\n3,4,B\n5,6,A\n7,8,B\n9,10,A\n11,12,B"
        csv_path = self.create_temp_csv(csv_content)

        try:
            splitter = DatasetSplitter()
            splitter.load_data(file_path=csv_path, target_column="label")
            splits = splitter.split(stratify=True)

            X_train, y_train = splits["train"]
            X_test, y_test = splits["test"]

            train_dist = y_train.value_counts()
            test_dist = y_test.value_counts()

            assert len(train_dist) > 0
            assert len(test_dist) > 0
        finally:
            Path(csv_path).unlink()

    def test_get_split_summary(self):
        """Test getting split summary."""
        csv_content = "feature1,feature2,label\n1,2,A\n3,4,B\n5,6,A\n7,8,B"
        csv_path = self.create_temp_csv(csv_content)

        try:
            splitter = DatasetSplitter()
            splitter.load_data(file_path=csv_path, target_column="label")
            splits = splitter.split()
            summary = splitter.get_split_summary(splits)

            assert "train" in summary
            assert "test" in summary
            assert "samples" in summary["train"]
        finally:
            Path(csv_path).unlink()

    def test_save_splits(self):
        """Test saving splits to files."""
        csv_content = "feature1,feature2,label\n1,2,A\n3,4,B\n5,6,A\n7,8,B"
        csv_path = self.create_temp_csv(csv_content)
        output_dir = tempfile.mkdtemp()

        try:
            splitter = DatasetSplitter()
            splitter.load_data(file_path=csv_path, target_column="label")
            splits = splitter.split()
            splitter.save_splits(splits, output_dir=output_dir)

            assert Path(f"{output_dir}/X_train.csv").exists()
            assert Path(f"{output_dir}/X_test.csv").exists()
            assert Path(f"{output_dir}/y_train.csv").exists()
            assert Path(f"{output_dir}/y_test.csv").exists()
        finally:
            Path(csv_path).unlink()

    def test_split_no_data(self):
        """Test that splitting without data raises error."""
        splitter = DatasetSplitter()
        with pytest.raises(ValueError, match="No data loaded"):
            splitter.split()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            DatasetSplitter(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
