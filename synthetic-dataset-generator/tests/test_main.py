"""Unit tests for synthetic dataset generator implementation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import SyntheticDatasetGenerator


class TestSyntheticDatasetGenerator:
    """Test SyntheticDatasetGenerator functionality."""

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
        generator = SyntheticDatasetGenerator()
        assert generator.n_samples == 1000
        assert generator.n_features == 10

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "generation": {
                "n_samples": 500,
                "n_features": 5,
                "random_state": 123,
                "noise": 0.2,
            },
            "logging": {"level": "WARNING", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            generator = SyntheticDatasetGenerator(config_path=config_path)
            assert generator.n_samples == 500
            assert generator.n_features == 5
        finally:
            Path(config_path).unlink()

    def test_generate_classification(self):
        """Test classification dataset generation."""
        generator = SyntheticDatasetGenerator()
        data, target = generator.generate_classification(
            n_samples=100, n_features=5, n_classes=2
        )

        assert isinstance(data, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert len(data) == 100
        assert len(data.columns) == 5
        assert len(target) == 100
        assert target.nunique() == 2

    def test_generate_classification_custom_params(self):
        """Test classification generation with custom parameters."""
        generator = SyntheticDatasetGenerator()
        data, target = generator.generate_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=5,
            class_sep=2.0,
        )

        assert len(data) == 200
        assert target.nunique() == 3

    def test_generate_regression(self):
        """Test regression dataset generation."""
        generator = SyntheticDatasetGenerator()
        data, target = generator.generate_regression(
            n_samples=100, n_features=5, noise=0.1
        )

        assert isinstance(data, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert len(data) == 100
        assert len(data.columns) == 5
        assert len(target) == 100
        assert pd.api.types.is_numeric_dtype(target)

    def test_generate_regression_custom_params(self):
        """Test regression generation with custom parameters."""
        generator = SyntheticDatasetGenerator()
        data, target = generator.generate_regression(
            n_samples=200, n_features=10, n_informative=8, noise=0.2
        )

        assert len(data) == 200
        assert len(data.columns) == 10

    def test_generate_custom_classification(self):
        """Test custom classification dataset generation."""
        generator = SyntheticDatasetGenerator()
        data, target = generator.generate_custom_classification(
            n_samples=100, n_features=5, n_classes=2
        )

        assert isinstance(data, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert len(data) == 100
        assert target.nunique() == 2

    def test_generate_custom_classification_distribution(self):
        """Test custom classification with class distribution."""
        generator = SyntheticDatasetGenerator()
        data, target = generator.generate_custom_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            class_distribution=[0.7, 0.3],
        )

        assert len(data) == 100
        class_counts = target.value_counts()
        assert class_counts.iloc[0] > class_counts.iloc[1]

    def test_generate_custom_classification_invalid_distribution(self):
        """Test custom classification with invalid distribution."""
        generator = SyntheticDatasetGenerator()
        with pytest.raises(ValueError, match="must sum to 1.0"):
            generator.generate_custom_classification(
                n_samples=100,
                n_features=5,
                n_classes=2,
                class_distribution=[0.7, 0.4],
            )

    def test_generate_custom_classification_feature_ranges(self):
        """Test custom classification with feature ranges."""
        generator = SyntheticDatasetGenerator()
        feature_ranges = [(-5, 5), (0, 10), (-10, 0), (1, 2), (100, 200)]
        data, target = generator.generate_custom_classification(
            n_samples=100, n_features=5, feature_ranges=feature_ranges
        )

        assert len(data.columns) == 5
        for i, col in enumerate(data.columns):
            assert data[col].min() >= feature_ranges[i][0]
            assert data[col].max() <= feature_ranges[i][1]

    def test_generate_custom_classification_invalid_ranges(self):
        """Test custom classification with invalid feature ranges."""
        generator = SyntheticDatasetGenerator()
        with pytest.raises(ValueError, match="must match n_features"):
            generator.generate_custom_classification(
                n_samples=100,
                n_features=5,
                feature_ranges=[(-5, 5), (0, 10)],
            )

    def test_get_dataset_info(self):
        """Test getting dataset information."""
        generator = SyntheticDatasetGenerator()
        generator.generate_classification(n_samples=100, n_classes=3)
        info = generator.get_dataset_info()

        assert "shape" in info
        assert "n_samples" in info
        assert "n_features" in info
        assert "task_type" in info
        assert info["task_type"] == "classification"
        assert info["n_classes"] == 3

    def test_get_dataset_info_no_data(self):
        """Test that getting info without data raises error."""
        generator = SyntheticDatasetGenerator()
        with pytest.raises(ValueError, match="No dataset generated"):
            generator.get_dataset_info()

    def test_save_dataset(self):
        """Test saving dataset to file."""
        generator = SyntheticDatasetGenerator()
        generator.generate_classification(n_samples=100, n_classes=2)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name

        try:
            generator.save_dataset(output_path)
            assert Path(output_path).exists()
            loaded = pd.read_csv(output_path)
            assert len(loaded) == 100
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_save_dataset_no_target(self):
        """Test saving dataset without target."""
        generator = SyntheticDatasetGenerator()
        generator.generate_classification(n_samples=100, n_classes=2)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name

        try:
            generator.save_dataset(output_path, include_target=False)
            loaded = pd.read_csv(output_path)
            assert "target" not in loaded.columns
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_save_dataset_no_data(self):
        """Test that saving without data raises error."""
        generator = SyntheticDatasetGenerator()
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name

        try:
            with pytest.raises(ValueError, match="No dataset generated"):
                generator.save_dataset(output_path)
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            SyntheticDatasetGenerator(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
