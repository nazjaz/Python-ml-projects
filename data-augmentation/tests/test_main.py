"""Unit tests for data augmentation implementation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import DataAugmenter


class TestDataAugmenter:
    """Test DataAugmenter functionality."""

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
        augmenter = DataAugmenter()
        assert augmenter.noise_type == "gaussian"
        assert augmenter.noise_std == 0.1

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "augmentation": {
                "noise_type": "uniform",
                "noise_std": 0.2,
                "noise_mean": 0.1,
                "scaling_type": "additive",
                "scaling_factor_min": 0.8,
                "scaling_factor_max": 1.2,
                "random_state": 123,
            },
            "logging": {"level": "WARNING", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            augmenter = DataAugmenter(config_path=config_path)
            assert augmenter.noise_type == "uniform"
            assert augmenter.noise_std == 0.2
        finally:
            Path(config_path).unlink()

    def test_load_data_from_file(self):
        """Test loading data from CSV file."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            df = augmenter.load_data(file_path=csv_path)
            assert len(df) == 2
            assert len(df.columns) == 2
        finally:
            Path(csv_path).unlink()

    def test_load_data_from_dataframe(self):
        """Test loading data from pandas DataFrame."""
        df = pd.DataFrame({"age": [25, 30], "score": [85.5, 92.0]})
        augmenter = DataAugmenter()
        result = augmenter.load_data(dataframe=df)
        assert len(result) == 2

    def test_load_data_no_source(self):
        """Test that loading without source raises error."""
        augmenter = DataAugmenter()
        with pytest.raises(ValueError, match="must be provided"):
            augmenter.load_data()

    def test_get_numeric_columns(self):
        """Test getting numerical columns."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            numeric_cols = augmenter.get_numeric_columns()
            assert "age" in numeric_cols
            assert "score" in numeric_cols
            assert "name" not in numeric_cols
        finally:
            Path(csv_path).unlink()

    def test_inject_noise_gaussian(self):
        """Test Gaussian noise injection."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            augmented = augmenter.inject_noise(noise_type="gaussian", noise_std=0.1)

            assert len(augmented) == 2
            assert not augmented.equals(augmenter.data)
        finally:
            Path(csv_path).unlink()

    def test_inject_noise_uniform(self):
        """Test uniform noise injection."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            augmented = augmenter.inject_noise(noise_type="uniform", noise_std=0.1)

            assert len(augmented) == 2
        finally:
            Path(csv_path).unlink()

    def test_inject_noise_laplace(self):
        """Test Laplace noise injection."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            augmented = augmenter.inject_noise(noise_type="laplace", noise_std=0.1)

            assert len(augmented) == 2
        finally:
            Path(csv_path).unlink()

    def test_inject_noise_invalid_type(self):
        """Test that invalid noise type raises error."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="Invalid noise type"):
                augmenter.inject_noise(noise_type="invalid")
        finally:
            Path(csv_path).unlink()

    def test_apply_scaling_variations_multiplicative(self):
        """Test multiplicative scaling variations."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            augmented = augmenter.apply_scaling_variations(
                scaling_type="multiplicative", scaling_factor_min=0.9, scaling_factor_max=1.1
            )

            assert len(augmented) == 2
        finally:
            Path(csv_path).unlink()

    def test_apply_scaling_variations_additive(self):
        """Test additive scaling variations."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            augmented = augmenter.apply_scaling_variations(
                scaling_type="additive", scaling_factor_min=-1, scaling_factor_max=1
            )

            assert len(augmented) == 2
        finally:
            Path(csv_path).unlink()

    def test_apply_scaling_variations_percentage(self):
        """Test percentage scaling variations."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            augmented = augmenter.apply_scaling_variations(
                scaling_type="percentage", scaling_factor_min=-0.1, scaling_factor_max=0.1
            )

            assert len(augmented) == 2
        finally:
            Path(csv_path).unlink()

    def test_apply_scaling_variations_invalid_type(self):
        """Test that invalid scaling type raises error."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="Invalid scaling type"):
                augmenter.apply_scaling_variations(scaling_type="invalid")
        finally:
            Path(csv_path).unlink()

    def test_apply_scaling_variations_invalid_range(self):
        """Test that invalid scaling range raises error."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="must be less than"):
                augmenter.apply_scaling_variations(
                    scaling_factor_min=1.1, scaling_factor_max=0.9
                )
        finally:
            Path(csv_path).unlink()

    def test_augment_all(self):
        """Test applying all augmentation techniques."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            augmented = augmenter.augment_all()

            assert len(augmented) == 2
            assert len(augmenter.augmentation_history) == 2
        finally:
            Path(csv_path).unlink()

    def test_get_augmentation_summary(self):
        """Test getting augmentation summary."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            augmenter.inject_noise()
            summary = augmenter.get_augmentation_summary()

            assert "total_operations" in summary
            assert summary["total_operations"] == 1
        finally:
            Path(csv_path).unlink()

    def test_save_augmented_data(self):
        """Test saving augmented data."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            augmenter.inject_noise()
            augmenter.save_augmented_data(output_path)

            assert Path(output_path).exists()
            loaded = pd.read_csv(output_path)
            assert len(loaded) == 2
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_inject_noise_specific_columns(self):
        """Test noise injection for specific columns."""
        csv_content = "age,score,height\n25,85.5,170\n30,92.0,175"
        csv_path = self.create_temp_csv(csv_content)

        try:
            augmenter = DataAugmenter()
            augmenter.load_data(file_path=csv_path)
            augmented = augmenter.inject_noise(columns=["age"])

            assert len(augmented) == 2
        finally:
            Path(csv_path).unlink()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            DataAugmenter(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
