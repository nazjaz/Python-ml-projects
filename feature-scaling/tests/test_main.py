"""Unit tests for feature scaling implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import FeatureScaler


class TestFeatureScaler:
    """Test FeatureScaler functionality."""

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
        scaler = FeatureScaler()
        assert scaler.min_max_range == (0, 1)
        assert scaler.inplace is False

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "scaling": {"min_max_range": [-1, 1], "inplace": True},
            "logging": {"level": "WARNING", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            scaler = FeatureScaler(config_path=config_path)
            assert scaler.min_max_range == (-1, 1)
            assert scaler.inplace is True
        finally:
            Path(config_path).unlink()

    def test_load_data_from_file(self):
        """Test loading data from CSV file."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            df = scaler.load_data(file_path=csv_path)
            assert len(df) == 2
            assert len(df.columns) == 2
        finally:
            Path(csv_path).unlink()

    def test_load_data_from_dataframe(self):
        """Test loading data from pandas DataFrame."""
        df = pd.DataFrame({"age": [25, 30], "score": [85.5, 92.0]})
        scaler = FeatureScaler()
        result = scaler.load_data(dataframe=df)
        assert len(result) == 2

    def test_load_data_no_source(self):
        """Test that loading without source raises error."""
        scaler = FeatureScaler()
        with pytest.raises(ValueError, match="must be provided"):
            scaler.load_data()

    def test_get_numeric_columns(self):
        """Test getting numerical columns."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            numeric_cols = scaler.get_numeric_columns()
            assert "age" in numeric_cols
            assert "score" in numeric_cols
            assert "name" not in numeric_cols
        finally:
            Path(csv_path).unlink()

    def test_get_numeric_columns_no_data(self):
        """Test that getting columns without data raises error."""
        scaler = FeatureScaler()
        with pytest.raises(ValueError, match="No data loaded"):
            scaler.get_numeric_columns()

    def test_min_max_scale(self):
        """Test min-max scaling."""
        csv_content = "age,score\n10,50\n20,100\n30,150"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            result = scaler.min_max_scale()

            assert result["age"].min() >= 0
            assert result["age"].max() <= 1
            assert result["score"].min() >= 0
            assert result["score"].max() <= 1
        finally:
            Path(csv_path).unlink()

    def test_min_max_scale_custom_range(self):
        """Test min-max scaling with custom range."""
        csv_content = "age,score\n10,50\n20,100\n30,150"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            result = scaler.min_max_scale(feature_range=(-1, 1))

            assert result["age"].min() >= -1
            assert result["age"].max() <= 1
        finally:
            Path(csv_path).unlink()

    def test_min_max_scale_specific_columns(self):
        """Test min-max scaling for specific columns."""
        csv_content = "age,score,height\n10,50,170\n20,100,175\n30,150,180"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            result = scaler.min_max_scale(columns=["age"])

            assert result["age"].min() >= 0
            assert result["age"].max() <= 1
            assert "age" in scaler.get_scaling_summary()
        finally:
            Path(csv_path).unlink()

    def test_min_max_scale_zero_range(self):
        """Test min-max scaling with zero range column."""
        csv_content = "age,score\n10,50\n10,100\n10,150"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            result = scaler.min_max_scale()

            assert (result["age"] == 0).all()
        finally:
            Path(csv_path).unlink()

    def test_z_score_normalize(self):
        """Test z-score normalization."""
        csv_content = "age,score\n25,85.5\n30,92.0\n35,98.5"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            result = scaler.z_score_normalize()

            assert abs(result["age"].mean()) < 0.01
            assert abs(result["age"].std() - 1.0) < 0.01
        finally:
            Path(csv_path).unlink()

    def test_z_score_normalize_specific_columns(self):
        """Test z-score normalization for specific columns."""
        csv_content = "age,score,height\n25,85.5,170\n30,92.0,175\n35,98.5,180"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            result = scaler.z_score_normalize(columns=["age"])

            assert abs(result["age"].mean()) < 0.01
            assert abs(result["age"].std() - 1.0) < 0.01
        finally:
            Path(csv_path).unlink()

    def test_z_score_normalize_zero_variance(self):
        """Test z-score normalization with zero variance column."""
        csv_content = "age,score\n10,50\n10,100\n10,150"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            result = scaler.z_score_normalize()

            assert (result["age"] == 0).all()
        finally:
            Path(csv_path).unlink()

    def test_inverse_transform_min_max(self):
        """Test inverse transform for min-max scaling."""
        csv_content = "age,score\n10,50\n20,100\n30,150"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            original_data = scaler.data.copy()
            scaled_data = scaler.min_max_scale()

            inverse_data = scaler.inverse_transform(scaled_data)

            pd.testing.assert_frame_equal(
                original_data[["age", "score"]],
                inverse_data[["age", "score"]],
                check_dtype=False,
            )
        finally:
            Path(csv_path).unlink()

    def test_inverse_transform_z_score(self):
        """Test inverse transform for z-score normalization."""
        csv_content = "age,score\n25,85.5\n30,92.0\n35,98.5"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            original_data = scaler.data.copy()
            normalized_data = scaler.z_score_normalize()

            inverse_data = scaler.inverse_transform(normalized_data)

            pd.testing.assert_frame_equal(
                original_data[["age", "score"]],
                inverse_data[["age", "score"]],
                check_dtype=False,
                atol=1e-5,
            )
        finally:
            Path(csv_path).unlink()

    def test_inverse_transform_no_params(self):
        """Test that inverse transform without params raises error."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="No scaling parameters"):
                scaler.inverse_transform()
        finally:
            Path(csv_path).unlink()

    def test_get_scaling_summary(self):
        """Test getting scaling summary."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            scaler.min_max_scale()

            summary = scaler.get_scaling_summary()
            assert len(summary) > 0
            assert "age" in summary
            assert summary["age"]["method"] == "min_max"
        finally:
            Path(csv_path).unlink()

    def test_save_scaled_data(self):
        """Test saving scaled data."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            scaler.min_max_scale()
            scaler.save_scaled_data(output_path)

            assert Path(output_path).exists()
            loaded = pd.read_csv(output_path)
            assert len(loaded) == 2
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_min_max_scale_invalid_range(self):
        """Test min-max scaling with invalid range."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="min must be less than max"):
                scaler.min_max_scale(feature_range=(1, 0))
        finally:
            Path(csv_path).unlink()

    def test_min_max_scale_invalid_column(self):
        """Test min-max scaling with invalid column name."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            scaler = FeatureScaler()
            scaler.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="not found"):
                scaler.min_max_scale(columns=["invalid_column"])
        finally:
            Path(csv_path).unlink()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            FeatureScaler(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
