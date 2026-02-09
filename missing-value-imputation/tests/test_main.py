"""Unit tests for missing value imputation implementation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import MissingValueImputer


class TestMissingValueImputer:
    """Test MissingValueImputer functionality."""

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
        imputer = MissingValueImputer()
        assert imputer.default_numeric_strategy == "mean"
        assert imputer.default_categorical_strategy == "mode"

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "imputation": {
                "default_numeric_strategy": "median",
                "default_categorical_strategy": "mode",
                "inplace": True,
            },
            "logging": {"level": "WARNING", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            imputer = MissingValueImputer(config_path=config_path)
            assert imputer.default_numeric_strategy == "median"
            assert imputer.inplace is True
        finally:
            Path(config_path).unlink()

    def test_load_data_from_file(self):
        """Test loading data from CSV file."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            df = imputer.load_data(file_path=csv_path)
            assert len(df) == 2
            assert len(df.columns) == 3
        finally:
            Path(csv_path).unlink()

    def test_load_data_from_dataframe(self):
        """Test loading data from pandas DataFrame."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        imputer = MissingValueImputer()
        result = imputer.load_data(dataframe=df)
        assert len(result) == 2

    def test_load_data_no_source(self):
        """Test that loading without source raises error."""
        imputer = MissingValueImputer()
        with pytest.raises(ValueError, match="must be provided"):
            imputer.load_data()

    def test_analyze_missing_values(self):
        """Test missing value analysis."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,,92.0\nCharlie,28,"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            analysis = imputer.analyze_missing_values()

            assert analysis["total_missing"] > 0
            assert "columns_with_missing" in analysis
            assert "columns_without_missing" in analysis
        finally:
            Path(csv_path).unlink()

    def test_analyze_missing_values_no_data(self):
        """Test that analysis without data raises error."""
        imputer = MissingValueImputer()
        with pytest.raises(ValueError, match="No data loaded"):
            imputer.analyze_missing_values()

    def test_impute_mean(self):
        """Test mean imputation."""
        csv_content = "age,score\n25,85.5\n,92.0\n28,"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            result = imputer.impute_mean()

            assert result["age"].isnull().sum() == 0
            assert result["score"].isnull().sum() == 0
        finally:
            Path(csv_path).unlink()

    def test_impute_mean_specific_columns(self):
        """Test mean imputation for specific columns."""
        csv_content = "age,score,height\n25,85.5,170\n,92.0,175\n28,,180"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            result = imputer.impute_mean(columns=["age"])

            assert result["age"].isnull().sum() == 0
            assert "age" in imputer.get_imputation_summary()
        finally:
            Path(csv_path).unlink()

    def test_impute_median(self):
        """Test median imputation."""
        csv_content = "age,score\n25,85.5\n,92.0\n28,"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            result = imputer.impute_median()

            assert result["age"].isnull().sum() == 0
            assert result["score"].isnull().sum() == 0
        finally:
            Path(csv_path).unlink()

    def test_impute_mode(self):
        """Test mode imputation."""
        csv_content = "name,category\nAlice,TypeA\n,TypeB\nBob,TypeA"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            result = imputer.impute_mode()

            assert result["name"].isnull().sum() == 0
        finally:
            Path(csv_path).unlink()

    def test_impute_all(self):
        """Test automatic imputation for all columns."""
        csv_content = "name,age,score\nAlice,25,85.5\n,30,\nBob,,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            result = imputer.impute_all()

            assert result.isnull().sum().sum() == 0
        finally:
            Path(csv_path).unlink()

    def test_impute_all_custom_strategies(self):
        """Test automatic imputation with custom strategies."""
        csv_content = "name,age,score\nAlice,25,85.5\n,30,\nBob,,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            result = imputer.impute_all(
                numeric_strategy="median", categorical_strategy="mode"
            )

            assert result.isnull().sum().sum() == 0
        finally:
            Path(csv_path).unlink()

    def test_impute_inplace(self):
        """Test inplace imputation."""
        csv_content = "age,score\n25,85.5\n,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            original_id = id(imputer.data)
            result = imputer.impute_mean(inplace=True)

            assert id(result) == original_id
        finally:
            Path(csv_path).unlink()

    def test_get_imputation_summary(self):
        """Test getting imputation summary."""
        csv_content = "age,score\n25,85.5\n,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            imputer.impute_mean()

            summary = imputer.get_imputation_summary()
            assert len(summary) > 0
        finally:
            Path(csv_path).unlink()

    def test_save_cleaned_data(self):
        """Test saving cleaned data."""
        csv_content = "age,score\n25,85.5\n,92.0"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            imputer.impute_mean()
            imputer.save_cleaned_data(output_path)

            assert Path(output_path).exists()
            loaded = pd.read_csv(output_path)
            assert loaded.isnull().sum().sum() == 0
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_impute_mean_no_missing(self):
        """Test mean imputation with no missing values."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            result = imputer.impute_mean()

            assert result.isnull().sum().sum() == 0
        finally:
            Path(csv_path).unlink()

    def test_impute_invalid_column(self):
        """Test imputation with invalid column name."""
        csv_content = "age,score\n25,85.5\n,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="not found"):
                imputer.impute_mean(columns=["invalid_column"])
        finally:
            Path(csv_path).unlink()

    def test_impute_all_invalid_strategy(self):
        """Test automatic imputation with invalid strategy."""
        csv_content = "age,score\n25,85.5\n,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            imputer = MissingValueImputer()
            imputer.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="Invalid"):
                imputer.impute_all(numeric_strategy="invalid")
        finally:
            Path(csv_path).unlink()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            MissingValueImputer(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
