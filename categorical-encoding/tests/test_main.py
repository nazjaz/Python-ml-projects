"""Unit tests for categorical encoding implementation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import CategoricalEncoder


class TestCategoricalEncoder:
    """Test CategoricalEncoder functionality."""

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
        encoder = CategoricalEncoder()
        assert encoder.drop_first is False
        assert encoder.prefix is None

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "encoding": {
                "drop_first": True,
                "prefix": "cat",
                "prefix_sep": "-",
                "inplace": True,
            },
            "logging": {"level": "WARNING", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            encoder = CategoricalEncoder(config_path=config_path)
            assert encoder.drop_first is True
            assert encoder.prefix == "cat"
            assert encoder.prefix_sep == "-"
        finally:
            Path(config_path).unlink()

    def test_load_data_from_file(self):
        """Test loading data from CSV file."""
        csv_content = "name,category\nAlice,TypeA\nBob,TypeB"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            df = encoder.load_data(file_path=csv_path)
            assert len(df) == 2
            assert len(df.columns) == 2
        finally:
            Path(csv_path).unlink()

    def test_load_data_from_dataframe(self):
        """Test loading data from pandas DataFrame."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "category": ["A", "B"]})
        encoder = CategoricalEncoder()
        result = encoder.load_data(dataframe=df)
        assert len(result) == 2

    def test_load_data_no_source(self):
        """Test that loading without source raises error."""
        encoder = CategoricalEncoder()
        with pytest.raises(ValueError, match="must be provided"):
            encoder.load_data()

    def test_get_categorical_columns(self):
        """Test getting categorical columns."""
        csv_content = "name,age,category\nAlice,25,TypeA\nBob,30,TypeB"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            categorical_cols = encoder.get_categorical_columns()
            assert "name" in categorical_cols
            assert "category" in categorical_cols
            assert "age" not in categorical_cols
        finally:
            Path(csv_path).unlink()

    def test_get_categorical_columns_no_data(self):
        """Test that getting columns without data raises error."""
        encoder = CategoricalEncoder()
        with pytest.raises(ValueError, match="No data loaded"):
            encoder.get_categorical_columns()

    def test_one_hot_encode(self):
        """Test one-hot encoding."""
        csv_content = "category\nTypeA\nTypeB\nTypeA"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            result = encoder.one_hot_encode()

            assert "category" not in result.columns
            assert any("category" in col for col in result.columns)
            assert result.sum().sum() == 3
        finally:
            Path(csv_path).unlink()

    def test_one_hot_encode_drop_first(self):
        """Test one-hot encoding with drop_first."""
        csv_content = "category\nTypeA\nTypeB\nTypeC"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            result = encoder.one_hot_encode(drop_first=True)

            category_cols = [col for col in result.columns if "category" in col]
            assert len(category_cols) == 2
        finally:
            Path(csv_path).unlink()

    def test_one_hot_encode_specific_columns(self):
        """Test one-hot encoding for specific columns."""
        csv_content = "name,category\nAlice,TypeA\nBob,TypeB"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            result = encoder.one_hot_encode(columns=["category"])

            assert "category" not in result.columns
            assert "name" in result.columns
        finally:
            Path(csv_path).unlink()

    def test_label_encode(self):
        """Test label encoding."""
        csv_content = "category\nTypeA\nTypeB\nTypeA"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            result = encoder.label_encode()

            assert "category" in result.columns
            assert result["category"].dtype in ["int64", "Int64"]
            assert result["category"].min() >= 0
        finally:
            Path(csv_path).unlink()

    def test_label_encode_specific_columns(self):
        """Test label encoding for specific columns."""
        csv_content = "name,category\nAlice,TypeA\nBob,TypeB"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            result = encoder.label_encode(columns=["category"])

            assert "category" in result.columns
            assert "name" in result.columns
            assert result["category"].dtype in ["int64", "Int64"]
        finally:
            Path(csv_path).unlink()

    def test_inverse_label_encode(self):
        """Test inverse label encoding."""
        csv_content = "category\nTypeA\nTypeB\nTypeA"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            original_data = encoder.data.copy()
            encoded_data = encoder.label_encode()

            inverse_data = encoder.inverse_label_encode(encoded_data)

            pd.testing.assert_series_equal(
                original_data["category"], inverse_data["category"]
            )
        finally:
            Path(csv_path).unlink()

    def test_inverse_label_encode_no_encodings(self):
        """Test that inverse transform without encodings raises error."""
        csv_content = "category\nTypeA\nTypeB"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="No label encodings"):
                encoder.inverse_label_encode()
        finally:
            Path(csv_path).unlink()

    def test_compare_encodings(self):
        """Test encoding comparison."""
        csv_content = "category,priority\nTypeA,Low\nTypeB,High\nTypeA,Medium"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            comparison = encoder.compare_encodings()

            assert len(comparison) > 0
            for col, info in comparison.items():
                if isinstance(info, dict) and "column" in info:
                    assert "one_hot" in info
                    assert "label" in info
                    assert "recommendation" in info
        finally:
            Path(csv_path).unlink()

    def test_compare_encodings_specific_columns(self):
        """Test encoding comparison for specific columns."""
        csv_content = "category,priority\nTypeA,Low\nTypeB,High"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            comparison = encoder.compare_encodings(columns=["category"])

            assert "category" in comparison
        finally:
            Path(csv_path).unlink()

    def test_get_encoding_summary(self):
        """Test getting encoding summary."""
        csv_content = "category\nTypeA\nTypeB\nTypeA"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            encoder.one_hot_encode()

            summary = encoder.get_encoding_summary()
            assert len(summary) > 0
            assert "category" in summary
            assert summary["category"]["method"] == "one_hot"
        finally:
            Path(csv_path).unlink()

    def test_save_encoded_data(self):
        """Test saving encoded data."""
        csv_content = "category\nTypeA\nTypeB"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            encoder.label_encode()
            encoder.save_encoded_data(output_path)

            assert Path(output_path).exists()
            loaded = pd.read_csv(output_path)
            assert len(loaded) == 2
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_one_hot_encode_invalid_column(self):
        """Test one-hot encoding with invalid column name."""
        csv_content = "category\nTypeA\nTypeB"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="not found"):
                encoder.one_hot_encode(columns=["invalid_column"])
        finally:
            Path(csv_path).unlink()

    def test_label_encode_invalid_column(self):
        """Test label encoding with invalid column name."""
        csv_content = "category\nTypeA\nTypeB"
        csv_path = self.create_temp_csv(csv_content)

        try:
            encoder = CategoricalEncoder()
            encoder.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="not found"):
                encoder.label_encode(columns=["invalid_column"])
        finally:
            Path(csv_path).unlink()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            CategoricalEncoder(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
