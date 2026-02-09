"""Unit tests for CSV data explorer implementation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import CSVDataExplorer


class TestCSVDataExplorer:
    """Test CSVDataExplorer functionality."""

    def create_temp_csv(self, content: str, separator: str = ",") -> str:
        """Create temporary CSV file for testing.

        Args:
            content: CSV content as string.
            separator: Column separator.

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
        explorer = CSVDataExplorer()
        assert explorer.separator == ","
        assert explorer.encoding == "utf-8"

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "csv_explorer": {
                "separator": ";",
                "encoding": "latin-1",
                "max_rows_preview": 5,
            },
            "logging": {"level": "WARNING", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            explorer = CSVDataExplorer(config_path=config_path)
            assert explorer.separator == ";"
            assert explorer.encoding == "latin-1"
            assert explorer.max_rows_preview == 5
        finally:
            Path(config_path).unlink()

    def test_load_csv_simple(self):
        """Test loading simple CSV file."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,30,92.0\nCharlie,28,78.5"
        csv_path = self.create_temp_csv(csv_content)

        try:
            explorer = CSVDataExplorer()
            df = explorer.load_csv(csv_path)
            assert len(df) == 3
            assert len(df.columns) == 3
            assert list(df.columns) == ["name", "age", "score"]
        finally:
            Path(csv_path).unlink()

    def test_load_csv_with_missing_values(self):
        """Test loading CSV with missing values."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,,92.0\nCharlie,28,"
        csv_path = self.create_temp_csv(csv_content)

        try:
            explorer = CSVDataExplorer()
            df = explorer.load_csv(csv_path)
            assert len(df) == 3
            assert df.isnull().sum().sum() > 0
        finally:
            Path(csv_path).unlink()

    def test_load_csv_file_not_found(self):
        """Test that loading non-existent file raises error."""
        explorer = CSVDataExplorer()
        with pytest.raises(FileNotFoundError):
            explorer.load_csv("nonexistent.csv")

    def test_get_basic_info(self):
        """Test getting basic dataset information."""
        csv_content = "col1,col2\n1,2\n3,4\n5,6"
        csv_path = self.create_temp_csv(csv_content)

        try:
            explorer = CSVDataExplorer()
            explorer.load_csv(csv_path)
            info = explorer.get_basic_info()

            assert info["rows"] == 3
            assert info["columns"] == 2
            assert info["shape"] == (3, 2)
            assert "col1" in info["column_names"]
            assert "col2" in info["column_names"]
        finally:
            Path(csv_path).unlink()

    def test_get_basic_info_no_data(self):
        """Test that getting info without data raises error."""
        explorer = CSVDataExplorer()
        with pytest.raises(ValueError, match="No data loaded"):
            explorer.get_basic_info()

    def test_get_data_types(self):
        """Test getting data types."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            explorer = CSVDataExplorer()
            explorer.load_csv(csv_path)
            dtypes = explorer.get_data_types()

            assert "name" in dtypes
            assert "age" in dtypes
            assert "score" in dtypes
        finally:
            Path(csv_path).unlink()

    def test_get_basic_statistics(self):
        """Test getting basic statistics."""
        csv_content = "age,score\n25,85.5\n30,92.0\n28,78.5"
        csv_path = self.create_temp_csv(csv_content)

        try:
            explorer = CSVDataExplorer()
            explorer.load_csv(csv_path)
            stats = explorer.get_basic_statistics()

            assert not stats.empty
            assert "age" in stats.columns
            assert "score" in stats.columns
        finally:
            Path(csv_path).unlink()

    def test_get_basic_statistics_no_numeric(self):
        """Test statistics with no numerical columns."""
        csv_content = "name,category\nAlice,TypeA\nBob,TypeB"
        csv_path = self.create_temp_csv(csv_content)

        try:
            explorer = CSVDataExplorer()
            explorer.load_csv(csv_path)
            stats = explorer.get_basic_statistics()

            assert stats.empty
        finally:
            Path(csv_path).unlink()

    def test_get_missing_value_analysis(self):
        """Test missing value analysis."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,,92.0\nCharlie,28,"
        csv_path = self.create_temp_csv(csv_content)

        try:
            explorer = CSVDataExplorer()
            explorer.load_csv(csv_path)
            missing = explorer.get_missing_value_analysis()

            assert missing["total_missing"] > 0
            assert "columns_with_missing" in missing
            assert "columns_without_missing" in missing
        finally:
            Path(csv_path).unlink()

    def test_get_missing_value_analysis_no_missing(self):
        """Test missing value analysis with no missing values."""
        csv_content = "name,age\nAlice,25\nBob,30"
        csv_path = self.create_temp_csv(csv_content)

        try:
            explorer = CSVDataExplorer()
            explorer.load_csv(csv_path)
            missing = explorer.get_missing_value_analysis()

            assert missing["total_missing"] == 0
            assert len(missing["columns_with_missing"]) == 0
        finally:
            Path(csv_path).unlink()

    def test_get_categorical_summary(self):
        """Test categorical summary."""
        csv_content = "name,category\nAlice,TypeA\nBob,TypeB\nAlice,TypeA"
        csv_path = self.create_temp_csv(csv_content)

        try:
            explorer = CSVDataExplorer()
            explorer.load_csv(csv_path)
            categorical = explorer.get_categorical_summary()

            assert "name" in categorical or "category" in categorical
        finally:
            Path(csv_path).unlink()

    def test_generate_report(self):
        """Test report generation."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            explorer = CSVDataExplorer()
            explorer.load_csv(csv_path)
            report = explorer.generate_report()

            assert "CSV Dataset Exploration Report" in report
            assert "BASIC INFORMATION" in report
            assert "DATA TYPES" in report
            assert "MISSING VALUE ANALYSIS" in report
        finally:
            Path(csv_path).unlink()

    def test_preview_data(self):
        """Test data preview."""
        csv_content = "col1,col2\n1,2\n3,4\n5,6\n7,8\n9,10"
        csv_path = self.create_temp_csv(csv_content)

        try:
            explorer = CSVDataExplorer()
            explorer.load_csv(csv_path)
            preview = explorer.preview_data(n_rows=3)

            assert len(preview) == 3
            assert len(preview.columns) == 2
        finally:
            Path(csv_path).unlink()

    def test_load_csv_custom_separator(self):
        """Test loading CSV with custom separator."""
        csv_content = "name;age;score\nAlice;25;85.5\nBob;30;92.0"
        csv_path = self.create_temp_csv(csv_content, separator=";")

        try:
            explorer = CSVDataExplorer()
            df = explorer.load_csv(csv_path, separator=";")
            assert len(df) == 2
            assert len(df.columns) == 3
        finally:
            Path(csv_path).unlink()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            CSVDataExplorer(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
