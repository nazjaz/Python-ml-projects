"""Unit tests for feature selection implementation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import FeatureSelector


class TestFeatureSelector:
    """Test FeatureSelector functionality."""

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
        selector = FeatureSelector()
        assert selector.variance_threshold == 0.0
        assert selector.correlation_threshold == 0.95

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "feature_selection": {
                "variance_threshold": 0.1,
                "correlation_threshold": 0.9,
                "univariate_k": 5,
                "univariate_score_func": "f_regression",
            },
            "logging": {"level": "WARNING", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            selector = FeatureSelector(config_path=config_path)
            assert selector.variance_threshold == 0.1
            assert selector.correlation_threshold == 0.9
        finally:
            Path(config_path).unlink()

    def test_load_data_from_file(self):
        """Test loading data from CSV file."""
        csv_content = "feature1,feature2,label\n1,2,A\n3,4,B"
        csv_path = self.create_temp_csv(csv_content)

        try:
            selector = FeatureSelector()
            X, y = selector.load_data(file_path=csv_path, target_column="label")
            assert len(X) == 2
            assert y is not None
        finally:
            Path(csv_path).unlink()

    def test_load_data_from_dataframe(self):
        """Test loading data from pandas DataFrame."""
        df = pd.DataFrame(
            {"feature1": [1, 2], "feature2": [3, 4], "label": ["A", "B"]}
        )
        selector = FeatureSelector()
        X, y = selector.load_data(dataframe=df, target_column="label")
        assert len(X) == 2

    def test_load_data_no_source(self):
        """Test that loading without source raises error."""
        selector = FeatureSelector()
        with pytest.raises(ValueError, match="must be provided"):
            selector.load_data()

    def test_get_numeric_columns(self):
        """Test getting numerical columns."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            selector = FeatureSelector()
            selector.load_data(file_path=csv_path)
            numeric_cols = selector.get_numeric_columns()
            assert "age" in numeric_cols
            assert "score" in numeric_cols
            assert "name" not in numeric_cols
        finally:
            Path(csv_path).unlink()

    def test_get_numeric_columns_no_data(self):
        """Test that getting columns without data raises error."""
        selector = FeatureSelector()
        with pytest.raises(ValueError, match="No data loaded"):
            selector.get_numeric_columns()

    def test_select_variance_threshold(self):
        """Test variance threshold selection."""
        csv_content = "feature1,feature2,feature3\n1,2,0\n3,4,0\n5,6,0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            selector = FeatureSelector()
            selector.load_data(file_path=csv_path)
            selected = selector.select_variance_threshold(threshold=0.1)

            assert isinstance(selected, list)
            assert "feature3" not in selected
        finally:
            Path(csv_path).unlink()

    def test_select_correlation(self):
        """Test correlation-based selection."""
        csv_content = "feature1,feature2,feature3\n1,1,5\n2,2,6\n3,3,7"
        csv_path = self.create_temp_csv(csv_content)

        try:
            selector = FeatureSelector()
            selector.load_data(file_path=csv_path)
            selected = selector.select_correlation(threshold=0.9)

            assert isinstance(selected, list)
        finally:
            Path(csv_path).unlink()

    def test_select_univariate(self):
        """Test univariate statistical selection."""
        csv_content = "feature1,feature2,label\n1,2,A\n3,4,B\n5,6,A\n7,8,B"
        csv_path = self.create_temp_csv(csv_content)

        try:
            selector = FeatureSelector()
            selector.load_data(file_path=csv_path, target_column="label")
            selected = selector.select_univariate(k=1)

            assert isinstance(selected, list)
            assert len(selected) <= 1
        finally:
            Path(csv_path).unlink()

    def test_select_univariate_no_target(self):
        """Test that univariate selection without target raises error."""
        csv_content = "feature1,feature2\n1,2\n3,4"
        csv_path = self.create_temp_csv(csv_content)

        try:
            selector = FeatureSelector()
            selector.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="Target column required"):
                selector.select_univariate()
        finally:
            Path(csv_path).unlink()

    def test_select_univariate_invalid_score_func(self):
        """Test univariate selection with invalid score function."""
        csv_content = "feature1,feature2,label\n1,2,A\n3,4,B"
        csv_path = self.create_temp_csv(csv_content)

        try:
            selector = FeatureSelector()
            selector.load_data(file_path=csv_path, target_column="label")
            with pytest.raises(ValueError, match="Invalid score function"):
                selector.select_univariate(score_func="invalid")
        finally:
            Path(csv_path).unlink()

    def test_select_all(self):
        """Test applying all selection methods."""
        csv_content = "feature1,feature2,feature3,label\n1,2,3,A\n4,5,6,B\n7,8,9,A"
        csv_path = self.create_temp_csv(csv_content)

        try:
            selector = FeatureSelector()
            selector.load_data(file_path=csv_path, target_column="label")
            selected = selector.select_all()

            assert isinstance(selected, list)
        finally:
            Path(csv_path).unlink()

    def test_get_selection_summary(self):
        """Test getting selection summary."""
        csv_content = "feature1,feature2,label\n1,2,A\n3,4,B"
        csv_path = self.create_temp_csv(csv_content)

        try:
            selector = FeatureSelector()
            selector.load_data(file_path=csv_path)
            selector.select_variance_threshold()
            summary = selector.get_selection_summary()

            assert "original_features" in summary
            assert "selected_features" in summary
        finally:
            Path(csv_path).unlink()

    def test_apply_selection(self):
        """Test applying selection to data."""
        csv_content = "feature1,feature2,label\n1,2,A\n3,4,B"
        csv_path = self.create_temp_csv(csv_content)

        try:
            selector = FeatureSelector()
            selector.load_data(file_path=csv_path)
            selector.select_variance_threshold()
            selected_data = selector.apply_selection()

            assert len(selected_data.columns) <= len(selector.data.columns)
        finally:
            Path(csv_path).unlink()

    def test_apply_selection_no_selection(self):
        """Test that applying without selection raises error."""
        csv_content = "feature1,feature2\n1,2\n3,4"
        csv_path = self.create_temp_csv(csv_content)

        try:
            selector = FeatureSelector()
            selector.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="No features selected"):
                selector.apply_selection()
        finally:
            Path(csv_path).unlink()

    def test_save_selected_data(self):
        """Test saving selected features."""
        csv_content = "feature1,feature2,label\n1,2,A\n3,4,B"
        csv_path = self.create_temp_csv(csv_content)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name

        try:
            selector = FeatureSelector()
            selector.load_data(file_path=csv_path)
            selector.select_variance_threshold()
            selector.save_selected_data(output_path)

            assert Path(output_path).exists()
        finally:
            Path(csv_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            FeatureSelector(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
