"""Unit tests for outlier detection implementation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import OutlierDetector


class TestOutlierDetector:
    """Test OutlierDetector functionality."""

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
        detector = OutlierDetector()
        assert detector.iqr_multiplier == 1.5
        assert detector.z_score_threshold == 3.0

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "outlier_detection": {
                "iqr_multiplier": 3.0,
                "z_score_threshold": 2.0,
                "isolation_contamination": 0.2,
                "isolation_random_state": 123,
            },
            "logging": {"level": "WARNING", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            detector = OutlierDetector(config_path=config_path)
            assert detector.iqr_multiplier == 3.0
            assert detector.z_score_threshold == 2.0
        finally:
            Path(config_path).unlink()

    def test_load_data_from_file(self):
        """Test loading data from CSV file."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            df = detector.load_data(file_path=csv_path)
            assert len(df) == 2
            assert len(df.columns) == 2
        finally:
            Path(csv_path).unlink()

    def test_load_data_from_dataframe(self):
        """Test loading data from pandas DataFrame."""
        df = pd.DataFrame({"age": [25, 30], "score": [85.5, 92.0]})
        detector = OutlierDetector()
        result = detector.load_data(dataframe=df)
        assert len(result) == 2

    def test_load_data_no_source(self):
        """Test that loading without source raises error."""
        detector = OutlierDetector()
        with pytest.raises(ValueError, match="must be provided"):
            detector.load_data()

    def test_get_numeric_columns(self):
        """Test getting numerical columns."""
        csv_content = "name,age,score\nAlice,25,85.5\nBob,30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            numeric_cols = detector.get_numeric_columns()
            assert "age" in numeric_cols
            assert "score" in numeric_cols
            assert "name" not in numeric_cols
        finally:
            Path(csv_path).unlink()

    def test_get_numeric_columns_no_data(self):
        """Test that getting columns without data raises error."""
        detector = OutlierDetector()
        with pytest.raises(ValueError, match="No data loaded"):
            detector.get_numeric_columns()

    def test_detect_iqr(self):
        """Test IQR outlier detection."""
        csv_content = "age,score\n25,85.5\n30,92.0\n100,200\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            outlier_mask = detector.detect_iqr()

            assert isinstance(outlier_mask, pd.Series)
            assert outlier_mask.sum() > 0
        finally:
            Path(csv_path).unlink()

    def test_detect_iqr_specific_columns(self):
        """Test IQR detection for specific columns."""
        csv_content = "age,score,height\n25,85.5,170\n30,92.0,175\n100,200,180"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            outlier_mask = detector.detect_iqr(columns=["age"])

            assert isinstance(outlier_mask, pd.Series)
        finally:
            Path(csv_path).unlink()

    def test_detect_zscore(self):
        """Test Z-score outlier detection."""
        csv_content = "age,score\n25,85.5\n30,92.0\n100,200\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            outlier_mask = detector.detect_zscore()

            assert isinstance(outlier_mask, pd.Series)
        finally:
            Path(csv_path).unlink()

    def test_detect_zscore_custom_threshold(self):
        """Test Z-score detection with custom threshold."""
        csv_content = "age,score\n25,85.5\n30,92.0\n100,200\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            outlier_mask = detector.detect_zscore(threshold=2.0)

            assert isinstance(outlier_mask, pd.Series)
        finally:
            Path(csv_path).unlink()

    def test_detect_isolation_forest(self):
        """Test Isolation Forest outlier detection."""
        csv_content = "age,score\n25,85.5\n30,92.0\n100,200\n28,88.0\n26,87.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            outlier_mask = detector.detect_isolation_forest()

            assert isinstance(outlier_mask, pd.Series)
        finally:
            Path(csv_path).unlink()

    def test_get_outlier_summary(self):
        """Test getting outlier summary."""
        csv_content = "age,score\n25,85.5\n30,92.0\n100,200\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            detector.detect_iqr()
            summary = detector.get_outlier_summary()

            assert "total_samples" in summary
            assert "outlier_count" in summary
            assert "outlier_percentage" in summary
        finally:
            Path(csv_path).unlink()

    def test_remove_outliers(self):
        """Test removing outliers."""
        csv_content = "age,score\n25,85.5\n30,92.0\n100,200\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            detector.detect_iqr()
            cleaned_data = detector.remove_outliers()

            assert len(cleaned_data) < len(detector.data)
        finally:
            Path(csv_path).unlink()

    def test_remove_outliers_no_detection(self):
        """Test that removing without detection raises error."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="No outliers detected"):
                detector.remove_outliers()
        finally:
            Path(csv_path).unlink()

    def test_cap_outliers_iqr(self):
        """Test capping outliers using IQR method."""
        csv_content = "age,score\n25,85.5\n30,92.0\n100,200\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            capped_data = detector.cap_outliers(method="iqr")

            assert len(capped_data) == len(detector.data)
        finally:
            Path(csv_path).unlink()

    def test_cap_outliers_zscore(self):
        """Test capping outliers using Z-score method."""
        csv_content = "age,score\n25,85.5\n30,92.0\n100,200\n28,88.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            capped_data = detector.cap_outliers(method="zscore")

            assert len(capped_data) == len(detector.data)
        finally:
            Path(csv_path).unlink()

    def test_cap_outliers_invalid_method(self):
        """Test capping with invalid method raises error."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="Invalid method"):
                detector.cap_outliers(method="invalid")
        finally:
            Path(csv_path).unlink()

    def test_detect_iqr_invalid_column(self):
        """Test IQR detection with invalid column."""
        csv_content = "age,score\n25,85.5\n30,92.0"
        csv_path = self.create_temp_csv(csv_content)

        try:
            detector = OutlierDetector()
            detector.load_data(file_path=csv_path)
            with pytest.raises(ValueError, match="not found"):
                detector.detect_iqr(columns=["invalid"])
        finally:
            Path(csv_path).unlink()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            OutlierDetector(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
