"""Unit tests for time series preprocessing implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import TimeSeriesPreprocessor


class TestTimeSeriesPreprocessor:
    """Test TimeSeriesPreprocessor functionality."""

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
        preprocessor = TimeSeriesPreprocessor()
        assert preprocessor.default_resample_freq == "D"

    def test_load_data_from_file(self):
        """Test loading data from CSV file."""
        csv_content = "date,value\n2023-01-01,10\n2023-01-02,20\n2023-01-03,30"
        csv_path = self.create_temp_csv(csv_content)

        try:
            preprocessor = TimeSeriesPreprocessor()
            df = preprocessor.load_data(file_path=csv_path)
            assert len(df) == 3
        finally:
            Path(csv_path).unlink()

    def test_load_data_from_dataframe(self):
        """Test loading data from pandas DataFrame."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5),
                "value": [10, 20, 30, 40, 50],
            }
        )
        preprocessor = TimeSeriesPreprocessor()
        result = preprocessor.load_data(dataframe=df, datetime_column="date")
        assert len(result) == 5

    def test_load_data_no_source(self):
        """Test that loading without source raises error."""
        preprocessor = TimeSeriesPreprocessor()
        with pytest.raises(ValueError, match="must be provided"):
            preprocessor.load_data()

    def test_resample(self):
        """Test resampling time series data."""
        dates = pd.date_range("2023-01-01", periods=10, freq="H")
        df = pd.DataFrame({"value": range(10)}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        resampled = preprocessor.resample(frequency="D", method="mean")
        assert len(resampled) <= len(df)

    def test_resample_invalid_method(self):
        """Test that invalid resample method raises error."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"value": range(5)}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        with pytest.raises(ValueError, match="Invalid method"):
            preprocessor.resample(frequency="D", method="invalid")

    def test_resample_no_datetime_index(self):
        """Test that resampling without datetime index raises error."""
        df = pd.DataFrame({"value": range(5)})
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        with pytest.raises(ValueError, match="datetime index"):
            preprocessor.resample(frequency="D")

    def test_interpolate_missing_linear(self):
        """Test linear interpolation of missing values."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"value": [1, None, 3, None, 5]}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        interpolated = preprocessor.interpolate_missing(method="linear")
        assert interpolated["value"].isna().sum() == 0

    def test_interpolate_missing_polynomial(self):
        """Test polynomial interpolation of missing values."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"value": [1, None, 3, None, 5]}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        interpolated = preprocessor.interpolate_missing(method="polynomial")
        assert interpolated["value"].isna().sum() == 0

    def test_interpolate_missing_time(self):
        """Test time-based interpolation of missing values."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"value": [1, None, 3, None, 5]}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        interpolated = preprocessor.interpolate_missing(method="time")
        assert interpolated["value"].isna().sum() == 0

    def test_interpolate_missing_invalid_method(self):
        """Test that invalid interpolation method raises error."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        with pytest.raises(ValueError, match="Invalid method"):
            preprocessor.interpolate_missing(method="invalid")

    def test_remove_trend_linear(self):
        """Test linear trend removal."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        trend = np.arange(10) * 2
        noise = np.random.normal(0, 0.1, 10)
        df = pd.DataFrame({"value": trend + noise}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        detrended = preprocessor.remove_trend(method="linear")
        assert len(detrended) == len(df)

    def test_remove_trend_polynomial(self):
        """Test polynomial trend removal."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        trend = np.arange(10) ** 2
        df = pd.DataFrame({"value": trend}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        detrended = preprocessor.remove_trend(method="polynomial", order=2)
        assert len(detrended) == len(df)

    def test_remove_trend_moving_average(self):
        """Test moving average trend removal."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        trend = np.arange(20) * 0.5
        noise = np.random.normal(0, 0.1, 20)
        df = pd.DataFrame({"value": trend + noise}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        detrended = preprocessor.remove_trend(method="moving_average")
        assert len(detrended) == len(df)

    def test_remove_trend_invalid_method(self):
        """Test that invalid trend removal method raises error."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        with pytest.raises(ValueError, match="Invalid method"):
            preprocessor.remove_trend(method="invalid")

    def test_get_time_series_info(self):
        """Test getting time series information."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({"value": range(10)}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)

        info = preprocessor.get_time_series_info()
        assert "shape" in info
        assert "has_datetime_index" in info
        assert info["has_datetime_index"] is True

    def test_get_time_series_info_no_data(self):
        """Test that getting info without data raises error."""
        preprocessor = TimeSeriesPreprocessor()
        with pytest.raises(ValueError, match="No data loaded"):
            preprocessor.get_time_series_info()

    def test_save_preprocessed_data(self):
        """Test saving preprocessed data."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"value": range(5)}, index=dates)
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_data(dataframe=df)
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name

        try:
            preprocessor.save_preprocessed_data(output_path)
            assert Path(output_path).exists()
            loaded = pd.read_csv(output_path)
            assert len(loaded) == 5
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_save_preprocessed_data_no_data(self):
        """Test that saving without data raises error."""
        preprocessor = TimeSeriesPreprocessor()
        output_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name

        try:
            with pytest.raises(ValueError, match="No data loaded"):
                preprocessor.save_preprocessed_data(output_path)
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            TimeSeriesPreprocessor(config_path="nonexistent.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
