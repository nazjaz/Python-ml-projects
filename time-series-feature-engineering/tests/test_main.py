"""Unit tests for Time Series Feature Engineering implementation."""

import numpy as np
import pandas as pd
import pytest

from src.main import TimeSeriesFeatureEngineering


class TestTimeSeriesFeatureEngineering:
    """Test Time Series Feature Engineering functionality."""

    def test_initialization(self):
        """Test initialization."""
        ts_fe = TimeSeriesFeatureEngineering(
            date_column="date", target_column="value", freq="D"
        )
        assert ts_fe.date_column == "date"
        assert ts_fe.target_column == "value"
        assert ts_fe.freq == "D"

    def test_create_lag_features(self):
        """Test lag feature creation."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        df = pd.DataFrame({
            "value": np.random.randn(50),
            "other": np.random.randn(50),
        }, index=dates)

        ts_fe = TimeSeriesFeatureEngineering(target_column="value")
        result_df = ts_fe.create_lag_features(df, lags=[1, 2, 3])

        assert "value_lag_1" in result_df.columns
        assert "value_lag_2" in result_df.columns
        assert "value_lag_3" in result_df.columns

    def test_create_rolling_statistics(self):
        """Test rolling statistics creation."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        df = pd.DataFrame({"value": np.random.randn(50)}, index=dates)

        ts_fe = TimeSeriesFeatureEngineering(target_column="value")
        result_df = ts_fe.create_rolling_statistics(df, windows=[3, 7], statistics=["mean", "std"])

        assert "value_rolling_mean_3" in result_df.columns
        assert "value_rolling_std_3" in result_df.columns
        assert "value_rolling_mean_7" in result_df.columns
        assert "value_rolling_std_7" in result_df.columns

    def test_create_expanding_statistics(self):
        """Test expanding statistics creation."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        df = pd.DataFrame({"value": np.random.randn(50)}, index=dates)

        ts_fe = TimeSeriesFeatureEngineering(target_column="value")
        result_df = ts_fe.create_expanding_statistics(df, statistics=["mean", "std"])

        assert "value_expanding_mean" in result_df.columns
        assert "value_expanding_std" in result_df.columns

    def test_create_time_features(self):
        """Test time feature creation."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        df = pd.DataFrame({"value": np.random.randn(50)}, index=dates)

        ts_fe = TimeSeriesFeatureEngineering()
        result_df = ts_fe.create_time_features(df, features=["year", "month", "dayofweek"])

        assert "year" in result_df.columns
        assert "month" in result_df.columns
        assert "dayofweek" in result_df.columns

    def test_create_difference_features(self):
        """Test difference feature creation."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        df = pd.DataFrame({"value": np.random.randn(50)}, index=dates)

        ts_fe = TimeSeriesFeatureEngineering(target_column="value")
        result_df = ts_fe.create_difference_features(df, periods=[1, 7])

        assert "value_diff_1" in result_df.columns
        assert "value_diff_7" in result_df.columns

    def test_create_ratio_features(self):
        """Test ratio feature creation."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        df = pd.DataFrame({"value": np.random.randn(50)}, index=dates)

        ts_fe = TimeSeriesFeatureEngineering(target_column="value")
        result_df = ts_fe.create_ratio_features(df, windows=[7, 30])

        assert "value_ratio_7" in result_df.columns
        assert "value_ratio_30" in result_df.columns

    def test_seasonal_decomposition(self):
        """Test seasonal decomposition."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        trend = np.linspace(0, 10, 100)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(100) / 7)
        noise = np.random.randn(100) * 0.5
        values = trend + seasonal + noise

        df = pd.DataFrame({"value": values}, index=dates)

        ts_fe = TimeSeriesFeatureEngineering(target_column="value", freq="D")
        result_df, components = ts_fe.seasonal_decomposition(df, "value", period=7)

        assert "value_trend" in result_df.columns
        assert "value_seasonal" in result_df.columns
        assert "value_residual" in result_df.columns
        assert "trend" in components
        assert "seasonal" in components
        assert "residual" in components

    def test_validate_dataframe_with_date_column(self):
        """Test dataframe validation with date column."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "value": np.random.randn(10),
        })

        ts_fe = TimeSeriesFeatureEngineering(date_column="date")
        result_df = ts_fe._validate_dataframe(df)

        assert isinstance(result_df.index, pd.DatetimeIndex)

    def test_validate_dataframe_with_datetime_index(self):
        """Test dataframe validation with datetime index."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({"value": np.random.randn(10)}, index=dates)

        ts_fe = TimeSeriesFeatureEngineering()
        result_df = ts_fe._validate_dataframe(df)

        assert isinstance(result_df.index, pd.DatetimeIndex)

    def test_validate_dataframe_error(self):
        """Test dataframe validation error."""
        df = pd.DataFrame({"value": np.random.randn(10)})

        ts_fe = TimeSeriesFeatureEngineering()
        with pytest.raises(ValueError, match="DatetimeIndex"):
            ts_fe._validate_dataframe(df)

    def test_all_features_together(self):
        """Test creating all feature types together."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({"value": np.random.randn(100)}, index=dates)

        ts_fe = TimeSeriesFeatureEngineering(target_column="value", freq="D")

        df = ts_fe.create_lag_features(df, lags=[1, 7])
        df = ts_fe.create_rolling_statistics(df, windows=[7], statistics=["mean"])
        df = ts_fe.create_time_features(df, features=["month", "dayofweek"])
        df = ts_fe.create_difference_features(df, periods=[1])

        assert len(ts_fe.feature_names_) > 0
        assert all(col in df.columns for col in ts_fe.feature_names_)
