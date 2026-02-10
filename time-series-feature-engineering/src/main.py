"""Time Series Feature Engineering.

This module provides functionality for time series feature engineering including
lag features, rolling statistics, and seasonal decomposition.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngineering:
    """Time Series Feature Engineering with lag features, rolling stats, and decomposition."""

    def __init__(
        self,
        date_column: Optional[str] = None,
        target_column: Optional[str] = None,
        freq: Optional[str] = None,
    ) -> None:
        """Initialize Time Series Feature Engineering.

        Args:
            date_column: Name of date/time column.
            target_column: Name of target column for feature engineering.
            freq: Frequency of time series (e.g., 'D', 'H', 'M').
        """
        self.date_column = date_column
        self.target_column = target_column
        self.freq = freq
        self.feature_names_ = []

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare dataframe.

        Args:
            df: Input dataframe.

        Returns:
            Validated dataframe with datetime index.
        """
        if self.date_column and self.date_column in df.columns:
            df = df.copy()
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            df = df.set_index(self.date_column)
            df = df.sort_index()

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Dataframe must have a DatetimeIndex or date_column must be specified")

        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lags: Union[int, List[int]] = [1, 2, 3, 7, 14, 30],
    ) -> pd.DataFrame:
        """Create lag features for specified columns.

        Args:
            df: Input dataframe.
            columns: Columns to create lags for (default: all numeric columns).
            lags: Lag periods (default: [1, 2, 3, 7, 14, 30]).

        Returns:
            Dataframe with lag features added.
        """
        df = self._validate_dataframe(df)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column and self.target_column in columns:
                columns = [self.target_column]

        if isinstance(lags, int):
            lags = [lags]

        result_df = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue

            for lag in lags:
                lag_col_name = f"{col}_lag_{lag}"
                result_df[lag_col_name] = df[col].shift(lag)
                self.feature_names_.append(lag_col_name)

        logger.info(f"Created {len(lags) * len(columns)} lag features")
        return result_df

    def create_rolling_statistics(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        windows: Union[int, List[int]] = [3, 7, 14, 30],
        statistics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create rolling statistics features.

        Args:
            df: Input dataframe.
            columns: Columns to create rolling stats for (default: all numeric columns).
            windows: Rolling window sizes (default: [3, 7, 14, 30]).
            statistics: Statistics to compute (default: ['mean', 'std', 'min', 'max']).

        Returns:
            Dataframe with rolling statistics features added.
        """
        df = self._validate_dataframe(df)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column and self.target_column in columns:
                columns = [self.target_column]

        if isinstance(windows, int):
            windows = [windows]

        if statistics is None:
            statistics = ["mean", "std", "min", "max"]

        result_df = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue

            for window in windows:
                rolling = df[col].rolling(window=window, min_periods=1)

                for stat in statistics:
                    if stat == "mean":
                        result_df[f"{col}_rolling_mean_{window}"] = rolling.mean()
                        self.feature_names_.append(f"{col}_rolling_mean_{window}")
                    elif stat == "std":
                        result_df[f"{col}_rolling_std_{window}"] = rolling.std()
                        self.feature_names_.append(f"{col}_rolling_std_{window}")
                    elif stat == "min":
                        result_df[f"{col}_rolling_min_{window}"] = rolling.min()
                        self.feature_names_.append(f"{col}_rolling_min_{window}")
                    elif stat == "max":
                        result_df[f"{col}_rolling_max_{window}"] = rolling.max()
                        self.feature_names_.append(f"{col}_rolling_max_{window}")
                    elif stat == "median":
                        result_df[f"{col}_rolling_median_{window}"] = rolling.median()
                        self.feature_names_.append(f"{col}_rolling_median_{window}")
                    elif stat == "sum":
                        result_df[f"{col}_rolling_sum_{window}"] = rolling.sum()
                        self.feature_names_.append(f"{col}_rolling_sum_{window}")

        logger.info(
            f"Created {len(windows) * len(columns) * len(statistics)} rolling statistics features"
        )
        return result_df

    def create_expanding_statistics(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        statistics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create expanding window statistics features.

        Args:
            df: Input dataframe.
            columns: Columns to create expanding stats for (default: all numeric columns).
            statistics: Statistics to compute (default: ['mean', 'std', 'min', 'max']).

        Returns:
            Dataframe with expanding statistics features added.
        """
        df = self._validate_dataframe(df)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column and self.target_column in columns:
                columns = [self.target_column]

        if statistics is None:
            statistics = ["mean", "std", "min", "max"]

        result_df = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue

            expanding = df[col].expanding(min_periods=1)

            for stat in statistics:
                if stat == "mean":
                    result_df[f"{col}_expanding_mean"] = expanding.mean()
                    self.feature_names_.append(f"{col}_expanding_mean")
                elif stat == "std":
                    result_df[f"{col}_expanding_std"] = expanding.std()
                    self.feature_names_.append(f"{col}_expanding_std")
                elif stat == "min":
                    result_df[f"{col}_expanding_min"] = expanding.min()
                    self.feature_names_.append(f"{col}_expanding_min")
                elif stat == "max":
                    result_df[f"{col}_expanding_max"] = expanding.max()
                    self.feature_names_.append(f"{col}_expanding_max")

        logger.info(f"Created {len(columns) * len(statistics)} expanding statistics features")
        return result_df

    def seasonal_decomposition(
        self,
        df: pd.DataFrame,
        column: str,
        model: str = "additive",
        period: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Perform seasonal decomposition.

        Args:
            df: Input dataframe.
            column: Column to decompose.
            model: Decomposition model ('additive' or 'multiplicative') (default: 'additive').
            period: Seasonal period (default: auto-detect).

        Returns:
            Tuple of (dataframe with decomposition features, decomposition components).
        """
        df = self._validate_dataframe(df)

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        series = df[column].dropna()

        if period is None:
            if self.freq:
                freq_map = {"D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1, "H": 24}
                period = freq_map.get(self.freq, 7)
            else:
                period = 7

        if len(series) < 2 * period:
            logger.warning(f"Series too short for decomposition, using period={min(period, len(series)//2)}")
            period = min(period, len(series) // 2) if len(series) > 1 else 1

        trend = self._calculate_trend(series, period)
        seasonal = self._calculate_seasonal(series, trend, period, model)
        residual = series - trend - seasonal if model == "additive" else series / (trend * seasonal)

        result_df = df.copy()
        result_df[f"{column}_trend"] = trend
        result_df[f"{column}_seasonal"] = seasonal
        result_df[f"{column}_residual"] = residual

        self.feature_names_.extend([
            f"{column}_trend",
            f"{column}_seasonal",
            f"{column}_residual",
        ])

        components = {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
        }

        logger.info(f"Performed seasonal decomposition on '{column}' with period={period}")
        return result_df, components

    def _calculate_trend(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate trend component using moving average.

        Args:
            series: Input time series.
            period: Seasonal period.

        Returns:
            Trend component.
        """
        if period % 2 == 0:
            trend = series.rolling(window=period, center=True, min_periods=1).mean()
            trend = trend.rolling(window=2, center=True, min_periods=1).mean()
        else:
            trend = series.rolling(window=period, center=True, min_periods=1).mean()

        return trend

    def _calculate_seasonal(
        self, series: pd.Series, trend: pd.Series, period: int, model: str
    ) -> pd.Series:
        """Calculate seasonal component.

        Args:
            series: Input time series.
            trend: Trend component.
            period: Seasonal period.
            model: Decomposition model.

        Returns:
            Seasonal component.
        """
        if model == "additive":
            detrended = series - trend
        else:
            detrended = series / trend.replace(0, np.nan)

        seasonal = detrended.groupby(detrended.index % period).mean()
        seasonal = seasonal.reindex(detrended.index, method="ffill")

        seasonal = seasonal - seasonal.mean() if model == "additive" else seasonal / seasonal.mean()

        return seasonal

    def create_time_features(
        self, df: pd.DataFrame, features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create time-based features from datetime index.

        Args:
            df: Input dataframe.
            features: Time features to create (default: all).

        Returns:
            Dataframe with time features added.
        """
        df = self._validate_dataframe(df)

        if features is None:
            features = [
                "year",
                "month",
                "day",
                "dayofweek",
                "dayofyear",
                "week",
                "quarter",
                "hour",
                "minute",
                "is_weekend",
            ]

        result_df = df.copy()
        index = df.index

        if "year" in features:
            result_df["year"] = index.year
            self.feature_names_.append("year")

        if "month" in features:
            result_df["month"] = index.month
            self.feature_names_.append("month")

        if "day" in features:
            result_df["day"] = index.day
            self.feature_names_.append("day")

        if "dayofweek" in features:
            result_df["dayofweek"] = index.dayofweek
            self.feature_names_.append("dayofweek")

        if "dayofyear" in features:
            result_df["dayofyear"] = index.dayofyear
            self.feature_names_.append("dayofyear")

        if "week" in features:
            result_df["week"] = index.isocalendar().week
            self.feature_names_.append("week")

        if "quarter" in features:
            result_df["quarter"] = index.quarter
            self.feature_names_.append("quarter")

        if "hour" in features:
            result_df["hour"] = index.hour
            self.feature_names_.append("hour")

        if "minute" in features:
            result_df["minute"] = index.minute
            self.feature_names_.append("minute")

        if "is_weekend" in features:
            result_df["is_weekend"] = (index.dayofweek >= 5).astype(int)
            self.feature_names_.append("is_weekend")

        logger.info(f"Created {len([f for f in features if f in result_df.columns])} time features")
        return result_df

    def create_difference_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        periods: Union[int, List[int]] = [1, 7, 30],
    ) -> pd.DataFrame:
        """Create difference features (first difference, seasonal difference).

        Args:
            df: Input dataframe.
            columns: Columns to create differences for (default: all numeric columns).
            periods: Difference periods (default: [1, 7, 30]).

        Returns:
            Dataframe with difference features added.
        """
        df = self._validate_dataframe(df)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column and self.target_column in columns:
                columns = [self.target_column]

        if isinstance(periods, int):
            periods = [periods]

        result_df = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue

            for period in periods:
                diff_col_name = f"{col}_diff_{period}"
                result_df[diff_col_name] = df[col].diff(period)
                self.feature_names_.append(diff_col_name)

        logger.info(f"Created {len(periods) * len(columns)} difference features")
        return result_df

    def create_ratio_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        windows: Union[int, List[int]] = [7, 30],
    ) -> pd.DataFrame:
        """Create ratio features (current value / rolling mean).

        Args:
            df: Input dataframe.
            columns: Columns to create ratios for (default: all numeric columns).
            windows: Rolling window sizes (default: [7, 30]).

        Returns:
            Dataframe with ratio features added.
        """
        df = self._validate_dataframe(df)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column and self.target_column in columns:
                columns = [self.target_column]

        if isinstance(windows, int):
            windows = [windows]

        result_df = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue

            for window in windows:
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                ratio_col_name = f"{col}_ratio_{window}"
                result_df[ratio_col_name] = df[col] / rolling_mean.replace(0, np.nan)
                self.feature_names_.append(ratio_col_name)

        logger.info(f"Created {len(windows) * len(columns)} ratio features")
        return result_df

    def plot_decomposition(
        self,
        components: Dict[str, pd.Series],
        column: str,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot seasonal decomposition components.

        Args:
            components: Decomposition components dictionary.
            column: Column name for title.
            save_path: Optional path to save figure.
            show: Whether to display plot.
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        fig.suptitle(f"Seasonal Decomposition: {column}", fontsize=14, fontweight="bold")

        original = components["trend"] + components["seasonal"] + components["residual"]

        axes[0].plot(original.index, original.values, label="Original", color="black")
        axes[0].set_ylabel("Original")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(components["trend"].index, components["trend"].values, label="Trend", color="blue")
        axes[1].set_ylabel("Trend")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(
            components["seasonal"].index,
            components["seasonal"].values,
            label="Seasonal",
            color="green",
        )
        axes[2].set_ylabel("Seasonal")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(
            components["residual"].index,
            components["residual"].values,
            label="Residual",
            color="red",
        )
        axes[3].set_ylabel("Residual")
        axes[3].set_xlabel("Time")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Decomposition plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Time Series Feature Engineering")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file with time series data",
    )
    parser.add_argument(
        "--date-column",
        type=str,
        default=None,
        help="Name of date/time column",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Name of target column for feature engineering",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save engineered features CSV",
    )
    parser.add_argument(
        "--lag-features",
        action="store_true",
        help="Create lag features",
    )
    parser.add_argument(
        "--rolling-stats",
        action="store_true",
        help="Create rolling statistics",
    )
    parser.add_argument(
        "--expanding-stats",
        action="store_true",
        help="Create expanding statistics",
    )
    parser.add_argument(
        "--seasonal-decomposition",
        action="store_true",
        help="Perform seasonal decomposition",
    )
    parser.add_argument(
        "--time-features",
        action="store_true",
        help="Create time-based features",
    )
    parser.add_argument(
        "--difference-features",
        action="store_true",
        help="Create difference features",
    )
    parser.add_argument(
        "--ratio-features",
        action="store_true",
        help="Create ratio features",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Create all feature types",
    )
    parser.add_argument(
        "--plot-decomposition",
        action="store_true",
        help="Plot seasonal decomposition",
    )
    parser.add_argument(
        "--save-decomposition-plot",
        type=str,
        default=None,
        help="Path to save decomposition plot",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        feature_config = config.get("features", {})

        df = pd.read_csv(args.input)
        print(f"\n=== Time Series Feature Engineering ===")
        print(f"Data shape: {df.shape}")

        ts_fe = TimeSeriesFeatureEngineering(
            date_column=args.date_column or feature_config.get("date_column"),
            target_column=args.target_column or feature_config.get("target_column"),
            freq=feature_config.get("freq"),
        )

        if args.all_features:
            args.lag_features = True
            args.rolling_stats = True
            args.expanding_stats = True
            args.seasonal_decomposition = True
            args.time_features = True
            args.difference_features = True
            args.ratio_features = True

        if not any([
            args.lag_features,
            args.rolling_stats,
            args.expanding_stats,
            args.seasonal_decomposition,
            args.time_features,
            args.difference_features,
            args.ratio_features,
        ]):
            print("No features selected. Use --all-features or select specific feature types.")
            return

        if args.lag_features:
            print("\nCreating lag features...")
            lags = feature_config.get("lags", [1, 2, 3, 7, 14, 30])
            df = ts_fe.create_lag_features(df, lags=lags)

        if args.rolling_stats:
            print("Creating rolling statistics...")
            windows = feature_config.get("rolling_windows", [3, 7, 14, 30])
            statistics = feature_config.get("rolling_statistics", ["mean", "std", "min", "max"])
            df = ts_fe.create_rolling_statistics(df, windows=windows, statistics=statistics)

        if args.expanding_stats:
            print("Creating expanding statistics...")
            statistics = feature_config.get("expanding_statistics", ["mean", "std", "min", "max"])
            df = ts_fe.create_expanding_statistics(df, statistics=statistics)

        if args.difference_features:
            print("Creating difference features...")
            periods = feature_config.get("difference_periods", [1, 7, 30])
            df = ts_fe.create_difference_features(df, periods=periods)

        if args.ratio_features:
            print("Creating ratio features...")
            windows = feature_config.get("ratio_windows", [7, 30])
            df = ts_fe.create_ratio_features(df, windows=windows)

        if args.time_features:
            print("Creating time features...")
            time_features = feature_config.get("time_features")
            df = ts_fe.create_time_features(df, features=time_features)

        if args.seasonal_decomposition:
            print("Performing seasonal decomposition...")
            if ts_fe.target_column:
                column = ts_fe.target_column
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                column = numeric_cols[0] if len(numeric_cols) > 0 else None

            if column:
                model = feature_config.get("decomposition_model", "additive")
                period = feature_config.get("decomposition_period")
                df, components = ts_fe.seasonal_decomposition(df, column, model=model, period=period)

                if args.plot_decomposition or args.save_decomposition_plot:
                    ts_fe.plot_decomposition(
                        components,
                        column,
                        save_path=args.save_decomposition_plot,
                        show=args.plot_decomposition,
                    )

        print(f"\n=== Feature Engineering Results ===")
        print(f"Final data shape: {df.shape}")
        print(f"Total features created: {len(ts_fe.feature_names_)}")
        print(f"\nFeature names:")
        for i, name in enumerate(ts_fe.feature_names_[:20], 1):
            print(f"  {i}. {name}")
        if len(ts_fe.feature_names_) > 20:
            print(f"  ... and {len(ts_fe.feature_names_) - 20} more")

        df.to_csv(args.output, index=True if isinstance(df.index, pd.DatetimeIndex) else False)
        print(f"\nEngineered features saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


if __name__ == "__main__":
    main()
