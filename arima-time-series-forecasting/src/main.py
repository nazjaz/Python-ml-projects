"""ARIMA Time Series Forecasting with Parameter Selection and Diagnostics.

This module provides implementations of ARIMA (AutoRegressive Integrated
Moving Average) models for time series forecasting, including automatic
parameter selection and comprehensive diagnostic tools.
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
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ARIMAModel:
    """ARIMA model for time series forecasting with parameter selection."""

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    ):
        """Initialize ARIMA model.

        Args:
            order: (p, d, q) order of the ARIMA model (default: None, auto-select)
            seasonal_order: (P, D, Q, s) seasonal order (default: None)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.results = None
        self.residuals = None
        self.fitted_values = None

    def fit(
        self,
        data: np.ndarray,
        method: str = "css-ml",
        maxiter: int = 50,
        **kwargs,
    ) -> "ARIMAModel":
        """Fit ARIMA model to data.

        Args:
            data: Time series data, shape (n_samples,)
            method: Fitting method - "css-ml", "ml", or "css" (default: "css-ml")
            maxiter: Maximum number of iterations (default: 50)
            **kwargs: Additional arguments for ARIMA.fit()

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is invalid or model fails to fit
        """
        data = np.array(data).flatten()

        if len(data) < 10:
            raise ValueError("Data must have at least 10 observations")

        if self.order is None:
            raise ValueError("Order must be specified. Use auto_select_order() first")

        try:
            self.model = ARIMA(
                data, order=self.order, seasonal_order=self.seasonal_order
            )
            self.fitted_model = self.model.fit(method=method, maxiter=maxiter, **kwargs)
            self.results = self.fitted_model
            self.residuals = self.fitted_model.resid
            self.fitted_values = self.fitted_model.fittedvalues

            logger.info(
                f"ARIMA{self.order} model fitted successfully. "
                f"AIC: {self.fitted_model.aic:.2f}, "
                f"BIC: {self.fitted_model.bic:.2f}"
            )

            return self
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {e}")
            raise ValueError(f"ARIMA model fitting failed: {e}") from e

    def predict(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        typ: str = "levels",
        dynamic: bool = False,
    ) -> np.ndarray:
        """Generate predictions from fitted model.

        Args:
            start: Start index for prediction (default: None, end of training)
            end: End index for prediction (default: None, same as start)
            typ: Type of prediction - "levels" or "linear" (default: "levels")
            dynamic: Whether to use dynamic prediction (default: False)

        Returns:
            Predicted values, shape (n_forecasts,)

        Raises:
            ValueError: If model is not fitted
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            predictions = self.fitted_model.predict(
                start=start, end=end, typ=typ, dynamic=dynamic
            )
            return np.array(predictions)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}") from e

    def forecast(self, steps: int, alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """Generate forecasts with confidence intervals.

        Args:
            steps: Number of steps ahead to forecast
            alpha: Significance level for confidence intervals (default: 0.05)

        Returns:
            Dictionary containing:
                - forecast: Forecasted values
                - conf_int_lower: Lower confidence interval
                - conf_int_upper: Upper confidence interval

        Raises:
            ValueError: If model is not fitted or steps is invalid
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        if steps <= 0:
            raise ValueError("steps must be positive")

        try:
            forecast_result = self.fitted_model.forecast(steps=steps, alpha=alpha)
            conf_int = self.fitted_model.get_forecast(steps=steps).conf_int(alpha=alpha)

            return {
                "forecast": np.array(forecast_result),
                "conf_int_lower": np.array(conf_int.iloc[:, 0]),
                "conf_int_upper": np.array(conf_int.iloc[:, 1]),
            }
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            raise ValueError(f"Forecasting failed: {e}") from e

    def get_summary(self) -> str:
        """Get model summary statistics.

        Returns:
            Formatted summary string

        Raises:
            ValueError: If model is not fitted
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before getting summary")

        return str(self.fitted_model.summary())


class ARIMAParameterSelector:
    """Automatic parameter selection for ARIMA models."""

    @staticmethod
    def auto_select_order(
        data: np.ndarray,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        seasonal: bool = False,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        m: int = 12,
        criterion: str = "aic",
        method: str = "css-ml",
    ) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]], float]:
        """Automatically select optimal ARIMA parameters.

        Uses grid search to find parameters that minimize AIC or BIC.

        Args:
            data: Time series data, shape (n_samples,)
            max_p: Maximum AR order (default: 5)
            max_d: Maximum differencing order (default: 2)
            max_q: Maximum MA order (default: 5)
            seasonal: Whether to include seasonal component (default: False)
            max_P: Maximum seasonal AR order (default: 2)
            max_D: Maximum seasonal differencing order (default: 1)
            max_Q: Maximum seasonal MA order (default: 2)
            m: Seasonal period (default: 12)
            criterion: Selection criterion - "aic" or "bic" (default: "aic")
            method: Fitting method (default: "css-ml")

        Returns:
            Tuple containing:
                - best_order: (p, d, q) order
                - best_seasonal_order: (P, D, Q, s) seasonal order or None
                - best_criterion_value: Best AIC or BIC value

        Raises:
            ValueError: If parameters are invalid
        """
        if criterion not in ["aic", "bic"]:
            raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion}")

        data = np.array(data).flatten()

        if len(data) < 20:
            raise ValueError("Data must have at least 20 observations for auto-selection")

        best_order = None
        best_seasonal_order = None
        best_criterion_value = np.inf

        logger.info("Starting parameter selection...")

        for d in range(max_d + 1):
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    try:
                        if seasonal:
                            for D in range(max_D + 1):
                                for P in range(max_P + 1):
                                    for Q in range(max_Q + 1):
                                        order = (p, d, q)
                                        seasonal_order = (P, D, Q, m)

                                        model = ARIMA(data, order=order, seasonal_order=seasonal_order)
                                        fitted = model.fit(method=method, disp=0)

                                        criterion_value = (
                                            fitted.aic if criterion == "aic" else fitted.bic
                                        )

                                        if criterion_value < best_criterion_value:
                                            best_criterion_value = criterion_value
                                            best_order = order
                                            best_seasonal_order = seasonal_order

                                        logger.debug(
                                            f"ARIMA{order} SARIMA{seasonal_order}: "
                                            f"{criterion.upper()}={criterion_value:.2f}"
                                        )
                        else:
                            order = (p, d, q)

                            model = ARIMA(data, order=order)
                            fitted = model.fit(method=method, disp=0)

                            criterion_value = (
                                fitted.aic if criterion == "aic" else fitted.bic
                            )

                            if criterion_value < best_criterion_value:
                                best_criterion_value = criterion_value
                                best_order = order
                                best_seasonal_order = None

                            logger.debug(
                                f"ARIMA{order}: {criterion.upper()}={criterion_value:.2f}"
                            )
                    except Exception as e:
                        logger.debug(f"Failed to fit ARIMA{p, d, q}: {e}")
                        continue

        if best_order is None:
            raise ValueError("Could not find valid ARIMA model")

        logger.info(
            f"Best model: ARIMA{best_order} "
            f"{f'SARIMA{best_seasonal_order}' if best_seasonal_order else ''} "
            f"with {criterion.upper()}={best_criterion_value:.2f}"
        )

        return best_order, best_seasonal_order, best_criterion_value

    @staticmethod
    def check_stationarity(
        data: np.ndarray, alpha: float = 0.05
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if time series is stationary using Augmented Dickey-Fuller test.

        Args:
            data: Time series data, shape (n_samples,)
            alpha: Significance level (default: 0.05)

        Returns:
            Tuple containing:
                - is_stationary: Boolean indicating stationarity
                - test_results: Dictionary with test statistics
        """
        data = np.array(data).flatten()

        result = adfuller(data)

        is_stationary = result[1] <= alpha

        test_results = {
            "adf_statistic": float(result[0]),
            "p_value": float(result[1]),
            "critical_values": {key: float(val) for key, val in result[4].items()},
            "is_stationary": is_stationary,
        }

        return is_stationary, test_results


class ARIMADiagnostics:
    """Diagnostic tools for ARIMA model evaluation."""

    @staticmethod
    def plot_acf_pacf(
        data: np.ndarray,
        lags: int = 40,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot ACF and PACF of time series.

        Args:
            data: Time series data, shape (n_samples,)
            lags: Number of lags to plot (default: 40)
            figsize: Figure size (default: (12, 6))
            save_path: Path to save figure (default: None)
        """
        data = np.array(data).flatten()

        fig, axes = plt.subplots(2, 1, figsize=figsize)

        acf_values = acf(data, nlags=lags, fft=True)
        pacf_values = pacf(data, nlags=lags)

        axes[0].stem(range(len(acf_values)), acf_values)
        axes[0].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        axes[0].axhline(
            y=1.96 / np.sqrt(len(data)),
            color="r",
            linestyle="--",
            linewidth=0.5,
            label="95% confidence",
        )
        axes[0].axhline(
            y=-1.96 / np.sqrt(len(data)),
            color="r",
            linestyle="--",
            linewidth=0.5,
        )
        axes[0].set_title("Autocorrelation Function (ACF)")
        axes[0].set_xlabel("Lag")
        axes[0].set_ylabel("ACF")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].stem(range(len(pacf_values)), pacf_values)
        axes[1].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        axes[1].axhline(
            y=1.96 / np.sqrt(len(data)),
            color="r",
            linestyle="--",
            linewidth=0.5,
            label="95% confidence",
        )
        axes[1].axhline(
            y=-1.96 / np.sqrt(len(data)),
            color="r",
            linestyle="--",
            linewidth=0.5,
        )
        axes[1].set_title("Partial Autocorrelation Function (PACF)")
        axes[1].set_xlabel("Lag")
        axes[1].set_ylabel("PACF")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"ACF/PACF plot saved to {save_path}")

        plt.close()

    @staticmethod
    def residual_analysis(
        residuals: np.ndarray, lags: int = 40
    ) -> Dict[str, Union[float, Dict, np.ndarray]]:
        """Perform comprehensive residual analysis.

        Args:
            residuals: Model residuals, shape (n_samples,)
            lags: Number of lags for tests (default: 40)

        Returns:
            Dictionary containing diagnostic statistics
        """
        residuals = np.array(residuals).flatten()

        analysis = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "skewness": float(stats.skew(residuals)),
            "kurtosis": float(stats.kurtosis(residuals)),
        }

        ljung_box_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
        analysis["ljung_box"] = {
            "statistic": float(ljung_box_result["lb_stat"].iloc[-1]),
            "p_value": float(ljung_box_result["lb_pvalue"].iloc[-1]),
            "is_white_noise": float(ljung_box_result["lb_pvalue"].iloc[-1]) > 0.05,
        }

        try:
            dw_statistic = durbin_watson(residuals)
            analysis["durbin_watson"] = {
                "statistic": float(dw_statistic),
                "interpretation": (
                    "No autocorrelation" if 1.5 < dw_statistic < 2.5 else "Autocorrelation present"
                ),
            }
        except Exception as e:
            logger.warning(f"Durbin-Watson test failed: {e}")
            analysis["durbin_watson"] = None

        shapiro_result = stats.shapiro(residuals[:5000])
        analysis["normality_test"] = {
            "statistic": float(shapiro_result.statistic),
            "p_value": float(shapiro_result.pvalue),
            "is_normal": shapiro_result.pvalue > 0.05,
        }

        return analysis

    @staticmethod
    def plot_residuals(
        residuals: np.ndarray,
        fitted_values: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot residual diagnostics.

        Args:
            residuals: Model residuals, shape (n_samples,)
            fitted_values: Fitted values (default: None)
            figsize: Figure size (default: (12, 8))
            save_path: Path to save figure (default: None)
        """
        residuals = np.array(residuals).flatten()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        axes[0, 0].plot(residuals)
        axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=0.5)
        axes[0, 0].set_title("Residuals Over Time")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        axes[0, 1].set_title("Residual Distribution")
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot (Normal Distribution)")
        axes[1, 0].grid(True, alpha=0.3)

        if fitted_values is not None:
            axes[1, 1].scatter(fitted_values, residuals, alpha=0.5)
            axes[1, 1].axhline(y=0, color="r", linestyle="--", linewidth=0.5)
            axes[1, 1].set_title("Residuals vs Fitted Values")
            axes[1, 1].set_xlabel("Fitted Values")
            axes[1, 1].set_ylabel("Residuals")
        else:
            acf_residuals = acf(residuals, nlags=20, fft=True)
            axes[1, 1].stem(range(len(acf_residuals)), acf_residuals)
            axes[1, 1].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
            axes[1, 1].axhline(
                y=1.96 / np.sqrt(len(residuals)),
                color="r",
                linestyle="--",
                linewidth=0.5,
            )
            axes[1, 1].axhline(
                y=-1.96 / np.sqrt(len(residuals)),
                color="r",
                linestyle="--",
                linewidth=0.5,
            )
            axes[1, 1].set_title("ACF of Residuals")
            axes[1, 1].set_xlabel("Lag")
            axes[1, 1].set_ylabel("ACF")

        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Residual plots saved to {save_path}")

        plt.close()


class ARIMAForecaster:
    """Main forecaster class combining all ARIMA functionality."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize ARIMA forecaster.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.model = None
        self.data = None
        self.original_data = None

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "logs/app.log")

        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(handler)

    def load_data(
        self, data: Union[np.ndarray, pd.Series, List], date_index: Optional[pd.DatetimeIndex] = None
    ) -> "ARIMAForecaster":
        """Load time series data.

        Args:
            data: Time series data
            date_index: Optional datetime index (default: None)

        Returns:
            Self for method chaining
        """
        if isinstance(data, pd.Series):
            self.original_data = data.values
            if date_index is None and isinstance(data.index, pd.DatetimeIndex):
                date_index = data.index
        else:
            self.original_data = np.array(data).flatten()

        self.data = self.original_data.copy()

        logger.info(f"Loaded data with {len(self.data)} observations")

        return self

    def auto_fit(
        self,
        max_p: Optional[int] = None,
        max_d: Optional[int] = None,
        max_q: Optional[int] = None,
        seasonal: Optional[bool] = None,
        criterion: Optional[str] = None,
    ) -> "ARIMAForecaster":
        """Automatically select parameters and fit model.

        Args:
            max_p: Maximum AR order (default: from config)
            max_d: Maximum differencing order (default: from config)
            max_q: Maximum MA order (default: from config)
            seasonal: Whether to include seasonal component (default: from config)
            criterion: Selection criterion - "aic" or "bic" (default: from config)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is not loaded
        """
        if self.data is None:
            raise ValueError("Data must be loaded before fitting")

        auto_config = self.config.get("auto_selection", {})
        max_p = max_p or auto_config.get("max_p", 5)
        max_d = max_d or auto_config.get("max_d", 2)
        max_q = max_q or auto_config.get("max_q", 5)
        seasonal = seasonal if seasonal is not None else auto_config.get("seasonal", False)
        criterion = criterion or auto_config.get("criterion", "aic")

        logger.info("Starting automatic parameter selection...")

        best_order, best_seasonal_order, best_criterion = (
            ARIMAParameterSelector.auto_select_order(
                self.data,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                seasonal=seasonal,
                criterion=criterion,
            )
        )

        self.model = ARIMAModel(order=best_order, seasonal_order=best_seasonal_order)
        self.model.fit(self.data)

        logger.info(f"Model fitted with order {best_order}")

        return self

    def fit(
        self,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    ) -> "ARIMAForecaster":
        """Fit ARIMA model with specified parameters.

        Args:
            order: (p, d, q) order of ARIMA model
            seasonal_order: (P, D, Q, s) seasonal order (default: None)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is not loaded
        """
        if self.data is None:
            raise ValueError("Data must be loaded before fitting")

        self.model = ARIMAModel(order=order, seasonal_order=seasonal_order)
        self.model.fit(self.data)

        return self

    def forecast(self, steps: int, alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """Generate forecasts.

        Args:
            steps: Number of steps ahead to forecast
            alpha: Significance level for confidence intervals (default: 0.05)

        Returns:
            Dictionary with forecast and confidence intervals

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None or self.model.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        return self.model.forecast(steps=steps, alpha=alpha)

    def get_diagnostics(self) -> Dict:
        """Get comprehensive model diagnostics.

        Returns:
            Dictionary containing diagnostic results

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None or self.model.fitted_model is None:
            raise ValueError("Model must be fitted before diagnostics")

        diagnostics = {
            "model_summary": self.model.get_summary(),
            "residual_analysis": ARIMADiagnostics.residual_analysis(
                self.model.residuals
            ),
            "aic": float(self.model.fitted_model.aic),
            "bic": float(self.model.fitted_model.bic),
            "order": self.model.order,
            "seasonal_order": self.model.seasonal_order,
        }

        return diagnostics

    def plot_diagnostics(
        self, save_dir: Optional[Path] = None
    ) -> None:
        """Generate diagnostic plots.

        Args:
            save_dir: Directory to save plots (default: None)

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None or self.model.fitted_model is None:
            raise ValueError("Model must be fitted before plotting diagnostics")

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        acf_pacf_path = save_dir / "acf_pacf.png" if save_dir else None
        ARIMADiagnostics.plot_acf_pacf(
            self.original_data, save_path=acf_pacf_path
        )

        residuals_path = save_dir / "residuals.png" if save_dir else None
        ARIMADiagnostics.plot_residuals(
            self.model.residuals,
            self.model.fitted_values,
            save_path=residuals_path,
        )


def main():
    """Main entry point for ARIMA forecaster."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ARIMA time series forecasting with parameter selection and diagnostics"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file or column name for time series",
    )
    parser.add_argument(
        "--column",
        type=str,
        help="Column name if input is CSV (default: first numeric column)",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        help="Date column name if input is CSV",
    )
    parser.add_argument(
        "--order",
        type=str,
        help="ARIMA order as 'p,d,q' (e.g., '1,1,1'). If not provided, auto-select",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically select optimal parameters",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=10,
        help="Number of steps to forecast (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output CSV file for forecasts",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=str,
        help="Path to output JSON file for diagnostics",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        help="Directory to save diagnostic plots",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    forecaster = ARIMAForecaster(
        config_path=Path(args.config) if args.config else None
    )

    input_path = Path(args.input)
    if input_path.exists() and input_path.suffix == ".csv":
        df = pd.read_csv(input_path)

        if args.date_col:
            df[args.date_col] = pd.to_datetime(df[args.date_col])
            df = df.set_index(args.date_col)

        if args.column:
            data = df[args.column].values
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in CSV")
            data = df[numeric_cols[0]].values
            logger.info(f"Using column: {numeric_cols[0]}")
    else:
        raise ValueError(f"Input file not found: {args.input}")

    forecaster.load_data(data)

    if args.auto or args.order is None:
        forecaster.auto_fit()
    else:
        order_parts = [int(x.strip()) for x in args.order.split(",")]
        if len(order_parts) != 3:
            raise ValueError("Order must be in format 'p,d,q'")
        forecaster.fit(order=tuple(order_parts))

    forecast_result = forecaster.forecast(steps=args.forecast_steps)

    if args.output:
        forecast_df = pd.DataFrame(
            {
                "forecast": forecast_result["forecast"],
                "lower": forecast_result["conf_int_lower"],
                "upper": forecast_result["conf_int_upper"],
            }
        )
        forecast_df.to_csv(args.output, index=False)
        logger.info(f"Forecasts saved to {args.output}")

    diagnostics = forecaster.get_diagnostics()

    if args.diagnostics_output:
        diagnostics_export = {
            "aic": diagnostics["aic"],
            "bic": diagnostics["bic"],
            "order": diagnostics["order"],
            "seasonal_order": diagnostics["seasonal_order"],
            "residual_analysis": diagnostics["residual_analysis"],
        }
        with open(args.diagnostics_output, "w") as f:
            json.dump(diagnostics_export, f, indent=2)
        logger.info(f"Diagnostics saved to {args.diagnostics_output}")

    if args.plots_dir:
        forecaster.plot_diagnostics(save_dir=Path(args.plots_dir))

    print("\nARIMA Model Summary:")
    print("=" * 50)
    print(diagnostics["model_summary"])
    print("\nResidual Analysis:")
    print(json.dumps(diagnostics["residual_analysis"], indent=2))
    print(f"\nForecast for {args.forecast_steps} steps:")
    print(f"Mean: {forecast_result['forecast']}")
    print(f"95% CI: [{forecast_result['conf_int_lower']}, {forecast_result['conf_int_upper']}]")


if __name__ == "__main__":
    main()
