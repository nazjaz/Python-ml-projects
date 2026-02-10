"""Tests for ARIMA time series forecasting module."""

import numpy as np
import pytest

from src.main import (
    ARIMADiagnostics,
    ARIMAForecaster,
    ARIMAModel,
    ARIMAParameterSelector,
)


class TestARIMAModel:
    """Test cases for ARIMA model."""

    def test_model_initialization(self):
        """Test ARIMA model initialization."""
        model = ARIMAModel(order=(1, 1, 1))
        assert model.order == (1, 1, 1)
        assert model.seasonal_order is None

    def test_model_fit_basic(self):
        """Test basic model fitting."""
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100)) + np.sin(np.arange(100) * 0.1)

        model = ARIMAModel(order=(1, 1, 1))
        model.fit(data)

        assert model.fitted_model is not None
        assert model.residuals is not None
        assert len(model.residuals) > 0

    def test_model_fit_insufficient_data(self):
        """Test model fitting with insufficient data."""
        data = np.array([1, 2, 3, 4, 5])

        model = ARIMAModel(order=(1, 1, 1))
        with pytest.raises(ValueError, match="Data must have at least 10"):
            model.fit(data)

    def test_model_fit_no_order(self):
        """Test model fitting without order specified."""
        data = np.random.randn(100)

        model = ARIMAModel()
        with pytest.raises(ValueError, match="Order must be specified"):
            model.fit(data)

    def test_model_predict(self):
        """Test model prediction."""
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100)) + np.sin(np.arange(100) * 0.1)

        model = ARIMAModel(order=(1, 1, 1))
        model.fit(data)

        predictions = model.predict(start=90, end=100)
        assert len(predictions) == 11

    def test_model_predict_not_fitted(self):
        """Test prediction without fitting."""
        model = ARIMAModel(order=(1, 1, 1))
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict()

    def test_model_forecast(self):
        """Test model forecasting."""
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100)) + np.sin(np.arange(100) * 0.1)

        model = ARIMAModel(order=(1, 1, 1))
        model.fit(data)

        forecast_result = model.forecast(steps=10)
        assert "forecast" in forecast_result
        assert "conf_int_lower" in forecast_result
        assert "conf_int_upper" in forecast_result
        assert len(forecast_result["forecast"]) == 10

    def test_model_forecast_invalid_steps(self):
        """Test forecasting with invalid steps."""
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100))

        model = ARIMAModel(order=(1, 1, 1))
        model.fit(data)

        with pytest.raises(ValueError, match="steps must be positive"):
            model.forecast(steps=-1)

    def test_model_get_summary(self):
        """Test getting model summary."""
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100))

        model = ARIMAModel(order=(1, 1, 1))
        model.fit(data)

        summary = model.get_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestARIMAParameterSelector:
    """Test cases for ARIMA parameter selection."""

    def test_check_stationarity_stationary(self):
        """Test stationarity check on stationary data."""
        np.random.seed(42)
        data = np.random.randn(100)

        is_stationary, results = ARIMAParameterSelector.check_stationarity(data)

        assert "adf_statistic" in results
        assert "p_value" in results
        assert "critical_values" in results
        assert isinstance(is_stationary, bool)

    def test_check_stationarity_non_stationary(self):
        """Test stationarity check on non-stationary data."""
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100))

        is_stationary, results = ARIMAParameterSelector.check_stationarity(data)

        assert "adf_statistic" in results
        assert "p_value" in results
        assert isinstance(is_stationary, bool)

    def test_auto_select_order_basic(self):
        """Test automatic order selection."""
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100)) + np.sin(np.arange(100) * 0.1)

        order, seasonal_order, criterion_value = (
            ARIMAParameterSelector.auto_select_order(
                data, max_p=2, max_d=1, max_q=2, criterion="aic"
            )
        )

        assert len(order) == 3
        assert all(isinstance(x, int) for x in order)
        assert isinstance(criterion_value, float)

    def test_auto_select_order_insufficient_data(self):
        """Test auto selection with insufficient data."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        with pytest.raises(ValueError, match="Data must have at least 20"):
            ARIMAParameterSelector.auto_select_order(data)

    def test_auto_select_order_invalid_criterion(self):
        """Test auto selection with invalid criterion."""
        np.random.seed(42)
        data = np.random.randn(100)

        with pytest.raises(ValueError, match="criterion must be"):
            ARIMAParameterSelector.auto_select_order(data, criterion="invalid")


class TestARIMADiagnostics:
    """Test cases for ARIMA diagnostics."""

    def test_residual_analysis(self):
        """Test residual analysis."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        analysis = ARIMADiagnostics.residual_analysis(residuals)

        assert "mean" in analysis
        assert "std" in analysis
        assert "skewness" in analysis
        assert "kurtosis" in analysis
        assert "ljung_box" in analysis
        assert "normality_test" in analysis

    def test_plot_acf_pacf(self):
        """Test ACF/PACF plotting."""
        np.random.seed(42)
        data = np.random.randn(100)

        ARIMADiagnostics.plot_acf_pacf(data, lags=20)

    def test_plot_residuals(self):
        """Test residual plotting."""
        np.random.seed(42)
        residuals = np.random.randn(100)
        fitted_values = np.random.randn(100)

        ARIMADiagnostics.plot_residuals(residuals, fitted_values)

    def test_plot_residuals_no_fitted(self):
        """Test residual plotting without fitted values."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        ARIMADiagnostics.plot_residuals(residuals)


class TestARIMAForecaster:
    """Test cases for ARIMA forecaster."""

    def test_forecaster_initialization(self):
        """Test forecaster initialization."""
        forecaster = ARIMAForecaster()
        assert forecaster.model is None
        assert forecaster.data is None

    def test_load_data(self):
        """Test loading data."""
        forecaster = ARIMAForecaster()
        data = np.random.randn(100)

        forecaster.load_data(data)

        assert forecaster.data is not None
        assert len(forecaster.data) == 100

    def test_load_data_list(self):
        """Test loading data from list."""
        forecaster = ARIMAForecaster()
        data = [1, 2, 3, 4, 5]

        forecaster.load_data(data)

        assert forecaster.data is not None
        assert len(forecaster.data) == 5

    def test_fit_with_order(self):
        """Test fitting with specified order."""
        forecaster = ARIMAForecaster()
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100))

        forecaster.load_data(data)
        forecaster.fit(order=(1, 1, 1))

        assert forecaster.model is not None
        assert forecaster.model.fitted_model is not None

    def test_fit_no_data(self):
        """Test fitting without loading data."""
        forecaster = ARIMAForecaster()

        with pytest.raises(ValueError, match="Data must be loaded"):
            forecaster.fit(order=(1, 1, 1))

    def test_auto_fit(self):
        """Test automatic fitting."""
        forecaster = ARIMAForecaster()
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100)) + np.sin(np.arange(100) * 0.1)

        forecaster.load_data(data)
        forecaster.auto_fit(max_p=2, max_d=1, max_q=2)

        assert forecaster.model is not None
        assert forecaster.model.fitted_model is not None

    def test_auto_fit_no_data(self):
        """Test auto fitting without loading data."""
        forecaster = ARIMAForecaster()

        with pytest.raises(ValueError, match="Data must be loaded"):
            forecaster.auto_fit()

    def test_forecast(self):
        """Test forecasting."""
        forecaster = ARIMAForecaster()
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100))

        forecaster.load_data(data)
        forecaster.fit(order=(1, 1, 1))

        forecast_result = forecaster.forecast(steps=10)

        assert "forecast" in forecast_result
        assert "conf_int_lower" in forecast_result
        assert "conf_int_upper" in forecast_result
        assert len(forecast_result["forecast"]) == 10

    def test_forecast_not_fitted(self):
        """Test forecasting without fitting."""
        forecaster = ARIMAForecaster()
        data = np.random.randn(100)

        forecaster.load_data(data)

        with pytest.raises(ValueError, match="Model must be fitted"):
            forecaster.forecast(steps=10)

    def test_get_diagnostics(self):
        """Test getting diagnostics."""
        forecaster = ARIMAForecaster()
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100))

        forecaster.load_data(data)
        forecaster.fit(order=(1, 1, 1))

        diagnostics = forecaster.get_diagnostics()

        assert "model_summary" in diagnostics
        assert "residual_analysis" in diagnostics
        assert "aic" in diagnostics
        assert "bic" in diagnostics
        assert "order" in diagnostics

    def test_get_diagnostics_not_fitted(self):
        """Test getting diagnostics without fitting."""
        forecaster = ARIMAForecaster()
        data = np.random.randn(100)

        forecaster.load_data(data)

        with pytest.raises(ValueError, match="Model must be fitted"):
            forecaster.get_diagnostics()

    def test_plot_diagnostics(self):
        """Test plotting diagnostics."""
        forecaster = ARIMAForecaster()
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100))

        forecaster.load_data(data)
        forecaster.fit(order=(1, 1, 1))

        forecaster.plot_diagnostics()

    def test_plot_diagnostics_not_fitted(self):
        """Test plotting diagnostics without fitting."""
        forecaster = ARIMAForecaster()
        data = np.random.randn(100)

        forecaster.load_data(data)

        with pytest.raises(ValueError, match="Model must be fitted"):
            forecaster.plot_diagnostics()


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(self):
        """Test complete forecasting workflow."""
        forecaster = ARIMAForecaster()
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100)) + np.sin(np.arange(100) * 0.1)

        forecaster.load_data(data)
        forecaster.auto_fit(max_p=2, max_d=1, max_q=2)

        forecast_result = forecaster.forecast(steps=10)
        diagnostics = forecaster.get_diagnostics()

        assert forecast_result is not None
        assert diagnostics is not None
        assert len(forecast_result["forecast"]) == 10

    def test_stationarity_check_integration(self):
        """Test stationarity check integration."""
        np.random.seed(42)
        stationary_data = np.random.randn(100)
        non_stationary_data = np.cumsum(np.random.randn(100))

        is_stat_1, _ = ARIMAParameterSelector.check_stationarity(stationary_data)
        is_stat_2, _ = ARIMAParameterSelector.check_stationarity(non_stationary_data)

        assert isinstance(is_stat_1, bool)
        assert isinstance(is_stat_2, bool)
