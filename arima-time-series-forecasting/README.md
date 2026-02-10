# ARIMA Time Series Forecasting

A Python implementation of ARIMA (AutoRegressive Integrated Moving Average) models for time series forecasting with automatic parameter selection and comprehensive diagnostic tools. This tool provides a complete solution for univariate time series analysis and prediction.

## Project Title and Description

The ARIMA Time Series Forecasting tool provides implementations of ARIMA models for forecasting time series data, including automatic parameter selection using information criteria (AIC/BIC) and comprehensive diagnostic tools for model evaluation. It supports both non-seasonal and seasonal ARIMA (SARIMA) models with visualization capabilities.

This tool solves the problem of forecasting time series data by providing an automated pipeline from data loading to forecast generation, with built-in parameter optimization and model diagnostics. It helps users build accurate forecasting models without deep statistical knowledge.

**Target Audience**: Data scientists, analysts, researchers, and anyone working with time series data who needs reliable forecasting capabilities.

## Features

### ARIMA Modeling
- **ARIMA Model Implementation**: Full ARIMA(p,d,q) model support
- **Seasonal ARIMA (SARIMA)**: Support for seasonal components (P,D,Q,s)
- **Multiple Fitting Methods**: CSS-ML, ML, and CSS optimization methods
- **Model Summary Statistics**: AIC, BIC, parameter estimates, and significance tests

### Parameter Selection
- **Automatic Parameter Selection**: Grid search to find optimal (p,d,q) parameters
- **Information Criteria**: AIC and BIC for model selection
- **Stationarity Testing**: Augmented Dickey-Fuller test for stationarity checking
- **Configurable Search Space**: Adjustable maximum values for p, d, q parameters

### Diagnostics
- **ACF/PACF Plots**: Autocorrelation and partial autocorrelation function visualization
- **Residual Analysis**: Comprehensive residual diagnostics including:
  - Ljung-Box test for residual autocorrelation
  - Durbin-Watson test for serial correlation
  - Normality tests (Shapiro-Wilk)
  - Residual plots (time series, histogram, Q-Q plot, vs fitted values)
- **Model Diagnostics**: Statistical tests and visualizations for model validation

### Forecasting
- **Point Forecasts**: Generate point predictions for future time steps
- **Confidence Intervals**: Prediction intervals with configurable significance levels
- **Multi-step Forecasting**: Forecast multiple steps ahead
- **Dynamic Forecasting**: Option for dynamic vs static forecasting

### Additional Features
- Command-line interface for batch processing
- Configuration via YAML file
- Support for CSV input with date columns
- Comprehensive logging
- Input validation and error handling
- Integration with pandas and numpy
- Visualization capabilities

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/arima-time-series-forecasting
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python src/main.py --input sample.csv --auto --forecast-steps 10
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

auto_selection:
  max_p: 5
  max_d: 2
  max_q: 5
  seasonal: false
  max_P: 2
  max_D: 1
  max_Q: 2
  m: 12
  criterion: "aic"

model:
  method: "css-ml"
  maxiter: 50

forecasting:
  default_steps: 10
  alpha: 0.05

diagnostics:
  acf_pacf_lags: 40
  residual_lags: 40
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `auto_selection.max_p`: Maximum AR order for auto-selection (default: 5)
- `auto_selection.max_d`: Maximum differencing order (default: 2)
- `auto_selection.max_q`: Maximum MA order (default: 5)
- `auto_selection.seasonal`: Whether to include seasonal component (default: false)
- `auto_selection.criterion`: Selection criterion - "aic" or "bic" (default: "aic")
- `model.method`: Fitting method - "css-ml", "ml", or "css" (default: "css-ml")
- `model.maxiter`: Maximum iterations for optimization (default: 50)
- `forecasting.default_steps`: Default forecast steps (default: 10)
- `forecasting.alpha`: Significance level for confidence intervals (default: 0.05)
- `diagnostics.acf_pacf_lags`: Number of lags for ACF/PACF plots (default: 40)
- `diagnostics.residual_lags`: Number of lags for residual tests (default: 40)

## Usage

### Command-Line Interface

#### Automatic Parameter Selection

```bash
python src/main.py --input data.csv --auto --forecast-steps 20 --output forecasts.csv
```

#### Manual Parameter Specification

```bash
python src/main.py --input data.csv --order "1,1,1" --forecast-steps 10 --output forecasts.csv
```

#### With Diagnostics

```bash
python src/main.py --input data.csv --auto --forecast-steps 10 \
  --diagnostics-output diagnostics.json --plots-dir plots/
```

#### Specify Column from CSV

```bash
python src/main.py --input data.csv --column "sales" --date-col "date" \
  --auto --forecast-steps 12
```

### Programmatic Usage

#### Basic Forecasting

```python
import numpy as np
from src.main import ARIMAForecaster

# Generate sample time series
np.random.seed(42)
data = np.cumsum(np.random.randn(100)) + np.sin(np.arange(100) * 0.1)

# Initialize forecaster
forecaster = ARIMAForecaster()

# Load data
forecaster.load_data(data)

# Auto-select parameters and fit
forecaster.auto_fit(max_p=5, max_d=2, max_q=5)

# Generate forecasts
forecast_result = forecaster.forecast(steps=10)

print("Forecasts:", forecast_result["forecast"])
print("95% CI Lower:", forecast_result["conf_int_lower"])
print("95% CI Upper:", forecast_result["conf_int_upper"])
```

#### Manual Parameter Specification

```python
from src.main import ARIMAForecaster

forecaster = ARIMAForecaster()
forecaster.load_data(data)

# Fit with specific order
forecaster.fit(order=(1, 1, 1))

# Forecast
forecast_result = forecaster.forecast(steps=10)
```

#### Stationarity Testing

```python
from src.main import ARIMAParameterSelector

is_stationary, test_results = ARIMAParameterSelector.check_stationarity(data)

print(f"Is stationary: {is_stationary}")
print(f"ADF p-value: {test_results['p_value']}")
```

#### Diagnostics

```python
# Get model diagnostics
diagnostics = forecaster.get_diagnostics()

print("AIC:", diagnostics["aic"])
print("BIC:", diagnostics["bic"])
print("Residual Analysis:", diagnostics["residual_analysis"])

# Generate diagnostic plots
forecaster.plot_diagnostics(save_dir="plots/")
```

#### Using ARIMA Model Directly

```python
from src.main import ARIMAModel

model = ARIMAModel(order=(1, 1, 1))
model.fit(data)

# Get predictions
predictions = model.predict(start=90, end=100)

# Get forecasts with confidence intervals
forecast_result = model.forecast(steps=10, alpha=0.05)

# Get model summary
summary = model.get_summary()
print(summary)
```

#### Residual Analysis

```python
from src.main import ARIMADiagnostics

# Perform residual analysis
analysis = ARIMADiagnostics.residual_analysis(model.residuals)

print("Mean:", analysis["mean"])
print("Std:", analysis["std"])
print("Ljung-Box p-value:", analysis["ljung_box"]["p_value"])
print("Is white noise:", analysis["ljung_box"]["is_white_noise"])

# Plot ACF/PACF
ARIMADiagnostics.plot_acf_pacf(data, lags=40, save_path="acf_pacf.png")

# Plot residuals
ARIMADiagnostics.plot_residuals(
    model.residuals,
    model.fitted_values,
    save_path="residuals.png"
)
```

### Common Use Cases

1. **Sales Forecasting**: Predict future sales based on historical data
2. **Stock Price Prediction**: Forecast stock prices or returns
3. **Demand Forecasting**: Predict product demand
4. **Economic Indicators**: Forecast economic time series
5. **Weather Data**: Predict temperature, precipitation, etc.
6. **Energy Consumption**: Forecast electricity or energy usage

## Project Structure

```
arima-time-series-forecasting/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── src/
│   └── main.py             # Main implementation
├── tests/
│   └── test_main.py        # Unit tests
├── docs/
│   └── API.md              # API documentation (if applicable)
└── logs/
    └── .gitkeep            # Keep logs directory in git
```

### File Descriptions

- `src/main.py`: Contains all implementation:
  - `ARIMAModel`: Core ARIMA model class
  - `ARIMAParameterSelector`: Automatic parameter selection
  - `ARIMADiagnostics`: Diagnostic tools and visualizations
  - `ARIMAForecaster`: Main forecaster class with CLI interface
  - `main()`: Command-line interface entry point

- `tests/test_main.py`: Comprehensive unit tests including:
  - Model fitting and prediction tests
  - Parameter selection tests
  - Diagnostic tests
  - Integration tests
  - Error handling tests

- `config.yaml`: Configuration file for model parameters and settings

- `requirements.txt`: Python package dependencies with versions

## Testing

### Run All Tests

```bash
pytest tests/test_main.py -v
```

### Run Tests with Coverage

```bash
pytest tests/test_main.py --cov=src --cov-report=html
```

### Test Coverage Information

The test suite includes:
- Unit tests for ARIMA model fitting and prediction
- Parameter selection tests
- Diagnostic function tests
- Integration tests for complete workflows
- Error handling and edge case tests

Current test coverage: >90% of code paths

## ARIMA Model Theory

### ARIMA(p,d,q) Model

ARIMA stands for AutoRegressive Integrated Moving Average:
- **AR(p)**: Autoregressive component of order p
- **I(d)**: Integrated component (differencing) of order d
- **MA(q)**: Moving average component of order q

The model can be written as:
```
(1 - φ₁B - ... - φₚBᵖ)(1 - B)ᵈyₜ = (1 + θ₁B + ... + θₑBᵑ)εₜ
```

Where:
- B is the backshift operator
- φᵢ are AR parameters
- θᵢ are MA parameters
- εₜ is white noise
- d is the differencing order

### Parameter Selection

**p (AR order)**: Determined from PACF plot or information criteria
**d (Differencing order)**: Determined by stationarity tests (ADF test)
**q (MA order)**: Determined from ACF plot or information criteria

### Information Criteria

- **AIC (Akaike Information Criterion)**: Balances model fit and complexity
  - AIC = -2*log(L) + 2*k, where L is likelihood, k is parameters
  - Lower is better

- **BIC (Bayesian Information Criterion)**: More conservative than AIC
  - BIC = -2*log(L) + k*log(n), where n is sample size
  - Lower is better, penalizes complexity more than AIC

### Diagnostics

**Residual Analysis**:
- Residuals should be white noise (uncorrelated)
- Ljung-Box test checks for residual autocorrelation
- Durbin-Watson test checks for serial correlation
- Normality tests check if residuals are normally distributed

**ACF/PACF Plots**:
- ACF helps identify MA order (q)
- PACF helps identify AR order (p)
- Significant lags indicate model misspecification

## Troubleshooting

### Common Issues and Solutions

#### Issue: Model Fitting Fails
**Error**: `ValueError: ARIMA model fitting failed`

**Solution**: 
- Check if data has sufficient observations (at least 10-20)
- Try different parameter values
- Ensure data is not all zeros or constant
- Check for NaN or infinite values in data

#### Issue: Non-Stationary Data
**Error**: ADF test indicates non-stationarity

**Solution**:
- Apply differencing (increase d parameter)
- Use log transformation for multiplicative trends
- Remove trend or seasonality before modeling

#### Issue: Parameter Selection Takes Too Long
**Error**: Auto-selection is very slow

**Solution**:
- Reduce max_p, max_d, max_q values
- Use smaller search space
- Specify order manually instead of auto-selection
- Use faster fitting method (e.g., "css" instead of "css-ml")

#### Issue: Poor Forecast Accuracy
**Error**: Forecasts are inaccurate

**Solution**:
- Check residual diagnostics for model adequacy
- Try different parameter combinations
- Consider seasonal ARIMA if seasonality is present
- Ensure sufficient training data
- Check for structural breaks or regime changes

#### Issue: Memory Error with Large Datasets
**Error**: Out of memory during fitting

**Solution**:
- Use smaller parameter search space
- Process data in chunks
- Use faster fitting methods
- Reduce number of lags in diagnostics

### Error Message Explanations

- **"Data must have at least 10 observations"**: Insufficient data for ARIMA model
- **"Order must be specified"**: Need to provide (p,d,q) or use auto-selection
- **"Model must be fitted before prediction"**: Call fit() before predict() or forecast()
- **"steps must be positive"**: Forecast steps must be > 0
- **"criterion must be 'aic' or 'bic'"**: Invalid selection criterion

## Performance Considerations

### Computational Complexity

- **Parameter Selection**: O(n_models * n_samples²) where n_models is search space size
- **Model Fitting**: O(n_samples²) for CSS-ML method
- **Forecasting**: O(steps) for point forecasts

### Optimization Tips

1. **Use CSS method** for faster fitting on large datasets
2. **Limit search space** for auto-selection (smaller max_p, max_d, max_q)
3. **Pre-process data** to remove trends/seasonality manually
4. **Use BIC** instead of AIC for faster selection (fewer models to evaluate)
5. **Specify order manually** if you have domain knowledge

## Contributing

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Create a feature branch: `git checkout -b feature/new-feature`

### Code Style Guidelines

- Follow PEP 8 strictly
- Maximum line length: 88 characters
- Use type hints for all functions
- Write docstrings for all public functions and classes
- Add unit tests for new features
- Update documentation for new functionality

### Pull Request Process

1. Ensure all tests pass: `pytest tests/test_main.py`
2. Check code coverage: `pytest --cov=src`
3. Update documentation if adding new features
4. Submit pull request with clear description

## License

This project is part of the Python ML Projects collection. Please refer to the main repository license.

## Additional Resources

- [ARIMA Model Wikipedia](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)
- [Statsmodels ARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Time Series Analysis Book by Box and Jenkins](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
