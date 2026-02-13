# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-13

### Added
- Core `CausalArima` class with `fit()`, `summary()`, `plot()`, and `impact()` methods
- Normal-based inference using psi weights (matching R CausalArima)
- Bootstrap inference via residual resampling (statsmodels `simulate()`)
- Full ARIMA and SARIMA support, including seasonal differencing
- Automatic ARIMA order selection via pmdarima (similar to R's `auto.arima`)
- Manual ARIMA order specification with direct statsmodels fitting
- Exogenous regressors (`xreg` parameter) support
- Visualization module: forecast, impact, and residual diagnostic plots
- Reporting module: numeric, HTML, and LaTeX table output
- Custom exception hierarchy (`CausalArimaError`, `ModelFittingError`, `CoefficientExtractionError`)
- Comprehensive test suite: 106 unit tests across 7 test files
- Pytest-based cross-validation suite: 211 parametrized tests across 30 DGPs (7 metrics each)
- Documentation: README, VALIDATION.md, CONTRIBUTING.md, comparison suite READMEs

### Validated Against R Package
- All 30 DGPs pass with point effect correlation > 0.9999
- Point effects, cumulative effects, and p-values match within 5% relative / 0.01 absolute tolerance
- Standard deviations match within 20% relative tolerance (accounts for sigma2 estimation differences between Python state-space MLE and R CSS-ML)
- Tested models: AR(1-3), MA(1-2), ARMA, ARIMA (d=0-2), full SARIMA with weekly/monthly/quarterly/bimonthly periods
