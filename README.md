# pycausalarima

A Python implementation of the C-ARIMA methodology for estimating causal effects of interventions on time series data.

> **Disclaimer:** This package is provided with no warranties of any kind, express or implied.

## Overview

`pycausalarima` estimates the causal effect of an intervention on a univariate time series using ARIMA models. It implements the methodology described in:

> Menchetti, F., Cipollini, F., & Mealli, F. (2023). "Combining counterfactual outcomes and ARIMA models for policy evaluation." *The Econometrics Journal*.

This package is a Python port of the [CausalArima R package](https://github.com/FMenchetti/CausalArima). Results have been validated against the R implementation across **30 test scenarios** covering ARIMA and SARIMA models. See [VALIDATION.md](https://github.com/RobsonTigre/pycausalarima/blob/main/VALIDATION.md) for details.

## Installation

### From source

```bash
git clone https://github.com/RobsonTigre/pycausalarima.git
cd pycausalarima
pip install -e .
```

### For development

```bash
pip install -e ".[dev]"
```

### Dependencies

Requires Python 3.10+ with:
- numpy, pandas, scipy
- statsmodels, pmdarima
- matplotlib

## Quick Start

```python
import numpy as np
import pandas as pd
from pycausalarima import CausalArima

# Create sample data
n = 100
np.random.seed(1)
y = np.cumsum(np.random.normal(0, 1, n)) + 100
y[70:] += 10  # Add intervention effect at day 70

dates = pd.date_range('2020-01-01', periods=n, freq='D')
intervention_date = dates[70]

# Fit model
ca = CausalArima(
    y=y,
    dates=dates,
    intervention_date=intervention_date,
)
result = ca.fit()

# View results
print(ca.summary())

# Visualize
ca.plot(type='forecast')
```

## Complete Example: Replicating the R Package Demo

This example replicates the canonical example from the original R CausalArima package, demonstrating detection of a +10 unit intervention effect with an exogenous regressor.

```python
import numpy as np
import pandas as pd
from pycausalarima import CausalArima

# Simulate data (matches R package example)
n = 100
np.random.seed(1)

# Generate AR(1)-like covariate
x1 = np.zeros(n)
x1[0] = 100
for t in range(1, n):
    x1[t] = 100 + 0.999 * (x1[t-1] - 100) + np.random.normal(0, 1)

# Response with relationship to x1
y = 1.2 * x1 + np.random.normal(0, 1, n)

# Add intervention effect (+10) starting at 71% of the series
intervention_start = int(n * 0.71)
y[intervention_start:] += 10

# Create dates
dates = pd.date_range(start='2014-01-05', periods=n, freq='D')
intervention_date = pd.Timestamp('2014-03-16')

# Fit model with exogenous regressor and bootstrap inference
ca = CausalArima(
    y=y,
    dates=dates,
    intervention_date=intervention_date,
    xreg=x1.reshape(-1, 1),  # Include covariate
    n_boot=1000              # Enable bootstrap inference
)
result = ca.fit()

# View summary
print(ca.summary())
```

**Expected Output:**
```
                              Estimate     SD    P-value (2-sided)
Point causal effect            12.257  1.211                 0.000
Cumulative causal effect      310.709  6.634                 0.000
Temporal average effect        10.357  0.221                 0.000
```

> **Note:** Output format shown is for illustration. Actual `summary()` returns a pandas DataFrame with full inference statistics including left-sided, bidirectional, and right-sided p-values. Values are from the R reference implementation; your results may differ slightly due to bootstrap sampling and numerical precision.

**Interpretation:**
- True intervention effect: +10 units
- Estimated temporal average: 10.357 (within 4% of true value)
- P-value ≈ 0: Strong evidence of a significant positive effect
- Cumulative effect (310.7): Total impact over 30 post-intervention days

### Visualizations

```python
# Forecast plot: observed vs counterfactual
ca.plot(type='forecast')

# Impact plot: point effects and cumulative effects
ca.plot(type='impact')

# Residual diagnostics: ACF, PACF, Q-Q plot
ca.plot(type='residuals')
```

## Main Features

- **Automatic ARIMA selection** via pmdarima (similar to R's `auto.arima`)
- **Normal-based and bootstrap inference** for uncertainty quantification
- **Three causal estimands**: point effect, cumulative effect, temporal average
- **Visualization**: forecast plots, impact plots, residual diagnostics
- **Exogenous regressors** support
- **Full SARIMA support**: Tested with 20+ seasonal model configurations

## API

### CausalArima Constructor

```python
CausalArima(
    y,                          # Time series (array-like)
    dates,                      # Dates (DatetimeIndex)
    intervention_date,          # Intervention date (Timestamp)
    auto=True,                  # Auto-select ARIMA order
    order=(0, 0, 0),            # Manual ARIMA order (p, d, q)
    seasonal_order=(0, 0, 0, 1),# Seasonal order (P, D, Q, s)
    xreg=None,                  # Exogenous regressors
    ic='aic',                   # Information criterion
    n_boot=None,                # Bootstrap iterations
    alpha=0.05                  # Significance level
)
```

### Methods

| Method | Description |
|--------|-------------|
| `fit()` | Fit model and compute effects |
| `summary()` | Summary table of effects |
| `plot(type=...)` | Visualizations (`'forecast'`, `'impact'`, `'residuals'`). Returns `matplotlib.figure.Figure` or dict of figures |
| `impact()` | Detailed impact tables |
| `get_residuals(standardized=False)` | Model residuals for diagnostics |

## Methodology

The C-ARIMA approach:
1. Fits an ARIMA model on pre-intervention data
2. Forecasts the counterfactual scenario (what would have happened without intervention)
3. Computes causal effects as observed minus counterfactual
4. Provides inference via analytical variance formulas or bootstrap

## Causal Assumptions

The C-ARIMA method produces valid causal estimates under three key assumptions:

| Assumption | Description | How to Check |
|------------|-------------|--------------|
| **Continuation** | Pre-intervention dynamics would have continued absent intervention | Examine residual ACF; verify stable pre-trend |
| **No Anticipation** | Intervention was not anticipated or acted upon early | Verify timing; check for pre-intervention drift |
| **Persistence** | Single, sustained intervention (not multiple or time-varying) | Review study design |
| **Exogeneity** | Exogenous regressors (`xreg`) are not affected by the intervention | Verify regressors are determined outside the causal pathway |

### State-Space Consistency

The method assumes the ARIMA model correctly captures pre-intervention dynamics:
- Residuals should be white noise (no autocorrelation)
- Model order should be appropriate (use `auto=True` or validate with AIC/BIC)
- Seasonal patterns must be explicitly modeled if present

**Check with:** `ca.plot(type='residuals')` to examine ACF/PACF of residuals.

### What Happens When Assumptions Fail

| Violation | Consequence | Symptom |
|-----------|-------------|---------|
| Pre-trend instability | Biased effect estimates | Poor fit in pre-intervention period |
| Anticipation effects | Effect appears before intervention | Gradual drift near intervention date |
| Multiple interventions | Confounded estimates | Large residuals after first intervention |
| Wrong ARIMA order | Invalid inference | Autocorrelated residuals |
| Unmodeled seasonality | Periodic bias | Seasonal patterns in residuals |

### When NOT to Use C-ARIMA

This method may not be appropriate if:
- Multiple interventions occur at different times
- The intervention effect varies over time (use dynamic regression instead)
- There's a control group available (use difference-in-differences or synthetic control)
- Pre-intervention series is very short (< 30 observations recommended)
- Strong non-stationarity that differencing cannot address

### Recommended Diagnostics

```python
result = ca.fit()

# 1. Residual diagnostics - should show no significant ACF/PACF
ca.plot(type='residuals')

# 2. Visual inspection - does the counterfactual look reasonable?
ca.plot(type='forecast')

# 3. Check model order (if auto=True)
print(f"Selected order: {result.order}")
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Current test suite includes:
- **Unit tests**: Core functionality, edge cases, reporting, visualization
- **R comparison tests**: Validates results match R implementation
- **Cross-validation**: 211 parametrized pytest tests across 30 DGPs (7 metrics each)

```bash
# Run cross-validation only
pytest tests/test_dgp_cross_validation.py -v

# Filter by suite
pytest -m dgp_main -v        # DGPs 1-8 (basic ARIMA)
pytest -m dgp_sarima -v      # DGPs 9-18 (seasonal differencing)
pytest -m dgp_extended -v    # DGPs 19-30 (edge cases)
```

See [VALIDATION.md](https://github.com/RobsonTigre/pycausalarima/blob/main/VALIDATION.md) for methodology and [comparison/](https://github.com/RobsonTigre/pycausalarima/tree/main/comparison) for full reports.

## Validation Summary

Cross-validation against the R `CausalArima` package across 30 Data Generating Processes:

| Metric | Tested | Pass | Status |
|--------|--------|------|--------|
| Point causal effects | 30 DGPs | 30 | All match (max diff < 1.8%) |
| Cumulative effects | 30 DGPs | 30 | All match |
| Temporal average effects | 30 DGPs | 30 | All match |
| Bidirectional p-values | 30 DGPs | 30 | All match (< 0.01 abs diff) |
| Tau time series correlation | 30 DGPs | 30 | All > 0.99 (29/30 = 1.0000) |
| Cumulative series correlation | 30 DGPs | 30 | All = 1.0000 |
| Standard deviations | 30 DGPs | 30 | All within 20% relative tolerance |

**Note on SD tolerance:** Standard deviations use a wider tolerance (20%) because Python (statsmodels state-space MLE) and R (CSS-ML) estimate the innovation variance (sigma2) differently. This is a known cross-library numerical characteristic, especially for models with seasonal differencing (D > 0) or high regular differencing (d >= 2). Point estimates and statistical significance are unaffected.

## Known Limitations

- **SD tolerance for seasonal models:** Standard deviations for seasonal and differenced ARIMA models may differ from R by up to 20% due to sigma2 estimation method differences (Python state-space MLE vs R CSS-ML). Point estimates and p-values match closely. See [VALIDATION.md](https://github.com/RobsonTigre/pycausalarima/blob/main/VALIDATION.md) for details.
- **Bootstrap compatibility:** Bootstrap simulation uses the statsmodels `simulate()` API, which may change in future statsmodels versions. Normal-based inference is unaffected.
- Very short time series (< 20 observations) may produce unstable estimates

## Next Steps

- [ ] **Exogenous regressor (xreg) cross-validation**: Current 30 DGPs test ARIMA-only models. Add DGPs with exogenous regressors to validate ARIMAX behavior against R.
- [ ] **Bootstrap inference cross-validation**: Add DGPs comparing bootstrap CIs and p-values between R and Python (current tests validate normal-based inference only).
- [ ] **Increase coverage for edge cases**: Near-unit-root with seasonal components, very long series (n > 500), multiple exogenous regressors.
- [ ] **CI/CD integration**: Add the cross-validation suite to CI pipeline (runs without R using pre-computed reference data).

## License

GPL-3.0 (same as original R package)

## References

```bibtex
@article{menchetti2023combining,
  title={Combining counterfactual outcomes and ARIMA models for policy evaluation},
  author={Menchetti, Fiammetta and Cipollini, Fabrizio and Mealli, Fabrizia},
  journal={The Econometrics Journal},
  year={2023}
}
```

## Acknowledgments

This is a Python port of the [CausalArima R package](https://github.com/FMenchetti/CausalArima) by Fabrizio Cipollini, Fiammetta Menchetti, and Eugenio Palmieri.

## See Also

- [Original R package](https://github.com/FMenchetti/CausalArima)
- [Paper on arXiv](https://arxiv.org/abs/2103.06740)
- [Validation documentation](https://github.com/RobsonTigre/pycausalarima/blob/main/VALIDATION.md)
- [YouTube webinar](https://www.youtube.com/watch?v=RjMEtv3C5S0) on C-ARIMA methodology
