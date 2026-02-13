# Validation: Python vs R Implementation

This document describes how `pycausalarima` is validated against the original R `CausalArima` package.

## Overview

The Python implementation is validated using **30 Data Generating Processes (DGPs)** covering a wide range of ARIMA and SARIMA configurations. All 30 DGPs pass validation with near-perfect agreement between R and Python results.

## Test Results Summary

| Metric | Tolerance | Result |
|--------|-----------|--------|
| Point Effect | 5% relative | **PASS** (all 30 DGPs) |
| Cumulative Effect | 5% relative | **PASS** (all 30 DGPs) |
| Average Effect | 5% relative | **PASS** (all 30 DGPs) |
| P-values | 0.01 absolute | **PASS** (all 30 DGPs) |
| Time Series Correlation | > 0.99 | **PASS** (most > 0.9999) |
| Standard Deviations | 20% relative | **PASS** (all 30 DGPs) |

**Note on SD tolerance:** Standard deviations depend on the innovation variance (sigma2), which is estimated via state-space MLE in Python (statsmodels) and CSS-ML in R (stats::arima). These methods can produce different sigma2 estimates, especially for models with seasonal differencing (D > 0) or high regular differencing (d >= 2). The 20% tolerance accommodates these cross-library numerical differences while ensuring the implementations are substantially equivalent. Point estimates and statistical significance are unaffected.

## DGP Coverage

### Main Suite (DGPs 1-8)
Basic ARIMA configurations:
- AR(1), AR(2)
- ARIMA(0,1,1), ARIMA(1,1,1)
- ARMA(1,1)
- Seasonal AR (weekly, monthly)

### SARIMA Suite (DGPs 9-18)
Seasonal ARIMA with differencing:
- Seasonal differencing (D > 0)
- Pure seasonal MA
- Full SARIMA with all components
- Quarterly, monthly, weekly periods

### Extended Suite (DGPs 19-30)
Edge cases and complex models:
- Higher-order: AR(3), MA(2), ARMA(2,2), ARMA(3,1)
- Double differencing (d=2)
- Effect size variations: small (+2), large (+50), negative (-15)
- Near-unit-root (AR coefficient 0.95)
- Bimonthly seasonality (s=6)
- Complex combined models

## Validation Structure

```
tests/
└── test_dgp_cross_validation.py  # 211 parametrized pytest tests (30 DGPs x 7 metrics)

comparison/
└── dgp_validation/
    ├── README.md             # Validation suite documentation
    ├── cross_validation_report.md  # Master report (all 30 DGPs)
    ├── data/                 # DGPs 1-8 generated data
    ├── results_r/            # R analysis results
    ├── results_python/       # Python analysis results
    ├── comparison_report.md  # Main suite comparison report
    │
    ├── sarima/               # DGPs 9-18
    │   ├── README.md
    │   ├── data/
    │   ├── results_r/
    │   ├── results_python/
    │   └── comparison_report.md
    │
    └── extended_dgp/         # DGPs 19-30
        ├── README.md
        ├── data/
        ├── results_r/
        ├── results_python/
        └── comparison_report.md
```

## Running Validation

### Quick: Pytest Cross-Validation
```bash
# All 211 tests (30 DGPs x 7 metrics + report generator)
pytest tests/test_dgp_cross_validation.py -v

# Filter by suite
pytest -m dgp_main -v        # DGPs 1-8
pytest -m dgp_sarima -v      # DGPs 9-18
pytest -m dgp_extended -v    # DGPs 19-30
```

### Full: Standalone Analysis Scripts

**Main Suite (DGPs 1-8):**
```bash
cd comparison/dgp_validation
python run_python_analysis.py
python compare_results.py
```

**SARIMA Suite (DGPs 9-18):**
```bash
cd comparison/dgp_validation/sarima
python run_python_sarima_analysis.py
python compare_sarima_results.py
```

**Extended Suite (DGPs 19-30):**
```bash
cd comparison/dgp_validation/extended_dgp
python run_python_extended_analysis.py
python compare_extended_results.py
```

### Regenerating R Results (requires R)
```bash
# Prerequisites: R with jsonlite and forecast packages
cd comparison/dgp_validation
Rscript generate_dgp_data.R && Rscript run_r_analysis.R

cd sarima && Rscript generate_sarima_data.R && Rscript run_r_sarima_analysis.R

cd ../extended_dgp && Rscript generate_extended_data.R && Rscript run_r_extended_analysis.R
```

## Interpreting Results

Each comparison script generates a `comparison_report.md` with:

1. **Summary Table**: PASS/FAIL status for each DGP
2. **Detailed Comparison**: R vs Python values with differences
3. **Correlation Analysis**: Time series correlation metrics
4. **DGP Configurations**: Model orders and true effect sizes

### What "PASS" Means
- Point estimates match within 5% relative tolerance (for values > 1)
- Standard deviations match within 20% relative tolerance
- P-values match within 0.01
- Time series of effects correlate > 0.99

## Adding New Validation Tests

To add a new DGP:

1. Add configuration to `dgp_*_configs.json`
2. Run data generation script
3. Run R and Python analysis scripts
4. Run comparison script
5. Verify PASS status

## See Also

- [comparison/dgp_validation/README.md](comparison/dgp_validation/README.md) - Full validation suite details
- [comparison/dgp_validation/sarima/README.md](comparison/dgp_validation/sarima/README.md) - SARIMA suite details
- [comparison/dgp_validation/extended_dgp/README.md](comparison/dgp_validation/extended_dgp/README.md) - Extended suite details
