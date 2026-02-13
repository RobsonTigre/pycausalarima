# DGP Validation: R vs Python Cross-Language Comparison

This folder contains a comprehensive validation suite comparing `pycausalarima` (Python) against `CausalArima` (R) using 30 Data Generating Processes (DGPs) with different ARIMA/SARIMA configurations organized in 3 test suites.

## Objective

Verify that the Python implementation produces results equivalent to the original R package across a variety of ARIMA/SARIMA model specifications, covering non-seasonal ARIMA, seasonal ARIMA (SARIMA), higher-order models, and edge cases.

## Test Suites Overview

| Suite | DGPs | Location | Focus |
|-------|------|----------|-------|
| **Main** | 1-8 | `./` (this directory) | Basic ARIMA + simple seasonal |
| **SARIMA** | 9-18 | `sarima/` | Seasonal differencing, full SARIMA |
| **Extended** | 19-30 | `extended_dgp/` | Higher-order, edge cases, effect sizes |

## All 30 DGP Configurations

### Main Suite (DGPs 1-8): Basic ARIMA & Simple Seasonal

| DGP | Name | Order (p,d,q) | Seasonal (P,D,Q,s) | Effect | Description |
|-----|------|---------------|---------------------|--------|-------------|
| 1 | ar1 | (1,0,0) | - | +10 | Simple AR(1) with phi=0.7 |
| 2 | arima011 | (0,1,1) | - | +15 | Integrated MA(1) |
| 3 | arma11 | (1,0,1) | - | +10 | Mixed ARMA(1,1) |
| 4 | ar2 | (2,0,0) | - | +12 | AR(2) process |
| 5 | seasonal_ar_weekly | (0,0,0) | (1,0,0,7) | +20 | Weekly seasonal AR |
| 6 | ar1_seasonal_ma_weekly | (1,0,0) | (0,0,1,7) | +15 | AR(1) + seasonal MA |
| 7 | seasonal_ar_monthly | (0,0,0) | (1,0,0,12) | +10 | Monthly seasonal AR |
| 8 | arima111 | (1,1,1) | - | +8 | Full ARIMA(1,1,1) |

### SARIMA Suite (DGPs 9-18): Seasonal Differencing & Complex SARIMA

| DGP | Name | Order (p,d,q) | Seasonal (P,D,Q,s) | Effect | Key Feature |
|-----|------|---------------|---------------------|--------|-------------|
| 9 | sar1_D1_monthly | (0,0,0) | (1,1,0,12) | +12 | SAR with D=1 |
| 10 | sma1_D1_quarterly | (0,0,0) | (0,1,1,4) | +8 | SMA with D=1, quarterly |
| 11 | pure_sma1_monthly | (0,0,0) | (0,0,1,12) | +10 | Pure seasonal MA |
| 12 | pure_sma2_weekly | (0,0,0) | (0,0,2,7) | +15 | Higher-order SMA |
| 13 | sarima_full_monthly | (1,0,1) | (1,0,1,12) | +10 | Full SARIMA |
| 14 | sarima_d1_D1 | (1,1,0) | (0,1,1,12) | +15 | Combined d=1 AND D=1 |
| 15 | sarima_quarterly | (0,1,1) | (1,1,0,4) | +10 | Quarterly with d+D |
| 16 | sar2_monthly | (0,0,0) | (2,0,0,12) | +12 | Higher-order SAR(2) |
| 17 | sarma_D1_monthly | (0,0,0) | (1,1,1,12) | +10 | SARMA with D=1 |
| 18 | sarima_full_weekly | (1,0,1) | (1,0,1,7) | +20 | Full weekly SARIMA |

### Extended Suite (DGPs 19-30): Edge Cases & Higher-Order Models

| DGP | Name | Order (p,d,q) | Seasonal (P,D,Q,s) | Effect | Description |
|-----|------|---------------|---------------------|--------|-------------|
| 19 | ar3 | (3,0,0) | - | +10 | Higher-order AR(3) |
| 20 | ma2 | (0,0,2) | - | +12 | Pure MA(2) |
| 21 | arma22 | (2,0,2) | - | +10 | ARMA(2,2) mixed model |
| 22 | arma31 | (3,0,1) | - | +15 | High AR(3) + MA(1) |
| 23 | arima_d2 | (1,2,1) | - | +10 | Double differencing (d=2) |
| 24 | ar1_small_effect | (1,0,0) | - | +2 | Small positive effect |
| 25 | ar1_large_effect | (1,0,0) | - | +50 | Large positive effect |
| 26 | arma11_negative | (1,0,1) | - | -15 | Negative effect |
| 27 | ar1_near_unit | (1,0,0) | - | +10 | Near unit root (phi=0.95) |
| 28 | sar1_bimonthly | (0,0,0) | (1,0,0,6) | +10 | Bimonthly seasonality |
| 29 | sarima_bimonthly | (1,0,1) | (1,0,1,6) | +15 | Full SARIMA with s=6 |
| 30 | arma21_seasonal | (2,0,1) | (1,0,1,12) | +12 | Complex combined model |

## Methodology

### Data Generation

Each DGP generates a time series with:
- Known ARIMA/SARIMA structure and coefficients
- A known intervention effect added at a specific point
- Random seed fixed at 42 for reproducibility

### Analysis

Both R and Python:
1. Fit an ARIMA model with specified order (auto=FALSE)
2. Compute counterfactual forecasts
3. Estimate causal effects (point, cumulative, temporal average)
4. Compute Gaussian inference (standard errors, p-values via psi weights)

### Comparison Metrics

For each DGP, we compare:
- **Point Effect (tau)**: Final value and full time series correlation
- **Cumulative Effect**: Final value and full time series correlation
- **Temporal Average**: Final value
- **Standard Deviations**: Point, cumulative, and average SD
- **P-values**: Bidirectional p-value for all effect types

### Tolerances

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| Point estimates (values > 1) | 5% relative | Allows minor numeric precision differences |
| Small values (values <= 1) | 0.1 absolute | P-values and near-zero values |
| P-values | 0.01 absolute | Standard significance threshold |
| Time series correlation | > 0.99 | Ensures same trajectory |
| Standard deviations | 20% relative | Wider tolerance due to sigma2 differences (see note) |

**Note on SD tolerance**: Standard deviations depend on the innovation variance (sigma2), which is computed via state-space MLE in Python (statsmodels) vs CSS-ML in R (stats::arima). These methods can produce different sigma2 estimates, especially for models with high differencing (d>=2) or combined regular+seasonal differencing. The 20% tolerance accommodates these cross-library numerical differences while still ensuring the implementations are substantially equivalent.

## Running the Validation

### Option 1: Pytest (Recommended)

```bash
# Run all 211 tests (30 DGPs x 7 test methods + report generator)
pytest tests/test_dgp_cross_validation.py -v

# Run by suite
pytest -m dgp_main -v       # DGPs 1-8
pytest -m dgp_sarima -v     # DGPs 9-18
pytest -m dgp_extended -v   # DGPs 19-30

# Generate master report
pytest tests/test_dgp_cross_validation.py::test_generate_cross_validation_report -v
```

### Option 2: Standalone Scripts

```bash
# Main suite (DGPs 1-8)
cd comparison/dgp_validation
python run_python_analysis.py
python compare_results.py

# SARIMA suite (DGPs 9-18)
cd sarima
python run_python_sarima_analysis.py
python compare_sarima_results.py

# Extended suite (DGPs 19-30)
cd ../extended_dgp
python run_python_extended_analysis.py
python compare_extended_results.py
```

### Regenerating R Results (requires R)

```bash
# Prerequisites: R with jsonlite and forecast packages
# Main suite
Rscript generate_dgp_data.R && Rscript run_r_analysis.R

# SARIMA suite
cd sarima && Rscript generate_sarima_data.R && Rscript run_r_sarima_analysis.R

# Extended suite
cd ../extended_dgp && Rscript generate_extended_data.R && Rscript run_r_extended_analysis.R
```

## File Structure

```
dgp_validation/
├── README.md                    # This file
├── dgp_configs.json             # DGP 1-8 configurations
├── generate_dgp_data.R          # R: generate DGP 1-8 data
├── run_r_analysis.R             # R: run CausalArima on DGPs 1-8
├── run_python_analysis.py       # Python: run pycausalarima on DGPs 1-8
├── compare_results.py           # Generate main suite comparison report
├── comparison_report.md         # Main suite comparison report
├── cross_validation_report.md   # Master report (all 30 DGPs, pytest-generated)
├── data/                        # DGP 1-8 CSV data files
├── results_r/                   # R results for DGPs 1-8
├── results_python/              # Python results for DGPs 1-8
│
├── sarima/                      # SARIMA suite (DGPs 9-18)
│   ├── README.md
│   ├── dgp_sarima_configs.json
│   ├── generate_sarima_data.R
│   ├── run_r_sarima_analysis.R
│   ├── run_python_sarima_analysis.py
│   ├── compare_sarima_results.py
│   ├── comparison_report.md
│   ├── data/
│   ├── results_r/
│   └── results_python/
│
└── extended_dgp/                # Extended suite (DGPs 19-30)
    ├── README.md
    ├── dgp_extended_configs.json
    ├── generate_extended_data.R
    ├── run_r_extended_analysis.R
    ├── run_python_extended_analysis.py
    ├── compare_extended_results.py
    ├── comparison_report.md
    ├── data/
    ├── results_r/
    └── results_python/
```

## Results

- **Master report**: `cross_validation_report.md` (all 30 DGPs)
- **Suite reports**: `comparison_report.md` in each suite directory

**Current status: 30 PASS, 0 FAIL across all DGPs.**

## Interpretation

- **PASS**: R and Python results match within tolerance on all metrics
- **FAIL**: One or more metrics exceed tolerance (investigate further)
