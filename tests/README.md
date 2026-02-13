# pycausalarima Test Suite

## Overview

This directory contains the pytest test suite for pycausalarima, including unit tests, integration tests, and cross-validation tests against the R CausalArima reference implementation.

## Test Files

| File | Purpose |
|------|---------|
| `conftest.py` | Shared fixtures and DGP cross-validation registry |
| `test_dgp_cross_validation.py` | Cross-validation: Python vs R across all 30 DGPs |
| `test_r_comparison.py` | Single deterministic dataset comparison against R |
| `test_causal_arima.py` | Core CausalArima unit tests |
| `test_seasonal.py` | Seasonal ARIMA model tests |
| `test_arma_utils.py` | ARMA utility function tests |
| `test_edge_cases.py` | Edge case and boundary condition tests |
| `test_reporting.py` | Summary and table reporting tests |
| `test_visualization.py` | Plotting tests |

## Running Tests

```bash
# All tests
pytest tests/

# Cross-validation only (all 30 DGPs)
pytest tests/test_dgp_cross_validation.py -v

# Filter by suite
pytest -m dgp_main -v        # DGPs 1-8 (basic ARIMA)
pytest -m dgp_sarima -v      # DGPs 9-18 (seasonal differencing)
pytest -m dgp_extended -v    # DGPs 19-30 (edge cases, higher-order)

# Single DGP
pytest tests/test_dgp_cross_validation.py -v -k "dgp_01_ar1"

# Generate comparison report only
pytest tests/test_dgp_cross_validation.py::test_generate_cross_validation_report -v
```

## Cross-Validation Tests (`test_dgp_cross_validation.py`)

### What Is Tested

30 Data Generating Processes (DGPs) spanning three suites:

- **Main (DGPs 1-8):** AR(1), ARIMA(0,1,1), ARMA(1,1), AR(2), seasonal AR weekly, AR(1)+seasonal MA weekly, seasonal AR monthly, ARIMA(1,1,1)
- **SARIMA (DGPs 9-18):** Seasonal differencing (D=1), quarterly models, full SARIMA, combined differencing (d+D), seasonal AR(2), seasonal ARMA
- **Extended (DGPs 19-30):** AR(3), MA(2), ARMA(2,2), double differencing (d=2), small/large/negative effects, near-unit root (phi=0.95), bimonthly seasonality (s=6), complex ARMA+SARMA

For each DGP, 7 metrics are compared against R:

| Metric | Tolerance |
|--------|-----------|
| Point effect (tau at final time) | 5% relative (when \|value\| > 1) or 0.1 absolute |
| Cumulative effect | 5% relative or 0.1 absolute |
| Temporal average effect | 5% relative or 0.1 absolute |
| Standard deviations (point, cumulative, average) | 5% relative or 0.1 absolute |
| Bidirectional p-values (point, cumulative, average) | 0.01 absolute |
| Tau time series correlation | Pearson > 0.99 |
| Cumulative time series correlation | Pearson > 0.99 |

### Prerequisites

R is **not** required at test runtime. Tests use pre-computed R results stored in:

- `comparison/dgp_validation/results_r/` (DGPs 1-8)
- `comparison/dgp_validation/sarima/results_r/` (DGPs 9-18)
- `comparison/dgp_validation/extended_dgp/results_r/` (DGPs 19-30)

Pre-generated DGP data is in the corresponding `data/` subdirectories.

To regenerate R reference results (requires R with CausalArima installed):

```bash
cd comparison/dgp_validation
Rscript generate_dgp_data.R && Rscript run_r_analysis.R

cd sarima
Rscript generate_sarima_data.R && Rscript run_r_sarima_analysis.R

cd ../extended_dgp
Rscript generate_extended_data.R && Rscript run_r_extended_analysis.R
```

### Interpreting Results

- **PASSED**: Python result matches R within tolerance
- **FAILED**: Discrepancy exceeds tolerance — investigate the specific metric and DGP
- **SKIPPED**: Data files or R results not found (run the R scripts first)

### Comparison Report

After running the cross-validation suite, a markdown report is generated at `comparison/dgp_validation/cross_validation_report.md` with:

1. Summary table (pass/fail per DGP with all metrics)
2. Detailed values (R vs Python side-by-side)
3. Time series correlations
4. DGP configurations reference
