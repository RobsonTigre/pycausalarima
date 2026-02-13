# SARIMA Validation: R vs Python Comparison

This folder contains a comprehensive validation suite for SARIMA models, comparing `pycausalarima` (Python) against `CausalArima` (R).

## Purpose

This test suite specifically addresses gaps in the main DGP validation:

| Gap | Status |
|-----|--------|
| Seasonal differencing (D > 0) | Tested in DGPs 9, 10, 14, 15, 17 |
| Pure seasonal MA (SMA only) | Tested in DGPs 11, 12 |
| Full SARIMA (all components) | Tested in DGPs 13, 18 |
| Combined d AND D differencing | Tested in DGPs 14, 15 |
| Quarterly seasonality (s=4) | Tested in DGPs 10, 15 |
| Higher-order seasonal | Tested in DGPs 12, 16 |

## DGP Configurations

| DGP | Name | Order (p,d,q) | Seasonal (P,D,Q,s) | Effect | Key Feature |
|-----|------|---------------|---------------------|--------|-------------|
| 9 | sar1_D1_monthly | (0,0,0) | (1,1,0,12) | +12 | SAR with D=1 |
| 10 | sma1_D1_quarterly | (0,0,0) | (0,1,1,4) | +8 | SMA with D=1, quarterly |
| 11 | pure_sma1_monthly | (0,0,0) | (0,0,1,12) | +10 | Pure SMA only |
| 12 | pure_sma2_weekly | (0,0,0) | (0,0,2,7) | +15 | Higher-order SMA |
| 13 | sarima_full_monthly | (1,0,1) | (1,0,1,12) | +10 | All 4 components |
| 14 | sarima_d1_D1 | (1,1,0) | (0,1,1,12) | +15 | Both d=1 AND D=1 |
| 15 | sarima_quarterly | (0,1,1) | (1,1,0,4) | +10 | Quarterly with d+D |
| 16 | sar2_monthly | (0,0,0) | (2,0,0,12) | +12 | Higher-order SAR |
| 17 | sarma_D1_monthly | (0,0,0) | (1,1,1,12) | +10 | SARMA with D=1 |
| 18 | sarima_full_weekly | (1,0,1) | (1,0,1,7) | +20 | Full weekly SARIMA |

## File Structure

```
sarima/
├── README.md                      # This file
├── dgp_sarima_configs.json        # DGP configuration file
├── generate_sarima_data.R         # R script to generate SARIMA data
├── run_r_sarima_analysis.R        # R CausalArima analysis
├── run_python_sarima_analysis.py  # Python pycausalarima analysis
├── compare_sarima_results.py      # Comparison and report generation
├── data/                          # Generated CSV data files
├── results_r/                     # R analysis results
├── results_python/                # Python analysis results
└── comparison_report.md           # Generated comparison report
```

## Execution

### Prerequisites

- R with packages: `jsonlite`, `forecast`
- R CausalArima package (from FMenchetti repo at `../../reference_r_package/`)
- Python with: `numpy`, `pandas`, `pycausalarima`

### Run the Validation

```bash
cd comparison/dgp_validation/sarima

# 1. Generate SARIMA data (R)
Rscript generate_sarima_data.R

# 2. Run R analysis
Rscript run_r_sarima_analysis.R

# 3. Run Python analysis
python run_python_sarima_analysis.py

# 4. Generate comparison report
python compare_sarima_results.py
```

## Validation Criteria

| Metric | Tolerance |
|--------|-----------|
| Values > 1 | 5% relative difference |
| Values ≤ 1 | 0.1 absolute difference |
| Correlation | > 0.99 |

## Expected Results

All 10 DGPs should **PASS** validation with:
- Effect estimates matching within 5%
- P-values matching within 0.01
- Time series correlations > 0.99

## Technical Details

### Seasonal Differencing (D > 0)

For a process with seasonal differencing, the series Y_t satisfies:
```
(1 - B^s)^D Y_t = W_t
```

Where W_t follows a SARMA process. To generate Y_t:
1. Generate SARMA innovations W_t
2. Integrate back: Y_t = Y_{t-s} + W_t (repeated D times)

### Coefficient Constraints

All coefficients are chosen to ensure:
- **Stationarity**: AR/SAR roots outside unit circle
- **Invertibility**: MA/SMA roots outside unit circle
- **Stability**: |coefficient| < 0.6

### Sample Sizes

| Period | Min n | Pre-intervention |
|--------|-------|------------------|
| s=4 (quarterly) | 80 | 60 |
| s=7 (weekly) | 140 | 112 |
| s=12 (monthly) | 120 | 96 |

## Comparison with Main DGP Validation

This suite complements the main DGP validation (`../`) which tests:
- Non-seasonal ARIMA (DGPs 1-4, 8)
- Basic seasonal AR (DGPs 5-7)

Together, they provide comprehensive coverage of ARIMA/SARIMA model validation.
