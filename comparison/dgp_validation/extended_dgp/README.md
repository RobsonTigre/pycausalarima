# Extended DGP Validation: R vs Python Comparison

This folder contains validation tests for extended ARIMA/SARIMA configurations comparing `pycausalarima` (Python) against `CausalArima` (R).

## Purpose

This test suite covers configurations **NOT tested** in the main (DGPs 1-8) or SARIMA (DGPs 9-18) suites. It focuses on:

| Category | DGPs | Key Features |
|----------|------|--------------|
| Higher-order non-seasonal AR | 19, 22 | AR(3), ARMA(3,1) |
| Higher-order non-seasonal MA | 20, 21 | MA(2), ARMA(2,2) |
| Double differencing | 23 | ARIMA(1,2,1) with d=2 |
| Effect size variations | 24, 25, 26 | Small (+2), Large (+50), Negative (-15) |
| Near-unit-root | 27 | AR(1) with phi=0.95 |
| Bimonthly seasonality | 28, 29 | s=6 period (not previously tested) |
| Complex combined | 30 | High-order AR+MA + seasonal components |

## DGP Configurations

| ID | Name | Order (p,d,q) | Seasonal (P,D,Q,s) | Effect | Description |
|----|------|---------------|---------------------|--------|-------------|
| 19 | ar3 | (3,0,0) | (0,0,0,1) | +10 | Higher-order AR(3) |
| 20 | ma2 | (0,0,2) | (0,0,0,1) | +12 | Pure MA(2) |
| 21 | arma22 | (2,0,2) | (0,0,0,1) | +10 | ARMA(2,2) mixed model |
| 22 | arma31 | (3,0,1) | (0,0,0,1) | +15 | High AR + MA |
| 23 | arima_d2 | (1,2,1) | (0,0,0,1) | +10 | Double differencing (d=2) |
| 24 | ar1_small_effect | (1,0,0) | (0,0,0,1) | +2 | Small positive effect |
| 25 | ar1_large_effect | (1,0,0) | (0,0,0,1) | +50 | Large positive effect |
| 26 | arma11_negative | (1,0,1) | (0,0,0,1) | -15 | Negative effect |
| 27 | ar1_near_unit | (1,0,0) | (0,0,0,1) | +10 | Near unit root (phi=0.95) |
| 28 | sar1_bimonthly | (0,0,0) | (1,0,0,6) | +10 | Bimonthly seasonality |
| 29 | sarima_bimonthly | (1,0,1) | (1,0,1,6) | +15 | Full SARIMA with s=6 |
| 30 | arma21_seasonal | (2,0,1) | (1,0,1,12) | +12 | Complex combined model |

## Technical Considerations

### Near-Unit-Root Models (DGP 27)
- AR coefficient at 0.95 (close to non-stationarity boundary)
- Tests model stability and forecast uncertainty handling
- Important for economic/financial time series applications

### Higher Differencing (DGP 23)
- d=2 models (I(2) processes)
- Tests proper double integration handling
- Longer sample sizes (n=150) for numerical stability

### Effect Size Sensitivity (DGPs 24-26)
- **Small effect (+2)**: Tests detectability at low signal-to-noise ratio
- **Large effect (+50)**: Tests numerical stability with large magnitudes
- **Negative effect (-15)**: Tests proper sign handling and p-value computation

### Bimonthly Seasonality (DGPs 28-29)
- s=6 period (not tested in other suites)
- Verifies both R and Python handle non-standard seasonal periods

## Execution

### Step 1: Generate DGP Data
```bash
cd comparison/dgp_validation/extended_dgp
Rscript generate_extended_data.R
```

### Step 2: Run R Analysis
```bash
Rscript run_r_extended_analysis.R
```

### Step 3: Run Python Analysis
```bash
python run_python_extended_analysis.py
```

### Step 4: Compare Results
```bash
python compare_extended_results.py
```

## File Structure

```
extended_dgp/
├── README.md                           # This file
├── dgp_extended_configs.json           # DGP configurations
├── generate_extended_data.R            # R data generation
├── run_r_extended_analysis.R           # R CausalArima analysis
├── run_python_extended_analysis.py     # Python pycausalarima analysis
├── compare_extended_results.py         # Comparison and reporting
├── data/                               # Generated CSV data files
│   ├── dgp_19_ar3.csv
│   └── ... (12 files total)
├── results_r/                          # R analysis results
│   ├── dgp_19_norm.csv                 # Time series results
│   ├── dgp_19_final.csv                # Summary metrics
│   ├── summary.csv                     # All DGPs summary
│   └── ...
├── results_python/                     # Python analysis results
│   ├── dgp_19_norm.csv
│   ├── dgp_19_final.csv
│   ├── summary.csv
│   └── ...
└── comparison_report.md                # Final comparison report
```

## Comparison Metrics

| Metric | Tolerance | Description |
|--------|-----------|-------------|
| Point Effect | 5% relative | Final period causal effect |
| Cumulative Effect | 5% relative | Sum of effects |
| Average Effect | 5% relative | Mean effect per period |
| P-value | 0.01 absolute | Bidirectional significance |
| Time Series Correlation | > 0.99 | Tau and cumulative series |

## Expected Results

All 12 DGPs should **PASS** validation, demonstrating that:
1. Python and R implementations produce equivalent results
2. Higher-order models are handled correctly
3. Double differencing (d=2) works properly
4. Effect size variations don't affect accuracy
5. Near-unit-root models remain stable
6. Non-standard seasonal periods (s=6) are supported
