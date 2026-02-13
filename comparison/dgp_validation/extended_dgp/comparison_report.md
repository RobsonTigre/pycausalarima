# Extended DGP: R vs Python Comparison Report

Generated: 2026-02-13 10:23:32

## Summary

| DGP | Name | Effect | Point Effect | Cumulative | Average | P-value | Status |
|-----|------|--------|--------------|------------|---------|---------|--------|
| 19 | ar3 | +10 | OK | OK | OK | OK | **PASS** |
| 20 | ma2 | +12 | OK | OK | OK | OK | **PASS** |
| 21 | arma22 | +10 | OK | OK | OK | OK | **PASS** |
| 22 | arma31 | +15 | OK | OK | OK | OK | **PASS** |
| 23 | arima_d2 | +10 | OK | OK | OK | OK | **PASS** |
| 24 | ar1_small_effect | +2 | OK | OK | OK | OK | **PASS** |
| 25 | ar1_large_effect | +50 | OK | OK | OK | OK | **PASS** |
| 26 | arma11_negative | -15 | OK | OK | OK | OK | **PASS** |
| 27 | ar1_near_unit | +10 | OK | OK | OK | OK | **PASS** |
| 28 | sar1_bimonthly | +10 | OK | OK | OK | OK | **PASS** |
| 29 | sarima_bimonthly | +15 | OK | OK | OK | OK | **PASS** |
| 30 | arma21_seasonal | +12 | OK | OK | OK | OK | **PASS** |

**Total: 12 PASS, 0 FAIL**

## Detailed Results

### Final Values Comparison

| DGP | True Effect | Metric | R | Python | Difference |
|-----|-------------|--------|---|--------|------------|
| 19 | +10 | Point Effect | 11.104 | 11.104 | +0.0000 |
| 19 | | Cumulative | 305.083 | 305.084 | +0.0007 |
| 19 | | Average | 10.169 | 10.169 | +0.0000 |
| 19 | | P-value | 0.0000 | 0.0000 | +0.000000 |
| 20 | +12 | Point Effect | 12.004 | 12.004 | +0.0000 |
| 20 | | Cumulative | 371.471 | 371.471 | +0.0002 |
| 20 | | Average | 12.382 | 12.382 | +0.0000 |
| 20 | | P-value | 0.0000 | 0.0000 | +0.000000 |
| 21 | +10 | Point Effect | 9.488 | 9.488 | -0.0000 |
| 21 | | Cumulative | 337.491 | 337.491 | -0.0002 |
| 21 | | Average | 11.250 | 11.250 | -0.0000 |
| 21 | | P-value | 0.0000 | 0.0000 | +0.000000 |
| 22 | +15 | Point Effect | 12.712 | 12.712 | +0.0000 |
| 22 | | Cumulative | 459.039 | 459.037 | -0.0013 |
| 22 | | Average | 15.301 | 15.301 | -0.0000 |
| 22 | | P-value | 0.0000 | 0.0000 | +0.000000 |
| 23 | +10 | Point Effect | -192.426 | -192.423 | +0.0029 |
| 23 | | Cumulative | -2006.094 | -2006.052 | +0.0419 |
| 23 | | Average | -66.870 | -66.868 | +0.0014 |
| 23 | | P-value | 0.0000 | 0.0000 | +0.000000 |
| 24 | +2 | Point Effect | 2.087 | 2.087 | +0.0000 |
| 24 | | Cumulative | 56.922 | 56.922 | +0.0001 |
| 24 | | Average | 1.897 | 1.897 | +0.0000 |
| 24 | | P-value | 0.0000 | 0.0000 | -0.000000 |
| 25 | +50 | Point Effect | 51.218 | 51.218 | +0.0000 |
| 25 | | Cumulative | 1503.416 | 1503.416 | +0.0002 |
| 25 | | Average | 50.114 | 50.114 | +0.0000 |
| 25 | | P-value | 0.0000 | 0.0000 | +0.000000 |
| 26 | -15 | Point Effect | -14.853 | -14.853 | -0.0000 |
| 26 | | Cumulative | -450.178 | -450.180 | -0.0019 |
| 26 | | Average | -15.006 | -15.006 | -0.0001 |
| 26 | | P-value | 0.0000 | 0.0000 | +0.000000 |
| 27 | +10 | Point Effect | 11.332 | 11.332 | +0.0001 |
| 27 | | Cumulative | 279.793 | 279.794 | +0.0015 |
| 27 | | Average | 9.326 | 9.326 | +0.0000 |
| 27 | | P-value | 0.0000 | 0.0000 | -0.000001 |
| 28 | +10 | Point Effect | 9.873 | 9.873 | +0.0000 |
| 28 | | Cumulative | 239.779 | 239.779 | +0.0003 |
| 28 | | Average | 9.991 | 9.991 | +0.0000 |
| 28 | | P-value | 0.0000 | 0.0000 | +0.000000 |
| 29 | +15 | Point Effect | 16.491 | 16.491 | +0.0000 |
| 29 | | Cumulative | 399.341 | 399.341 | +0.0001 |
| 29 | | Average | 16.639 | 16.639 | +0.0000 |
| 29 | | P-value | 0.0000 | 0.0000 | +0.000000 |
| 30 | +12 | Point Effect | 11.477 | 11.270 | -0.2066 |
| 30 | | Cumulative | 453.122 | 445.548 | -7.5742 |
| 30 | | Average | 12.587 | 12.376 | -0.2104 |
| 30 | | P-value | 0.0000 | 0.0000 | +0.000000 |

### Time Series Correlations

| DGP | Name | Tau Correlation | Cumulative Correlation |
|-----|------|-----------------|------------------------|
| 19 | ar3 | 1.0000 | 1.0000 |
| 20 | ma2 | 1.0000 | 1.0000 |
| 21 | arma22 | 1.0000 | 1.0000 |
| 22 | arma31 | 1.0000 | 1.0000 |
| 23 | arima_d2 | 1.0000 | 1.0000 |
| 24 | ar1_small_effect | 1.0000 | 1.0000 |
| 25 | ar1_large_effect | 1.0000 | 1.0000 |
| 26 | arma11_negative | 1.0000 | 1.0000 |
| 27 | ar1_near_unit | 1.0000 | 1.0000 |
| 28 | sar1_bimonthly | 1.0000 | 1.0000 |
| 29 | sarima_bimonthly | 1.0000 | 1.0000 |
| 30 | arma21_seasonal | 0.9954 | 1.0000 |

## Methodology

### Tolerances Used

- Relative tolerance (for values > 1): 5.0%
- Absolute tolerance (for values <= 1): 0.1
- P-value tolerance: 0.01
- Correlation threshold: 0.99

### DGP Categories Tested

| Category | DGPs | Description |
|----------|------|-------------|
| Higher-order AR | 19, 22 | AR(3), ARMA(3,1) |
| Higher-order MA | 20, 21 | MA(2), ARMA(2,2) |
| Double differencing | 23 | ARIMA(1,2,1) with d=2 |
| Effect sizes | 24, 25, 26 | Small (+2), Large (+50), Negative (-15) |
| Near-unit-root | 27 | AR(1) with phi=0.95 |
| Bimonthly seasonality | 28, 29 | s=6 period |
| Complex combined | 30 | ARMA(2,1) + SARMA(1,1,12) |

### DGP Configurations

| DGP | Order (p,d,q) | Seasonal (P,D,Q,s) | True Effect |
|-----|---------------|---------------------|-------------|
| 19 - ar3 | (3,0,0) | (0,0,0,1) | +10 |
| 20 - ma2 | (0,0,2) | (0,0,0,1) | +12 |
| 21 - arma22 | (2,0,2) | (0,0,0,1) | +10 |
| 22 - arma31 | (3,0,1) | (0,0,0,1) | +15 |
| 23 - arima_d2 | (1,2,1) | (0,0,0,1) | +10 |
| 24 - ar1_small_effect | (1,0,0) | (0,0,0,1) | +2 |
| 25 - ar1_large_effect | (1,0,0) | (0,0,0,1) | +50 |
| 26 - arma11_negative | (1,0,1) | (0,0,0,1) | -15 |
| 27 - ar1_near_unit | (1,0,0) | (0,0,0,1) | +10 |
| 28 - sar1_bimonthly | (0,0,0) | (1,0,0,6) | +10 |
| 29 - sarima_bimonthly | (1,0,1) | (1,0,1,6) | +15 |
| 30 - arma21_seasonal | (2,0,1) | (1,0,1,12) | +12 |