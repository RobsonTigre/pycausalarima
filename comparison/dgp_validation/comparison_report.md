# R vs Python DGP Comparison Report

Generated: 2026-02-13 10:23:30

## Summary

| DGP | Name | Point Effect | Cumulative | Average | P-value | Status |
|-----|------|--------------|------------|---------|---------|--------|
| 1 | ar1 | OK | OK | OK | OK | **PASS** |
| 2 | arima011 | OK | OK | OK | OK | **PASS** |
| 3 | arma11 | OK | OK | OK | OK | **PASS** |
| 4 | ar2 | OK | OK | OK | OK | **PASS** |
| 5 | seasonal_ar_weekly | OK | OK | OK | OK | **PASS** |
| 6 | ar1_seasonal_ma_weekly | OK | OK | OK | OK | **PASS** |
| 7 | seasonal_ar_monthly | OK | OK | OK | OK | **PASS** |
| 8 | arima111 | OK | OK | OK | OK | **PASS** |

**Total: 8 PASS, 0 FAIL**

## Detailed Results

### Final Values Comparison

| DGP | Metric | R | Python | Difference | Rel/Abs Diff |
|-----|--------|---|--------|------------|--------------|
| 1 | Point Effect | 11.998 | 11.998 | -0.000 | 0.0000 |
| 1 | Cumulative | 304.172 | 304.172 | -0.000 | 0.0000 |
| 1 | Average | 10.139 | 10.139 | -0.000 | 0.0000 |
| 2 | Point Effect | 9.750 | 9.750 | +0.000 | 0.0000 |
| 2 | Cumulative | 560.245 | 560.245 | +0.000 | 0.0000 |
| 2 | Average | 18.675 | 18.675 | +0.000 | 0.0000 |
| 3 | Point Effect | 11.573 | 11.573 | +0.000 | 0.0000 |
| 3 | Cumulative | 312.902 | 312.903 | +0.001 | 0.0000 |
| 3 | Average | 10.430 | 10.430 | +0.000 | 0.0000 |
| 4 | Point Effect | 12.083 | 12.083 | +0.000 | 0.0000 |
| 4 | Cumulative | 363.383 | 363.383 | +0.000 | 0.0000 |
| 4 | Average | 12.113 | 12.113 | +0.000 | 0.0000 |
| 5 | Point Effect | 18.041 | 18.041 | +0.000 | 0.0000 |
| 5 | Cumulative | 569.842 | 569.842 | +0.001 | 0.0000 |
| 5 | Average | 20.351 | 20.352 | +0.000 | 0.0000 |
| 6 | Point Effect | 16.221 | 16.221 | -0.000 | 0.0000 |
| 6 | Cumulative | 395.637 | 395.636 | -0.000 | 0.0000 |
| 6 | Average | 14.130 | 14.130 | -0.000 | 0.0000 |
| 7 | Point Effect | 12.676 | 12.676 | +0.000 | 0.0000 |
| 7 | Cumulative | 243.363 | 243.363 | +0.000 | 0.0000 |
| 7 | Average | 10.140 | 10.140 | +0.000 | 0.0000 |
| 8 | Point Effect | -4.418 | -4.418 | -0.000 | 0.0000 |
| 8 | Cumulative | -43.936 | -43.939 | -0.003 | 0.0001 |
| 8 | Average | -1.465 | -1.465 | -0.000 | 0.0001 |

### Time Series Correlations

| DGP | Name | Tau Correlation | Cumulative Correlation |
|-----|------|-----------------|----------------------|
| 1 | ar1 | 1.0000 | 1.0000 |
| 2 | arima011 | 1.0000 | 1.0000 |
| 3 | arma11 | 1.0000 | 1.0000 |
| 4 | ar2 | 1.0000 | 1.0000 |
| 5 | seasonal_ar_weekly | 1.0000 | 1.0000 |
| 6 | ar1_seasonal_ma_weekly | 1.0000 | 1.0000 |
| 7 | seasonal_ar_monthly | 1.0000 | 1.0000 |
| 8 | arima111 | 1.0000 | 1.0000 |

## Methodology

### Tolerances Used

- Relative tolerance (for values > 1): 5.0%
- Absolute tolerance (for values <= 1): 0.1
- P-value tolerance: 0.01
- Correlation threshold: 0.99

### Interpretation

- **PASS**: All metrics match within tolerance
- **FAIL**: One or more metrics exceed tolerance

### DGP Configurations

| DGP | Order (p,d,q) | Seasonal (P,D,Q,s) | True Effect |
|-----|---------------|---------------------|-------------|
| 1 - ar1 | (1,0,0) | (0,0,0,1) | +10 |
| 2 - arima011 | (0,1,1) | (0,0,0,1) | +15 |
| 3 - arma11 | (1,0,1) | (0,0,0,1) | +10 |
| 4 - ar2 | (2,0,0) | (0,0,0,1) | +12 |
| 5 - seasonal_ar_weekly | (0,0,0) | (1,0,0,7) | +20 |
| 6 - ar1_seasonal_ma_weekly | (1,0,0) | (0,0,1,7) | +15 |
| 7 - seasonal_ar_monthly | (0,0,0) | (1,0,0,12) | +10 |
| 8 - arima111 | (1,1,1) | (0,0,0,1) | +8 |