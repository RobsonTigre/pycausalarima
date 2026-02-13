# R vs Python SARIMA DGP Comparison Report

Generated: 2026-02-13 10:23:31

## Summary

| DGP | Name | Point Effect | Cumulative | Average | P-value | Status |
|-----|------|--------------|------------|---------|---------|--------|
| 9 | sar1_D1_monthly | OK | OK | OK | OK | **PASS** |
| 10 | sma1_D1_quarterly | OK | OK | OK | OK | **PASS** |
| 11 | pure_sma1_monthly | OK | OK | OK | OK | **PASS** |
| 12 | pure_sma2_weekly | OK | OK | OK | OK | **PASS** |
| 13 | sarima_full_monthly | OK | OK | OK | OK | **PASS** |
| 14 | sarima_d1_D1 | OK | OK | OK | OK | **PASS** |
| 15 | sarima_quarterly | OK | OK | OK | OK | **PASS** |
| 16 | sar2_monthly | OK | OK | OK | OK | **PASS** |
| 17 | sarma_D1_monthly | OK | OK | OK | OK | **PASS** |
| 18 | sarima_full_weekly | OK | OK | OK | OK | **PASS** |

**Total: 10 PASS, 0 FAIL**

## Detailed Results

### Final Values Comparison

| DGP | Metric | R | Python | Difference | Rel/Abs Diff |
|-----|--------|---|--------|------------|--------------|
| 9 | Point Effect | 12.661 | 12.661 | -0.000 | 0.0000 |
| 9 | Cumulative | 436.678 | 436.678 | -0.000 | 0.0000 |
| 9 | Average | 12.130 | 12.130 | -0.000 | 0.0000 |
| 10 | Point Effect | 12.368 | 12.368 | -0.000 | 0.0000 |
| 10 | Cumulative | 166.465 | 166.465 | +0.000 | 0.0000 |
| 10 | Average | 8.323 | 8.323 | +0.000 | 0.0000 |
| 11 | Point Effect | 12.003 | 12.003 | +0.000 | 0.0000 |
| 11 | Cumulative | 249.024 | 249.024 | +0.000 | 0.0000 |
| 11 | Average | 10.376 | 10.376 | +0.000 | 0.0000 |
| 12 | Point Effect | 14.390 | 14.390 | +0.000 | 0.0000 |
| 12 | Cumulative | 412.112 | 412.112 | +0.000 | 0.0000 |
| 12 | Average | 14.718 | 14.718 | +0.000 | 0.0000 |
| 13 | Point Effect | 8.796 | 8.796 | -0.000 | 0.0000 |
| 13 | Cumulative | 203.629 | 203.629 | -0.000 | 0.0000 |
| 13 | Average | 8.485 | 8.485 | -0.000 | 0.0000 |
| 14 | Point Effect | -4.649 | -4.651 | -0.002 | 0.0003 |
| 14 | Cumulative | 220.485 | 220.453 | -0.032 | 0.0001 |
| 14 | Average | 6.125 | 6.124 | -0.001 | 0.0001 |
| 15 | Point Effect | 26.301 | 26.301 | +0.001 | 0.0000 |
| 15 | Cumulative | 272.132 | 272.138 | +0.006 | 0.0000 |
| 15 | Average | 13.607 | 13.607 | +0.000 | 0.0000 |
| 16 | Point Effect | 12.667 | 12.668 | +0.000 | 0.0000 |
| 16 | Cumulative | 290.181 | 290.181 | +0.000 | 0.0000 |
| 16 | Average | 12.091 | 12.091 | +0.000 | 0.0000 |
| 17 | Point Effect | 9.785 | 9.785 | -0.000 | 0.0000 |
| 17 | Cumulative | 371.631 | 371.631 | -0.000 | 0.0000 |
| 17 | Average | 10.323 | 10.323 | -0.000 | 0.0000 |
| 18 | Point Effect | 22.315 | 22.315 | -0.000 | 0.0000 |
| 18 | Cumulative | 568.495 | 568.489 | -0.005 | 0.0000 |
| 18 | Average | 20.303 | 20.303 | -0.000 | 0.0000 |

### Time Series Correlations

| DGP | Name | Tau Correlation | Cumulative Correlation |
|-----|------|-----------------|----------------------|
| 9 | sar1_D1_monthly | 1.0000 | 1.0000 |
| 10 | sma1_D1_quarterly | 1.0000 | 1.0000 |
| 11 | pure_sma1_monthly | 1.0000 | 1.0000 |
| 12 | pure_sma2_weekly | 1.0000 | 1.0000 |
| 13 | sarima_full_monthly | 1.0000 | 1.0000 |
| 14 | sarima_d1_D1 | 1.0000 | 1.0000 |
| 15 | sarima_quarterly | 1.0000 | 1.0000 |
| 16 | sar2_monthly | 1.0000 | 1.0000 |
| 17 | sarma_D1_monthly | 1.0000 | 1.0000 |
| 18 | sarima_full_weekly | 1.0000 | 1.0000 |

## Methodology

### Tolerances Used

- Relative tolerance (for values > 1): 5.0%
- Absolute tolerance (for values <= 1): 0.1
- Correlation threshold: 0.99

### Interpretation

- **PASS**: All metrics match within tolerance
- **FAIL**: One or more metrics exceed tolerance

### DGP Configurations

| DGP | Order (p,d,q) | Seasonal (P,D,Q,s) | True Effect |
|-----|---------------|---------------------|-------------|
| 9 - sar1_D1_monthly | (0,0,0) | (1,1,0,12) | +12 |
| 10 - sma1_D1_quarterly | (0,0,0) | (0,1,1,4) | +8 |
| 11 - pure_sma1_monthly | (0,0,0) | (0,0,1,12) | +10 |
| 12 - pure_sma2_weekly | (0,0,0) | (0,0,2,7) | +15 |
| 13 - sarima_full_monthly | (1,0,1) | (1,0,1,12) | +10 |
| 14 - sarima_d1_D1 | (1,1,0) | (0,1,1,12) | +15 |
| 15 - sarima_quarterly | (0,1,1) | (1,1,0,4) | +10 |
| 16 - sar2_monthly | (0,0,0) | (2,0,0,12) | +12 |
| 17 - sarma_D1_monthly | (0,0,0) | (1,1,1,12) | +10 |
| 18 - sarima_full_weekly | (1,0,1) | (1,0,1,7) | +20 |