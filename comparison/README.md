# R/Python Validation Suite

This folder contains validation tests that prove `pycausalarima` (Python) matches `CausalArima` (R).

## Test Coverage: 30 DGPs

| Suite | DGPs | Location | Coverage |
|-------|------|----------|----------|
| **Main** | 1-8 | [dgp_validation/](dgp_validation/) | AR, MA, ARIMA, basic seasonal |
| **SARIMA** | 9-18 | [dgp_validation/sarima/](dgp_validation/sarima/) | Seasonal differencing, full SARIMA |
| **Extended** | 19-30 | [dgp_validation/extended_dgp/](dgp_validation/extended_dgp/) | Higher-order, edge cases, bimonthly |

## Results Summary

**All 30 DGPs PASS validation:**
- Time series correlations > 0.9999
- Point/cumulative/average effects match within 5% relative tolerance
- P-values match within 0.01 absolute tolerance

## Quick Start

### Pytest cross-validation (recommended, no R required)

```bash
# All 30 DGPs
pytest tests/test_dgp_cross_validation.py -v

# Filter by suite
pytest -m dgp_main -v        # DGPs 1-8
pytest -m dgp_sarima -v      # DGPs 9-18
pytest -m dgp_extended -v    # DGPs 19-30
```

### Standalone scripts (requires R)

```bash
# Main suite (DGPs 1-8)
cd dgp_validation
Rscript generate_dgp_data.R && Rscript run_r_analysis.R
python run_python_analysis.py && python compare_results.py

# SARIMA suite (DGPs 9-18)
cd sarima
Rscript generate_sarima_data.R && Rscript run_r_sarima_analysis.R
python run_python_sarima_analysis.py && python compare_sarima_results.py

# Extended suite (DGPs 19-30)
cd ../extended_dgp
Rscript generate_extended_data.R && Rscript run_r_extended_analysis.R
python run_python_extended_analysis.py && python compare_extended_results.py
```

## Folder Structure

```
comparison/
├── README.md                    # This file
├── dgp_validation/
│   ├── README.md                # Main suite documentation
│   ├── dgp_configs.json         # DGP 1-8 configurations
│   ├── generate_dgp_data.R      # Data generation
│   ├── run_r_analysis.R         # R analysis
│   ├── run_python_analysis.py   # Python analysis
│   ├── compare_results.py       # Comparison script
│   ├── comparison_report.md     # Standalone comparison report
│   ├── cross_validation_report.md # Pytest-generated report
│   ├── data/                    # Generated CSV files
│   ├── results_r/               # R output
│   ├── results_python/          # Python output
│   │
│   ├── sarima/                  # SARIMA suite (DGPs 9-18)
│   │   └── ... (same structure)
│   │
│   └── extended_dgp/            # Extended suite (DGPs 19-30)
│       └── ... (same structure)
```

## Model Coverage Matrix

| Component | DGPs Tested | Status |
|-----------|-------------|--------|
| AR(p), p=1-3 | 1, 4, 19, 22 | PASS |
| MA(q), q=1-2 | 2, 20 | PASS |
| ARIMA (d=1) | 2, 8 | PASS |
| ARIMA (d=2) | 23 | PASS |
| SAR(P), P=1-2 | 5, 7, 16, 28 | PASS |
| SMA(Q), Q=1-2 | 6, 11, 12 | PASS |
| Seasonal D=1 | 9, 10, 14, 15, 17 | PASS |
| Period s=4 | 10, 15 | PASS |
| Period s=6 | 28, 29 | PASS |
| Period s=7 | 5, 6, 12, 18 | PASS |
| Period s=12 | 7, 9, 11, 13, 14, 16, 17, 30 | PASS |
| Negative effects | 26 | PASS |
| Near-unit-root | 27 | PASS |

## See Also

- [VALIDATION.md](../VALIDATION.md) - Top-level validation overview
- [tests/test_dgp_cross_validation.py](../tests/test_dgp_cross_validation.py) - Pytest cross-validation (211 tests)
- [tests/test_r_comparison.py](../tests/test_r_comparison.py) - Deterministic R comparison test
- [cross_validation_report.md](dgp_validation/cross_validation_report.md) - Detailed results
