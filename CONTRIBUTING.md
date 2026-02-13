# Contributing to pycausalarima

Thank you for your interest in contributing to pycausalarima!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/RobsonTigre/pycausalarima.git
cd pycausalarima
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pycausalarima --cov-report=html

# Run specific test file
pytest tests/test_causal_arima.py -v
```

## Code Style

This project uses:
- **black** for code formatting
- **isort** for import sorting
- **ruff** for linting
- **mypy** for type checking

Run all checks:
```bash
black pycausalarima tests
isort pycausalarima tests
ruff check pycausalarima tests
mypy pycausalarima
```

## R/Python Validation

When modifying core algorithms, run the validation suite to ensure results still match R. There are three suites covering 30 DGPs total. See [VALIDATION.md](VALIDATION.md) for full details.

**Main Suite (DGPs 1-8):**
```bash
cd comparison/dgp_validation
python run_python_analysis.py
python compare_results.py
# Check comparison_report.md for PASS/FAIL status
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

**Quick check (unit tests only):**
```bash
pytest tests/test_r_comparison.py -v
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Run the full test suite and linting
4. Update documentation if needed
5. Submit a pull request with a clear description

## Reporting Issues

Please include:
- Python version and OS
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback if applicable

## Questions?

Open an issue for questions or discussion.
