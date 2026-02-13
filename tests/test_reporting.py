"""Tests for reporting module (tables and summary).

These tests cover the HTML/LaTeX export functionality and summary generation
to improve coverage of reporting/tables.py and reporting/summary.py.
"""

import numpy as np
import pandas as pd
import pytest

from pycausalarima import CausalArima


@pytest.fixture
def fitted_model():
    """Create a fitted CausalArima model for testing."""
    np.random.seed(42)
    n = 100
    y = np.cumsum(np.random.normal(0, 1, n)) + 100
    y[70:] += 10
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    ca = CausalArima(y=y, dates=dates, intervention_date=dates[70])
    ca.fit()
    return ca


@pytest.fixture
def fitted_model_with_bootstrap():
    """Create a fitted model with bootstrap inference."""
    np.random.seed(42)
    n = 100
    y = np.cumsum(np.random.normal(0, 1, n)) + 100
    y[70:] += 10
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    ca = CausalArima(y=y, dates=dates, intervention_date=dates[70], n_boot=100)
    ca.fit()
    return ca


class TestImpactTables:
    """Tests for impact table generation."""

    def test_impact_numeric_format(self, fitted_model):
        """Impact tables should return DataFrames in numeric format."""
        result = fitted_model.impact(format="numeric")
        assert isinstance(result, dict)
        assert "arima" in result
        assert "impact_norm" in result

    def test_impact_html_format(self, fitted_model):
        """Impact tables should convert to HTML strings."""
        result = fitted_model.impact(format="html")
        assert isinstance(result, dict)
        # HTML tables should contain <table> tag
        norm_html = result["impact_norm"]["average"]
        assert "<table" in norm_html

    def test_impact_latex_format(self, fitted_model):
        """Impact tables should convert to LaTeX strings."""
        result = fitted_model.impact(format="latex")
        assert isinstance(result, dict)
        # LaTeX tables should contain tabular or begin
        norm_latex = result["impact_norm"]["average"]
        assert "tabular" in norm_latex or "\\begin" in norm_latex

    def test_impact_invalid_format_raises_error(self, fitted_model):
        """Invalid format should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown format"):
            fitted_model.impact(format="invalid")

    def test_impact_with_horizon(self, fitted_model):
        """Impact with specific horizon dates."""
        horizon = [fitted_model.result_.dates[80]]
        result = fitted_model.impact(format="numeric", horizon=horizon)
        assert isinstance(result, dict)
        assert "impact_norm" in result

    def test_impact_arima_tables_structure(self, fitted_model):
        """ARIMA tables should have expected structure."""
        result = fitted_model.impact(format="numeric")
        arima = result["arima"]
        assert "arima_order" in arima
        assert isinstance(arima["arima_order"], pd.DataFrame)

    def test_impact_norm_tables_structure(self, fitted_model):
        """Normal inference tables should have expected structure."""
        result = fitted_model.impact(format="numeric")
        norm = result["impact_norm"]
        assert "point_effect" in norm
        assert "sum" in norm
        assert "average" in norm
        assert isinstance(norm["average"], pd.DataFrame)


class TestImpactTablesWithBootstrap:
    """Tests for impact tables with bootstrap inference."""

    def test_boot_tables_generated(self, fitted_model_with_bootstrap):
        """Bootstrap tables should be generated when n_boot > 0."""
        result = fitted_model_with_bootstrap.impact(format="numeric")
        assert result["impact_boot"] is not None

    def test_boot_tables_html_format(self, fitted_model_with_bootstrap):
        """Bootstrap tables should convert to HTML."""
        result = fitted_model_with_bootstrap.impact(format="html")
        assert result["impact_boot"] is not None

    def test_boot_tables_latex_format(self, fitted_model_with_bootstrap):
        """Bootstrap tables should convert to LaTeX."""
        result = fitted_model_with_bootstrap.impact(format="latex")
        assert result["impact_boot"] is not None

    def test_boot_tables_with_horizon(self, fitted_model_with_bootstrap):
        """Bootstrap tables should work with horizon."""
        horizon = [fitted_model_with_bootstrap.result_.dates[80]]
        result = fitted_model_with_bootstrap.impact(format="numeric", horizon=horizon)
        assert result["impact_boot"] is not None


class TestSummary:
    """Tests for summary generation."""

    def test_summary_norm_type(self, fitted_model):
        """Summary with norm type should return DataFrame."""
        result = fitted_model.summary(type="norm")
        assert isinstance(result, pd.DataFrame)
        assert "Point causal effect" in result.index

    def test_summary_boot_type(self, fitted_model_with_bootstrap):
        """Summary with boot type should work when bootstrap available."""
        result = fitted_model_with_bootstrap.summary(type="boot")
        assert isinstance(result, pd.DataFrame)
        assert "Point causal effect" in result.index

    def test_summary_boot_without_bootstrap_raises_error(self, fitted_model):
        """Summary boot type without bootstrap should raise error."""
        with pytest.raises(ValueError, match="Bootstrap inference not available"):
            fitted_model.summary(type="boot")

    def test_summary_invalid_type_raises_error(self, fitted_model):
        """Summary with invalid type should raise error."""
        with pytest.raises(ValueError, match="Unknown inference type"):
            fitted_model.summary(type="invalid")

    def test_summary_with_horizon(self, fitted_model):
        """Summary with horizon should include specific dates."""
        horizon = [fitted_model.result_.dates[80], fitted_model.result_.dates[90]]
        result = fitted_model.summary(horizon=horizon)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 2


