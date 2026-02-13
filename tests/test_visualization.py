"""Tests for visualization module.

These are smoke tests to ensure plotting functions don't crash
and return the expected types. We don't test matplotlib internals.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pycausalarima import CausalArima


@pytest.fixture
def fitted_model():
    """Create a fitted CausalArima model for testing plots.

    Uses 100 observations with 70 pre-intervention to ensure:
    - Enough samples for PACF (max_lags=30 requires >60 samples)
    - Robust model fitting
    """
    np.random.seed(42)
    n = 100
    y = np.cumsum(np.random.normal(0, 1, n)) + 100
    y[70:] += 10  # Add intervention effect

    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    intervention_date = dates[70]

    ca = CausalArima(
        y=y,
        dates=dates,
        intervention_date=intervention_date,
    )
    ca.fit()
    return ca


class TestForecastPlot:
    """Tests for forecast plot generation."""

    def test_forecast_plot_returns_figure(self, fitted_model):
        """Forecast plot should return a matplotlib Figure."""
        fig = fitted_model.plot(type="forecast")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_forecast_plot_has_axes(self, fitted_model):
        """Forecast plot figure should have axes."""
        fig = fitted_model.plot(type="forecast")
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_forecast_plot_with_custom_kwargs(self, fitted_model):
        """Forecast plot should accept custom kwargs."""
        # This tests that kwargs are passed through without error
        fig = fitted_model.plot(type="forecast", figsize=(10, 6))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestImpactPlot:
    """Tests for impact plot generation."""

    def test_impact_plot_returns_dict(self, fitted_model):
        """Impact plot should return a dict with expected keys."""
        result = fitted_model.plot(type="impact")
        assert isinstance(result, dict)
        assert "plot" in result
        assert "cumulative_plot" in result
        plt.close('all')

    def test_impact_plot_figures_are_valid(self, fitted_model):
        """Impact plot dict values should be matplotlib Figures."""
        result = fitted_model.plot(type="impact")
        assert isinstance(result["plot"], plt.Figure)
        assert isinstance(result["cumulative_plot"], plt.Figure)
        plt.close('all')

    def test_impact_plot_has_axes(self, fitted_model):
        """Impact plot figures should have axes."""
        result = fitted_model.plot(type="impact")
        assert len(result["plot"].axes) > 0
        assert len(result["cumulative_plot"].axes) > 0
        plt.close('all')


class TestResidualsPlot:
    """Tests for residuals diagnostic plot generation."""

    def test_residuals_plot_returns_dict(self, fitted_model):
        """Residuals plot should return a dict."""
        result = fitted_model.plot(type="residuals")
        assert isinstance(result, dict)
        plt.close('all')

    def test_residuals_plot_has_expected_keys(self, fitted_model):
        """Residuals plot dict should have diagnostic plot keys."""
        result = fitted_model.plot(type="residuals")
        # Should have ACF, PACF, and possibly Q-Q plot
        assert len(result) >= 2
        plt.close('all')

    def test_residuals_plot_figures_are_valid(self, fitted_model):
        """All residuals plot values should be matplotlib Figures."""
        result = fitted_model.plot(type="residuals")
        for key, fig in result.items():
            assert isinstance(fig, plt.Figure), f"Plot '{key}' is not a Figure"
        plt.close('all')


class TestPlotBeforeFit:
    """Tests for error handling when plotting before fit."""

    def test_plot_before_fit_raises_error(self):
        """Plotting before fit() should raise ValueError."""
        np.random.seed(42)
        n = 50
        y = np.random.normal(0, 1, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=dates[35],
        )

        with pytest.raises(ValueError, match="Must call fit"):
            ca.plot(type="forecast")


class TestPlotWithBootstrap:
    """Tests for plots with bootstrap inference."""

    @pytest.fixture
    def fitted_model_with_bootstrap(self):
        """Create a fitted model with bootstrap inference."""
        np.random.seed(42)
        n = 100
        y = np.cumsum(np.random.normal(0, 1, n)) + 100
        y[70:] += 10

        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[70]

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
            n_boot=100,  # Small number for fast testing
        )
        ca.fit()
        return ca

    def test_forecast_plot_with_bootstrap(self, fitted_model_with_bootstrap):
        """Forecast plot should work with bootstrap model."""
        fig = fitted_model_with_bootstrap.plot(type="forecast")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_impact_plot_with_bootstrap(self, fitted_model_with_bootstrap):
        """Impact plot should work with bootstrap model."""
        result = fitted_model_with_bootstrap.plot(type="impact")
        assert isinstance(result, dict)
        assert "plot" in result
        plt.close('all')
