"""Tests for CausalArima class comparing against R package results."""

import numpy as np
import pandas as pd
import pytest

from pycausalarima import CausalArima


class TestCausalArimaBasic:
    """Basic tests for CausalArima functionality."""

    def test_initialization(self, simple_ar1_data):
        """Test that CausalArima initializes correctly."""
        ca = CausalArima(
            y=simple_ar1_data["y"],
            dates=simple_ar1_data["dates"],
            intervention_date=simple_ar1_data["intervention_date"],
        )

        assert ca.auto is True
        assert ca.alpha == 0.05
        assert ca.result_ is None

    def test_fit_returns_result(self, simple_ar1_data):
        """Test that fit() returns a CausalArimaResult."""
        ca = CausalArima(
            y=simple_ar1_data["y"],
            dates=simple_ar1_data["dates"],
            intervention_date=simple_ar1_data["intervention_date"],
        )
        result = ca.fit()

        assert result is not None
        assert ca.result_ is not None
        assert hasattr(result, "causal_effect")
        assert hasattr(result, "norm")
        assert hasattr(result, "forecast")

    def test_causal_effect_length(self, simple_ar1_data):
        """Test that causal effect has correct length."""
        ca = CausalArima(
            y=simple_ar1_data["y"],
            dates=simple_ar1_data["dates"],
            intervention_date=simple_ar1_data["intervention_date"],
        )
        result = ca.fit()

        # Post-intervention period should have 30 observations (100 - 70)
        n_post = sum(simple_ar1_data["dates"] >= simple_ar1_data["intervention_date"])
        assert len(result.causal_effect) == n_post

    def test_positive_causal_effect(self, simple_ar1_data):
        """Test that we detect a positive causal effect when there is one."""
        ca = CausalArima(
            y=simple_ar1_data["y"],
            dates=simple_ar1_data["dates"],
            intervention_date=simple_ar1_data["intervention_date"],
        )
        result = ca.fit()

        # Average effect should be positive (we added +5)
        avg_effect = np.mean(result.causal_effect[~np.isnan(result.causal_effect)])
        assert avg_effect > 0


class TestCausalArimaWithExog:
    """Tests for CausalArima with exogenous regressors."""

    def test_fit_with_xreg(self, simple_ar1_data):
        """Test fitting with exogenous regressors."""
        n = len(simple_ar1_data["y"])
        xreg = np.random.normal(0, 1, n)

        ca = CausalArima(
            y=simple_ar1_data["y"],
            dates=simple_ar1_data["dates"],
            intervention_date=simple_ar1_data["intervention_date"],
            xreg=xreg,
        )
        result = ca.fit()

        assert result is not None
        assert result.xreg is not None


class TestCausalArimaBootstrap:
    """Tests for bootstrap inference."""

    def test_bootstrap_inference(self, simple_ar1_data):
        """Test that bootstrap inference works."""
        ca = CausalArima(
            y=simple_ar1_data["y"],
            dates=simple_ar1_data["dates"],
            intervention_date=simple_ar1_data["intervention_date"],
            n_boot=100,  # Small number for testing speed
        )
        result = ca.fit()

        assert result.boot is not None
        assert result.boot.boot_distribution is not None
        assert result.boot.boot_distribution.shape[1] == 100


class TestSummaryAndPlot:
    """Tests for summary and plotting methods."""

    def test_summary(self, simple_ar1_data):
        """Test that summary() returns a DataFrame."""
        ca = CausalArima(
            y=simple_ar1_data["y"],
            dates=simple_ar1_data["dates"],
            intervention_date=simple_ar1_data["intervention_date"],
        )
        ca.fit()
        summary = ca.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "Point causal effect" in summary.index

    def test_summary_with_horizon(self, simple_ar1_data):
        """Test summary with specific horizons."""
        ca = CausalArima(
            y=simple_ar1_data["y"],
            dates=simple_ar1_data["dates"],
            intervention_date=simple_ar1_data["intervention_date"],
        )
        ca.fit()

        # Get a valid horizon date
        horizon = [simple_ar1_data["dates"][80]]
        summary = ca.summary(horizon=horizon)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary.columns) == 1

    def test_impact(self, simple_ar1_data):
        """Test that impact() returns expected structure."""
        ca = CausalArima(
            y=simple_ar1_data["y"],
            dates=simple_ar1_data["dates"],
            intervention_date=simple_ar1_data["intervention_date"],
        )
        ca.fit()
        impact = ca.impact(format="numeric")

        assert isinstance(impact, dict)
        assert "arima" in impact
        assert "impact_norm" in impact


class TestValidation:
    """Tests for input validation."""

    def test_invalid_y_length(self):
        """Test that mismatched y and dates raises error."""
        y = np.array([1, 2, 3])
        dates = pd.date_range("2020-01-01", periods=5, freq="D")

        with pytest.raises(ValueError):
            CausalArima(
                y=y,
                dates=dates,
                intervention_date=pd.Timestamp("2020-01-03"),
            )

    def test_intervention_before_start(self):
        """Test that intervention date before data raises error."""
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        dates = pd.date_range("2020-01-01", periods=10, freq="D")

        with pytest.raises(ValueError):
            CausalArima(
                y=y,
                dates=dates,
                intervention_date=pd.Timestamp("2019-12-01"),
            )


# Note: The R comparison tests below require the exact same random number
# generation as R, which is difficult to achieve. These tests use approximate
# tolerances and may need adjustment.

class TestRComparison:
    """Tests comparing results against R package output.

    Note: Due to differences in random number generation between R and Python,
    these tests use relaxed tolerances. For exact comparison, run the R package
    and Python package on the same pre-generated data file.
    """

    @pytest.mark.skip(reason="Requires matching R random number generation")
    def test_r_example_point_effect(self, r_example_data, r_expected_results):
        """Test that point effect matches R output."""
        ca = CausalArima(
            y=r_example_data["y"],
            dates=r_example_data["dates"],
            intervention_date=r_example_data["intervention_date"],
            xreg=r_example_data["x1"],
        )
        result = ca.fit()

        # Get final point effect
        final_tau = result.norm.tau[-1]
        expected = r_expected_results["final"]["point_effect"]

        # Use relatively loose tolerance due to RNG differences
        np.testing.assert_allclose(final_tau, expected, rtol=0.1)

    @pytest.mark.skip(reason="Requires matching R random number generation")
    def test_r_example_cumulative_effect(self, r_example_data, r_expected_results):
        """Test that cumulative effect matches R output."""
        ca = CausalArima(
            y=r_example_data["y"],
            dates=r_example_data["dates"],
            intervention_date=r_example_data["intervention_date"],
            xreg=r_example_data["x1"],
        )
        result = ca.fit()

        final_cum = result.norm.cumulative[-1]
        expected = r_expected_results["final"]["cumulative_effect"]

        np.testing.assert_allclose(final_cum, expected, rtol=0.1)
