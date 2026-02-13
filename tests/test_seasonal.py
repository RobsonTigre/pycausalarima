"""Tests for seasonal ARIMA models.

These tests verify that seasonal ARIMA models work correctly,
based on real-world usage patterns from the book example.
"""

import numpy as np
import pandas as pd
import pytest

from pycausalarima import CausalArima


class TestWeeklySeasonality:
    """Tests for weekly seasonality (period=7)."""

    @pytest.fixture
    def weekly_seasonal_data(self):
        """Generate data with weekly seasonality and intervention effect.

        This replicates the pattern from the book example:
        - Day-of-week effects (Saturday/Sunday higher, Monday lower)
        - Intervention effect of +20 starting at day 75
        """
        np.random.seed(456)
        n = 100

        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

        # Day-of-week effects
        day_effects = {
            "Saturday": 30, "Sunday": 25, "Monday": -5, "Tuesday": 0,
            "Wednesday": 5, "Thursday": 10, "Friday": 20
        }
        dow_effect = np.array([day_effects[d.day_name()] for d in dates])

        # Base series with day-of-week seasonality
        y = 100 + dow_effect + np.random.normal(0, 3, n)

        # Add intervention effect at day 75
        intervention_idx = 74  # 0-indexed
        y[intervention_idx:] += 20

        intervention_date = dates[intervention_idx]

        return {
            "y": y,
            "dates": dates,
            "intervention_date": intervention_date,
            "true_effect": 20,
        }

    def test_weekly_seasonality_detects_effect(self, weekly_seasonal_data):
        """Model with weekly seasonality should detect the intervention effect."""
        ca = CausalArima(
            y=weekly_seasonal_data["y"],
            dates=weekly_seasonal_data["dates"],
            intervention_date=weekly_seasonal_data["intervention_date"],
            seasonal_order=(0, 0, 0, 7),  # Weekly seasonality
        )
        result = ca.fit()

        # The temporal average effect should be close to the true effect
        estimated_avg = result.norm.average[-1]
        true_effect = weekly_seasonal_data["true_effect"]

        # Allow 30% tolerance due to model uncertainty
        assert abs(estimated_avg - true_effect) < 0.3 * true_effect, \
            f"Estimated average effect {estimated_avg:.2f} too far from true {true_effect}"

    def test_weekly_seasonality_with_bootstrap(self, weekly_seasonal_data):
        """Bootstrap inference should work with weekly seasonality."""
        ca = CausalArima(
            y=weekly_seasonal_data["y"],
            dates=weekly_seasonal_data["dates"],
            intervention_date=weekly_seasonal_data["intervention_date"],
            seasonal_order=(0, 0, 0, 7),
            n_boot=100,  # Small number for fast testing
        )
        result = ca.fit()

        # Bootstrap results should exist
        assert result.boot is not None
        assert result.boot.average is not None

        # Both estimates should be in a reasonable range around the true effect (20)
        # Note: Normal and bootstrap estimates can differ significantly because
        # they use different estimation methods. Both should be reasonable though.
        norm_avg = result.norm.average[-1]
        boot_avg = result.boot.average[-1]
        true_effect = weekly_seasonal_data["true_effect"]

        # Both estimates should be within 50% of true effect
        assert 0.5 * true_effect < norm_avg < 1.5 * true_effect, \
            f"Normal estimate {norm_avg:.2f} outside reasonable range"
        assert 0.5 * true_effect < boot_avg < 1.5 * true_effect, \
            f"Bootstrap estimate {boot_avg:.2f} outside reasonable range"

    def test_weekly_seasonality_p_values(self, weekly_seasonal_data):
        """P-values should indicate significant effect for weekly seasonal model."""
        ca = CausalArima(
            y=weekly_seasonal_data["y"],
            dates=weekly_seasonal_data["dates"],
            intervention_date=weekly_seasonal_data["intervention_date"],
            seasonal_order=(0, 0, 0, 7),
        )
        result = ca.fit()

        # Two-sided p-value for average effect should be significant
        p_value = result.norm.pvalue_avg_bidirectional[-1]
        assert p_value < 0.05, f"P-value {p_value:.4f} not significant at 0.05 level"


class TestPlaceboWithSeasonality:
    """Tests for placebo analysis with seasonal models."""

    def test_placebo_no_effect(self):
        """Placebo test (no real intervention) should show no significant effect."""
        np.random.seed(456)
        n = 70  # Pre-intervention data only

        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

        # Day-of-week effects
        day_effects = {
            "Saturday": 30, "Sunday": 25, "Monday": -5, "Tuesday": 0,
            "Wednesday": 5, "Thursday": 10, "Friday": 20
        }
        dow_effect = np.array([day_effects[d.day_name()] for d in dates])

        # Base series with NO intervention
        y = 100 + dow_effect + np.random.normal(0, 3, n)

        # Pretend intervention at day 57
        intervention_date = dates[56]

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
            seasonal_order=(0, 0, 0, 7),
        )
        result = ca.fit()

        # The average effect should be close to zero
        estimated_avg = result.norm.average[-1]
        assert abs(estimated_avg) < 5, \
            f"Placebo effect {estimated_avg:.2f} should be near zero"


class TestDifferentSeasonalPeriods:
    """Tests for various seasonal periods."""

    def test_monthly_seasonality(self):
        """Test with monthly seasonality (period=12)."""
        np.random.seed(42)
        n = 120  # 10 years of monthly data

        dates = pd.date_range(start="2014-01-01", periods=n, freq="ME")  # ME = Month End

        # Monthly seasonal pattern
        monthly_effect = np.tile([0, 2, 5, 8, 10, 12, 12, 10, 8, 5, 2, 0], n // 12 + 1)[:n]
        y = 100 + monthly_effect + np.random.normal(0, 2, n)

        # Intervention at month 90
        y[90:] += 15

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=dates[90],
            seasonal_order=(0, 0, 0, 12),
        )
        result = ca.fit()

        # Should detect effect reasonably close to 15
        estimated_avg = result.norm.average[-1]
        assert 10 < estimated_avg < 20, \
            f"Monthly seasonal model estimated {estimated_avg:.2f}, expected ~15"

    def test_no_seasonality(self):
        """Test without seasonality (period=1)."""
        np.random.seed(42)
        n = 100

        dates = pd.date_range(start="2020-01-01", periods=n, freq="D")

        # Simple AR(1) process without seasonality
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.7 * y[t-1] + np.random.normal(0, 1)
        y += 50
        y[70:] += 10

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=dates[70],
            seasonal_order=(0, 0, 0, 1),  # No seasonality
        )
        result = ca.fit()

        # Should detect effect
        estimated_avg = result.norm.average[-1]
        assert 5 < estimated_avg < 15, \
            f"Non-seasonal model estimated {estimated_avg:.2f}, expected ~10"


class TestSeasonalWithExogenous:
    """Tests for seasonal models with exogenous regressors."""

    def test_weekly_with_covariate(self):
        """Test weekly seasonality with exogenous regressor."""
        np.random.seed(456)
        n = 100

        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

        # Exogenous regressor (not affected by intervention)
        x1 = np.cumsum(np.random.normal(0, 1, n)) + 50

        # Day-of-week effects
        day_effects = {
            "Saturday": 30, "Sunday": 25, "Monday": -5, "Tuesday": 0,
            "Wednesday": 5, "Thursday": 10, "Friday": 20
        }
        dow_effect = np.array([day_effects[d.day_name()] for d in dates])

        # y depends on x1 and day-of-week
        y = 100 + 0.5 * x1 + dow_effect + np.random.normal(0, 3, n)
        y[74:] += 20  # Intervention effect

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=dates[74],
            seasonal_order=(0, 0, 0, 7),
            xreg=x1,
        )
        result = ca.fit()

        # Should detect effect close to 20
        estimated_avg = result.norm.average[-1]
        assert 10 < estimated_avg < 30, \
            f"Model with covariate estimated {estimated_avg:.2f}, expected ~20"


class TestSeasonalResiduals:
    """Tests for residual access with seasonal models."""

    def test_get_residuals_seasonal(self):
        """get_residuals() should work with seasonal models."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

        # Simple seasonal pattern
        dow = np.array([0, 1, 2, 3, 4, 5, 6] * 8)[:n]
        y = 100 + dow * 2 + np.random.normal(0, 1, n)
        y[35:] += 5

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=dates[35],
            seasonal_order=(0, 0, 0, 7),
        )
        ca.fit()

        # Get raw residuals
        resid = ca.get_residuals()
        assert len(resid) > 0
        assert not np.any(np.isnan(resid))

        # Get standardized residuals
        std_resid = ca.get_residuals(standardized=True)
        assert len(std_resid) > 0

        # Standardized residuals should have approximately unit variance
        assert 0.5 < np.std(std_resid) < 2.0
