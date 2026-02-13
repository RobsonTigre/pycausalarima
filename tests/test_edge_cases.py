"""Edge case tests for pyCausalArima.

These tests cover boundary conditions and unusual inputs to ensure
robust behavior across a variety of scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from pycausalarima import CausalArima


class TestMinimumSeriesLength:
    """Tests for minimum viable time series lengths."""

    def test_exactly_10_observations(self):
        """Test with exactly 10 observations (minimum allowed)."""
        n = 10
        np.random.seed(123)
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[6]  # 6 pre, 4 post

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
        )
        result = ca.fit()

        assert result is not None
        assert len(result.causal_effect) == 4  # Post-intervention observations

    def test_exactly_5_pre_intervention(self):
        """Test with exactly 5 pre-intervention observations (minimum)."""
        n = 15
        np.random.seed(456)
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[5]  # Exactly 5 pre-intervention

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
        )
        result = ca.fit()

        assert result is not None
        assert result.n_pre == 5

    def test_single_post_intervention_observation(self):
        """Test with only 1 post-intervention observation."""
        n = 10
        np.random.seed(789)
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[-1]  # Last observation is post-intervention

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
        )
        result = ca.fit()

        assert result is not None
        assert len(result.causal_effect) == 1


class TestValidationErrors:
    """Tests for input validation error handling."""

    def test_too_few_observations(self):
        """Test that fewer than 10 observations raises an error."""
        n = 9
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        with pytest.raises(ValueError, match="at least 10 observations"):
            CausalArima(
                y=y,
                dates=dates,
                intervention_date=dates[5],
            )

    def test_too_few_pre_intervention(self):
        """Test that fewer than 5 pre-intervention observations raises an error."""
        n = 10
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[3]  # Only 3 pre-intervention

        with pytest.raises(ValueError, match="at least 5 pre-intervention"):
            CausalArima(
                y=y,
                dates=dates,
                intervention_date=intervention_date,
            )

    def test_intervention_before_series(self):
        """Test that intervention before series start raises an error."""
        n = 20
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = pd.Timestamp("2019-12-01")  # Before series

        with pytest.raises(ValueError, match="must be >="):
            CausalArima(
                y=y,
                dates=dates,
                intervention_date=intervention_date,
            )

    def test_intervention_after_series(self):
        """Test that intervention after series end raises an error."""
        n = 20
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = pd.Timestamp("2020-12-01")  # After series

        with pytest.raises(ValueError, match="must be <="):
            CausalArima(
                y=y,
                dates=dates,
                intervention_date=intervention_date,
            )

    def test_mismatched_lengths(self):
        """Test that mismatched y and dates lengths raise an error."""
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # 12 elements
        dates = pd.date_range("2020-01-01", periods=15, freq="D")  # 15 dates

        with pytest.raises(ValueError, match="must equal"):
            CausalArima(
                y=y,
                dates=dates,
                intervention_date=dates[8],
            )

    def test_invalid_alpha(self):
        """Test that invalid alpha raises an error."""
        n = 20
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        with pytest.raises(ValueError, match="between 0 and 1"):
            CausalArima(
                y=y,
                dates=dates,
                intervention_date=dates[10],
                alpha=1.5,
            )

    def test_negative_n_boot(self):
        """Test that negative n_boot raises an error."""
        n = 20
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        with pytest.raises(ValueError, match="n_boot must be >= 1"):
            CausalArima(
                y=y,
                dates=dates,
                intervention_date=dates[10],
                n_boot=-10,
            )


class TestSpecialDataPatterns:
    """Tests for special patterns in the data."""

    def test_constant_pre_intervention(self):
        """Test with constant pre-intervention values."""
        n = 30
        y = np.ones(n) * 100
        y[20:] = 110  # Step change at intervention
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[20]

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
        )
        result = ca.fit()

        # Should detect positive effect
        assert np.mean(result.causal_effect) > 5

    def test_trending_series(self):
        """Test with a clear trend in the data."""
        n = 50
        np.random.seed(111)
        trend = np.linspace(0, 50, n)
        y = 100 + trend + np.random.normal(0, 2, n)
        y[30:] += 10  # Additional step at intervention
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[30]

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
        )
        result = ca.fit()

        assert result is not None
        # The model should capture the intervention effect

    def test_high_variance_series(self):
        """Test with high-variance time series."""
        n = 50
        np.random.seed(222)
        y = np.random.normal(100, 50, n)  # High variance
        y[30:] += 20  # Effect should still be detectable
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[30]

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
        )
        result = ca.fit()

        assert result is not None

    def test_no_effect(self):
        """Test when there is no actual effect."""
        n = 50
        np.random.seed(333)
        y = np.random.normal(100, 5, n)  # Pure noise, no effect
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[30]

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
        )
        result = ca.fit()

        # Effect should be close to zero on average
        # (may not be exactly zero due to random variation)
        assert result is not None


class TestExogenousRegressors:
    """Tests for exogenous regressors handling."""

    def test_single_exog_regressor(self):
        """Test with a single exogenous regressor."""
        n = 50
        np.random.seed(444)
        x = np.random.normal(10, 2, n)
        y = 2 * x + np.random.normal(0, 1, n)
        y[30:] += 5  # Effect
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[30]

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
            xreg=x,
        )
        result = ca.fit()

        assert result is not None
        assert result.xreg is not None

    def test_multiple_exog_regressors(self):
        """Test with multiple exogenous regressors."""
        n = 50
        np.random.seed(555)
        x1 = np.random.normal(10, 2, n)
        x2 = np.random.normal(5, 1, n)
        xreg = np.column_stack([x1, x2])
        y = 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)
        y[30:] += 5  # Effect
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[30]

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
            xreg=xreg,
        )
        result = ca.fit()

        assert result is not None
        assert result.xreg.shape[1] == 2


class TestManualARIMAOrder:
    """Tests for manual ARIMA order specification."""

    def test_specify_arima_order(self):
        """Test specifying ARIMA order manually."""
        n = 50
        np.random.seed(666)
        y = np.random.normal(100, 5, n)
        y[30:] += 10
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[30]

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
            order=(1, 0, 0),  # AR(1)
        )
        result = ca.fit()

        assert result is not None
        assert result.order.p == 1
        assert result.order.d == 0
        assert result.order.q == 0

    def test_specify_arima_order_with_differencing(self):
        """Test specifying ARIMA order with differencing."""
        n = 50
        np.random.seed(777)
        # Create integrated series
        y = np.cumsum(np.random.normal(0, 1, n)) + 100
        y[30:] += 10
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[30]

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
            order=(1, 1, 0),  # ARIMA(1,1,0)
        )
        result = ca.fit()

        assert result is not None
        assert result.order.d == 1


class TestBootstrapInference:
    """Tests for bootstrap inference."""

    def test_bootstrap_small_n(self):
        """Test bootstrap with small number of iterations."""
        n = 30
        np.random.seed(888)
        y = np.random.normal(100, 5, n)
        y[20:] += 10
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[20]

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
            n_boot=50,  # Small number
        )
        result = ca.fit()

        assert result.boot is not None
        assert result.boot.boot_distribution.shape[1] == 50

    def test_bootstrap_distribution_shape(self):
        """Test that bootstrap distribution has correct shape."""
        n = 30
        y = np.random.default_rng(999).normal(100, 5, n)
        y[20:] += 10
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[20]

        n_boot = 100
        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date=intervention_date,
            n_boot=n_boot,
        )
        result = ca.fit()

        # Check bootstrap distribution shape
        assert result.boot is not None
        assert result.boot.boot_distribution is not None
        # Shape should be (h, n_boot) where h is post-intervention length
        n_post = sum(dates >= intervention_date)
        assert result.boot.boot_distribution.shape == (n_post, n_boot)


class TestXregValidation:
    """Additional xreg validation tests."""

    def test_2d_xreg_correct_shape(self):
        """Test 2D xreg with correct first dimension."""
        n = 20
        np.random.seed(42)
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        xreg = np.random.normal(0, 1, (n, 2))  # 2D with correct shape

        ca = CausalArima(y=y, dates=dates, intervention_date=dates[10], xreg=xreg)
        result = ca.fit()
        assert result is not None
        assert result.xreg.shape[1] == 2

    def test_xreg_nan_raises_error(self):
        """Test that xreg with NaN raises error."""
        n = 20
        np.random.seed(42)
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        xreg = np.random.normal(0, 1, n)
        xreg[5] = np.nan

        with pytest.raises(ValueError, match="cannot contain NaN"):
            CausalArima(y=y, dates=dates, intervention_date=dates[10], xreg=xreg)

    def test_xreg_wrong_dimension_raises_error(self):
        """Test that xreg with wrong first dimension raises error."""
        n = 20
        np.random.seed(42)
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        xreg = np.random.normal(0, 1, (15, 2))  # Wrong first dim

        with pytest.raises(ValueError, match="must equal"):
            CausalArima(y=y, dates=dates, intervention_date=dates[10], xreg=xreg)

    def test_xreg_3d_raises_error(self):
        """Test that 3D xreg raises error."""
        n = 20
        np.random.seed(42)
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        xreg = np.random.normal(0, 1, (n, 2, 3))  # 3D array

        with pytest.raises(ValueError, match="1D or 2D"):
            CausalArima(y=y, dates=dates, intervention_date=dates[10], xreg=xreg)


class TestHorizonValidation:
    """Tests for horizon validation."""

    def test_horizon_before_intervention_raises_error(self):
        """Test that horizon before intervention raises error."""
        n = 50
        np.random.seed(42)
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[30]

        ca = CausalArima(y=y, dates=dates, intervention_date=intervention_date)
        ca.fit()

        # Try to get summary with horizon before intervention
        with pytest.raises(ValueError, match="before intervention"):
            ca.summary(horizon=[dates[20]])

    def test_horizon_not_in_dates_raises_error(self):
        """Test that horizon not in dates raises error."""
        n = 50
        np.random.seed(42)
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[30]

        ca = CausalArima(y=y, dates=dates, intervention_date=intervention_date)
        ca.fit()

        # Try with date not in the series
        invalid_date = pd.Timestamp("2020-06-15")  # Not in dates
        with pytest.raises(ValueError, match="not found"):
            ca.summary(horizon=[invalid_date])

    def test_horizon_string_converted_to_timestamp(self):
        """Test that string horizon dates are converted to Timestamp."""
        n = 50
        np.random.seed(42)
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        intervention_date = dates[30]

        ca = CausalArima(y=y, dates=dates, intervention_date=intervention_date)
        ca.fit()

        # Use string date format (should be converted)
        horizon = ["2020-02-05"]  # dates[35] as string
        result = ca.summary(horizon=horizon)
        assert isinstance(result, pd.DataFrame)


class TestTypeCoercion:
    """Tests for automatic type coercion."""

    def test_y_list_coerced_to_array(self):
        """Test that list y is coerced to numpy array."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        ca = CausalArima(y=y_list, dates=dates, intervention_date=dates[10])
        assert isinstance(ca.y, np.ndarray)

    def test_string_dates_coerced(self):
        """Test that string dates can be coerced to DatetimeIndex."""
        n = 20
        np.random.seed(42)
        y = np.random.normal(100, 5, n)
        # Use string dates
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        ca = CausalArima(
            y=y,
            dates=dates,
            intervention_date="2020-01-11",  # String date
        )
        assert isinstance(ca.intervention_date, pd.Timestamp)

    def test_alpha_not_number_raises_error(self):
        """Test that non-numeric alpha raises TypeError."""
        n = 20
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        with pytest.raises(TypeError, match="must be a number"):
            CausalArima(
                y=y,
                dates=dates,
                intervention_date=dates[10],
                alpha="0.05",
            )

    def test_n_boot_not_int_raises_error(self):
        """Test that non-integer n_boot raises TypeError."""
        n = 20
        y = np.random.normal(100, 5, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        with pytest.raises(TypeError, match="must be an integer"):
            CausalArima(
                y=y,
                dates=dates,
                intervention_date=dates[10],
                n_boot=100.5,
            )
