"""Tests for ARMA utility functions."""

import numpy as np
import pytest

from pycausalarima.core.arma_utils import (
    _merge_polynomials,
    compute_psi_weights,
    sarma_to_larma,
)


class TestMergePolynomials:
    """Tests for _merge_polynomials function."""

    def test_empty_inputs(self):
        """Test with empty inputs."""
        result = _merge_polynomials(np.array([]), np.array([]), 12)
        assert len(result) == 0

    def test_ar_only(self):
        """Test with only AR coefficients."""
        ar = np.array([0.5, 0.3])
        result = _merge_polynomials(ar, np.array([]), 12)
        np.testing.assert_array_almost_equal(result, ar)

    def test_seasonal_only(self):
        """Test with only seasonal coefficients."""
        sar = np.array([0.4])
        result = _merge_polynomials(np.array([]), sar, 4)

        # Should have zeros in positions 0-2 and 0.4 in position 3
        expected = np.zeros(4)
        expected[3] = 0.4
        np.testing.assert_array_almost_equal(result, expected)

    def test_ar_and_seasonal(self):
        """Test with both AR and seasonal coefficients."""
        ar = np.array([0.5])
        sar = np.array([0.3])
        result = _merge_polynomials(ar, sar, 4)

        # Polynomial multiplication: (1 + 0.5*B) * (1 + 0.3*B^4)
        # = 1 + 0.5*B + 0.3*B^4 + 0.15*B^5
        # Coefficients (without leading 1): [0.5, 0, 0, 0.3, 0.15]
        expected = np.array([0.5, 0, 0, 0.3, 0.15])
        np.testing.assert_array_almost_equal(result, expected)


class TestComputePsiWeights:
    """Tests for compute_psi_weights function."""

    def test_ma_only(self):
        """Test MA(1) model: psi_0=1, psi_1=theta, psi_j=0 for j>1."""
        ar = np.array([])
        ma = np.array([0.6])
        psi = compute_psi_weights(ar, ma, 5)

        assert psi[0] == 1.0
        assert psi[1] == 0.6
        np.testing.assert_array_almost_equal(psi[2:], [0, 0, 0, 0])

    def test_ar_only(self):
        """Test AR(1) model: psi_j = phi^j."""
        ar = np.array([0.8])
        ma = np.array([])
        psi = compute_psi_weights(ar, ma, 5)

        expected = 0.8 ** np.arange(6)
        np.testing.assert_array_almost_equal(psi, expected)

    def test_arma_11(self):
        """Test ARMA(1,1) model."""
        ar = np.array([0.7])
        ma = np.array([0.3])
        psi = compute_psi_weights(ar, ma, 4)

        # psi_0 = 1
        # psi_1 = theta_1 + phi_1 * psi_0 = 0.3 + 0.7 * 1 = 1.0
        # psi_2 = phi_1 * psi_1 = 0.7 * 1.0 = 0.7
        # psi_3 = phi_1 * psi_2 = 0.7 * 0.7 = 0.49
        # psi_4 = phi_1 * psi_3 = 0.7 * 0.49 = 0.343
        expected = np.array([1.0, 1.0, 0.7, 0.49, 0.343])
        np.testing.assert_array_almost_equal(psi, expected)

    def test_ar2(self):
        """Test AR(2) model."""
        ar = np.array([0.5, 0.3])
        ma = np.array([])
        psi = compute_psi_weights(ar, ma, 4)

        # psi_0 = 1
        # psi_1 = phi_1 * psi_0 = 0.5
        # psi_2 = phi_1 * psi_1 + phi_2 * psi_0 = 0.5*0.5 + 0.3*1 = 0.55
        # psi_3 = phi_1 * psi_2 + phi_2 * psi_1 = 0.5*0.55 + 0.3*0.5 = 0.425
        # psi_4 = phi_1 * psi_3 + phi_2 * psi_2 = 0.5*0.425 + 0.3*0.55 = 0.3775
        expected = np.array([1.0, 0.5, 0.55, 0.425, 0.3775])
        np.testing.assert_array_almost_equal(psi, expected)


class TestSarmaToLarma:
    """Tests for sarma_to_larma function."""

    def test_no_seasonal(self):
        """Test conversion with no seasonal components."""
        ar = np.array([0.5])
        ma = np.array([0.3])

        ar_long, ma_long = sarma_to_larma(ar, ma, np.array([]), np.array([]), 1)

        # Without seasonal, should return original (with sign adjustment)
        np.testing.assert_array_almost_equal(ar_long, ar)
        np.testing.assert_array_almost_equal(ma_long, ma)

    def test_empty_all(self):
        """Test with all empty inputs."""
        ar_long, ma_long = sarma_to_larma(
            np.array([]), np.array([]), np.array([]), np.array([]), 12
        )

        assert len(ar_long) == 0
        assert len(ma_long) == 0


