"""ARMA utility functions for pycausalarima.

This module contains functions for converting SARMA models to long-form ARMA
and computing MA(infinity) psi weights, which are essential for variance
calculation in causal inference.

These are Python translations of the R functions:
- .sarma2larma() -> sarma_to_larma()
- .long() -> _merge_polynomials()
- ARMAtoMA() -> compute_psi_weights()
"""

from typing import Tuple

import numpy as np


def sarma_to_larma(
    ar: np.ndarray,
    ma: np.ndarray,
    sar: np.ndarray,
    sma: np.ndarray,
    s: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert SARMA representation to long-form ARMA.

    This is a Python translation of the R .sarma2larma() function.
    Merges AR(p) with SAR(P) and MA(q) with SMA(Q).

    Parameters
    ----------
    ar : np.ndarray
        AR coefficients (phi). Can be empty array.
    ma : np.ndarray
        MA coefficients (theta). Can be empty array.
    sar : np.ndarray
        Seasonal AR coefficients. Can be empty array.
    sma : np.ndarray
        Seasonal MA coefficients. Can be empty array.
    s : int
        Seasonal period (e.g., 12 for monthly, 7 for daily with weekly pattern).

    Returns
    -------
    tuple of (ar_long, ma_long)
        Long-form ARMA coefficients after merging seasonal components.

    Notes
    -----
    The R code adjusts signs: ar and sar are negated before merging,
    then the result is negated again. This is because R's ARIMA uses
    the convention: (1 - phi*B) y_t = (1 + theta*B) e_t

    Examples
    --------
    >>> ar = np.array([0.5])
    >>> ma = np.array([0.3])
    >>> sar = np.array([0.2])
    >>> sma = np.array([0.1])
    >>> ar_long, ma_long = sarma_to_larma(ar, ma, sar, sma, s=12)
    """
    # Ensure inputs are numpy arrays
    ar = np.atleast_1d(ar) if ar is not None and len(ar) > 0 else np.array([])
    ma = np.atleast_1d(ma) if ma is not None and len(ma) > 0 else np.array([])
    sar = np.atleast_1d(sar) if sar is not None and len(sar) > 0 else np.array([])
    sma = np.atleast_1d(sma) if sma is not None and len(sma) > 0 else np.array([])

    # Adjust signs (R convention: ar and sar are negated)
    ar_adj = -ar if len(ar) > 0 else ar
    sar_adj = -sar if len(sar) > 0 else sar

    # Merge and adjust final sign
    ar_long = -_merge_polynomials(ar_adj, sar_adj, s)
    ma_long = _merge_polynomials(ma, sma, s)

    return ar_long, ma_long


def _merge_polynomials(p: np.ndarray, ps: np.ndarray, s: int) -> np.ndarray:
    """Merge short and seasonal polynomial components.

    Python translation of R .long() function.
    Multiplies the short-term polynomial (1 + p1*B + p2*B^2 + ...)
    with the seasonal polynomial (1 + ps1*B^s + ps2*B^{2s} + ...)
    using convolution.

    Parameters
    ----------
    p : np.ndarray
        Short-term coefficients (without the leading 1).
    ps : np.ndarray
        Seasonal coefficients (without the leading 1).
    s : int
        Seasonal period.

    Returns
    -------
    np.ndarray
        Merged polynomial coefficients (without the leading 1).

    Notes
    -----
    The function builds coefficient vectors with a leading 1,
    convolves them, and returns the result without the leading 1.
    """
    # Ensure inputs are numpy arrays
    p = np.atleast_1d(p) if p is not None else np.array([])
    ps = np.atleast_1d(ps) if ps is not None else np.array([])

    np_len = len(p)
    nps_len = len(ps)

    # Build coefficient vector for short-term component: [1, p1, p2, ...]
    cp = np.concatenate([[1], p]) if np_len > 0 else np.array([1])

    # Build coefficient vector for seasonal component
    if nps_len > 0:
        # Create sparse vector with seasonal coefficients at positions s, 2s, 3s, ...
        # e.g., for s=12 and ps=[0.2]: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2]
        cps = np.zeros(s * nps_len + 1)
        cps[0] = 1
        indices = np.arange(s, s * nps_len + 1, s)
        cps[indices] = ps
    else:
        cps = np.array([1])

    # Convolve polynomials (equivalent to polynomial multiplication)
    # R's convolve with type="open" and rev() is equivalent to np.convolve
    result = np.convolve(cp, cps, mode="full")

    # Remove leading 1 (return coefficients only)
    return result[1:] if len(result) > 1 else np.array([])


def compute_psi_weights(
    ar: np.ndarray,
    ma: np.ndarray,
    lag_max: int,
) -> np.ndarray:
    """Compute MA(infinity) psi weights from ARMA coefficients.

    Equivalent to R's ARMAtoMA() function.
    Converts ARMA(p,q) representation to MA(infinity) representation
    by computing the impulse response function.

    The psi weights satisfy:
    psi_0 = 1
    psi_j = theta_j + sum_{i=1}^{min(j,p)} phi_i * psi_{j-i}  for j >= 1

    Parameters
    ----------
    ar : np.ndarray
        AR coefficients (phi_1, phi_2, ..., phi_p).
    ma : np.ndarray
        MA coefficients (theta_1, theta_2, ..., theta_q).
    lag_max : int
        Maximum lag for psi weights (returns lag_max + 1 weights).

    Returns
    -------
    np.ndarray
        Psi weights [psi_0=1, psi_1, ..., psi_{lag_max}].

    Examples
    --------
    >>> # MA(1) model: psi_0=1, psi_1=theta, psi_j=0 for j>1
    >>> ar = np.array([])
    >>> ma = np.array([0.6])
    >>> psi = compute_psi_weights(ar, ma, 5)
    >>> # psi = [1.0, 0.6, 0.0, 0.0, 0.0, 0.0]

    >>> # AR(1) model: psi_j = phi^j
    >>> ar = np.array([0.8])
    >>> ma = np.array([])
    >>> psi = compute_psi_weights(ar, ma, 5)
    >>> # psi = [1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768]
    """
    # Ensure inputs are numpy arrays
    ar = np.atleast_1d(ar) if ar is not None and len(ar) > 0 else np.array([])
    ma = np.atleast_1d(ma) if ma is not None and len(ma) > 0 else np.array([])

    psi = np.zeros(lag_max + 1)
    psi[0] = 1.0

    p = len(ar)
    q = len(ma)

    for j in range(1, lag_max + 1):
        # MA contribution: theta_j if j <= q, else 0
        psi[j] = ma[j - 1] if j <= q else 0.0

        # AR contribution: sum of phi_i * psi_{j-i} for i = 1 to min(j, p)
        for i in range(1, min(j, p) + 1):
            psi[j] += ar[i - 1] * psi[j - i]

    return psi


