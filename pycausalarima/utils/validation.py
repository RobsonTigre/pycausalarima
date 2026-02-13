"""Input validation utilities for pycausalarima."""

from typing import List, Optional

import numpy as np
import pandas as pd

from pycausalarima.exceptions import ValidationError


def validate_inputs(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    intervention_date: pd.Timestamp,
    xreg: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_boot: Optional[int] = None,
) -> None:
    """Validate all input parameters for CausalArima.

    Parameters
    ----------
    y : np.ndarray
        Time series observations.
    dates : pd.DatetimeIndex
        Dates corresponding to observations.
    intervention_date : pd.Timestamp
        Date when intervention occurred.
    xreg : np.ndarray, optional
        Exogenous regressors.
    alpha : float
        Significance level.
    n_boot : int, optional
        Number of bootstrap iterations.

    Raises
    ------
    ValueError
        If any validation check fails.
    TypeError
        If inputs have wrong types.
    """
    # Check y
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a numpy array, got {type(y)}")
    if y.ndim != 1:
        raise ValidationError(f"y must be 1-dimensional, got shape {y.shape}")
    if len(y) < 10:
        raise ValidationError(f"y must have at least 10 observations, got {len(y)}")
    if np.any(np.isnan(y)):
        raise ValidationError("y cannot contain NaN values")

    # Check dates
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError(f"dates must be a DatetimeIndex, got {type(dates)}")
    if len(dates) != len(y):
        raise ValidationError(
            f"Length of dates ({len(dates)}) must equal length of y ({len(y)})"
        )

    # Check intervention_date
    if not isinstance(intervention_date, pd.Timestamp):
        raise TypeError(
            f"intervention_date must be a Timestamp, got {type(intervention_date)}"
        )
    if intervention_date < dates.min():
        raise ValidationError(
            f"intervention_date ({intervention_date}) must be >= first date ({dates.min()})"
        )
    if intervention_date > dates.max():
        raise ValidationError(
            f"intervention_date ({intervention_date}) must be <= last date ({dates.max()})"
        )

    # Check that there are observations before and after intervention
    n_pre = np.sum(dates < intervention_date)
    n_post = np.sum(dates >= intervention_date)
    if n_pre < 5:
        raise ValidationError(
            f"Need at least 5 pre-intervention observations, got {n_pre}"
        )
    if n_post < 1:
        raise ValidationError(
            f"Need at least 1 post-intervention observation, got {n_post}"
        )

    # Check xreg
    if xreg is not None:
        if not isinstance(xreg, np.ndarray):
            raise TypeError(f"xreg must be a numpy array, got {type(xreg)}")
        if xreg.ndim == 1:
            if len(xreg) != len(y):
                raise ValidationError(
                    f"Length of xreg ({len(xreg)}) must equal length of y ({len(y)})"
                )
        elif xreg.ndim == 2:
            if xreg.shape[0] != len(y):
                raise ValidationError(
                    f"First dimension of xreg ({xreg.shape[0]}) must equal length of y ({len(y)})"
                )
        else:
            raise ValidationError(f"xreg must be 1D or 2D, got shape {xreg.shape}")
        if np.any(np.isnan(xreg)):
            raise ValidationError("xreg cannot contain NaN values")

    # Check alpha
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be a number, got {type(alpha)}")
    if not 0 < alpha < 1:
        raise ValidationError(f"alpha must be between 0 and 1, got {alpha}")

    # Check n_boot
    if n_boot is not None:
        if not isinstance(n_boot, int):
            raise TypeError(f"n_boot must be an integer, got {type(n_boot)}")
        if n_boot < 1:
            raise ValidationError(f"n_boot must be >= 1, got {n_boot}")


def validate_horizon(
    horizon: Optional[List[pd.Timestamp]],
    dates: pd.DatetimeIndex,
    intervention_date: pd.Timestamp,
) -> Optional[List[int]]:
    """Validate horizon dates and convert to indices.

    Parameters
    ----------
    horizon : list of Timestamp, optional
        Specific dates to report effects for.
    dates : pd.DatetimeIndex
        All dates in the time series.
    intervention_date : pd.Timestamp
        Date when intervention occurred.

    Returns
    -------
    list of int or None
        Indices into post-intervention period, or None if no horizon specified.

    Raises
    ------
    ValueError
        If horizon dates are invalid.
    """
    if horizon is None:
        return None

    post_dates = dates[dates >= intervention_date]
    indices = []

    for h in horizon:
        if not isinstance(h, pd.Timestamp):
            h = pd.Timestamp(h)
        if h < intervention_date:
            raise ValidationError(
                f"Horizon date {h} is before intervention date {intervention_date}"
            )
        if h not in post_dates:
            # Find closest date
            closest_idx = np.argmin(np.abs(post_dates - h))
            closest_date = post_dates[closest_idx]
            raise ValidationError(
                f"Horizon date {h} not found in dates. Closest is {closest_date}"
            )
        idx = np.where(post_dates == h)[0][0]
        indices.append(idx)

    return indices
