"""Summary generation functions for pycausalarima.

This module contains functions for generating formatted summaries of causal effects.
Python translation of R's summary.cArima() and print.cArima() functions.
"""

from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from pycausalarima.utils.types import CausalArimaResult
from pycausalarima.utils.validation import validate_horizon


def generate_summary(
    result: CausalArimaResult,
    type: Literal["norm", "boot"] = "norm",
    horizon: Optional[List] = None,
    digits: int = 3,
) -> pd.DataFrame:
    """Generate formatted summary of causal effects.

    Equivalent to R's summary.cArima() method.

    Parameters
    ----------
    result : CausalArimaResult
        Fitted model results.
    type : str, default='norm'
        Type of inference ('norm' or 'boot').
    horizon : list of dates, optional
        Specific dates to report effects for.
    digits : int, default=3
        Number of decimal places.

    Returns
    -------
    pd.DataFrame
        Summary table of effects.
    """
    # Get the appropriate inference result
    if type == "norm":
        inf = result.norm
    elif type == "boot":
        if result.boot is None:
            raise ValueError("Bootstrap inference not available. Run with n_boot > 0.")
        inf = result.boot
    else:
        raise ValueError(f"Unknown inference type: {type}")

    # Get post-intervention dates
    dates = result.dates[result.dates >= result.intervention_date]
    valid_mask = ~np.isnan(result.causal_effect)
    dates = dates[valid_mask]

    # Determine indices to report
    if horizon is not None:
        # Convert horizon to timestamps and find indices
        indices = validate_horizon(horizon, result.dates, result.intervention_date)
        col_names = [str(pd.Timestamp(h).date()) for h in horizon]
    else:
        # Just report final observation
        indices = [-1]
        col_names = [""]

    # Build summary rows (matching R output format)
    rows = [
        "Point causal effect",
        "Standard error",
        "Left-sided p-value",
        "Bidirectional p-value",
        "Right-sided p-value",
        "",
        "Cumulative causal effect",
        "Standard error",
        "Left-sided p-value",
        "Bidirectional p-value",
        "Right-sided p-value",
        "",
        "Temporal average causal effect",
        "Standard error",
        "Left-sided p-value",
        "Bidirectional p-value",
        "Right-sided p-value",
    ]

    data = {}
    for i, col in zip(indices, col_names):
        data[col] = [
            round(inf.tau[i], digits),
            round(inf.sd_tau[i], digits),
            round(inf.pvalue_tau_left[i], digits),
            round(inf.pvalue_tau_bidirectional[i], digits),
            round(inf.pvalue_tau_right[i], digits),
            "",
            round(inf.cumulative[i], digits),
            round(inf.sd_cumulative[i], digits),
            round(inf.pvalue_sum_left[i], digits),
            round(inf.pvalue_sum_bidirectional[i], digits),
            round(inf.pvalue_sum_right[i], digits),
            "",
            round(inf.average[i], digits),
            round(inf.sd_average[i], digits),
            round(inf.pvalue_avg_left[i], digits),
            round(inf.pvalue_avg_bidirectional[i], digits),
            round(inf.pvalue_avg_right[i], digits),
        ]

    return pd.DataFrame(data, index=rows)


