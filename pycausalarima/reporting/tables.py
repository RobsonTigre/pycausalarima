"""Table generation functions for pycausalarima.

This module contains functions for generating formatted tables of impact results
in numeric, HTML, or LaTeX format.

Python translation of R's impact() function.
"""

from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from pycausalarima.utils.types import CausalArimaResult
from pycausalarima.utils.validation import validate_horizon


def generate_impact_tables(
    result: CausalArimaResult,
    format: Literal["numeric", "html", "latex"] = "numeric",
    horizon: Optional[List] = None,
    digits: int = 3,
    **kwargs,
) -> Dict[str, Any]:
    """Generate formatted impact tables.

    Equivalent to R's impact() function.

    Parameters
    ----------
    result : CausalArimaResult
        Fitted model results.
    format : str, default='numeric'
        Output format ('numeric', 'html', 'latex').
    horizon : list of dates, optional
        Specific dates to include.
    digits : int, default=3
        Number of decimal places.
    **kwargs
        Additional formatting arguments for HTML/LaTeX.

    Returns
    -------
    dict
        Dictionary containing formatted tables with keys:
        - 'arima': ARIMA model information
        - 'impact_norm': Normal inference results
        - 'impact_boot': Bootstrap inference results (or None)
    """
    tables = {
        "arima": _build_arima_tables(result, digits),
        "impact_norm": _build_norm_tables(result, horizon, digits),
        "impact_boot": _build_boot_tables(result, horizon, digits)
        if result.boot is not None
        else None,
    }

    if format == "numeric":
        return tables
    elif format == "html":
        return _convert_to_html(tables, **kwargs)
    elif format == "latex":
        return _convert_to_latex(tables, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")


def _build_arima_tables(result: CausalArimaResult, digits: int) -> Dict[str, pd.DataFrame]:
    """Build ARIMA model summary tables.

    Parameters
    ----------
    result : CausalArimaResult
        Fitted model results.
    digits : int
        Number of decimal places.

    Returns
    -------
    dict
        Dictionary with 'arima_order', 'param', 'accuracy', 'log_stats' DataFrames.
    """
    order = result.order

    # ARIMA order table
    order_df = pd.DataFrame(
        {"p": [order.p], "d": [order.d], "q": [order.q]}, index=["arima_order"]
    )

    # If seasonal, add seasonal order
    if order.P > 0 or order.D > 0 or order.Q > 0 or order.s > 1:
        order_df["P"] = order.P
        order_df["D"] = order.D
        order_df["Q"] = order.Q
        order_df["s"] = order.s

    # Coefficients table
    param_df = None
    model = result.model

    try:
        # Try to get parameters from the model
        if hasattr(model, "params") and hasattr(model, "bse"):
            params = model.params
            se = model.bse
            if hasattr(params, "index"):
                param_names = params.index
            else:
                param_names = [f"param_{i}" for i in range(len(params))]

            param_df = pd.DataFrame(
                {
                    "coef": np.round(np.array(params), digits),
                    "se": np.round(np.array(se), digits),
                    "t value": np.round(np.array(params) / np.array(se), digits),
                },
                index=param_names,
            )
        elif hasattr(model, "arima_res_"):
            # pmdarima
            sm_model = model.arima_res_
            if hasattr(sm_model, "params") and hasattr(sm_model, "bse"):
                params = sm_model.params
                se = sm_model.bse
                param_names = (
                    params.index if hasattr(params, "index") else range(len(params))
                )
                param_df = pd.DataFrame(
                    {
                        "coef": np.round(np.array(params), digits),
                        "se": np.round(np.array(se), digits),
                        "t value": np.round(np.array(params) / np.array(se), digits),
                    },
                    index=param_names,
                )
    except (AttributeError, KeyError, ValueError, TypeError):
        param_df = None

    # Model statistics
    log_stats = {}
    try:
        if hasattr(model, "llf"):
            log_stats["loglik"] = round(model.llf, digits)
        elif hasattr(model, "arima_res_"):
            log_stats["loglik"] = round(model.arima_res_.llf, digits)

        if hasattr(model, "aic"):
            log_stats["aic"] = round(model.aic, digits)
        elif hasattr(model, "arima_res_"):
            log_stats["aic"] = round(model.arima_res_.aic, digits)

        if hasattr(model, "bic"):
            log_stats["bic"] = round(model.bic, digits)
        elif hasattr(model, "arima_res_"):
            log_stats["bic"] = round(model.arima_res_.bic, digits)

        if hasattr(model, "aicc"):
            log_stats["aicc"] = round(model.aicc, digits)
        elif hasattr(model, "arima_res_") and hasattr(model.arima_res_, "aicc"):
            log_stats["aicc"] = round(model.arima_res_.aicc, digits)
    except (AttributeError, KeyError, ValueError, TypeError):
        pass

    log_stats_df = pd.DataFrame(log_stats, index=["metrics"]) if log_stats else None

    return {
        "arima_order": order_df,
        "param": param_df,
        "log_stats": log_stats_df,
    }


def _build_norm_tables(
    result: CausalArimaResult, horizon: Optional[List], digits: int
) -> Dict[str, pd.DataFrame]:
    """Build normal inference tables.

    Parameters
    ----------
    result : CausalArimaResult
        Fitted model results.
    horizon : list of dates, optional
        Specific dates to include.
    digits : int
        Number of decimal places.

    Returns
    -------
    dict
        Dictionary with 'average', 'sum', 'point_effect' DataFrames.
    """
    inf = result.norm

    if horizon:
        indices = validate_horizon(horizon, result.dates, result.intervention_date)
        index_names = [str(pd.Timestamp(h).date()) for h in horizon]
    else:
        indices = [-1]
        index_names = ["Final"]

    # Point effect table
    point_df = pd.DataFrame(
        {
            "estimate": [round(inf.tau[i], digits) for i in indices],
            "sd": [round(inf.sd_tau[i], digits) for i in indices],
            "p_value_left": [round(inf.pvalue_tau_left[i], digits) for i in indices],
            "p_value_bidirectional": [
                round(inf.pvalue_tau_bidirectional[i], digits) for i in indices
            ],
            "p_value_right": [round(inf.pvalue_tau_right[i], digits) for i in indices],
        },
        index=index_names,
    )

    # Cumulative effect table
    sum_df = pd.DataFrame(
        {
            "estimate": [round(inf.cumulative[i], digits) for i in indices],
            "sd": [round(inf.sd_cumulative[i], digits) for i in indices],
            "p_value_left": [round(inf.pvalue_sum_left[i], digits) for i in indices],
            "p_value_bidirectional": [
                round(inf.pvalue_sum_bidirectional[i], digits) for i in indices
            ],
            "p_value_right": [round(inf.pvalue_sum_right[i], digits) for i in indices],
        },
        index=index_names,
    )

    # Average effect table
    avg_df = pd.DataFrame(
        {
            "estimate": [round(inf.average[i], digits) for i in indices],
            "sd": [round(inf.sd_average[i], digits) for i in indices],
            "p_value_left": [round(inf.pvalue_avg_left[i], digits) for i in indices],
            "p_value_bidirectional": [
                round(inf.pvalue_avg_bidirectional[i], digits) for i in indices
            ],
            "p_value_right": [round(inf.pvalue_avg_right[i], digits) for i in indices],
        },
        index=index_names,
    )

    return {"point_effect": point_df, "sum": sum_df, "average": avg_df}


def _build_boot_tables(
    result: CausalArimaResult, horizon: Optional[List], digits: int
) -> Optional[Dict[str, pd.DataFrame]]:
    """Build bootstrap inference tables.

    Parameters
    ----------
    result : CausalArimaResult
        Fitted model results.
    horizon : list of dates, optional
        Specific dates to include.
    digits : int
        Number of decimal places.

    Returns
    -------
    dict or None
        Dictionary with bootstrap tables, or None if no bootstrap results.
    """
    if result.boot is None:
        return None

    inf = result.boot

    if horizon:
        indices = validate_horizon(horizon, result.dates, result.intervention_date)
        index_names = [str(pd.Timestamp(h).date()) for h in horizon]
    else:
        indices = [-1]
        index_names = ["Final"]

    # Similar structure to R's output
    tables = {}

    for idx, name in zip(indices, index_names):
        # Compute confidence intervals from bootstrap
        alpha = result.alpha
        z = stats.norm.ppf(1 - alpha / 2)

        # For average
        avg_estimate = inf.average[idx]
        avg_sd = inf.sd_average[idx]
        avg_lower = avg_estimate - z * avg_sd
        avg_upper = avg_estimate + z * avg_sd

        # For cumulative
        cum_estimate = inf.cumulative[idx]
        cum_sd = inf.sd_cumulative[idx]
        cum_lower = cum_estimate - z * cum_sd
        cum_upper = cum_estimate + z * cum_sd

        # Build table
        avg_table = pd.DataFrame(
            {
                "estimates": [round(avg_estimate, digits)],
                "inf": [round(avg_lower, digits)],
                "sup": [round(avg_upper, digits)],
                "sd": [round(avg_sd, digits)],
            },
            index=["absolute_effect"],
        )

        cum_table = pd.DataFrame(
            {
                "estimates": [round(cum_estimate, digits)],
                "inf": [round(cum_lower, digits)],
                "sup": [round(cum_upper, digits)],
                "sd": [round(cum_sd, digits)],
            },
            index=["absolute_effect"],
        )

        pvalue_table = pd.DataFrame(
            {
                "x": [alpha, round(inf.pvalue_avg_bidirectional[idx], digits)],
            },
            index=["alpha", "p"],
        )

        tables[name] = {"average": avg_table, "effect_cum": cum_table, "p_values": pvalue_table}

    return tables


def _convert_to_html(tables: Dict, **kwargs) -> Dict:
    """Convert all tables to HTML format.

    Parameters
    ----------
    tables : dict
        Dictionary of DataFrames.
    **kwargs
        Additional arguments for to_html().

    Returns
    -------
    dict
        Dictionary with HTML strings.
    """

    def df_to_html(df):
        if df is None:
            return None
        if isinstance(df, pd.DataFrame):
            return df.to_html(**kwargs)
        return df

    return _apply_to_nested(tables, df_to_html)


def _convert_to_latex(tables: Dict, **kwargs) -> Dict:
    """Convert all tables to LaTeX format.

    Parameters
    ----------
    tables : dict
        Dictionary of DataFrames.
    **kwargs
        Additional arguments for to_latex().

    Returns
    -------
    dict
        Dictionary with LaTeX strings.
    """

    def df_to_latex(df):
        if df is None:
            return None
        if isinstance(df, pd.DataFrame):
            return df.to_latex(**kwargs)
        return df

    return _apply_to_nested(tables, df_to_latex)


def _apply_to_nested(d: Dict, func) -> Dict:
    """Apply function to all DataFrames in nested dict.

    Parameters
    ----------
    d : dict
        Nested dictionary.
    func : callable
        Function to apply to DataFrames.

    Returns
    -------
    dict
        Transformed dictionary.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _apply_to_nested(v, func)
        elif isinstance(v, pd.DataFrame):
            result[k] = func(v)
        else:
            result[k] = v
    return result
