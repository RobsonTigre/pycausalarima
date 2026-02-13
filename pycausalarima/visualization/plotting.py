"""Visualization functions for pycausalarima.

This module contains functions for creating visualizations of CausalArima results,
including forecast plots, impact plots, and residual diagnostics.

Python translation of R's plot.cArima() function.
"""

from typing import Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from pycausalarima.utils.types import CausalArimaResult


def plot_causal_arima(
    result: CausalArimaResult,
    type: Literal["forecast", "impact", "residuals"] = "forecast",
    horizon: Optional[List] = None,
    **kwargs,
) -> Union[Figure, Dict[str, Figure]]:
    """Create visualizations of CausalArima results.

    Parameters
    ----------
    result : CausalArimaResult
        Fitted model results.
    type : str
        Type of plot ('forecast', 'impact', 'residuals').
    horizon : list of dates, optional
        Dates to highlight.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    Figure or dict of Figures
    """
    if type == "forecast":
        return _plot_forecast(result, horizon, **kwargs)
    elif type == "impact":
        return _plot_impact(result, horizon, alpha=result.alpha, **kwargs)
    elif type == "residuals":
        return _plot_residuals(result, **kwargs)
    else:
        raise ValueError(f"Unknown plot type: {type}")


def _plot_forecast(
    result: CausalArimaResult,
    horizon: Optional[List] = None,
    win: float = 0.4,
    colors: tuple = ("darkblue", "black"),
    fill_color: str = "lightsteelblue",
    line_width: float = 1.0,
    figsize: tuple = (12, 6),
) -> Figure:
    """Plot observed vs forecasted time series.

    Parameters
    ----------
    result : CausalArimaResult
        Fitted model results.
    horizon : list of dates, optional
        Dates to highlight with vertical lines.
    win : float, default=0.4
        Proportion of pre-intervention data to show.
    colors : tuple, default=('darkblue', 'black')
        Colors for (forecast, observed) lines.
    fill_color : str, default='lightsteelblue'
        Color for confidence interval fill.
    line_width : float, default=1.0
        Line width.
    figsize : tuple, default=(12, 6)
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    dates = result.dates
    int_date = result.intervention_date

    # Get observed values
    observed = result.y

    # Get fitted values for pre-intervention period
    pre_mask = dates < int_date
    n_pre = np.sum(pre_mask)

    # Get model fitted values
    # Note: pmdarima's fittedvalues is a method, statsmodels' is a property
    if hasattr(result.model, "fittedvalues"):
        fv = result.model.fittedvalues
        if callable(fv):
            fv = fv()  # pmdarima: call the method
        if hasattr(fv, "values"):
            fv = fv.values  # Convert pandas Series to numpy
        fitted = np.atleast_1d(np.asarray(fv))
    elif hasattr(result.model, "arima_res_"):
        fv = result.model.arima_res_.fittedvalues
        if callable(fv):
            fv = fv()
        if hasattr(fv, "values"):
            fv = fv.values
        fitted = np.atleast_1d(np.asarray(fv))
    else:
        fitted = observed[pre_mask]  # Fallback

    # Ensure fitted has correct length
    if len(fitted) < n_pre:
        fitted = np.concatenate([np.full(n_pre - len(fitted), np.nan), fitted])

    # Combine fitted and forecast
    forecasted = np.concatenate([fitted[:n_pre], result.forecast])

    # Confidence bands (only for post-intervention)
    forecast_upper = np.concatenate([np.full(n_pre, np.nan), result.forecast_upper])
    forecast_lower = np.concatenate([np.full(n_pre, np.nan), result.forecast_lower])

    # Determine display window
    start_idx = int(n_pre - win * n_pre)

    x = dates[start_idx:]
    obs_cut = observed[start_idx:]
    fcast_cut = forecasted[start_idx:]
    upper_cut = forecast_upper[start_idx:]
    lower_cut = forecast_lower[start_idx:]

    # Plot confidence interval
    ax.fill_between(
        x, lower_cut, upper_cut, alpha=0.3, color=fill_color, label="_nolegend_"
    )

    # Plot forecasted
    ax.plot(
        x,
        fcast_cut,
        color=colors[0],
        linestyle="--",
        linewidth=line_width,
        label="Forecast",
    )

    # Plot observed
    ax.plot(x, obs_cut, color=colors[1], linewidth=line_width, label="Observed")

    # Intervention line
    ax.axvline(
        x=int_date,
        color="gray",
        linestyle="--",
        linewidth=line_width,
        label=f"Intervention ({int_date.date()})",
    )

    # Horizon lines
    if horizon:
        for h in horizon:
            h_ts = pd.Timestamp(h)
            ax.axvline(
                x=h_ts, color="gray", linestyle="-.", linewidth=line_width * 0.8
            )

    ax.set_title("Forecasted Series")
    ax.legend()
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    return fig


def _plot_impact(
    result: CausalArimaResult,
    horizon: Optional[List] = None,
    alpha: float = 0.05,
    color_line: str = "darkblue",
    color_intervals: str = "lightsteelblue",
    line_width: float = 1.0,
    figsize: tuple = (12, 5),
) -> Dict[str, Figure]:
    """Plot point and cumulative causal effects.

    Parameters
    ----------
    result : CausalArimaResult
        Fitted model results.
    horizon : list of dates, optional
        Dates to highlight.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    color_line : str, default='darkblue'
        Line color.
    color_intervals : str, default='lightsteelblue'
        Confidence interval fill color.
    line_width : float, default=1.0
        Line width.
    figsize : tuple, default=(12, 5)
        Figure size for each plot.

    Returns
    -------
    dict
        Dictionary with 'plot' and 'cumulative_plot' keys.
    """
    dates = result.dates
    int_date = result.intervention_date

    # Post-intervention dates (excluding NaN effects)
    post_dates = dates[dates >= int_date]
    valid_mask = ~np.isnan(result.causal_effect)
    x = post_dates[valid_mask]

    # Point effect
    y = result.causal_effect[valid_mask]
    sd = result.norm.sd_tau
    z = stats.norm.ppf(1 - alpha / 2)
    y_upper = y + sd * z
    y_lower = y - sd * z

    # Create point effect plot
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.fill_between(x, y_lower, y_upper, alpha=0.3, color=color_intervals)
    ax1.axhline(y=0, color="gray", linewidth=line_width)
    ax1.plot(x, y, color=color_line, linestyle="--", linewidth=line_width)

    if horizon:
        for h in horizon:
            ax1.axvline(
                x=pd.Timestamp(h),
                color="gray",
                linestyle="--",
                linewidth=line_width * 0.8,
            )

    ax1.set_title("Point Effect")
    plt.tight_layout()

    # Cumulative effect
    y_cum = result.norm.cumulative
    sd_cum = result.norm.sd_cumulative
    y_cum_upper = y_cum + sd_cum * z
    y_cum_lower = y_cum - sd_cum * z

    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.fill_between(x, y_cum_lower, y_cum_upper, alpha=0.3, color=color_intervals)
    ax2.axhline(y=0, color="gray", linewidth=line_width)
    ax2.plot(x, y_cum, color=color_line, linestyle="--", linewidth=line_width)

    if horizon:
        for h in horizon:
            ax2.axvline(
                x=pd.Timestamp(h),
                color="gray",
                linestyle="--",
                linewidth=line_width * 0.8,
            )

    ax2.set_title("Cumulative Effect")
    plt.tight_layout()

    return {"plot": fig1, "cumulative_plot": fig2}


def _plot_residuals(
    result: CausalArimaResult,
    max_lags: int = 30,
    figsize: tuple = (12, 4),
) -> Dict[str, Figure]:
    """Plot residual diagnostics (ACF, PACF, QQ plot).

    Parameters
    ----------
    result : CausalArimaResult
        Fitted model results.
    max_lags : int, default=30
        Maximum lags for ACF/PACF plots.
    figsize : tuple, default=(12, 4)
        Figure size for each plot.

    Returns
    -------
    dict
        Dictionary with 'ACF', 'PACF', and 'QQ_plot' keys.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # Get residuals
    if hasattr(result.model, "resid"):
        residuals = (
            result.model.resid()
            if callable(result.model.resid)
            else np.array(result.model.resid)
        )
    elif hasattr(result.model, "arima_res_"):
        residuals = np.array(result.model.arima_res_.resid)
    else:
        raise ValueError("Cannot extract residuals from model")

    # Remove NaN
    residuals = residuals[~np.isnan(residuals)]

    # Standardize
    std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    # ACF plot
    fig_acf, ax_acf = plt.subplots(figsize=figsize)
    plot_acf(std_residuals, lags=max_lags, ax=ax_acf)
    ax_acf.set_title("Autocorrelation Function")
    plt.tight_layout()

    # PACF plot
    fig_pacf, ax_pacf = plt.subplots(figsize=figsize)
    plot_pacf(std_residuals, lags=max_lags, ax=ax_pacf)
    ax_pacf.set_title("Partial Autocorrelation Function")
    plt.tight_layout()

    # QQ plot
    fig_qq, ax_qq = plt.subplots(figsize=figsize)
    stats.probplot(std_residuals, dist="norm", plot=ax_qq)
    ax_qq.set_title("Normal Q-Q Plot")
    plt.tight_layout()

    return {"ACF": fig_acf, "PACF": fig_pacf, "QQ_plot": fig_qq}
