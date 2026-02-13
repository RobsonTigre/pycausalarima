"""Type definitions and dataclasses for pycausalarima."""

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class ARIMAOrder:
    """ARIMA model order specification.

    Attributes
    ----------
    p : int
        AR order (autoregressive).
    d : int
        Differencing order.
    q : int
        MA order (moving average).
    P : int
        Seasonal AR order.
    D : int
        Seasonal differencing order.
    Q : int
        Seasonal MA order.
    s : int
        Seasonal period (e.g., 12 for monthly, 7 for daily with weekly seasonality).
    """

    p: int = 0
    d: int = 0
    q: int = 0
    P: int = 0
    D: int = 0
    Q: int = 0
    s: int = 1

    @property
    def order(self) -> tuple:
        """Return non-seasonal order as tuple (p, d, q)."""
        return (self.p, self.d, self.q)

    @property
    def seasonal_order(self) -> tuple:
        """Return seasonal order as tuple (P, D, Q, s)."""
        return (self.P, self.D, self.Q, self.s)

    def __str__(self) -> str:
        """Return string representation."""
        if self.P == 0 and self.D == 0 and self.Q == 0:
            return f"ARIMA({self.p},{self.d},{self.q})"
        return f"ARIMA({self.p},{self.d},{self.q})({self.P},{self.D},{self.Q})[{self.s}]"


@dataclass
class InferenceResult:
    """Results from statistical inference (normal or bootstrap).

    This class holds the inference statistics for causal effects,
    including point estimates, standard errors, and p-values for
    three types of effects: point (tau), cumulative (sum), and
    temporal average (avg).

    Attributes
    ----------
    type : str
        Type of inference: 'norm' for normal-based, 'boot' for bootstrap.
    tau : np.ndarray
        Point causal effects at each time point.
    sd_tau : np.ndarray
        Standard deviations of point effects.
    pvalue_tau_left : np.ndarray
        Left-sided p-values for point effects.
    pvalue_tau_bidirectional : np.ndarray
        Two-sided p-values for point effects.
    pvalue_tau_right : np.ndarray
        Right-sided p-values for point effects.
    cumulative : np.ndarray
        Cumulative causal effects.
    sd_cumulative : np.ndarray
        Standard deviations of cumulative effects.
    pvalue_sum_left : np.ndarray
        Left-sided p-values for cumulative effects.
    pvalue_sum_bidirectional : np.ndarray
        Two-sided p-values for cumulative effects.
    pvalue_sum_right : np.ndarray
        Right-sided p-values for cumulative effects.
    average : np.ndarray
        Temporal average causal effects.
    sd_average : np.ndarray
        Standard deviations of average effects.
    pvalue_avg_left : np.ndarray
        Left-sided p-values for average effects.
    pvalue_avg_bidirectional : np.ndarray
        Two-sided p-values for average effects.
    pvalue_avg_right : np.ndarray
        Right-sided p-values for average effects.
    boot_distribution : np.ndarray, optional
        Bootstrap distribution matrix (h x n_boot) if bootstrap inference.
    """

    type: Literal["norm", "boot"]
    tau: np.ndarray
    sd_tau: np.ndarray
    pvalue_tau_left: np.ndarray
    pvalue_tau_bidirectional: np.ndarray
    pvalue_tau_right: np.ndarray
    cumulative: np.ndarray
    sd_cumulative: np.ndarray
    pvalue_sum_left: np.ndarray
    pvalue_sum_bidirectional: np.ndarray
    pvalue_sum_right: np.ndarray
    average: np.ndarray
    sd_average: np.ndarray
    pvalue_avg_left: np.ndarray
    pvalue_avg_bidirectional: np.ndarray
    pvalue_avg_right: np.ndarray
    boot_distribution: Optional[np.ndarray] = None


@dataclass
class CausalArimaResult:
    """Complete results from CausalArima analysis.

    This class contains all the results from fitting a causal ARIMA model,
    including inference results, model information, and forecasts.

    Attributes
    ----------
    norm : InferenceResult
        Normal-based inference results.
    boot : InferenceResult, optional
        Bootstrap inference results (if n_boot was specified).
    causal_effect : np.ndarray
        Raw causal effects (observed - forecasted) at each post-intervention time.
    model : object
        Fitted ARIMA model from statsmodels/pmdarima.
    order : ARIMAOrder
        ARIMA order specification.
    y : np.ndarray
        Original time series data.
    dates : pd.DatetimeIndex
        Dates corresponding to observations.
    intervention_date : pd.Timestamp
        Date when intervention occurred.
    xreg : np.ndarray, optional
        Exogenous regressors if provided.
    forecast : np.ndarray
        Forecasted counterfactual values for post-intervention period.
    forecast_lower : np.ndarray
        Lower confidence bound for forecasts.
    forecast_upper : np.ndarray
        Upper confidence bound for forecasts.
    alpha : float
        Significance level used for confidence intervals.
    """

    norm: InferenceResult
    boot: Optional[InferenceResult]
    causal_effect: np.ndarray
    model: Any  # statsmodels ARIMAResults or pmdarima ARIMA
    order: ARIMAOrder
    y: np.ndarray
    dates: pd.DatetimeIndex
    intervention_date: pd.Timestamp
    xreg: Optional[np.ndarray]
    forecast: np.ndarray
    forecast_lower: np.ndarray
    forecast_upper: np.ndarray
    alpha: float

    @property
    def post_intervention_dates(self) -> pd.DatetimeIndex:
        """Get dates for post-intervention period."""
        return self.dates[self.dates >= self.intervention_date]

    @property
    def pre_intervention_dates(self) -> pd.DatetimeIndex:
        """Get dates for pre-intervention period."""
        return self.dates[self.dates < self.intervention_date]

    @property
    def n_pre(self) -> int:
        """Number of pre-intervention observations."""
        return len(self.pre_intervention_dates)

    @property
    def n_post(self) -> int:
        """Number of post-intervention observations."""
        return len(self.post_intervention_dates)
