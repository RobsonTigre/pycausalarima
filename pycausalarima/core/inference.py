"""Statistical inference functions for pycausalarima.

This module contains functions for computing normal-based and bootstrap-based
inference for causal effects. These are Python translations of the R functions:
- .norm.inf() -> normal_inference()
- .boot.inf() -> bootstrap_inference()
"""

import logging
from typing import Optional

import numpy as np
from scipy import stats

from pycausalarima.utils.types import InferenceResult

logger = logging.getLogger("pycausalarima")


def normal_inference(
    tau: np.ndarray,
    cumulative: np.ndarray,
    average: np.ndarray,
    sigma2: float,
    psi: np.ndarray,
) -> InferenceResult:
    """Compute Gaussian-based inference for causal effects.

    Python translation of R's .norm.inf() function.

    Parameters
    ----------
    tau : np.ndarray
        Point causal effects at each time point.
    cumulative : np.ndarray
        Cumulative effects (cumsum of tau).
    average : np.ndarray
        Temporal average effects (cumulative / t).
    sigma2 : float
        Model residual variance (sigma^2).
    psi : np.ndarray
        MA(infinity) psi weights [psi_0=1, psi_1, ..., psi_{h-1}].

    Returns
    -------
    InferenceResult
        Object containing all inference statistics including:
        - Point effect (tau) statistics
        - Cumulative effect statistics
        - Temporal average effect statistics
        - Standard errors and p-values for each

    Notes
    -----
    The variance formulas are:
    - Var(tau_t) = sigma^2 * sum(psi_j^2, j=0 to t-1)
    - Var(Delta_t) = sigma^2 * sum((sum(psi_j, j=0 to k))^2, k=0 to t-1)
    - Var(tau_bar_t) = Var(Delta_t) / t^2

    P-values are computed using the standard normal distribution:
    - Left-sided: P(Z < z)
    - Right-sided: P(Z > z) = 1 - P(Z < z)
    - Two-sided: 2 * (1 - P(Z < |z|))
    """
    n = len(tau)

    # Validate psi length matches tau length
    if len(psi) != n:
        raise ValueError(
            f"Psi weights length ({len(psi)}) must equal tau length ({n}). "
            "This indicates a bug in psi weight computation."
        )

    # Stat 1: tau (point effect)
    # Var(tau_t) = sigma^2 * sum(psi_j^2, j=0 to t-1)
    sd_tau = np.sqrt(sigma2 * np.cumsum(psi**2))
    z_tau = tau / sd_tau

    # Stat 2: cumulative effect (Delta)
    # Var(Delta_t) = sigma^2 * sum((cumsum(psi))_k^2, k=0 to t-1)
    psi_cumsum = np.cumsum(psi)
    sd_cumulative = np.sqrt(sigma2 * np.cumsum(psi_cumsum**2))

    z_cumulative = cumulative / sd_cumulative

    # Stat 3: temporal average effect
    # Var(avg_t) = Var(Delta_t) / t^2
    t_indices = np.arange(1, n + 1)
    sd_average = sd_cumulative / t_indices
    z_average = average / sd_average

    return InferenceResult(
        type="norm",
        tau=tau,
        sd_tau=sd_tau,
        pvalue_tau_left=stats.norm.cdf(z_tau),
        pvalue_tau_bidirectional=2 * (1 - stats.norm.cdf(np.abs(z_tau))),
        pvalue_tau_right=1 - stats.norm.cdf(z_tau),
        cumulative=cumulative,
        sd_cumulative=sd_cumulative,
        pvalue_sum_left=stats.norm.cdf(z_cumulative),
        pvalue_sum_bidirectional=2 * (1 - stats.norm.cdf(np.abs(z_cumulative))),
        pvalue_sum_right=1 - stats.norm.cdf(z_cumulative),
        average=average,
        sd_average=sd_average,
        pvalue_avg_left=stats.norm.cdf(z_average),
        pvalue_avg_bidirectional=2 * (1 - stats.norm.cdf(np.abs(z_average))),
        pvalue_avg_right=1 - stats.norm.cdf(z_average),
        boot_distribution=None,
    )


def bootstrap_inference(
    model,
    h: int,
    n_boot: int,
    y_post: np.ndarray,
    xreg: Optional[np.ndarray] = None,
) -> InferenceResult:
    """Compute bootstrap-based inference for causal effects.

    Generates bootstrap simulations by resampling model residuals
    and computing empirical distributions of effects.

    Python translation of R's .boot.inf() function.

    Parameters
    ----------
    model : object
        Fitted ARIMA model (statsmodels or pmdarima).
    h : int
        Forecast horizon (number of post-intervention observations).
    n_boot : int
        Number of bootstrap iterations.
    y_post : np.ndarray
        Observed post-intervention values.
    xreg : np.ndarray, optional
        Exogenous regressors for post-intervention period.

    Returns
    -------
    InferenceResult
        Object containing bootstrap inference statistics including:
        - Point effect (tau) statistics
        - Cumulative effect statistics
        - Temporal average effect statistics
        - Standard errors and p-values for each
        - Bootstrap distribution matrix

    Notes
    -----
    The bootstrap procedure:
    1. For each iteration, simulate future paths using bootstrapped residuals
    2. Compute effects as: observed - simulated
    3. Calculate empirical mean, sd, and p-values from the distribution

    P-values are computed empirically:
    - Left-sided: proportion of bootstrap effects > 0
    - Right-sided: proportion of bootstrap effects < 0
    - Two-sided: 2 - 2 * max(prop < 0, prop > 0)
    """
    # Generate bootstrap simulations
    simulated = np.zeros((h, n_boot))

    for i in range(n_boot):
        # Simulate from model with bootstrapped residuals
        sim = _simulate_arima(model, h, xreg, bootstrap=True)
        simulated[:, i] = sim

    # Handle NaN in y_post
    valid_mask = ~np.isnan(y_post)
    y_valid = y_post[valid_mask]
    sim_valid = simulated[valid_mask, :]

    # Compute effect distributions
    # dist1: tau distribution (observed - simulated)
    dist1 = y_valid[:, np.newaxis] - sim_valid
    tau = np.mean(dist1, axis=1)
    sd_tau = np.std(dist1, axis=1, ddof=1)

    # dist2: cumulative distribution
    dist2 = np.cumsum(dist1, axis=0)
    cumulative = np.mean(dist2, axis=1)
    sd_cumulative = np.std(dist2, axis=1, ddof=1)

    # dist3: temporal average distribution
    t_indices = np.arange(1, len(y_valid) + 1)[:, np.newaxis]
    dist3 = dist2 / t_indices
    average = np.mean(dist3, axis=1)
    sd_average = np.std(dist3, axis=1, ddof=1)

    # Compute empirical p-values
    def empirical_pvalues(dist):
        """Compute empirical p-values from bootstrap distribution."""
        p_left = np.mean(dist > 0, axis=1)
        p_right = np.mean(dist < 0, axis=1)
        p_bidirectional = 2 - 2 * np.maximum(
            np.mean(dist < 0, axis=1),
            np.mean(dist > 0, axis=1),
        )
        return p_left, p_bidirectional, p_right

    pv_tau_l, pv_tau_b, pv_tau_r = empirical_pvalues(dist1)
    pv_sum_l, pv_sum_b, pv_sum_r = empirical_pvalues(dist2)
    pv_avg_l, pv_avg_b, pv_avg_r = empirical_pvalues(dist3)

    return InferenceResult(
        type="boot",
        tau=tau,
        sd_tau=sd_tau,
        pvalue_tau_left=pv_tau_l,
        pvalue_tau_bidirectional=pv_tau_b,
        pvalue_tau_right=pv_tau_r,
        cumulative=cumulative,
        sd_cumulative=sd_cumulative,
        pvalue_sum_left=pv_sum_l,
        pvalue_sum_bidirectional=pv_sum_b,
        pvalue_sum_right=pv_sum_r,
        average=average,
        sd_average=sd_average,
        pvalue_avg_left=pv_avg_l,
        pvalue_avg_bidirectional=pv_avg_b,
        pvalue_avg_right=pv_avg_r,
        boot_distribution=simulated,
    )


def _simulate_arima(
    model,
    nsim: int,
    xreg: Optional[np.ndarray] = None,
    bootstrap: bool = True,
) -> np.ndarray:
    """Simulate future values from fitted ARIMA model.

    This uses statsmodels' simulate() method for proper recursive ARIMA
    simulation, matching R's simulate.Arima() behavior. For bootstrap,
    innovations are resampled from model residuals.

    Parameters
    ----------
    model : object
        Fitted ARIMA model (statsmodels or pmdarima).
    nsim : int
        Number of steps to simulate.
    xreg : np.ndarray, optional
        Exogenous regressors for simulation period.
    bootstrap : bool, default=True
        If True, use bootstrapped residuals as innovations; otherwise,
        sample from normal distribution.

    Returns
    -------
    np.ndarray
        Simulated values of length nsim.

    Notes
    -----
    R's CausalArima uses: simulate(model, future=TRUE, nsim=h, bootstrap=TRUE)
    which performs proper ARIMA recursion:
        y_t = phi_1*y_{t-1} + ... + e_t + theta_1*e_{t-1} + ...

    This implementation matches that behavior using statsmodels' simulate()
    method with anchor='end' to continue from the fitted series.
    """
    # Get underlying statsmodels result if pmdarima
    if hasattr(model, "arima_res_"):
        sm_model = model.arima_res_
    else:
        sm_model = model

    # Get residuals for bootstrapping
    residuals = _get_model_residuals(sm_model)

    # Get sigma2 for normal sampling fallback
    sigma2 = _get_model_sigma2(sm_model, residuals)

    # Generate bootstrap innovations by resampling residuals
    if bootstrap and len(residuals) > 0:
        boot_innovations = np.random.choice(residuals, size=nsim, replace=True)
    else:
        sigma = np.sqrt(sigma2) if sigma2 > 0 else 1.0
        boot_innovations = np.random.normal(0, sigma, nsim)

    # Try to use statsmodels simulate() for proper ARIMA recursion
    # This matches R's simulate.Arima(model, future=TRUE, nsim=h, bootstrap=TRUE)
    if hasattr(sm_model, 'simulate'):
        try:
            # In statsmodels state-space framework, ARIMA innovations enter via
            # state_shocks. For ARIMA models, k_posdef=1 (single innovation).
            # pretransformed_state_shocks=True means shocks are already on the
            # sigma scale (our residuals are), so no further transformation.
            state_shocks = boot_innovations.reshape(-1, 1)
            simulated = sm_model.simulate(
                nsimulations=nsim,
                anchor='end',
                exog=xreg,
                state_shocks=state_shocks,
                pretransformed_state_shocks=True,
            )
            # Convert to numpy array if needed
            if hasattr(simulated, 'values'):
                simulated = simulated.values
            return np.asarray(simulated).flatten()
        except Exception as e:
            logger.warning(
                f"ARIMA simulate() failed: {e}. Falling back to forecast+noise "
                "approximation which ignores AR/MA recursive structure. "
                "Bootstrap confidence intervals may be less accurate."
            )
    else:
        logger.warning(
            "Model does not support simulate(). Using forecast+noise fallback "
            "for bootstrap simulation."
        )

    # Fallback: forecast mean + noise (less accurate — ignores AR/MA recursion)
    mean_forecast = _get_model_forecast(sm_model, nsim, xreg)
    return mean_forecast + boot_innovations


def _get_model_forecast(sm_model, nsim: int, xreg: Optional[np.ndarray]) -> np.ndarray:
    """Get forecast mean from statsmodels model."""
    try:
        if hasattr(sm_model, "get_forecast"):
            fcast = sm_model.get_forecast(steps=nsim, exog=xreg)
            mean_fcast = fcast.predicted_mean
            if hasattr(mean_fcast, "values"):
                mean_fcast = mean_fcast.values
            return np.asarray(mean_fcast)
        elif hasattr(sm_model, "forecast"):
            mean_fcast = sm_model.forecast(steps=nsim, exog=xreg)
            if hasattr(mean_fcast, "values"):
                mean_fcast = mean_fcast.values
            return np.asarray(mean_fcast)
    except (AttributeError, ValueError, TypeError) as e:
        logger.warning(f"Forecast method failed: {e}, using zeros as baseline")

    # Fallback: return zeros (simulation will just be noise)
    return np.zeros(nsim)


def _get_model_residuals(sm_model) -> np.ndarray:
    """Extract residuals from statsmodels model.

    Returns empty array if residuals cannot be extracted, with a warning.
    Bootstrap will fall back to normal sampling in that case.
    """
    if hasattr(sm_model, "resid"):
        resid = sm_model.resid
        if hasattr(resid, "values"):
            resid = resid.values
        resid = np.asarray(resid)
        valid = resid[~np.isnan(resid)]
        if len(valid) > 0:
            return valid
    logger.warning(
        "Could not extract valid residuals from the fitted model. "
        "Bootstrap will use parametric (normal) sampling instead of "
        "residual resampling."
    )
    return np.array([])


def _get_model_sigma2(sm_model, residuals: np.ndarray) -> float:
    """Get residual variance from model."""
    if hasattr(sm_model, "sigma2"):
        return float(sm_model.sigma2)
    elif len(residuals) > 1:
        return float(np.var(residuals, ddof=1))
    return 1.0


