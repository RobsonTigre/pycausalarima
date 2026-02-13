"""Main CausalArima class for causal effect estimation.

This module contains the CausalArima class, which is the main entry point
for estimating causal effects of interventions on time series using ARIMA models.

This is a Python translation of the R CausalArima() function.
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from pycausalarima.core.arma_utils import compute_psi_weights, sarma_to_larma
from pycausalarima.core.inference import bootstrap_inference, normal_inference
from pycausalarima.exceptions import CoefficientExtractionError, ModelFittingError
from pycausalarima.utils.types import ARIMAOrder, CausalArimaResult, InferenceResult
from pycausalarima.utils.validation import validate_inputs

logger = logging.getLogger("pycausalarima")


class CausalArima:
    """Estimate causal effects of interventions using ARIMA models.

    This class implements the C-ARIMA methodology for estimating causal
    effects in time series interrupted by interventions.

    Parameters
    ----------
    y : array-like
        Univariate time series observations.
    dates : array-like of datetime
        Dates corresponding to each observation.
    intervention_date : datetime
        Date when the intervention occurred.
    auto : bool, default=True
        If True, use auto_arima to select best model.
    order : tuple of (p, d, q), optional
        ARIMA order. Required when auto=False.
    seasonal_order : tuple of (P, D, Q, s), optional
        Seasonal ARIMA order. Default is (0, 0, 0, 1).
    xreg : array-like, optional
        Exogenous regressors.
    ic : str, default='aic'
        Information criterion for model selection ('aic', 'bic', 'aicc').
    n_boot : int, optional
        Number of bootstrap iterations for inference.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    **arima_kwargs
        Additional arguments passed to ARIMA/auto_arima.

    Attributes
    ----------
    result_ : CausalArimaResult
        Results after calling fit().
    model_ : ARIMAResults
        Fitted ARIMA model from statsmodels.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pycausalarima import CausalArima
    >>>
    >>> # Generate sample data
    >>> n = 100
    >>> dates = pd.date_range('2014-01-05', periods=n, freq='D')
    >>> y = 0.5 * np.arange(n) + np.random.normal(0, 6, n)
    >>> intervention_date = pd.Timestamp('2014-03-16')
    >>> y[dates >= intervention_date] *= 1.25
    >>>
    >>> # Fit model
    >>> ca = CausalArima(y, dates, intervention_date, n_boot=1000)
    >>> result = ca.fit()
    >>>
    >>> # View summary
    >>> ca.summary()
    >>>
    >>> # Plot results
    >>> ca.plot(type='forecast')
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.Series, List[float]],
        dates: Union[np.ndarray, pd.DatetimeIndex, List],
        intervention_date: Union[pd.Timestamp, str],
        auto: bool = True,
        order: Optional[tuple] = None,
        seasonal_order: Optional[tuple] = None,
        xreg: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        ic: Literal["aic", "bic", "aicc"] = "aic",
        n_boot: Optional[int] = None,
        alpha: float = 0.05,
        **arima_kwargs,
    ):
        # Convert inputs to standard formats
        self.y = np.asarray(y, dtype=float)
        self.dates = pd.DatetimeIndex(dates)
        self.intervention_date = pd.Timestamp(intervention_date)
        self.auto = auto
        self.order = order if order is not None else (0, 0, 0)
        self.seasonal_order = seasonal_order if seasonal_order is not None else (0, 0, 0, 1)
        self.xreg = np.asarray(xreg, dtype=float) if xreg is not None else None
        self.ic = ic
        self.n_boot = n_boot
        self.alpha = alpha
        self.arima_kwargs = arima_kwargs

        # If order is specified, disable auto
        if order is not None and sum(order) > 0:
            self.auto = False
        if seasonal_order is not None and sum(seasonal_order[:3]) > 0:
            self.auto = False

        # Results (populated after fit)
        self.result_: Optional[CausalArimaResult] = None
        self.model_: Optional[Any] = None

        # Validate inputs
        validate_inputs(
            self.y,
            self.dates,
            self.intervention_date,
            self.xreg,
            self.alpha,
            self.n_boot,
        )

    def fit(self) -> CausalArimaResult:
        """Fit the causal ARIMA model and compute effects.

        Returns
        -------
        CausalArimaResult
            Object containing all estimation results.
        """
        # Step 1: Split data at intervention
        pre_mask = self.dates < self.intervention_date
        post_mask = ~pre_mask

        y_pre = self.y[pre_mask]
        y_post = self.y[post_mask]

        xreg_pre = self.xreg[pre_mask] if self.xreg is not None else None
        xreg_post = self.xreg[post_mask] if self.xreg is not None else None

        # Ensure xreg is 2D
        if xreg_pre is not None and xreg_pre.ndim == 1:
            xreg_pre = xreg_pre.reshape(-1, 1)
        if xreg_post is not None and xreg_post.ndim == 1:
            xreg_post = xreg_post.reshape(-1, 1)

        # Step 2: Fit ARIMA on pre-intervention data
        self.model_ = self._fit_arima(y_pre, xreg_pre)

        # Step 3: Forecast counterfactual
        h = len(y_post)
        forecast_result = self._forecast(h, xreg_post)
        mean_forecast = forecast_result["mean"]
        forecast_lower = forecast_result["lower"]
        forecast_upper = forecast_result["upper"]

        # Step 4: Compute causal effects
        causal_effect = y_post - mean_forecast

        # Step 5: Handle NaN values
        valid_mask = ~np.isnan(causal_effect)
        tau = causal_effect[valid_mask]

        # Step 6: Compute psi weights for variance computation
        # NOTE: We use psi weights (not Kalman filter variance) to match R's CausalArima
        # package exactly. R ignores the differencing parameter (d) in variance computation
        # and uses: sd = sqrt(sigma2 * cumsum(psi^2)). While Kalman filter variance is
        # more statistically rigorous for integrated ARIMA models, using psi weights
        # ensures consistency with R's published methodology.
        psi = self._compute_psi_weights(h)
        psi_valid = psi[valid_mask]

        # Step 7: Compute statistics
        cumulative = np.cumsum(tau)
        average = cumulative / np.arange(1, len(tau) + 1)

        # Step 8: Normal-based inference (using psi weights to match R)
        sigma2 = self._get_sigma2()
        norm_inf = normal_inference(
            tau, cumulative, average, sigma2, psi_valid
        )

        # Step 9: Bootstrap inference (if requested)
        boot_inf = None
        if self.n_boot is not None and self.n_boot > 0:
            boot_inf = bootstrap_inference(
                self.model_, h, self.n_boot, y_post, xreg_post
            )

        # Step 10: Compile results
        self.result_ = CausalArimaResult(
            norm=norm_inf,
            boot=boot_inf,
            causal_effect=causal_effect,
            model=self.model_,
            order=self._get_order(),
            y=self.y,
            dates=self.dates,
            intervention_date=self.intervention_date,
            xreg=self.xreg,
            forecast=mean_forecast,
            forecast_lower=forecast_lower,
            forecast_upper=forecast_upper,
            alpha=self.alpha,
        )

        return self.result_

    def get_residuals(self, standardized: bool = False) -> np.ndarray:
        """Get model residuals for diagnostics.

        This provides a clean public API for accessing residuals, avoiding
        the need to access internal model attributes directly.

        Parameters
        ----------
        standardized : bool, default=False
            If True, return standardized residuals (divided by residual std).
            Standardized residuals are useful for diagnostic tests like
            Ljung-Box and normality checks.

        Returns
        -------
        np.ndarray
            Model residuals from the pre-intervention ARIMA fit.

        Raises
        ------
        ValueError
            If fit() has not been called yet.

        Examples
        --------
        >>> ca = CausalArima(y, dates, intervention_date)
        >>> result = ca.fit()
        >>> residuals = ca.get_residuals()
        >>> standardized_residuals = ca.get_residuals(standardized=True)
        >>>
        >>> # Use for Ljung-Box test
        >>> from statsmodels.stats.diagnostic import acorr_ljungbox
        >>> ljung_box = acorr_ljungbox(standardized_residuals, lags=[10])
        """
        if self.result_ is None:
            raise ValueError("Must call fit() before get_residuals()")

        resid = self._get_raw_residuals()

        if len(resid) == 0:
            raise ValueError("Could not extract residuals from model")

        if standardized:
            # Try to get standardized forecast errors (Kalman filter innovations)
            # which are the proper residuals for state-space diagnostic tests
            sm_model = None
            if hasattr(self.model_, "arima_res_"):
                sm_model = self.model_.arima_res_
            else:
                sm_model = self.model_

            if hasattr(sm_model, "standardized_forecasts_error"):
                std_resid = sm_model.standardized_forecasts_error
                if hasattr(std_resid, "__getitem__"):
                    std_resid = std_resid[0]  # First state variable
                return np.asarray(std_resid)

            # Fallback: standardize manually
            sigma = np.sqrt(self._get_sigma2())
            return resid / sigma

        return resid

    def _fit_arima(self, y: np.ndarray, xreg: Optional[np.ndarray]) -> Any:
        """Fit ARIMA model using auto or manual specification.

        Raises
        ------
        ModelFittingError
            If ARIMA model fitting fails.
        """
        if self.auto:
            # Use pmdarima for auto ARIMA
            import pmdarima as pm

            # Determine if seasonal modeling is needed
            # Only pass seasonal parameters when explicitly requested (m > 1)
            seasonal_period = self.seasonal_order[3] if len(self.seasonal_order) > 3 else 1
            use_seasonal = seasonal_period > 1

            try:
                if use_seasonal:
                    model = pm.auto_arima(
                        y,
                        X=xreg,
                        information_criterion=self.ic,
                        seasonal=True,
                        m=seasonal_period,
                        suppress_warnings=True,
                        error_action="ignore",
                        **self.arima_kwargs,
                    )
                else:
                    # Let pmdarima use defaults for non-seasonal models
                    # This matches R's auto.arima behavior more closely
                    model = pm.auto_arima(
                        y,
                        X=xreg,
                        information_criterion=self.ic,
                        suppress_warnings=True,
                        error_action="ignore",
                        **self.arima_kwargs,
                    )
            except Exception as e:
                raise ModelFittingError(
                    f"auto_arima fitting failed: {e}"
                ) from e
            # Store extracted order
            self._fitted_order = model.order
            self._fitted_seasonal_order = model.seasonal_order
            return model
        else:
            # Use statsmodels directly
            from statsmodels.tsa.arima.model import ARIMA

            # Handle seasonal order
            seasonal_order = self.seasonal_order
            if len(seasonal_order) == 3:
                seasonal_order = (*seasonal_order, 1)

            # Build model kwargs - only include seasonal_order if needed
            use_seasonal = seasonal_order[3] > 1
            model_kwargs = {
                "endog": y,
                "order": self.order,
                "exog": xreg,
                **self.arima_kwargs,
            }
            if use_seasonal:
                model_kwargs["seasonal_order"] = seasonal_order

            try:
                model = ARIMA(**model_kwargs)
                result = model.fit()
            except Exception as e:
                raise ModelFittingError(
                    f"ARIMA({self.order}) fitting failed: {e}"
                ) from e
            self._fitted_order = self.order
            self._fitted_seasonal_order = seasonal_order if use_seasonal else (0, 0, 0, 1)
            return result

    def _forecast(self, h: int, xreg: Optional[np.ndarray]) -> dict:
        """Generate forecasts with confidence intervals.

        Returns a dict with keys: mean, lower, upper.
        """
        # Handle pmdarima model
        if hasattr(self.model_, "predict") and hasattr(self.model_, "arima_res_"):
            # pmdarima ARIMA
            fcast = self.model_.predict(n_periods=h, X=xreg, return_conf_int=True, alpha=self.alpha)
            if isinstance(fcast, tuple):
                mean_fcast = fcast[0]
                conf_int = fcast[1]

                return {
                    "mean": np.array(mean_fcast),
                    "lower": np.array(conf_int[:, 0]),
                    "upper": np.array(conf_int[:, 1]),
                }

        # Handle statsmodels model directly
        if hasattr(self.model_, "get_forecast"):
            forecast_obj = self.model_.get_forecast(steps=h, exog=xreg)
            conf_int = forecast_obj.conf_int(alpha=self.alpha)

            # Handle both pandas Series/DataFrame and numpy arrays
            mean_pred = forecast_obj.predicted_mean
            if hasattr(mean_pred, "values"):
                mean_pred = mean_pred.values
            mean_pred = np.asarray(mean_pred)

            if hasattr(conf_int, "iloc"):
                lower = conf_int.iloc[:, 0].values
                upper = conf_int.iloc[:, 1].values
            else:
                conf_int = np.asarray(conf_int)
                lower = conf_int[:, 0]
                upper = conf_int[:, 1]

            return {
                "mean": mean_pred,
                "lower": lower,
                "upper": upper,
            }

        raise ValueError("Model does not support forecasting")

    def _compute_psi_weights(self, h: int) -> np.ndarray:
        """Compute MA(infinity) psi weights from ARIMA model.

        Extracts AR, MA, SAR, and SMA coefficients from the fitted model,
        matching R's approach of extracting coefficients by name:
            ar  <- coef[substr(names(coef), 1, 2) == "ar"]
            sar <- coef[substr(names(coef), 1, 3) == "sar"]
        """
        # Extract coefficients
        if hasattr(self.model_, "arparams"):
            # pmdarima or statsmodels with arparams method
            ar = self.model_.arparams() if callable(self.model_.arparams) else np.array(self.model_.arparams if self.model_.arparams is not None else [])
            ma = self.model_.maparams() if callable(self.model_.maparams) else np.array(self.model_.maparams if self.model_.maparams is not None else [])

            # Extract seasonal coefficients from underlying statsmodels result
            # This matches R's approach: coef[substr(names(coef), 1, 3) == "sar"]
            sar = np.array([])
            sma = np.array([])
            s = 1

            # Get seasonal period from fitted order
            if hasattr(self, "_fitted_seasonal_order"):
                if hasattr(self._fitted_seasonal_order, "__len__") and len(self._fitted_seasonal_order) > 3:
                    s = self._fitted_seasonal_order[3]

            # Try to extract SAR/SMA from pmdarima's underlying statsmodels result
            sm_result = None
            if hasattr(self.model_, 'arima_res_'):
                sm_result = self.model_.arima_res_
            elif hasattr(self.model_, 'specification'):
                # Direct statsmodels ARIMA result (auto=False path)
                sm_result = self.model_

            if sm_result is not None:
                # Get seasonal period from model specification if available
                if hasattr(sm_result, 'specification'):
                    spec_s = sm_result.specification.get('seasonal_periods', None)
                    if spec_s is not None and spec_s > 1:
                        s = spec_s

                # First try: extract from parameter names (works for both paths)
                if hasattr(sm_result, 'params'):
                    params = sm_result.params
                    param_names = list(params.index) if hasattr(params, 'index') else []

                    # statsmodels uses 'ar.S.L{s}', 'ma.S.L{s}' for seasonal params
                    # Match patterns like 'ar.S.L7', 'ar.S.L12', 'ma.S.L7', etc.
                    sar_names = sorted([n for n in param_names if 'ar.S.' in n or 'ar.s.' in n.lower()])
                    sma_names = sorted([n for n in param_names if 'ma.S.' in n or 'ma.s.' in n.lower()])

                    if sar_names:
                        sar = np.array([float(params[n]) for n in sar_names])
                        logger.debug(f"Extracted SAR coefficients: {sar} from {sar_names}")
                    if sma_names:
                        sma = np.array([float(params[n]) for n in sma_names])
                        logger.debug(f"Extracted SMA coefficients: {sma} from {sma_names}")

                # Fallback: extract from polynomial_seasonal_ar/ma attributes
                if len(sar) == 0 and hasattr(sm_result, "polynomial_seasonal_ar"):
                    poly_sar = np.array(sm_result.polynomial_seasonal_ar)
                    if len(poly_sar) > 1 and s > 1:
                        sar_indices = np.arange(s, len(poly_sar), s)
                        sar_coeffs = [-poly_sar[i] for i in sar_indices if i < len(poly_sar) and poly_sar[i] != 0]
                        if sar_coeffs:
                            sar = np.array(sar_coeffs)
                            logger.debug(f"Extracted SAR from polynomial: {sar}")

                if len(sma) == 0 and hasattr(sm_result, "polynomial_seasonal_ma"):
                    poly_sma = np.array(sm_result.polynomial_seasonal_ma)
                    if len(poly_sma) > 1 and s > 1:
                        sma_indices = np.arange(s, len(poly_sma), s)
                        sma_coeffs = [poly_sma[i] for i in sma_indices if i < len(poly_sma) and poly_sma[i] != 0]
                        if sma_coeffs:
                            sma = np.array(sma_coeffs)
                            logger.debug(f"Extracted SMA from polynomial: {sma}")

        elif hasattr(self.model_, "polynomial_ar"):
            # statsmodels SARIMAX format
            poly_ar = np.array(self.model_.polynomial_ar)
            poly_ma = np.array(self.model_.polynomial_ma)
            ar = -poly_ar[1:] if len(poly_ar) > 1 else np.array([])
            ma = poly_ma[1:] if len(poly_ma) > 1 else np.array([])

            # Extract seasonal coefficients from polynomial representation
            sar = np.array([])
            sma = np.array([])
            s = 1

            if hasattr(self.model_, "specification"):
                s = self.model_.specification.get("seasonal_periods", 1) or 1

            # Extract SAR from polynomial_seasonal_ar
            if hasattr(self.model_, "polynomial_seasonal_ar"):
                poly_sar = np.array(self.model_.polynomial_seasonal_ar)
                if len(poly_sar) > 1 and s > 1:
                    # Extract non-zero seasonal coefficients
                    # polynomial_seasonal_ar has form [1, 0, 0, ..., -sar1, 0, 0, ..., -sar2, ...]
                    sar_indices = np.arange(s, len(poly_sar), s)
                    sar_coeffs = [-poly_sar[i] for i in sar_indices if i < len(poly_sar) and poly_sar[i] != 0]
                    if sar_coeffs:
                        sar = np.array(sar_coeffs)
                        logger.debug(f"Extracted SAR from polynomial: {sar}")

            # Extract SMA from polynomial_seasonal_ma
            if hasattr(self.model_, "polynomial_seasonal_ma"):
                poly_sma = np.array(self.model_.polynomial_seasonal_ma)
                if len(poly_sma) > 1 and s > 1:
                    # polynomial_seasonal_ma has form [1, 0, 0, ..., sma1, 0, 0, ..., sma2, ...]
                    sma_indices = np.arange(s, len(poly_sma), s)
                    sma_coeffs = [poly_sma[i] for i in sma_indices if i < len(poly_sma) and poly_sma[i] != 0]
                    if sma_coeffs:
                        sma = np.array(sma_coeffs)
                        logger.debug(f"Extracted SMA from polynomial: {sma}")
        else:
            raise CoefficientExtractionError(
                "Could not extract AR/MA coefficients from fitted model. "
                "The model object does not have 'arparams' or 'polynomial_ar' attributes. "
                "Ensure you are using a supported statsmodels or pmdarima ARIMA model."
            )

        # Convert SARMA to LARMA
        ar_long, ma_long = sarma_to_larma(ar, ma, sar, sma, s)

        # Compute psi weights (h-1 to match R's lag.max = h-1)
        psi = compute_psi_weights(ar_long, ma_long, h - 1)
        return psi

    def _get_sigma2(self) -> float:
        """Get innovation variance (sigma^2) from fitted model.

        This extracts sigma2 from the model's estimated parameters, matching R's
        approach. For statsmodels SARIMAX/ARIMA, sigma2 is the last parameter
        in the params array. This is the MLE estimate of the innovation variance,
        NOT the sample variance of residuals.

        Returns
        -------
        float
            Innovation variance (sigma^2) from the fitted model.

        Raises
        ------
        ValueError
            If sigma2 cannot be extracted from the model.
        """
        # Get underlying statsmodels model
        if hasattr(self.model_, "arima_res_"):
            sm_model = self.model_.arima_res_
        else:
            sm_model = self.model_

        # Method 1: Extract from params (statsmodels SARIMAX stores sigma2 as last param)
        if hasattr(sm_model, "params"):
            params = sm_model.params
            if hasattr(params, "values"):
                params = params.values
            params = np.asarray(params)
            # sigma2 is the last parameter in statsmodels SARIMAX
            sigma2 = float(params[-1])
            if sigma2 > 0:
                logger.debug(f"Extracted sigma2 from model params: {sigma2:.6f}")
                return sigma2

        # Method 2: Try model's sigma2 attribute directly
        if hasattr(sm_model, "sigma2") and sm_model.sigma2 is not None:
            sigma2 = float(sm_model.sigma2)
            if sigma2 > 0:
                logger.debug(f"Extracted sigma2 from model attribute: {sigma2:.6f}")
                return sigma2

        # Method 3: Fallback to residual variance (less accurate for ARIMA)
        resid = self._get_raw_residuals()
        if len(resid) > 0:
            sigma2 = float(np.var(resid, ddof=1))
            logger.warning(
                f"Could not extract sigma2 from model, falling back to residual variance: {sigma2:.6f}"
            )
            return sigma2

        raise ValueError(
            "Could not extract sigma2 from fitted model. "
            "Please ensure the model is properly fitted."
        )

    def _get_raw_residuals(self) -> np.ndarray:
        """Extract raw residuals from the fitted model.

        Returns
        -------
        np.ndarray
            Model residuals with NaN values removed.
        """
        resid = None

        # Try pmdarima model first
        if hasattr(self.model_, "arima_res_"):
            sm_model = self.model_.arima_res_
            if hasattr(sm_model, "resid"):
                resid = sm_model.resid
        # Try direct resid attribute
        elif hasattr(self.model_, "resid"):
            resid = self.model_.resid() if callable(self.model_.resid) else self.model_.resid

        if resid is None:
            return np.array([])

        # Convert to numpy and remove NaN
        if hasattr(resid, "values"):
            resid = resid.values
        resid = np.asarray(resid)
        resid = resid[~np.isnan(resid)]

        return resid

    def _get_order(self) -> ARIMAOrder:
        """Extract ARIMA order from fitted model."""
        order = getattr(self, "_fitted_order", self.order)
        seasonal = getattr(self, "_fitted_seasonal_order", self.seasonal_order)

        return ARIMAOrder(
            p=order[0] if len(order) > 0 else 0,
            d=order[1] if len(order) > 1 else 0,
            q=order[2] if len(order) > 2 else 0,
            P=seasonal[0] if len(seasonal) > 0 else 0,
            D=seasonal[1] if len(seasonal) > 1 else 0,
            Q=seasonal[2] if len(seasonal) > 2 else 0,
            s=seasonal[3] if len(seasonal) > 3 else 1,
        )

    def summary(
        self,
        type: Literal["norm", "boot"] = "norm",
        horizon: Optional[List] = None,
        digits: int = 3,
    ) -> pd.DataFrame:
        """Generate summary of causal effects.

        Parameters
        ----------
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
        if self.result_ is None:
            raise ValueError("Must call fit() before summary()")

        from pycausalarima.reporting.summary import generate_summary

        return generate_summary(self.result_, type, horizon, digits)

    def plot(
        self,
        type: Literal["forecast", "impact", "residuals"] = "forecast",
        horizon: Optional[List] = None,
        **kwargs,
    ):
        """Create visualization of results.

        Parameters
        ----------
        type : str, default='forecast'
            Type of plot ('forecast', 'impact', 'residuals').
        horizon : list of dates, optional
            Dates to highlight in impact plots.
        **kwargs
            Additional plotting arguments.

        Returns
        -------
        matplotlib.figure.Figure or dict of figures
        """
        if self.result_ is None:
            raise ValueError("Must call fit() before plot()")

        from pycausalarima.visualization.plotting import plot_causal_arima

        return plot_causal_arima(self.result_, type, horizon, **kwargs)

    def impact(
        self,
        format: Literal["numeric", "html", "latex"] = "numeric",
        horizon: Optional[List] = None,
        digits: int = 3,
        **kwargs,
    ) -> Dict:
        """Generate formatted tables of impact results.

        Parameters
        ----------
        format : str, default='numeric'
            Output format ('numeric', 'html', 'latex').
        horizon : list of dates, optional
            Specific dates to include.
        digits : int, default=3
            Number of decimal places.
        **kwargs
            Additional formatting arguments.

        Returns
        -------
        dict
            Dictionary containing formatted tables.
        """
        if self.result_ is None:
            raise ValueError("Must call fit() before impact()")

        from pycausalarima.reporting.tables import generate_impact_tables

        return generate_impact_tables(self.result_, format, horizon, digits, **kwargs)

    def __repr__(self) -> str:
        """Return string representation."""
        status = "fitted" if self.result_ is not None else "not fitted"
        return f"CausalArima(intervention_date={self.intervention_date.date()}, auto={self.auto}, status={status})"
