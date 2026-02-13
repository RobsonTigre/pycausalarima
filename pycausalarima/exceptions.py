"""Custom exceptions for pycausalarima.

This module defines domain-specific exceptions that make error handling
more precise and debugging easier.
"""


class CausalArimaError(Exception):
    """Base exception for all pycausalarima errors.

    All custom exceptions in this package inherit from this class,
    making it easy to catch any pycausalarima-specific error.

    Examples
    --------
    >>> try:
    ...     ca.fit()
    ... except CausalArimaError as e:
    ...     print(f"pycausalarima error: {e}")
    """

    pass


class ValidationError(CausalArimaError, ValueError):
    """Raised when input validation fails.

    This exception is raised when:
    - Input arrays have incompatible shapes
    - Required parameters are missing or invalid
    - Date/intervention parameters are inconsistent

    Also subclasses ``ValueError`` for backward compatibility with code
    that catches ``ValueError`` from validation functions.

    Examples
    --------
    >>> ca = CausalArima(y=[1, 2], dates=[...], ...)  # Too few observations
    ValidationError: Time series must have at least 10 observations
    """

    pass


class ModelFittingError(CausalArimaError):
    """Raised when ARIMA model fitting fails.

    This exception is raised when:
    - auto_arima fails to find a suitable model
    - statsmodels ARIMA fitting fails to converge
    - Model parameters cannot be estimated

    Examples
    --------
    >>> ca = CausalArima(y=constant_series, ...)
    >>> ca.fit()
    ModelFittingError: ARIMA fitting failed: series has no variance
    """

    pass


class CoefficientExtractionError(CausalArimaError):
    """Raised when extracting model coefficients fails.

    This exception is raised when:
    - AR/MA coefficients cannot be extracted from the fitted model
    - The model structure is not recognized
    - Psi weight computation fails

    Examples
    --------
    >>> ca.fit()
    CoefficientExtractionError: Could not extract AR coefficients from model
    """

    pass


class SimulationError(CausalArimaError):
    """Raised when bootstrap simulation fails.

    This exception is raised when:
    - ARIMA simulation cannot be performed
    - Residuals cannot be extracted for bootstrapping
    - Bootstrap inference computation fails

    Examples
    --------
    >>> ca = CausalArima(..., n_boot=1000)
    >>> ca.fit()
    SimulationError: Bootstrap simulation failed: no valid residuals
    """

    pass


class InferenceError(CausalArimaError):
    """Raised when statistical inference computation fails.

    This exception is raised when:
    - Variance computation produces invalid results
    - P-value calculation fails
    - Inference results contain NaN values

    Examples
    --------
    >>> ca.fit()
    InferenceError: Variance computation produced NaN values
    """

    pass
