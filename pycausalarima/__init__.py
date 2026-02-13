"""
pyCausalArima: Causal Effect Estimation using ARIMA Models
==========================================================

A Python implementation of the C-ARIMA methodology for estimating
causal effects of interventions on time series data.

Main Classes
------------
CausalArima : Main class for causal effect estimation

Example
-------
>>> from pycausalarima import CausalArima
>>> import pandas as pd
>>> import numpy as np
>>>
>>> # Generate sample data
>>> n = 100
>>> np.random.seed(1)
>>> dates = pd.date_range('2014-01-05', periods=n, freq='D')
>>> y = np.cumsum(np.random.normal(0, 1, n)) + 100
>>> intervention_date = pd.Timestamp('2014-03-16')
>>> y[dates >= intervention_date] += 10  # Add intervention effect
>>>
>>> # Fit model and estimate causal effects
>>> ca = CausalArima(y, dates, intervention_date, n_boot=1000)
>>> result = ca.fit()
>>>
>>> # View summary
>>> ca.summary()

References
----------
Menchetti, F., Cipollini, F., & Mealli, F. (2023).
"Combining counterfactual outcomes and ARIMA models for policy evaluation."
The Econometrics Journal.
"""

from pycausalarima.core.causal_arima import CausalArima
from pycausalarima.exceptions import (
    CausalArimaError,
    CoefficientExtractionError,
    InferenceError,
    ModelFittingError,
    SimulationError,
    ValidationError,
)
from pycausalarima.utils.types import ARIMAOrder, CausalArimaResult, InferenceResult

__version__ = "0.1.0"
__author__ = "Robson Tigre"

__all__ = [
    # Main class
    "CausalArima",
    # Result types
    "CausalArimaResult",
    "InferenceResult",
    "ARIMAOrder",
    # Exceptions
    "CausalArimaError",
    "ValidationError",
    "ModelFittingError",
    "CoefficientExtractionError",
    "SimulationError",
    "InferenceError",
]
