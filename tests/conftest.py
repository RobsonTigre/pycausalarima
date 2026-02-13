"""Shared test fixtures for pycausalarima."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# DGP cross-validation registry
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
COMPARISON_DIR = PROJECT_ROOT / "comparison" / "dgp_validation"

SUITE_REGISTRY = {
    "main": {
        "config": COMPARISON_DIR / "dgp_configs.json",
        "data_dir": COMPARISON_DIR / "data",
        "results_r_dir": COMPARISON_DIR / "results_r",
    },
    "sarima": {
        "config": COMPARISON_DIR / "sarima" / "dgp_sarima_configs.json",
        "data_dir": COMPARISON_DIR / "sarima" / "data",
        "results_r_dir": COMPARISON_DIR / "sarima" / "results_r",
    },
    "extended": {
        "config": COMPARISON_DIR / "extended_dgp" / "dgp_extended_configs.json",
        "data_dir": COMPARISON_DIR / "extended_dgp" / "data",
        "results_r_dir": COMPARISON_DIR / "extended_dgp" / "results_r",
    },
}


@dataclass
class DGPTestCase:
    """A single DGP test case with resolved file paths."""

    dgp_id: int
    name: str
    description: str
    suite: str
    order: tuple
    seasonal_order: tuple
    effect: float
    n: int
    n_pre: int
    sigma: float
    data_path: Path
    r_final_path: Path
    r_norm_path: Path


def load_all_dgp_test_cases() -> List[DGPTestCase]:
    """Load all 30 DGP test cases from the three suite config files."""
    cases: List[DGPTestCase] = []
    for suite_name, suite_info in SUITE_REGISTRY.items():
        config_path = suite_info["config"]
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = json.load(f)
        for dgp in config["dgps"]:
            dgp_id = dgp["id"]
            dgp_name = dgp["name"]
            cases.append(
                DGPTestCase(
                    dgp_id=dgp_id,
                    name=dgp_name,
                    description=dgp["description"],
                    suite=suite_name,
                    order=tuple(dgp["order"]),
                    seasonal_order=tuple(dgp["seasonal_order"]),
                    effect=dgp["effect"],
                    n=dgp["n"],
                    n_pre=dgp["n_pre"],
                    sigma=dgp.get("sigma", 1.0),
                    data_path=suite_info["data_dir"]
                    / f"dgp_{dgp_id}_{dgp_name}.csv",
                    r_final_path=suite_info["results_r_dir"]
                    / f"dgp_{dgp_id}_final.csv",
                    r_norm_path=suite_info["results_r_dir"]
                    / f"dgp_{dgp_id}_norm.csv",
                )
            )
    return cases


ALL_DGP_CASES = load_all_dgp_test_cases()


@pytest.fixture
def r_example_data():
    """Generate the exact same data as the R README example.

    This replicates the R code:
        n <- 100
        set.seed(1)
        x1 <- 100 + arima.sim(model = list(ar = 0.999), n = n)
        y <- 1.2 * x1 + rnorm(n)
        y[floor(n*.71):n] <- y[floor(n*.71):n] + 10
    """
    n = 100
    np.random.seed(1)

    # Generate AR(1) process with phi = 0.999
    # R's arima.sim uses innovations, so we simulate step by step
    x1 = np.zeros(n)
    innovations = np.random.normal(0, 1, n)

    for t in range(n):
        if t == 0:
            x1[t] = innovations[t]
        else:
            x1[t] = 0.999 * x1[t - 1] + innovations[t]

    x1 = 100 + x1

    # y <- 1.2 * x1 + rnorm(n)
    np.random.seed(1)  # Reset seed for reproducibility
    # Skip the first n random numbers (used for AR process)
    _ = np.random.normal(0, 1, n)
    y = 1.2 * x1 + np.random.normal(0, 1, n)

    # Add intervention effect: y[floor(n*.71):n] <- y[floor(n*.71):n] + 10
    # R uses 1-based indexing, floor(100*.71) = 71, so indices 71:100 (R) = 70:100 (Python)
    y[70:] = y[70:] + 10

    # Dates
    dates = pd.date_range("2014-01-05", periods=n, freq="D")
    intervention_date = pd.Timestamp("2014-03-16")

    return {
        "y": y,
        "x1": x1,
        "dates": dates,
        "intervention_date": intervention_date,
        "n": n,
    }


@pytest.fixture
def r_expected_results():
    """Expected results from the R package README.

    These are the exact values shown in the R README for the example data.
    """
    return {
        # Final observation results
        "final": {
            "point_effect": 12.257,
            "point_effect_sd": 1.211,
            "cumulative_effect": 310.709,
            "cumulative_effect_sd": 6.634,
            "temporal_average": 10.357,
            "temporal_average_sd": 0.221,
        },
        # Results at specific horizons
        "horizons": {
            "2014-03-25": {
                "point_effect": 9.962,
                "cumulative_effect": 104.356,
                "temporal_average": 10.436,
            },
            "2014-04-05": {
                "point_effect": 9.673,
                "cumulative_effect": 216.327,
                "temporal_average": 10.301,
            },
        },
    }


@pytest.fixture
def simple_ar1_data():
    """Simple AR(1) data for unit testing."""
    np.random.seed(42)
    n = 100

    # Simple AR(1) with known parameters
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.8 * y[t - 1] + np.random.normal(0, 1)

    # Add intervention
    y[70:] += 5

    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    intervention_date = dates[70]

    return {
        "y": y,
        "dates": dates,
        "intervention_date": intervention_date,
    }
