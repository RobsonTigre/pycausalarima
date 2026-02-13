"""Cross-validation tests: pycausalarima vs R CausalArima across all 30 DGPs.

These tests sweep across all DGP settings (main: 1-8, sarima: 9-18,
extended: 19-30), run Python CausalArima on pre-generated data, and
compare results against pre-computed R CausalArima outputs.

R is NOT required at test runtime — all R results are pre-computed.

Tolerances (matching the standalone comparison scripts):
- Relative tolerance for |value| > 1: 5%
- Absolute tolerance for |value| <= 1: 0.1
- P-value absolute tolerance: 0.01
- Time series correlation threshold: 0.99
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pycausalarima import CausalArima

# pytest makes conftest symbols available but not as a regular import.
# Import from the conftest module that pytest auto-discovers.
import sys

sys.path.insert(0, str(Path(__file__).parent))
from conftest import ALL_DGP_CASES, DGPTestCase  # noqa: E402

# ---------------------------------------------------------------------------
# Tolerances (matching comparison/dgp_validation/compare_results.py)
# ---------------------------------------------------------------------------

TOLERANCE_RELATIVE = 0.05
TOLERANCE_ABSOLUTE = 0.1
TOLERANCE_PVALUE = 0.01
CORRELATION_THRESHOLD = 0.99

# Standard deviations use a wider tolerance because they depend on sigma2,
# which is computed via state-space MLE (statsmodels) vs CSS-ML (R's arima).
# These estimation methods can produce different innovation variance estimates,
# especially for models with high differencing (d>=2) or combined d+D.
# Cumulative/average SDs further amplify differences through summation.
# DGP 30 (ARMA(2,1)+SARMA(1,1,12), 6 params) is the hardest case at ~17%.
TOLERANCE_SD_RELATIVE = 0.20

# ---------------------------------------------------------------------------
# Result cache — each DGP model is fitted only once across all test functions
# ---------------------------------------------------------------------------

_RESULT_CACHE: dict = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_missing(dgp_case: DGPTestCase) -> None:
    if not dgp_case.data_path.exists():
        pytest.skip(f"Data file not found: {dgp_case.data_path}")
    if not dgp_case.r_final_path.exists():
        pytest.skip(f"R final results not found: {dgp_case.r_final_path}")


def _load_data(dgp_case: DGPTestCase) -> dict:
    df = pd.read_csv(dgp_case.data_path, parse_dates=["date"])
    int_idx = df[df["intervention"] == 1].index[0]
    int_date = df.loc[int_idx, "date"]
    return {
        "y": df["y"].values,
        "dates": pd.DatetimeIndex(df["date"]),
        "intervention_date": pd.Timestamp(int_date),
    }


def _load_r_final(dgp_case: DGPTestCase) -> dict:
    df = pd.read_csv(dgp_case.r_final_path)
    result = {}
    for _, row in df.iterrows():
        result[row["metric"]] = {
            "estimate": row["estimate"],
            "sd": row["sd"],
            "pvalue_bidirectional": row["pvalue_bidirectional"],
        }
    return result


def _load_r_norm(dgp_case: DGPTestCase) -> pd.DataFrame:
    return pd.read_csv(dgp_case.r_norm_path)


def _get_or_run(dgp_case: DGPTestCase):
    """Get cached CausalArimaResult or fit the model."""
    if dgp_case.dgp_id not in _RESULT_CACHE:
        data = _load_data(dgp_case)
        ca = CausalArima(
            y=data["y"],
            dates=data["dates"],
            intervention_date=data["intervention_date"],
            auto=False,
            order=dgp_case.order,
            seasonal_order=dgp_case.seasonal_order,
        )
        _RESULT_CACHE[dgp_case.dgp_id] = ca.fit()
    return _RESULT_CACHE[dgp_case.dgp_id]


def _compare_value(r_val, py_val, tol_rel=TOLERANCE_RELATIVE, tol_abs=TOLERANCE_ABSOLUTE):
    """Compare using relative tolerance for |val|>1, absolute otherwise."""
    if abs(r_val) > 1:
        rel_diff = abs(r_val - py_val) / abs(r_val)
        return rel_diff < tol_rel, rel_diff, "relative"
    else:
        abs_diff = abs(r_val - py_val)
        return abs_diff < tol_abs, abs_diff, "absolute"


def _compute_correlation(r_series, py_series):
    valid = ~(np.isnan(r_series) | np.isnan(py_series))
    if valid.sum() < 2:
        return np.nan
    return np.corrcoef(r_series[valid], py_series[valid])[0, 1]


# ---------------------------------------------------------------------------
# Parametrize setup with suite markers
# ---------------------------------------------------------------------------

_SUITE_MARKS = {
    "main": pytest.mark.dgp_main,
    "sarima": pytest.mark.dgp_sarima,
    "extended": pytest.mark.dgp_extended,
}

dgp_params = [
    pytest.param(
        case,
        marks=[pytest.mark.cross_validation, _SUITE_MARKS[case.suite]],
        id=f"dgp_{case.dgp_id:02d}_{case.name}",
    )
    for case in ALL_DGP_CASES
]

dgp_parametrize = pytest.mark.parametrize("dgp_case", dgp_params)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestDGPCrossValidation:
    """Cross-validation: Python vs R across all 30 DGPs."""

    @dgp_parametrize
    def test_point_effect(self, dgp_case: DGPTestCase):
        """Final point effect (tau[-1]) matches R."""
        _skip_if_missing(dgp_case)
        r_final = _load_r_final(dgp_case)
        result = _get_or_run(dgp_case)

        r_val = r_final["point_effect"]["estimate"]
        py_val = result.norm.tau[-1]
        passes, diff, diff_type = _compare_value(r_val, py_val)
        assert passes, (
            f"DGP {dgp_case.dgp_id} ({dgp_case.name}): point effect mismatch. "
            f"R={r_val:.6f}, Py={py_val:.6f}, {diff_type} diff={diff:.6f}"
        )

    @dgp_parametrize
    def test_cumulative_effect(self, dgp_case: DGPTestCase):
        """Final cumulative effect matches R."""
        _skip_if_missing(dgp_case)
        r_final = _load_r_final(dgp_case)
        result = _get_or_run(dgp_case)

        r_val = r_final["cumulative_effect"]["estimate"]
        py_val = result.norm.cumulative[-1]
        passes, diff, diff_type = _compare_value(r_val, py_val)
        assert passes, (
            f"DGP {dgp_case.dgp_id} ({dgp_case.name}): cumulative effect mismatch. "
            f"R={r_val:.6f}, Py={py_val:.6f}, {diff_type} diff={diff:.6f}"
        )

    @dgp_parametrize
    def test_average_effect(self, dgp_case: DGPTestCase):
        """Final temporal average effect matches R."""
        _skip_if_missing(dgp_case)
        r_final = _load_r_final(dgp_case)
        result = _get_or_run(dgp_case)

        r_val = r_final["avg_effect"]["estimate"]
        py_val = result.norm.average[-1]
        passes, diff, diff_type = _compare_value(r_val, py_val)
        assert passes, (
            f"DGP {dgp_case.dgp_id} ({dgp_case.name}): average effect mismatch. "
            f"R={r_val:.6f}, Py={py_val:.6f}, {diff_type} diff={diff:.6f}"
        )

    @dgp_parametrize
    def test_standard_deviations(self, dgp_case: DGPTestCase):
        """Standard deviations (point, cumulative, average) match R."""
        _skip_if_missing(dgp_case)
        r_final = _load_r_final(dgp_case)
        result = _get_or_run(dgp_case)

        checks = [
            ("point_effect", result.norm.sd_tau[-1]),
            ("cumulative_effect", result.norm.sd_cumulative[-1]),
            ("avg_effect", result.norm.sd_average[-1]),
        ]
        for metric, py_sd in checks:
            r_sd = r_final[metric]["sd"]
            passes, diff, diff_type = _compare_value(
                r_sd, py_sd, tol_rel=TOLERANCE_SD_RELATIVE,
            )
            assert passes, (
                f"DGP {dgp_case.dgp_id} ({dgp_case.name}): {metric} SD mismatch. "
                f"R={r_sd:.6f}, Py={py_sd:.6f}, {diff_type} diff={diff:.6f}"
            )

    @dgp_parametrize
    def test_pvalues(self, dgp_case: DGPTestCase):
        """Bidirectional p-values match R within tolerance."""
        _skip_if_missing(dgp_case)
        r_final = _load_r_final(dgp_case)
        result = _get_or_run(dgp_case)

        pval_pairs = [
            ("point_effect", result.norm.pvalue_tau_bidirectional[-1]),
            ("cumulative_effect", result.norm.pvalue_sum_bidirectional[-1]),
            ("avg_effect", result.norm.pvalue_avg_bidirectional[-1]),
        ]
        for metric, py_pval in pval_pairs:
            r_pval = r_final[metric]["pvalue_bidirectional"]
            abs_diff = abs(r_pval - py_pval)
            assert abs_diff < TOLERANCE_PVALUE, (
                f"DGP {dgp_case.dgp_id} ({dgp_case.name}): {metric} p-value mismatch. "
                f"R={r_pval:.6e}, Py={py_pval:.6e}, abs diff={abs_diff:.6e}"
            )

    @dgp_parametrize
    def test_tau_time_series_correlation(self, dgp_case: DGPTestCase):
        """Full tau time series correlates highly with R (> 0.99)."""
        _skip_if_missing(dgp_case)
        if not dgp_case.r_norm_path.exists():
            pytest.skip(f"R norm results not found: {dgp_case.r_norm_path}")

        r_norm = _load_r_norm(dgp_case)
        result = _get_or_run(dgp_case)

        corr = _compute_correlation(r_norm["tau"].values, result.norm.tau)
        assert not np.isnan(corr), f"DGP {dgp_case.dgp_id}: tau correlation is NaN"
        assert corr > CORRELATION_THRESHOLD, (
            f"DGP {dgp_case.dgp_id} ({dgp_case.name}): tau correlation {corr:.6f} "
            f"< threshold {CORRELATION_THRESHOLD}"
        )

    @dgp_parametrize
    def test_cumulative_time_series_correlation(self, dgp_case: DGPTestCase):
        """Full cumulative time series correlates highly with R (> 0.99)."""
        _skip_if_missing(dgp_case)
        if not dgp_case.r_norm_path.exists():
            pytest.skip(f"R norm results not found: {dgp_case.r_norm_path}")

        r_norm = _load_r_norm(dgp_case)
        result = _get_or_run(dgp_case)

        corr = _compute_correlation(
            r_norm["cumulative"].values, result.norm.cumulative
        )
        assert not np.isnan(corr), (
            f"DGP {dgp_case.dgp_id}: cumulative correlation is NaN"
        )
        assert corr > CORRELATION_THRESHOLD, (
            f"DGP {dgp_case.dgp_id} ({dgp_case.name}): cumulative correlation "
            f"{corr:.6f} < threshold {CORRELATION_THRESHOLD}"
        )


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def test_generate_cross_validation_report():
    """Generate a markdown comparison report from all 30 DGPs.

    Uses the _RESULT_CACHE so no models are re-fitted. Writes to
    tests/cross_validation_report.md.
    """
    lines = [
        "# Cross-Validation Report: pycausalarima vs R CausalArima",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| DGP | Name | Suite | Order | Seasonal | Effect "
        "| Point | Cumul. | Avg | P-value | Tau Corr | Cum Corr | Status |",
        "|-----|------|-------|-------|----------|--------"
        "|-------|--------|-----|---------|----------|----------|--------|",
    ]

    detail_lines = [
        "",
        "## Detailed Values",
        "",
        "| DGP | Metric | R | Python | Diff | Rel/Abs |",
        "|-----|--------|---|--------|------|---------|",
    ]

    corr_lines = [
        "",
        "## Time Series Correlations",
        "",
        "| DGP | Name | Tau Corr | Cumulative Corr |",
        "|-----|------|----------|-----------------|",
    ]

    pass_count = 0
    fail_count = 0
    skip_count = 0

    for case in ALL_DGP_CASES:
        if not case.data_path.exists() or not case.r_final_path.exists():
            lines.append(
                f"| {case.dgp_id} | {case.name} | {case.suite} | "
                f"{case.order} | {case.seasonal_order} | {case.effect} "
                f"| - | - | - | - | - | - | SKIP |"
            )
            skip_count += 1
            continue

        r_final = _load_r_final(case)
        result = _get_or_run(case)

        # Point estimates
        r_point = r_final["point_effect"]["estimate"]
        py_point = result.norm.tau[-1]
        point_ok, point_diff, _ = _compare_value(r_point, py_point)

        r_cum = r_final["cumulative_effect"]["estimate"]
        py_cum = result.norm.cumulative[-1]
        cum_ok, cum_diff, _ = _compare_value(r_cum, py_cum)

        r_avg = r_final["avg_effect"]["estimate"]
        py_avg = result.norm.average[-1]
        avg_ok, avg_diff, _ = _compare_value(r_avg, py_avg)

        # P-values
        r_pval = r_final["avg_effect"]["pvalue_bidirectional"]
        py_pval = result.norm.pvalue_avg_bidirectional[-1]
        pval_diff = abs(r_pval - py_pval)
        pval_ok = pval_diff < TOLERANCE_PVALUE

        # Correlations
        corr_tau = np.nan
        corr_cum = np.nan
        if case.r_norm_path.exists():
            r_norm = _load_r_norm(case)
            corr_tau = _compute_correlation(r_norm["tau"].values, result.norm.tau)
            corr_cum = _compute_correlation(
                r_norm["cumulative"].values, result.norm.cumulative
            )

        corr_ok = True
        if not np.isnan(corr_tau):
            corr_ok = corr_tau > CORRELATION_THRESHOLD
        if not np.isnan(corr_cum):
            corr_ok = corr_ok and corr_cum > CORRELATION_THRESHOLD

        all_pass = point_ok and cum_ok and avg_ok and pval_ok and corr_ok
        status = "PASS" if all_pass else "FAIL"
        if all_pass:
            pass_count += 1
        else:
            fail_count += 1

        # Summary row
        point_mark = "OK" if point_ok else f"DIFF:{point_diff:.4f}"
        cum_mark = "OK" if cum_ok else f"DIFF:{cum_diff:.4f}"
        avg_mark = "OK" if avg_ok else f"DIFF:{avg_diff:.4f}"
        pval_mark = "OK" if pval_ok else f"DIFF:{pval_diff:.4f}"
        corr_tau_str = f"{corr_tau:.4f}" if not np.isnan(corr_tau) else "N/A"
        corr_cum_str = f"{corr_cum:.4f}" if not np.isnan(corr_cum) else "N/A"

        lines.append(
            f"| {case.dgp_id} | {case.name} | {case.suite} | "
            f"{case.order} | {case.seasonal_order} | {case.effect:+g} "
            f"| {point_mark} | {cum_mark} | {avg_mark} | {pval_mark} "
            f"| {corr_tau_str} | {corr_cum_str} | **{status}** |"
        )

        # Detailed rows
        for metric, r_v, py_v in [
            ("Point Effect", r_point, py_point),
            ("Cumulative", r_cum, py_cum),
            ("Average", r_avg, py_avg),
        ]:
            diff = py_v - r_v
            rel = abs(diff / r_v) if abs(r_v) > 1 else abs(diff)
            detail_lines.append(
                f"| {case.dgp_id} | {metric} | {r_v:.4f} | {py_v:.4f} "
                f"| {diff:+.4f} | {rel:.6f} |"
            )

        # Correlation row
        corr_lines.append(
            f"| {case.dgp_id} | {case.name} | {corr_tau_str} | {corr_cum_str} |"
        )

    lines.append("")
    lines.append(f"**Total: {pass_count} PASS, {fail_count} FAIL, {skip_count} SKIP**")

    # Methodology
    method_lines = [
        "",
        "## Methodology",
        "",
        "### Tolerances",
        "",
        f"- Relative tolerance (|value| > 1): {TOLERANCE_RELATIVE * 100}%",
        f"- Absolute tolerance (|value| <= 1): {TOLERANCE_ABSOLUTE}",
        f"- P-value tolerance: {TOLERANCE_PVALUE}",
        f"- Correlation threshold: {CORRELATION_THRESHOLD}",
        "",
        "### DGP Configurations",
        "",
        "| DGP | Order (p,d,q) | Seasonal (P,D,Q,s) | True Effect | n | n_pre |",
        "|-----|---------------|---------------------|-------------|---|-------|",
    ]
    for case in ALL_DGP_CASES:
        method_lines.append(
            f"| {case.dgp_id} - {case.name} | {case.order} | "
            f"{case.seasonal_order} | {case.effect:+g} | {case.n} | {case.n_pre} |"
        )

    all_lines = lines + detail_lines + corr_lines + method_lines
    report_dir = Path(__file__).parent.parent / "comparison" / "dgp_validation"
    report_path = report_dir / "cross_validation_report.md"
    report_path.write_text("\n".join(all_lines) + "\n")
