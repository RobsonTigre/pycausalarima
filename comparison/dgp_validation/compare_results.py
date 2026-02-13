"""Compare R and Python CausalArima Results.

This script compares results from R's CausalArima and Python's pycausalarima
across all DGPs and generates a detailed comparison report.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

# Load configuration
with open("dgp_configs.json", "r") as f:
    config = json.load(f)

dgps = config["dgps"]
settings = config["settings"]

TOLERANCE_RELATIVE = settings["tolerance_relative"]
TOLERANCE_ABSOLUTE = settings["tolerance_absolute"]
CORRELATION_THRESHOLD = settings["correlation_threshold"]


def compare_values(r_val, py_val, tolerance_rel=0.05, tolerance_abs=0.1):
    """Compare two values with appropriate tolerance."""
    if abs(r_val) > 1:
        # Relative tolerance for larger values
        rel_diff = abs(r_val - py_val) / abs(r_val)
        return rel_diff < tolerance_rel, rel_diff
    else:
        # Absolute tolerance for small values
        abs_diff = abs(r_val - py_val)
        return abs_diff < tolerance_abs, abs_diff


def compute_correlation(r_series, py_series):
    """Compute correlation between two series."""
    valid = ~(np.isnan(r_series) | np.isnan(py_series))
    if sum(valid) < 2:
        return np.nan
    return np.corrcoef(r_series[valid], py_series[valid])[0, 1]


print("Comparing R vs Python Results...")
print("=" * 50)
print()

# Initialize report
report_lines = [
    "# R vs Python DGP Comparison Report",
    "",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Summary",
    "",
    "| DGP | Name | Point Effect | Cumulative | Average | P-value | Status |",
    "|-----|------|--------------|------------|---------|---------|--------|",
]

all_results = []
pass_count = 0
fail_count = 0

for dgp in dgps:
    dgp_id = dgp["id"]
    dgp_name = dgp["name"]

    r_final_file = f"results_r/dgp_{dgp_id}_final.csv"
    py_final_file = f"results_python/dgp_{dgp_id}_final.csv"
    r_norm_file = f"results_r/dgp_{dgp_id}_norm.csv"
    py_norm_file = f"results_python/dgp_{dgp_id}_norm.csv"

    print(f"DGP {dgp_id}: {dgp_name}")

    # Check if files exist
    if not os.path.exists(r_final_file):
        print(f"  - R results not found: {r_final_file}")
        report_lines.append(f"| {dgp_id} | {dgp_name} | - | - | - | - | SKIP (no R) |")
        continue

    if not os.path.exists(py_final_file):
        print(f"  - Python results not found: {py_final_file}")
        report_lines.append(f"| {dgp_id} | {dgp_name} | - | - | - | - | SKIP (no Py) |")
        continue

    # Load final results
    r_final = pd.read_csv(r_final_file)
    py_final = pd.read_csv(py_final_file)

    # Extract values
    r_point = r_final[r_final["metric"] == "point_effect"]["estimate"].values[0]
    r_cum = r_final[r_final["metric"] == "cumulative_effect"]["estimate"].values[0]
    r_avg = r_final[r_final["metric"] == "avg_effect"]["estimate"].values[0]
    r_pval = r_final[r_final["metric"] == "avg_effect"]["pvalue_bidirectional"].values[0]

    py_point = py_final[py_final["metric"] == "point_effect"]["estimate"].values[0]
    py_cum = py_final[py_final["metric"] == "cumulative_effect"]["estimate"].values[0]
    py_avg = py_final[py_final["metric"] == "avg_effect"]["estimate"].values[0]
    py_pval = py_final[py_final["metric"] == "avg_effect"]["pvalue_bidirectional"].values[0]

    # Compare values
    point_ok, point_diff = compare_values(r_point, py_point)
    cum_ok, cum_diff = compare_values(r_cum, py_cum)
    avg_ok, avg_diff = compare_values(r_avg, py_avg)
    pval_ok, pval_diff = compare_values(r_pval, py_pval, tolerance_abs=0.01)

    # Compute correlations if time series exist
    corr_tau = np.nan
    corr_cum = np.nan
    if os.path.exists(r_norm_file) and os.path.exists(py_norm_file):
        r_norm = pd.read_csv(r_norm_file)
        py_norm = pd.read_csv(py_norm_file)

        corr_tau = compute_correlation(r_norm["tau"].values, py_norm["tau"].values)
        corr_cum = compute_correlation(r_norm["cumulative"].values, py_norm["cumulative"].values)

    all_pass = point_ok and cum_ok and avg_ok and pval_ok
    if not np.isnan(corr_tau):
        all_pass = all_pass and (corr_tau > CORRELATION_THRESHOLD)

    status = "PASS" if all_pass else "FAIL"
    if all_pass:
        pass_count += 1
    else:
        fail_count += 1

    # Format status markers
    point_mark = "OK" if point_ok else f"DIFF:{point_diff:.3f}"
    cum_mark = "OK" if cum_ok else f"DIFF:{cum_diff:.3f}"
    avg_mark = "OK" if avg_ok else f"DIFF:{avg_diff:.3f}"
    pval_mark = "OK" if pval_ok else f"DIFF:{pval_diff:.4f}"

    report_lines.append(
        f"| {dgp_id} | {dgp_name} | {point_mark} | {cum_mark} | {avg_mark} | {pval_mark} | **{status}** |"
    )

    # Store detailed results
    all_results.append({
        "dgp_id": dgp_id,
        "dgp_name": dgp_name,
        "r_point": r_point,
        "py_point": py_point,
        "r_cum": r_cum,
        "py_cum": py_cum,
        "r_avg": r_avg,
        "py_avg": py_avg,
        "r_pval": r_pval,
        "py_pval": py_pval,
        "corr_tau": corr_tau,
        "corr_cum": corr_cum,
        "status": status,
    })

    print(f"  - Point: R={r_point:.3f}, Py={py_point:.3f} -> {point_mark}")
    print(f"  - Cumulative: R={r_cum:.3f}, Py={py_cum:.3f} -> {cum_mark}")
    print(f"  - Average: R={r_avg:.3f}, Py={py_avg:.3f} -> {avg_mark}")
    print(f"  - P-value: R={r_pval:.4f}, Py={py_pval:.4f} -> {pval_mark}")
    if not np.isnan(corr_tau):
        print(f"  - Correlation (tau): {corr_tau:.4f}")
        print(f"  - Correlation (cum): {corr_cum:.4f}")
    print(f"  - Status: {status}")
    print()

# Add summary statistics
report_lines.extend([
    "",
    f"**Total: {pass_count} PASS, {fail_count} FAIL**",
    "",
    "## Detailed Results",
    "",
])

# Add detailed table
report_lines.extend([
    "### Final Values Comparison",
    "",
    "| DGP | Metric | R | Python | Difference | Rel/Abs Diff |",
    "|-----|--------|---|--------|------------|--------------|",
])

for res in all_results:
    # Point effect
    diff_point = res["py_point"] - res["r_point"]
    rel_point = abs(diff_point / res["r_point"]) if abs(res["r_point"]) > 1 else abs(diff_point)
    report_lines.append(
        f"| {res['dgp_id']} | Point Effect | {res['r_point']:.3f} | {res['py_point']:.3f} | {diff_point:+.3f} | {rel_point:.4f} |"
    )

    # Cumulative
    diff_cum = res["py_cum"] - res["r_cum"]
    rel_cum = abs(diff_cum / res["r_cum"]) if abs(res["r_cum"]) > 1 else abs(diff_cum)
    report_lines.append(
        f"| {res['dgp_id']} | Cumulative | {res['r_cum']:.3f} | {res['py_cum']:.3f} | {diff_cum:+.3f} | {rel_cum:.4f} |"
    )

    # Average
    diff_avg = res["py_avg"] - res["r_avg"]
    rel_avg = abs(diff_avg / res["r_avg"]) if abs(res["r_avg"]) > 1 else abs(diff_avg)
    report_lines.append(
        f"| {res['dgp_id']} | Average | {res['r_avg']:.3f} | {res['py_avg']:.3f} | {diff_avg:+.3f} | {rel_avg:.4f} |"
    )

# Add correlation summary
report_lines.extend([
    "",
    "### Time Series Correlations",
    "",
    "| DGP | Name | Tau Correlation | Cumulative Correlation |",
    "|-----|------|-----------------|----------------------|",
])

for res in all_results:
    corr_tau_str = f"{res['corr_tau']:.4f}" if not np.isnan(res["corr_tau"]) else "N/A"
    corr_cum_str = f"{res['corr_cum']:.4f}" if not np.isnan(res["corr_cum"]) else "N/A"
    report_lines.append(
        f"| {res['dgp_id']} | {res['dgp_name']} | {corr_tau_str} | {corr_cum_str} |"
    )

# Add methodology notes
report_lines.extend([
    "",
    "## Methodology",
    "",
    "### Tolerances Used",
    "",
    f"- Relative tolerance (for values > 1): {TOLERANCE_RELATIVE * 100}%",
    f"- Absolute tolerance (for values <= 1): {TOLERANCE_ABSOLUTE}",
    f"- P-value tolerance: 0.01",
    f"- Correlation threshold: {CORRELATION_THRESHOLD}",
    "",
    "### Interpretation",
    "",
    "- **PASS**: All metrics match within tolerance",
    "- **FAIL**: One or more metrics exceed tolerance",
    "",
    "### DGP Configurations",
    "",
    "| DGP | Order (p,d,q) | Seasonal (P,D,Q,s) | True Effect |",
    "|-----|---------------|---------------------|-------------|",
])

for dgp in dgps:
    order = dgp["order"]
    seasonal = dgp["seasonal_order"]
    report_lines.append(
        f"| {dgp['id']} - {dgp['name']} | ({order[0]},{order[1]},{order[2]}) | "
        f"({seasonal[0]},{seasonal[1]},{seasonal[2]},{seasonal[3]}) | +{dgp['effect']} |"
    )

# Write report
report_text = "\n".join(report_lines)
with open("comparison_report.md", "w") as f:
    f.write(report_text)

print()
print("=" * 50)
print(f"Comparison complete: {pass_count} PASS, {fail_count} FAIL")
print("Report saved to comparison_report.md")

# Print the report to console as well
print()
print("=" * 50)
print("COMPARISON REPORT")
print("=" * 50)
print()
print(report_text)
