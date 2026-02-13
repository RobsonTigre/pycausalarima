"""Compare R vs Python SARIMA Results and Generate Report."""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

# Load configuration
with open("dgp_sarima_configs.json", "r") as f:
    config = json.load(f)

dgps = config["dgps"]
settings = config["settings"]

tolerance_relative = settings["tolerance_relative"]
tolerance_absolute = settings["tolerance_absolute"]
correlation_threshold = settings["correlation_threshold"]

print("Comparing R vs Python SARIMA Results...")
print("=" * 50)
print()

# Results storage
results = []
detailed_results = []


def compare_values(r_val, py_val, metric_name):
    """Compare R and Python values with appropriate tolerance."""
    if abs(r_val) > 1:
        # Use relative tolerance for large values
        rel_diff = abs(r_val - py_val) / abs(r_val)
        ok = rel_diff < tolerance_relative
        return ok, rel_diff
    else:
        # Use absolute tolerance for small values
        abs_diff = abs(r_val - py_val)
        ok = abs_diff < tolerance_absolute
        return ok, abs_diff


def compute_correlation(r_series, py_series):
    """Compute Pearson correlation between R and Python series."""
    # Remove NaN values
    valid = ~(np.isnan(r_series) | np.isnan(py_series))
    if valid.sum() < 2:
        return np.nan
    return np.corrcoef(r_series[valid], py_series[valid])[0, 1]


for dgp in dgps:
    dgp_id = dgp["id"]
    dgp_name = dgp["name"]

    print(f"DGP {dgp_id}: {dgp_name}")

    # Load R results
    r_final_file = f"results_r/dgp_{dgp_id}_final.csv"
    r_norm_file = f"results_r/dgp_{dgp_id}_norm.csv"

    # Load Python results
    py_final_file = f"results_python/dgp_{dgp_id}_final.csv"
    py_norm_file = f"results_python/dgp_{dgp_id}_norm.csv"

    # Check files exist
    if not all(os.path.exists(f) for f in [r_final_file, py_final_file]):
        print(f"  - ERROR: Missing result files")
        results.append({
            "dgp_id": dgp_id,
            "dgp_name": dgp_name,
            "status": "ERROR",
            "point_ok": False,
            "cum_ok": False,
            "avg_ok": False,
            "pval_ok": False,
        })
        print()
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
    point_ok, point_diff = compare_values(r_point, py_point, "point")
    cum_ok, cum_diff = compare_values(r_cum, py_cum, "cumulative")
    avg_ok, avg_diff = compare_values(r_avg, py_avg, "average")
    pval_ok, pval_diff = compare_values(r_pval, py_pval, "pvalue")

    # Compute correlations if time series files exist
    tau_corr = np.nan
    cum_corr = np.nan

    if os.path.exists(r_norm_file) and os.path.exists(py_norm_file):
        r_norm = pd.read_csv(r_norm_file)
        py_norm = pd.read_csv(py_norm_file)

        tau_corr = compute_correlation(r_norm["tau"].values, py_norm["tau"].values)
        cum_corr = compute_correlation(r_norm["cumulative"].values, py_norm["cumulative"].values)

    # Determine overall status
    all_ok = point_ok and cum_ok and avg_ok and pval_ok
    corr_ok = np.isnan(tau_corr) or (tau_corr >= correlation_threshold and cum_corr >= correlation_threshold)

    status = "PASS" if (all_ok and corr_ok) else "FAIL"

    # Print results
    point_str = "OK" if point_ok else f"DIFF:{point_diff:.4f}"
    cum_str = "OK" if cum_ok else f"DIFF:{cum_diff:.4f}"
    avg_str = "OK" if avg_ok else f"DIFF:{avg_diff:.4f}"
    pval_str = "OK" if pval_ok else f"DIFF:{pval_diff:.4f}"

    print(f"  - Point: R={r_point:.3f}, Py={py_point:.3f} -> {point_str}")
    print(f"  - Cumulative: R={r_cum:.3f}, Py={py_cum:.3f} -> {cum_str}")
    print(f"  - Average: R={r_avg:.3f}, Py={py_avg:.3f} -> {avg_str}")
    print(f"  - P-value: R={r_pval:.4f}, Py={py_pval:.4f} -> {pval_str}")
    print(f"  - Correlation (tau): {tau_corr:.4f}")
    print(f"  - Correlation (cum): {cum_corr:.4f}")
    print(f"  - Status: {status}")
    print()

    # Store results
    results.append({
        "dgp_id": dgp_id,
        "dgp_name": dgp_name,
        "status": status,
        "point_ok": point_str,
        "cum_ok": cum_str,
        "avg_ok": avg_str,
        "pval_ok": pval_str,
        "tau_corr": tau_corr,
        "cum_corr": cum_corr,
    })

    # Store detailed results
    detailed_results.append({
        "dgp_id": dgp_id,
        "r_point": r_point,
        "py_point": py_point,
        "r_cum": r_cum,
        "py_cum": py_cum,
        "r_avg": r_avg,
        "py_avg": py_avg,
        "r_pval": r_pval,
        "py_pval": py_pval,
    })

# Summary
n_pass = sum(1 for r in results if r["status"] == "PASS")
n_fail = sum(1 for r in results if r["status"] == "FAIL")
n_error = sum(1 for r in results if r["status"] == "ERROR")

print()
print("=" * 50)
print(f"Comparison complete: {n_pass} PASS, {n_fail} FAIL, {n_error} ERROR")

# Generate markdown report
report = []
report.append("# R vs Python SARIMA DGP Comparison Report")
report.append("")
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append("")
report.append("## Summary")
report.append("")
report.append("| DGP | Name | Point Effect | Cumulative | Average | P-value | Status |")
report.append("|-----|------|--------------|------------|---------|---------|--------|")

for r in results:
    report.append(f"| {r['dgp_id']} | {r['dgp_name']} | {r['point_ok']} | {r['cum_ok']} | {r['avg_ok']} | {r['pval_ok']} | **{r['status']}** |")

report.append("")
report.append(f"**Total: {n_pass} PASS, {n_fail} FAIL**")
report.append("")

# Detailed results
report.append("## Detailed Results")
report.append("")
report.append("### Final Values Comparison")
report.append("")
report.append("| DGP | Metric | R | Python | Difference | Rel/Abs Diff |")
report.append("|-----|--------|---|--------|------------|--------------|")

for d in detailed_results:
    dgp_id = d["dgp_id"]

    # Point effect
    diff = d["py_point"] - d["r_point"]
    rel_diff = abs(diff / d["r_point"]) if abs(d["r_point"]) > 1 else abs(diff)
    report.append(f"| {dgp_id} | Point Effect | {d['r_point']:.3f} | {d['py_point']:.3f} | {diff:+.3f} | {rel_diff:.4f} |")

    # Cumulative
    diff = d["py_cum"] - d["r_cum"]
    rel_diff = abs(diff / d["r_cum"]) if abs(d["r_cum"]) > 1 else abs(diff)
    report.append(f"| {dgp_id} | Cumulative | {d['r_cum']:.3f} | {d['py_cum']:.3f} | {diff:+.3f} | {rel_diff:.4f} |")

    # Average
    diff = d["py_avg"] - d["r_avg"]
    rel_diff = abs(diff / d["r_avg"]) if abs(d["r_avg"]) > 1 else abs(diff)
    report.append(f"| {dgp_id} | Average | {d['r_avg']:.3f} | {d['py_avg']:.3f} | {diff:+.3f} | {rel_diff:.4f} |")

report.append("")
report.append("### Time Series Correlations")
report.append("")
report.append("| DGP | Name | Tau Correlation | Cumulative Correlation |")
report.append("|-----|------|-----------------|----------------------|")

for r in results:
    report.append(f"| {r['dgp_id']} | {r['dgp_name']} | {r['tau_corr']:.4f} | {r['cum_corr']:.4f} |")

report.append("")
report.append("## Methodology")
report.append("")
report.append("### Tolerances Used")
report.append("")
report.append(f"- Relative tolerance (for values > 1): {tolerance_relative * 100:.1f}%")
report.append(f"- Absolute tolerance (for values <= 1): {tolerance_absolute}")
report.append(f"- Correlation threshold: {correlation_threshold}")
report.append("")
report.append("### Interpretation")
report.append("")
report.append("- **PASS**: All metrics match within tolerance")
report.append("- **FAIL**: One or more metrics exceed tolerance")
report.append("")
report.append("### DGP Configurations")
report.append("")
report.append("| DGP | Order (p,d,q) | Seasonal (P,D,Q,s) | True Effect |")
report.append("|-----|---------------|---------------------|-------------|")

for dgp in dgps:
    order = dgp["order"]
    seasonal = dgp["seasonal_order"]
    report.append(f"| {dgp['id']} - {dgp['name']} | ({order[0]},{order[1]},{order[2]}) | ({seasonal[0]},{seasonal[1]},{seasonal[2]},{seasonal[3]}) | +{dgp['effect']} |")

# Save report
report_text = "\n".join(report)
with open("comparison_report.md", "w") as f:
    f.write(report_text)

print("Report saved to comparison_report.md")
print()
print("=" * 50)
print("COMPARISON REPORT")
print("=" * 50)
print()
print(report_text)
