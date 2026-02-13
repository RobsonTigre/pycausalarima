"""Run Python pycausalarima Analysis on All DGPs.

This script runs pycausalarima on each generated DGP dataset and saves results
in the same format as the R analysis for comparison.
"""

import json
import sys
import os

import numpy as np
import pandas as pd

# Add parent directory to path for pycausalarima import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pycausalarima import CausalArima

# Load configuration
with open("dgp_configs.json", "r") as f:
    config = json.load(f)

dgps = config["dgps"]
settings = config["settings"]

np.random.seed(settings["seed"])

print("Running Python pycausalarima Analysis...")
print("=" * 40)
print()

results_summary = []

for dgp in dgps:
    dgp_id = dgp["id"]
    dgp_name = dgp["name"]

    print(f"DGP {dgp_id}: {dgp_name}")

    # Load data
    filename = f"data/dgp_{dgp_id}_{dgp_name}.csv"

    if not os.path.exists(filename):
        print(f"  - ERROR: Data file not found: {filename}")
        print("  - Run generate_dgp_data.R first")
        print()
        continue

    df = pd.read_csv(filename, parse_dates=["date"])

    # Find intervention date
    int_idx = df[df["intervention"] == 1].index[0]
    int_date = df.loc[int_idx, "date"]

    # Extract parameters
    order = tuple(dgp["order"])
    seasonal_order = tuple(dgp["seasonal_order"])
    s = seasonal_order[3] if len(seasonal_order) > 3 else 1

    print(f"  - Order: {order}")
    print(f"  - Seasonal: {seasonal_order}")
    print(f"  - Intervention date: {int_date.date()}")

    try:
        # Run CausalArima
        ca = CausalArima(
            y=df["y"].values,
            dates=pd.DatetimeIndex(df["date"]),
            intervention_date=pd.Timestamp(int_date),
            auto=False,
            order=order,
            seasonal_order=seasonal_order,
        )
        result = ca.fit()

        # Extract results
        norm = result.norm
        n_post = len(norm.tau)

        # Save full time series results
        norm_results = pd.DataFrame({
            "time": range(1, n_post + 1),
            "tau": norm.tau,
            "sd_tau": norm.sd_tau,
            "pvalue_tau_left": norm.pvalue_tau_left,
            "pvalue_tau_bidirectional": norm.pvalue_tau_bidirectional,
            "pvalue_tau_right": norm.pvalue_tau_right,
            "cumulative": norm.cumulative,
            "sd_cumulative": norm.sd_cumulative,
            "pvalue_sum_left": norm.pvalue_sum_left,
            "pvalue_sum_bidirectional": norm.pvalue_sum_bidirectional,
            "pvalue_sum_right": norm.pvalue_sum_right,
            "average": norm.average,
            "sd_average": norm.sd_average,
            "pvalue_avg_left": norm.pvalue_avg_left,
            "pvalue_avg_bidirectional": norm.pvalue_avg_bidirectional,
            "pvalue_avg_right": norm.pvalue_avg_right,
        })

        norm_filename = f"results_python/dgp_{dgp_id}_norm.csv"
        norm_results.to_csv(norm_filename, index=False)

        # Get final values
        final_tau = norm.tau[-1]
        final_sd_tau = norm.sd_tau[-1]
        final_cum = norm.cumulative[-1]
        final_sd_cum = norm.sd_cumulative[-1]
        final_avg = norm.average[-1]
        final_sd_avg = norm.sd_average[-1]
        final_pval = norm.pvalue_avg_bidirectional[-1]

        # Save final results
        final_results = pd.DataFrame({
            "metric": ["point_effect", "cumulative_effect", "avg_effect"],
            "estimate": [final_tau, final_cum, final_avg],
            "sd": [final_sd_tau, final_sd_cum, final_sd_avg],
            "pvalue_bidirectional": [
                norm.pvalue_tau_bidirectional[-1],
                norm.pvalue_sum_bidirectional[-1],
                final_pval,
            ],
        })

        final_filename = f"results_python/dgp_{dgp_id}_final.csv"
        final_results.to_csv(final_filename, index=False)

        # Add to summary
        results_summary.append({
            "dgp_id": dgp_id,
            "dgp_name": dgp_name,
            "point_effect": final_tau,
            "sd_point": final_sd_tau,
            "cumulative_effect": final_cum,
            "sd_cumulative": final_sd_cum,
            "avg_effect": final_avg,
            "sd_avg": final_sd_avg,
            "pvalue_avg": final_pval,
        })

        print(f"  - Point effect: {final_tau:.3f} (sd: {final_sd_tau:.3f})")
        print(f"  - Cumulative: {final_cum:.3f} (sd: {final_sd_cum:.3f})")
        print(f"  - Average: {final_avg:.3f} (sd: {final_sd_avg:.3f})")
        print(f"  - P-value (avg, bidirectional): {final_pval:.4f}")
        print("  - SUCCESS")
        print()

    except Exception as e:
        print(f"  - ERROR: {e}")
        print()

# Save summary
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv("results_python/summary.csv", index=False)

print()
print("=" * 40)
print("Python analysis complete!")
print("Results saved to results_python/")
