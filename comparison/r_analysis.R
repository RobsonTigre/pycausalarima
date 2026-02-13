# R script to generate test data and run CausalArima
# Results are saved to CSV for comparison with Python

library(CausalArima)

# Set seed and generate data (same as README example)
n <- 100
set.seed(1)
x1 <- 100 + arima.sim(model = list(ar = 0.999), n = n)
y <- 1.2 * x1 + rnorm(n)
y[floor(n * 0.71):n] <- y[floor(n * 0.71):n] + 10

# Create dates
dates <- seq.Date(from = as.Date("2014-01-05"), by = "days", length.out = n)
int.date <- as.Date("2014-03-16")

# Save the input data
input_data <- data.frame(
  date = dates,
  y = as.numeric(y),
  x1 = as.numeric(x1)
)
write.csv(input_data, "comparison/input_data.csv", row.names = FALSE)

cat("Input data saved to comparison/input_data.csv\n")
cat("Intervention date:", as.character(int.date), "\n")

# Fit the model
cat("\nFitting CausalArima model...\n")
ce <- CausalArima(
  y = ts(y, frequency = 1),
  dates = dates,
  int.date = int.date,
  xreg = x1,
  nboot = 1000
)

# Extract normal inference results
norm_inf <- ce$norm$inf
post_dates <- dates[dates >= int.date]

results_norm <- data.frame(
  date = post_dates,
  tau = norm_inf[, "tau"],
  sd_tau = norm_inf[, "sd.tau"],
  pvalue_tau_left = norm_inf[, "pvalue.tau.l"],
  pvalue_tau_bidirectional = norm_inf[, "pvalue.tau.b"],
  pvalue_tau_right = norm_inf[, "pvalue.tau.r"],
  cumulative = norm_inf[, "sum"],
  sd_cumulative = norm_inf[, "sd.sum"],
  pvalue_sum_left = norm_inf[, "pvalue.sum.l"],
  pvalue_sum_bidirectional = norm_inf[, "pvalue.sum.b"],
  pvalue_sum_right = norm_inf[, "pvalue.sum.r"],
  average = norm_inf[, "avg"],
  sd_average = norm_inf[, "sd.avg"],
  pvalue_avg_left = norm_inf[, "pvalue.avg.l"],
  pvalue_avg_bidirectional = norm_inf[, "pvalue.avg.b"],
  pvalue_avg_right = norm_inf[, "pvalue.avg.r"]
)

write.csv(results_norm, "comparison/r_results_norm.csv", row.names = FALSE)
cat("Normal inference results saved to comparison/r_results_norm.csv\n")

# Extract bootstrap inference results
boot_inf <- ce$boot$inf
results_boot <- data.frame(
  date = post_dates,
  tau = boot_inf[, "tau"],
  sd_tau = boot_inf[, "sd.tau"],
  pvalue_tau_left = boot_inf[, "pvalue.tau.l"],
  pvalue_tau_bidirectional = boot_inf[, "pvalue.tau.b"],
  pvalue_tau_right = boot_inf[, "pvalue.tau.r"],
  cumulative = boot_inf[, "sum"],
  sd_cumulative = boot_inf[, "sd.sum"],
  pvalue_sum_left = boot_inf[, "pvalue.sum.l"],
  pvalue_sum_bidirectional = boot_inf[, "pvalue.sum.b"],
  pvalue_sum_right = boot_inf[, "pvalue.sum.r"],
  average = boot_inf[, "avg"],
  sd_average = boot_inf[, "sd.avg"],
  pvalue_avg_left = boot_inf[, "pvalue.avg.l"],
  pvalue_avg_bidirectional = boot_inf[, "pvalue.avg.b"],
  pvalue_avg_right = boot_inf[, "pvalue.avg.r"]
)

write.csv(results_boot, "comparison/r_results_boot.csv", row.names = FALSE)
cat("Bootstrap inference results saved to comparison/r_results_boot.csv\n")

# Print summary
cat("\n=== R CausalArima Summary ===\n")
print(summary(ce))

# Save model info
model_info <- data.frame(
  metric = c("sigma2", "aic", "bic", "loglik"),
  value = c(ce$model$sigma2, ce$model$aic, ce$model$bic, ce$model$loglik)
)
write.csv(model_info, "comparison/r_model_info.csv", row.names = FALSE)
cat("\nModel info saved to comparison/r_model_info.csv\n")

# Save final results for easy comparison
final_results <- data.frame(
  metric = c(
    "point_effect", "point_effect_sd",
    "cumulative_effect", "cumulative_effect_sd",
    "temporal_average", "temporal_average_sd"
  ),
  value = c(
    tail(norm_inf[, "tau"], 1),
    tail(norm_inf[, "sd.tau"], 1),
    tail(norm_inf[, "sum"], 1),
    tail(norm_inf[, "sd.sum"], 1),
    tail(norm_inf[, "avg"], 1),
    tail(norm_inf[, "sd.avg"], 1)
  )
)
write.csv(final_results, "comparison/r_final_results.csv", row.names = FALSE)
cat("Final results saved to comparison/r_final_results.csv\n")

cat("\n=== Final Effect Estimates (R) ===\n")
cat("Point effect:", round(tail(norm_inf[, "tau"], 1), 3), "\n")
cat("Cumulative effect:", round(tail(norm_inf[, "sum"], 1), 3), "\n")
cat("Temporal average:", round(tail(norm_inf[, "avg"], 1), 3), "\n")
