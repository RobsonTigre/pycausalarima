# Run R CausalArima Analysis on SARIMA DGPs

library(jsonlite)
library(forecast)

# Source CausalArima from FMenchetti repo (at repo root)
source("../../../reference_r_package/R/CausalARIMA.R")
source("../../../reference_r_package/R/print_arima.R")
source("../../../reference_r_package/R/table_arima.R")
source("../../../reference_r_package/R/plot_arima.R")

# Load configuration
config <- fromJSON("dgp_sarima_configs.json")
dgps <- config$dgps
settings <- config$settings

set.seed(settings$seed)

cat("Running R CausalArima SARIMA Analysis...\n")
cat("=========================================\n\n")

results_summary <- data.frame(
  dgp_id = integer(),
  dgp_name = character(),
  point_effect = numeric(),
  sd_point = numeric(),
  cumulative_effect = numeric(),
  sd_cumulative = numeric(),
  avg_effect = numeric(),
  sd_avg = numeric(),
  pvalue_avg = numeric(),
  stringsAsFactors = FALSE
)

for (i in seq_len(nrow(dgps))) {
  dgp <- dgps[i, ]

  cat(sprintf("DGP %d: %s\n", dgp$id, dgp$name))

  # Load data
  filename <- sprintf("data/dgp_%d_%s.csv", dgp$id, dgp$name)

  if (!file.exists(filename)) {
    cat(sprintf("  - ERROR: Data file not found: %s\n\n", filename))
    next
  }

  df <- read.csv(filename)
  df$date <- as.Date(df$date)

  # Find intervention date
  int_idx <- which(df$intervention == 1)[1]
  int_date <- df$date[int_idx]

  # Extract parameters
  order <- unlist(dgp$order)
  seasonal_order <- unlist(dgp$seasonal_order)
  s <- seasonal_order[4]

  cat(sprintf("  - Order: (%d,%d,%d)\n", order[1], order[2], order[3]))
  cat(sprintf("  - Seasonal: (%d,%d,%d,%d)\n",
              seasonal_order[1], seasonal_order[2], seasonal_order[3], s))
  cat(sprintf("  - Intervention date: %s\n", int_date))

  tryCatch({
    # Run CausalArima with specified order
    ce <- CausalArima(
      y = ts(df$y, frequency = s),
      dates = df$date,
      int.date = int_date,
      auto = FALSE,
      order = order[1:3],
      seasonal = seasonal_order[1:3]  # Note: R uses P,D,Q without s
    )

    # Extract results - inf is a matrix
    inf <- ce$norm$inf
    n_post <- nrow(inf)
    final <- inf[n_post, ]

    # Save full time series results
    norm_results <- data.frame(
      time = 1:n_post,
      tau = inf[, "tau"],
      sd_tau = inf[, "sd.tau"],
      pvalue_tau_left = inf[, "pvalue.tau.l"],
      pvalue_tau_bidirectional = inf[, "pvalue.tau.b"],
      pvalue_tau_right = inf[, "pvalue.tau.r"],
      cumulative = inf[, "sum"],
      sd_cumulative = inf[, "sd.sum"],
      pvalue_sum_left = inf[, "pvalue.sum.l"],
      pvalue_sum_bidirectional = inf[, "pvalue.sum.b"],
      pvalue_sum_right = inf[, "pvalue.sum.r"],
      average = inf[, "avg"],
      sd_average = inf[, "sd.avg"],
      pvalue_avg_left = inf[, "pvalue.avg.l"],
      pvalue_avg_bidirectional = inf[, "pvalue.avg.b"],
      pvalue_avg_right = inf[, "pvalue.avg.r"]
    )

    write.csv(norm_results, sprintf("results_r/dgp_%d_norm.csv", dgp$id), row.names = FALSE)

    # Save final results
    final_results <- data.frame(
      metric = c("point_effect", "cumulative_effect", "avg_effect"),
      estimate = c(final["tau"], final["sum"], final["avg"]),
      sd = c(final["sd.tau"], final["sd.sum"], final["sd.avg"]),
      pvalue_bidirectional = c(final["pvalue.tau.b"], final["pvalue.sum.b"], final["pvalue.avg.b"])
    )

    write.csv(final_results, sprintf("results_r/dgp_%d_final.csv", dgp$id), row.names = FALSE)

    # Add to summary
    results_summary <- rbind(results_summary, data.frame(
      dgp_id = dgp$id,
      dgp_name = dgp$name,
      point_effect = final["tau"],
      sd_point = final["sd.tau"],
      cumulative_effect = final["sum"],
      sd_cumulative = final["sd.sum"],
      avg_effect = final["avg"],
      sd_avg = final["sd.avg"],
      pvalue_avg = final["pvalue.avg.b"],
      stringsAsFactors = FALSE
    ))

    cat(sprintf("  - Point effect: %.3f (sd: %.3f)\n", final["tau"], final["sd.tau"]))
    cat(sprintf("  - Cumulative: %.3f (sd: %.3f)\n", final["sum"], final["sd.sum"]))
    cat(sprintf("  - Average: %.3f (sd: %.3f)\n", final["avg"], final["sd.avg"]))
    cat(sprintf("  - P-value (avg, bidirectional): %.4f\n", final["pvalue.avg.b"]))
    cat("  - SUCCESS\n\n")

  }, error = function(e) {
    cat(sprintf("  - ERROR: %s\n\n", e$message))
  })
}

write.csv(results_summary, "results_r/summary.csv", row.names = FALSE)

cat("\n=========================================\n")
cat("R SARIMA analysis complete!\n")
cat("Results saved to results_r/\n")
