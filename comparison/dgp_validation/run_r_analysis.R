# Run R CausalArima Analysis on All DGPs
# This script runs CausalArima on each generated DGP dataset and saves results.

library(jsonlite)

# Source CausalArima package (adjust path as needed)
# Option 1: If installed as package
# library(CausalArima)

# Option 2: Source from local files
source("../../reference_r_package/R/CausalARIMA.R")
source("../../reference_r_package/R/print_arima.R")
source("../../reference_r_package/R/table_arima.R")
source("../../reference_r_package/R/plot_arima.R")

# Load forecast for ARIMA functions
library(forecast)

# Load configuration
config <- fromJSON("dgp_configs.json")
dgps <- config$dgps
settings <- config$settings

set.seed(settings$seed)

cat("Running R CausalArima Analysis...\n")
cat("==================================\n\n")

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

for (i in 1:nrow(dgps)) {
  dgp <- dgps[i, ]

  cat(sprintf("DGP %d: %s\n", dgp$id, dgp$name))

  # Load data
  filename <- sprintf("data/dgp_%d_%s.csv", dgp$id, dgp$name)
  df <- read.csv(filename)
  df$date <- as.Date(df$date)

  # Find intervention date (first post-intervention observation)
  int_idx <- which(df$intervention == 1)[1]
  int_date <- df$date[int_idx]

  # Extract parameters
  order <- unlist(dgp$order)
  seasonal_order <- unlist(dgp$seasonal_order)
  s <- seasonal_order[4]
  if (is.null(s) || is.na(s)) s <- 1

  cat(sprintf("  - Order: (%d,%d,%d)\n", order[1], order[2], order[3]))
  cat(sprintf("  - Seasonal: (%d,%d,%d,%d)\n",
              seasonal_order[1], seasonal_order[2], seasonal_order[3], s))
  cat(sprintf("  - Intervention date: %s\n", int_date))

  tryCatch({
    # Run CausalArima
    # Use auto=FALSE to force the specified order
    if (s > 1) {
      # Seasonal model
      ce <- CausalArima(
        y = ts(df$y, frequency = s),
        dates = df$date,
        int.date = int_date,
        auto = FALSE,
        order = order[1:3],
        seasonal = seasonal_order[1:3]
      )
    } else {
      # Non-seasonal model
      ce <- CausalArima(
        y = ts(df$y, frequency = 1),
        dates = df$date,
        int.date = int_date,
        auto = FALSE,
        order = order[1:3],
        seasonal = c(0, 0, 0)
      )
    }

    # Extract results - inf is a matrix, not data.frame, so use column indexing
    inf <- ce$norm$inf

    # Get final observation results
    n_post <- nrow(inf)
    final <- inf[n_post, ]

    # Save full time series results (use matrix column names)
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

    norm_filename <- sprintf("results_r/dgp_%d_norm.csv", dgp$id)
    write.csv(norm_results, norm_filename, row.names = FALSE)

    # Save final results (use named vector indexing)
    final_results <- data.frame(
      metric = c("point_effect", "cumulative_effect", "avg_effect"),
      estimate = c(final["tau"], final["sum"], final["avg"]),
      sd = c(final["sd.tau"], final["sd.sum"], final["sd.avg"]),
      pvalue_bidirectional = c(final["pvalue.tau.b"], final["pvalue.sum.b"], final["pvalue.avg.b"])
    )

    final_filename <- sprintf("results_r/dgp_%d_final.csv", dgp$id)
    write.csv(final_results, final_filename, row.names = FALSE)

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

# Save summary
write.csv(results_summary, "results_r/summary.csv", row.names = FALSE)

cat("\n==================================\n")
cat("R analysis complete!\n")
cat("Results saved to results_r/\n")
