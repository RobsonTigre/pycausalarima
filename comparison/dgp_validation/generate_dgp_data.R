# Generate DGP Data for R vs Python Comparison
# This script generates time series data with known ARIMA/SARIMA structure
# and known intervention effects for validation testing.

library(jsonlite)

# Load configuration
config <- fromJSON("dgp_configs.json")
dgps <- config$dgps
settings <- config$settings

set.seed(settings$seed)

cat("Generating DGP data...\n")
cat("======================\n\n")

for (i in 1:nrow(dgps)) {
  dgp <- dgps[i, ]

  cat(sprintf("DGP %d: %s - %s\n", dgp$id, dgp$name, dgp$description))

  n <- dgp$n
  n_pre <- dgp$n_pre
  effect <- dgp$effect
  sigma <- dgp$sigma

  # Extract ARIMA parameters
  order <- unlist(dgp$order)
  seasonal_order <- unlist(dgp$seasonal_order)
  ar <- unlist(dgp$ar_coefs)
  ma <- unlist(dgp$ma_coefs)
  sar <- unlist(dgp$sar_coefs)
  sma <- unlist(dgp$sma_coefs)

  # Seasonal period
  s <- seasonal_order[4]
  if (is.null(s) || is.na(s)) s <- 1

  # Build model specification for arima.sim
  # For non-seasonal models
  if (s <= 1) {
    model <- list()
    if (length(ar) > 0) model$ar <- ar
    if (length(ma) > 0) model$ma <- ma

    # Generate series
    if (length(model) == 0) {
      # White noise
      y <- rnorm(n, mean = 100, sd = sigma)
    } else {
      innov <- rnorm(n + 100, 0, sigma)
      y <- 100 + arima.sim(model = model, n = n, innov = innov[(101):(100+n)])
    }
  } else {
    # For seasonal models, we need to simulate manually or use a different approach
    # Using a simple seasonal AR simulation
    y <- rep(0, n)
    innov <- rnorm(n, 0, sigma)

    for (t in 1:n) {
      y[t] <- 100 + innov[t]

      # Non-seasonal AR
      if (length(ar) > 0) {
        for (j in 1:length(ar)) {
          if (t > j) y[t] <- y[t] + ar[j] * (y[t-j] - 100)
        }
      }

      # Seasonal AR
      if (length(sar) > 0) {
        for (j in 1:length(sar)) {
          lag <- j * s
          if (t > lag) y[t] <- y[t] + sar[j] * (y[t-lag] - 100)
        }
      }

      # Non-seasonal MA (simplified)
      if (length(ma) > 0) {
        for (j in 1:length(ma)) {
          if (t > j) y[t] <- y[t] + ma[j] * innov[t-j]
        }
      }

      # Seasonal MA (simplified)
      if (length(sma) > 0) {
        for (j in 1:length(sma)) {
          lag <- j * s
          if (t > lag) y[t] <- y[t] + sma[j] * innov[t-lag]
        }
      }
    }
  }

  # Apply differencing if d > 0
  d <- order[2]
  if (d > 0) {
    # Generate integrated series by cumulative sum
    y_diff <- y - 100
    for (dd in 1:d) {
      y_diff <- cumsum(y_diff)
    }
    y <- 100 + y_diff
  }

  # Add intervention effect
  y[(n_pre + 1):n] <- y[(n_pre + 1):n] + effect

  # Generate dates
  if (s == 7) {
    dates <- seq.Date(from = as.Date("2020-01-01"), by = "day", length.out = n)
  } else if (s == 12) {
    dates <- seq.Date(from = as.Date("2014-01-01"), by = "month", length.out = n)
  } else {
    dates <- seq.Date(from = as.Date("2020-01-01"), by = "day", length.out = n)
  }

  intervention_date <- dates[n_pre + 1]

  # Create data frame
  df <- data.frame(
    date = dates,
    y = as.numeric(y),
    intervention = c(rep(0, n_pre), rep(1, n - n_pre))
  )

  # Save to CSV
  filename <- sprintf("data/dgp_%d_%s.csv", dgp$id, dgp$name)
  write.csv(df, filename, row.names = FALSE)

  cat(sprintf("  - Saved to %s\n", filename))
  cat(sprintf("  - n=%d, n_pre=%d, effect=%.1f\n", n, n_pre, effect))
  cat(sprintf("  - Intervention date: %s\n\n", intervention_date))
}

cat("Done! Generated", nrow(dgps), "DGP datasets.\n")
