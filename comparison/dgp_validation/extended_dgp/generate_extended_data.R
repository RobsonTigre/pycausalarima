# Generate Extended DGP Data for R vs Python Comparison
# This script handles higher-order models, d=2 differencing, effect variations,
# near-unit-root, and bimonthly seasonality

library(jsonlite)

# Load configuration
config <- fromJSON("dgp_extended_configs.json")
dgps <- config$dgps
settings <- config$settings

set.seed(settings$seed)

cat("Generating Extended DGP data...\n")
cat("================================\n\n")

# Function to generate SARIMA data with support for d=2
generate_extended_data <- function(n, order, seasonal_order, ar, ma, sar, sma, sigma, burn_in = 500) {
  d <- order[2]
  D <- seasonal_order[2]
  s <- seasonal_order[4]

  # Total series length needed (accounting for burn-in)
  total_n <- n + burn_in

  # Generate innovations
  innov <- rnorm(total_n, 0, sigma)

  # Initialize series
  w <- numeric(total_n)

  # Generate SARMA process (the process AFTER differencing)
  for (t in 1:total_n) {
    w[t] <- innov[t]

    # AR contribution
    if (length(ar) > 0) {
      for (j in seq_along(ar)) {
        if (t > j) w[t] <- w[t] + ar[j] * w[t-j]
      }
    }

    # SAR contribution (at seasonal lags)
    if (length(sar) > 0 && s > 1) {
      for (j in seq_along(sar)) {
        lag <- j * s
        if (t > lag) w[t] <- w[t] + sar[j] * w[t-lag]
      }
    }

    # MA contribution
    if (length(ma) > 0) {
      for (j in seq_along(ma)) {
        if (t > j) w[t] <- w[t] + ma[j] * innov[t-j]
      }
    }

    # SMA contribution (at seasonal lags)
    if (length(sma) > 0 && s > 1) {
      for (j in seq_along(sma)) {
        lag <- j * s
        if (t > lag) w[t] <- w[t] + sma[j] * innov[t-lag]
      }
    }
  }

  y <- w

  # Integrate back seasonal differencing (D times)
  for (dd in seq_len(D)) {
    y_new <- numeric(length(y))
    y_new[1:s] <- cumsum(y[1:s])
    for (t in (s+1):length(y)) {
      y_new[t] <- y_new[t-s] + y[t]
    }
    y <- y_new
  }

  # Integrate back regular differencing (d times) - handles d=2
  for (dd in seq_len(d)) {
    y <- cumsum(y)
  }

  # Take last n observations and center around 100
  y <- tail(y, n)
  y <- y - mean(y) + 100

  return(y)
}

# Generate date sequence based on frequency
generate_dates <- function(start_date, n, frequency) {
  start <- as.Date(start_date)
  switch(frequency,
    "day" = seq.Date(start, by = "day", length.out = n),
    "week" = seq.Date(start, by = "week", length.out = n),
    "month" = seq.Date(start, by = "month", length.out = n),
    "quarter" = seq.Date(start, by = "3 months", length.out = n),
    seq.Date(start, by = "day", length.out = n)
  )
}

# Process each DGP
for (i in seq_len(nrow(dgps))) {
  dgp <- dgps[i, ]

  cat(sprintf("DGP %d: %s - %s\n", dgp$id, dgp$name, dgp$description))

  # Extract parameters
  order <- unlist(dgp$order)
  seasonal_order <- unlist(dgp$seasonal_order)
  ar <- if (length(dgp$ar_coefs[[1]]) > 0) unlist(dgp$ar_coefs) else numeric(0)
  ma <- if (length(dgp$ma_coefs[[1]]) > 0) unlist(dgp$ma_coefs) else numeric(0)
  sar <- if (length(dgp$sar_coefs[[1]]) > 0) unlist(dgp$sar_coefs) else numeric(0)
  sma <- if (length(dgp$sma_coefs[[1]]) > 0) unlist(dgp$sma_coefs) else numeric(0)

  cat(sprintf("  - Order: (%d,%d,%d)\n", order[1], order[2], order[3]))
  cat(sprintf("  - Seasonal: (%d,%d,%d,%d)\n",
              seasonal_order[1], seasonal_order[2], seasonal_order[3], seasonal_order[4]))

  # Generate series
  y <- generate_extended_data(
    n = dgp$n,
    order = order,
    seasonal_order = seasonal_order,
    ar = ar,
    ma = ma,
    sar = sar,
    sma = sma,
    sigma = dgp$sigma
  )

  # Add intervention effect (can be positive or negative)
  y[(dgp$n_pre + 1):dgp$n] <- y[(dgp$n_pre + 1):dgp$n] + dgp$effect

  # Generate dates
  dates <- generate_dates(dgp$start_date, dgp$n, dgp$date_frequency)

  # Create data frame
  df <- data.frame(
    date = dates,
    y = y,
    intervention = c(rep(0, dgp$n_pre), rep(1, dgp$n - dgp$n_pre))
  )

  # Save
  filename <- sprintf("data/dgp_%d_%s.csv", dgp$id, dgp$name)
  write.csv(df, filename, row.names = FALSE)

  cat(sprintf("  - Effect: %+.1f\n", dgp$effect))
  cat(sprintf("  - Saved: %s\n", filename))
  cat(sprintf("  - n=%d, n_pre=%d\n\n", dgp$n, dgp$n_pre))
}

cat("Done! Generated", nrow(dgps), "Extended DGP datasets.\n")
