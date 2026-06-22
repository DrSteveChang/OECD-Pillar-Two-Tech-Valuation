# 05_validate_heterogeneity.R
# Causal Forest heterogeneity analysis — Athey, Tibshirani & Wager (2019)
# Upgraded with Best Linear Projection (BLP) and Rank-Weighted ATE (RATE).

firm <- tryCatch(read.csv("data/silver/fact_firm_financial_year.csv"), error = function(e) NULL)
if (is.null(firm)) {
  cat("GRF: firm data not available, skipping.\n")
  return(invisible(NULL))
}

wide <- reshape(
  firm[c("Ticker", "FiscalYear", "Log_Revenue", "Leverage", "RD_Intensity", "ETR", "pillar_two_in_scope_proxy")],
  idvar = "Ticker",
  timevar = "FiscalYear",
  direction = "wide"
)
needed <- c("Log_Revenue.2022", "Log_Revenue.2025", "Leverage.2025",
            "RD_Intensity.2025", "ETR.2025", "pillar_two_in_scope_proxy.2025")
missing_needed <- setdiff(needed, names(wide))
if (length(missing_needed) > 0) {
  cat("GRF: required columns missing after reshape:", paste(missing_needed, collapse = ", "), "\n")
  write.csv(data.frame(note = "GRF heterogeneity not estimated; required columns missing."),
            "data/gold/statistical/r_validation/r_grf_blp.csv", row.names = FALSE)
  write.csv(data.frame(note = "GRF RATE not estimated; required columns missing."),
            "data/gold/statistical/r_validation/r_grf_rates.csv", row.names = FALSE)
  heterogeneity_result <- list(status = "not_estimated", reason = "required_columns_missing")
  return(invisible(NULL))
}
wide <- wide[complete.cases(wide[needed]), ]

heterogeneity_result <- list(status = "not_estimated")
grf_available <- requireNamespace("grf", quietly = TRUE)

if (grf_available && nrow(wide) >= 50) {
  library(grf)

  X <- as.matrix(wide[c("Leverage.2025", "RD_Intensity.2025", "ETR.2025")])
  Y <- wide$Log_Revenue.2025 - wide$Log_Revenue.2022
  W <- wide$pillar_two_in_scope_proxy.2025

  # Main causal forest
  forest <- causal_forest(X, Y, W, num.trees = 2000, seed = 42)

  # Average Treatment Effect on the Treated
  att <- average_treatment_effect(forest, target.sample = "treated")

  # Best Linear Projection — which covariates drive heterogeneity?
  blp_written <- tryCatch({
    blp <- best_linear_projection(forest)
    blp_df <- tryCatch(as.data.frame(blp), error = function(e) NULL)
    if (!is.null(blp_df) && nrow(blp_df) > 0) {
      blp_df$term <- rownames(blp_df)
      canonical <- c("term", "estimate", "std.err", "p.value")
      if (all(canonical %in% names(blp_df))) {
        write.csv(blp_df[canonical],
                  "data/gold/statistical/r_validation/r_grf_blp.csv", row.names = FALSE)
      } else {
        write.csv(blp_df[c("term", setdiff(names(blp_df), "term"))],
                  "data/gold/statistical/r_validation/r_grf_blp.csv", row.names = FALSE)
      }
    } else {
      write.csv(data.frame(note = "GRF BLP result could not be coerced to a data frame."),
                "data/gold/statistical/r_validation/r_grf_blp.csv", row.names = FALSE)
    }
    TRUE
  }, error = function(e) {
    cat("GRF BLP export failed:", conditionMessage(e), "\n")
    FALSE
  })
  if (!blp_written) {
    write.csv(data.frame(note = "GRF BLP not estimated."),
              "data/gold/statistical/r_validation/r_grf_blp.csv", row.names = FALSE)
  }

  # Rank-Weighted ATE — firm-level treatment effect ranking
  rate_written <- tryCatch({
    rates <- rank_average_treatment_effect(forest, target.sample = "treated")
    rates_df <- as.data.frame(rates)
    rates_df$rank <- seq_len(nrow(rates_df))
    write.csv(rates_df,
              "data/gold/statistical/r_validation/r_grf_rates.csv", row.names = FALSE)
    TRUE
  }, error = function(e) {
    cat("GRF RATE export failed:", conditionMessage(e), "\n")
    FALSE
  })
  if (!rate_written) {
    write.csv(data.frame(note = "GRF RATE not estimated."),
              "data/gold/statistical/r_validation/r_grf_rates.csv", row.names = FALSE)
  }

  # Calibration test
  calibration <- tryCatch(test_calibration(forest), error = function(e) NULL)
  calibration_mean_pval <- NA_real_
  calibration_diff_pval <- NA_real_
  if (!is.null(calibration) && nrow(calibration) >= 3 && ncol(calibration) >= 4) {
    calibration_mean_pval <- unname(calibration[2, 4])
    calibration_diff_pval <- unname(calibration[3, 4])
  }

  heterogeneity_result <- list(
    status = "exploratory_with_blp_and_rate",
    estimate = unname(att[1]),
    std_error = unname(att[2]),
    sample_size = nrow(wide),
    calibration_mean_pval = calibration_mean_pval,
    calibration_diff_pval = calibration_diff_pval
  )

  cat("GRF complete. ATT:", round(att[1], 6), "SE:", round(att[2], 6),
      "N:", nrow(wide), "\n")
} else {
  cat("GRF: insufficient data (n=", nrow(wide), ") or grf not available, skipping.\n", sep = "")
  # Write placeholder outputs
  write.csv(data.frame(note = "GRF heterogeneity not estimated."),
            "data/gold/statistical/r_validation/r_grf_blp.csv", row.names = FALSE)
  write.csv(data.frame(note = "GRF RATE not estimated."),
            "data/gold/statistical/r_validation/r_grf_rates.csv", row.names = FALSE)
}
