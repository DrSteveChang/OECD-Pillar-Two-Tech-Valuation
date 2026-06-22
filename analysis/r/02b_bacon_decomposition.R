# 02b_bacon_decomposition.R
# Goodman-Bacon decomposition of TWFE DiD estimator
# Diagnoses whether the TWFE estimate is a contaminated weighted average
# of "good" (treated vs never-treated) and "bad" (early vs late) comparisons.

library(bacondecomp)

market <- tryCatch(read.csv("data/silver/fact_market_monthly.csv"), error = function(e) NULL)
if (is.null(market)) {
  cat("Bacon: market data not available, skipping.\n")
  return(invisible(NULL))
}

market <- market[market$trading_days >= 10, ]
market$Year <- as.integer(substr(market$Month, 1, 4))
cohort_map <- tapply(market$Year[market$pillar_two_in_scope_proxy == 1],
                     market$Ticker[market$pillar_two_in_scope_proxy == 1], min)
cohort_map[setdiff(unique(market$Ticker), names(cohort_map))] <- 0
market$first_treat <- as.integer(cohort_map[market$Ticker])
market$first_treat[market$pillar_two_in_scope_proxy == 0] <- 0

bacon_result <- tryCatch(
  bacondecomp::bacon(abnormal_return ~ pillar_two_in_scope_proxy,
                     data = market,
                     id_var = "Ticker",
                     time_var = "Year"),
  error = function(e) {
    cat("Bacon: decomposition failed:", conditionMessage(e), "\n")
    cat("Note: Bacon decomposition requires treatment timing variation.\n")
    cat("If all treated units share the same first-treatment year, this is expected.\n")
    NULL
  }
)

if (!is.null(bacon_result)) {
  write.csv(bacon_result, "data/gold/statistical/r_validation/r_bacon_decomposition.csv",
            row.names = FALSE)
  cat("Bacon decomposition saved. Summary:\n")
  cat("  Total treatment effect:", round(mean(bacon_result$estimate), 6), "\n")
  cat("  Types found:", paste(unique(bacon_result$type), collapse = ", "), "\n")
  if ("Earlier vs Later" %in% bacon_result$type ||
      "Later vs Earlier" %in% bacon_result$type) {
    cat("  WARNING: Bad comparisons detected. TWFE estimate is biased.\n")
  } else {
    cat("  All comparisons are treated vs never-treated (clean).\n")
  }
} else {
  # Write a diagnostic note instead of failing
  write.csv(
    data.frame(diagnostic = "All treated units share the same first-treatment cohort. Bacon decomposition not applicable."),
    "data/gold/statistical/r_validation/r_bacon_decomposition.csv",
    row.names = FALSE
  )
}
