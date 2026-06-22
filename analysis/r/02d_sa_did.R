# 02d_sa_did.R
# Sun & Abraham (2021) heterogeneity-robust event-study estimator
# Uses fixest::feols() + fixest::sunab() for dynamic treatment effects.

library(fixest)

skip_sa_did <- FALSE
firm_exposure_path <- "data/gold/analytical/fact_firm_jurisdiction_pillar_two_exposure.csv"
required_exposure_columns <- c("Ticker", "jurisdiction_code", "FiscalYear", "exposure_weight", "first_treat_year", "treated")
exposure_valid <- FALSE
if (file.exists(firm_exposure_path)) {
  exposure <- tryCatch(read.csv(firm_exposure_path), error = function(e) data.frame())
  missing_columns <- setdiff(required_exposure_columns, names(exposure))
  if (nrow(exposure) > 0 && length(missing_columns) == 0) {
    cohorts <- unique(na.omit(exposure$first_treat_year[exposure$treated == 1]))
    has_support <- any(exposure$treated == 0, na.rm = TRUE) ||
      any(exposure$first_treat_year > exposure$FiscalYear, na.rm = TRUE)
    weights_ok <- all(!is.na(exposure$exposure_weight)) &&
      all(exposure$exposure_weight >= 0 & exposure$exposure_weight <= 1)
    exposure_valid <- length(cohorts) >= 2 && has_support && weights_ok
  }
}
if (!exposure_valid) {
  message <- "not_applicable: missing or invalid firm-jurisdiction revenue/profit exposure for staggered cohorts"
  write.csv(data.frame(status = "not_applicable", note = message),
            "data/gold/statistical/r_validation/r_sa_did_event_study.csv",
            row.names = FALSE)
  cat("SA DiD not applicable:", message, "\n")
  skip_sa_did <- TRUE
} else {
  message <- "not_implemented: validated firm-jurisdiction exposure is present, but SA DiD estimator is not wired to it yet"
  write.csv(data.frame(status = "not_implemented", note = message),
            "data/gold/statistical/r_validation/r_sa_did_event_study.csv",
            row.names = FALSE)
  cat("SA DiD not implemented for validated exposure:", message, "\n")
  skip_sa_did <- TRUE
}

if (!skip_sa_did) {
market <- tryCatch(read.csv("data/silver/fact_market_monthly.csv"), error = function(e) NULL)
if (is.null(market)) {
  cat("SA DiD: market data not available, skipping.\n")
  return(invisible(NULL))
}

market <- market[market$trading_days >= 10, ]
market$Year <- as.integer(substr(market$Month, 1, 4))

# Build cohort variable
cohort_map <- tapply(market$Year[market$pillar_two_in_scope_proxy == 1],
                     market$Ticker[market$pillar_two_in_scope_proxy == 1], min)
cohort_map[setdiff(unique(market$Ticker), names(cohort_map))] <- 0
market$first_treat <- as.integer(cohort_map[market$Ticker])
# Set never-treated cohort to a value outside treatment range
market$first_treat[market$first_treat == 0] <- 10000

sa_result <- tryCatch({
  # Sun & Abraham requires the cohort variable to be a factor
  model <- feols(abnormal_return ~ sunab(first_treat, Year) | Ticker + Year,
                 data = market,
                 cluster = ~Ticker)
  model
}, error = function(e) {
  cat("SA DiD: estimation failed:", conditionMessage(e), "\n")
  cat("Note: SA DiD requires cohort timing variation. If all treated units are in the same cohort,\n")
  cat("or if there are no never-treated units, sunab() may fail. This is expected for this design.\n")
  NULL
})

if (!is.null(sa_result)) {
  coefs <- coef(sa_result)
  ses <- sqrt(diag(vcov(sa_result)))
  sa_coefs <- grep("^sunab\\(|^year::|^cohort::", names(coefs), value = TRUE)
  if (length(sa_coefs) > 0) {
    sa_df <- data.frame(
      term = sa_coefs,
      estimate = unname(coefs[sa_coefs]),
      std_error = unname(ses[sa_coefs])
    )
    # Extract event time from sunab term names
    times <- regmatches(sa_df$term, regexpr("-?\\d+", sa_df$term))
    sa_df$event_time <- as.integer(times)
    sa_df <- sa_df[order(sa_df$event_time), ]
    write.csv(sa_df, "data/gold/statistical/r_validation/r_sa_did_event_study.csv",
              row.names = FALSE)
    cat("SA DiD complete. Event-study coefficients saved.\n")
  } else {
    cat("SA DiD: no sunab coefficients extracted.\n")
  }
} else {
  cat("SA DiD: writing placeholder.\n")
  write.csv(data.frame(status = "not_applicable", note = "not_applicable: uniform treatment timing"),
            "data/gold/statistical/r_validation/r_sa_did_event_study.csv",
            row.names = FALSE)
}
}
