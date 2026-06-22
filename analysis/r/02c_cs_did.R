# 02c_cs_did.R
# Callaway & Sant'Anna (2021) staggered DiD estimator
# Uses did::att_gt() for group-time ATTs, did::aggte() for aggregation.

library(did)

skip_cs_did <- FALSE
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
            "data/gold/statistical/r_validation/r_cs_did_att.csv", row.names = FALSE)
  write.csv(data.frame(status = "not_applicable", note = message),
            "data/gold/statistical/r_validation/r_cs_did_event_study.csv", row.names = FALSE)
  jsonlite::write_json(
    list(method = "Callaway & Sant'Anna (2021)", status = "not_applicable", note = message),
    "data/gold/statistical/r_validation/r_cs_did_aggregate.json",
    pretty = TRUE, auto_unbox = TRUE
  )
  cat("CS DiD not applicable:", message, "\n")
  skip_cs_did <- TRUE
} else {
  message <- "not_implemented: validated firm-jurisdiction exposure is present, but CS DiD estimator is not wired to it yet"
  write.csv(data.frame(status = "not_implemented", note = message),
            "data/gold/statistical/r_validation/r_cs_did_att.csv", row.names = FALSE)
  write.csv(data.frame(status = "not_implemented", note = message),
            "data/gold/statistical/r_validation/r_cs_did_event_study.csv", row.names = FALSE)
  jsonlite::write_json(
    list(method = "Callaway & Sant'Anna (2021)", status = "not_implemented", note = message),
    "data/gold/statistical/r_validation/r_cs_did_aggregate.json",
    pretty = TRUE, auto_unbox = TRUE
  )
  cat("CS DiD not implemented for validated exposure:", message, "\n")
  skip_cs_did <- TRUE
}

if (!skip_cs_did) {
market <- tryCatch(read.csv("data/silver/fact_market_monthly.csv"), error = function(e) NULL)
if (is.null(market)) {
  cat("CS DiD: market data not available, skipping.\n")
  return(invisible(NULL))
}

market <- market[market$trading_days >= 10, ]
market$Year <- as.integer(substr(market$Month, 1, 4))
market$MonthNum <- as.integer(factor(market$Month, levels = sort(unique(market$Month))))

# Build cohort variable: first year a firm is treated
cohort_map <- tapply(market$Year[market$pillar_two_in_scope_proxy == 1],
                     market$Ticker[market$pillar_two_in_scope_proxy == 1], min)
cohort_map[setdiff(unique(market$Ticker), names(cohort_map))] <- 0
market$first_treat <- as.integer(cohort_map[market$Ticker])

# CS DiD estimation
cs_att <- tryCatch(
  att_gt(yname = "abnormal_return",
         tname = "Year",
         idname = "Ticker",
         gname = "first_treat",
         data = market,
         control_group = "notyettreated",
         allow_unbalanced_panel = TRUE,
         base_period = "universal"),
  error = function(e) {
    cat("CS DiD att_gt failed:", conditionMessage(e), "\n")
    NULL
  }
)

if (!is.null(cs_att)) {
  # Simple aggregation
  cs_agg <- aggte(cs_att, type = "simple")
  write.csv(
    data.frame(estimate = cs_agg$overall.att, std_error = cs_agg$overall.se,
               ci_low = cs_agg$overall.att - 1.96 * cs_agg$overall.se,
               ci_high = cs_agg$overall.att + 1.96 * cs_agg$overall.se),
    "data/gold/statistical/r_validation/r_cs_did_att.csv", row.names = FALSE
  )

  # Dynamic event study
  cs_dyn <- aggte(cs_att, type = "dynamic")
  write.csv(
    data.frame(event_time = cs_dyn$egt, estimate = cs_dyn$att.egt,
               std_error = cs_dyn$se.egt),
    "data/gold/statistical/r_validation/r_cs_did_event_study.csv", row.names = FALSE
  )

  # Aggregate summary JSON
  jsonlite::write_json(
    list(method = "Callaway & Sant'Anna (2021)",
         estimate = unname(cs_agg$overall.att),
         std_error = unname(cs_agg$overall.se),
         p_value = unname(2 * pnorm(abs(cs_agg$overall.att / cs_agg$overall.se), lower.tail = FALSE)),
         nobs = nrow(market)),
    "data/gold/statistical/r_validation/r_cs_did_aggregate.json",
    pretty = TRUE, auto_unbox = TRUE
  )

  cat("CS DiD complete. ATT:", round(cs_agg$overall.att, 6),
      "SE:", round(cs_agg$overall.se, 6), "\n")
} else {
  # Write empty placeholder to avoid downstream failures
  cat("CS DiD: writing placeholder outputs.\n")
  write.csv(data.frame(status = "not_applicable", note = "not_applicable: CS DiD estimation failed"),
            "data/gold/statistical/r_validation/r_cs_did_att.csv", row.names = FALSE)
  write.csv(data.frame(status = "not_applicable", note = "not_applicable: CS DiD event study estimation failed"),
            "data/gold/statistical/r_validation/r_cs_did_event_study.csv", row.names = FALSE)
  jsonlite::write_json(
    list(method = "Callaway & Sant'Anna (2021)", status = "not_applicable"),
    "data/gold/statistical/r_validation/r_cs_did_aggregate.json",
    pretty = TRUE, auto_unbox = TRUE
  )
}
}
