market <- read.csv("data/silver/fact_market_monthly.csv")
pre_months <- unique(market$Month[market$Month < "2024-01"])
post_months <- unique(market$Month[market$Month >= "2024-01"])
if (length(pre_months) < 12 || length(post_months) < 12) {
  stop("Insufficient genuine pre/post periods for SCM validation")
}
scm_validation <- list(
  status = "input_conditions_passed",
  genuine_pre_months = length(pre_months),
  genuine_post_months = length(post_months),
  imputed_periods = 0,
  complete_case_only = TRUE
)

wide <- reshape(
  market[c("Ticker", "Month", "abnormal_return", "pillar_two_in_scope_proxy")],
  idvar = "Ticker",
  timevar = "Month",
  direction = "wide"
)
outcome_columns <- grep("^abnormal_return\\.", names(wide), value = TRUE)
wide <- wide[complete.cases(wide[outcome_columns]), ]
group_column <- grep("^pillar_two_in_scope_proxy\\.", names(wide), value = TRUE)[1]
wide <- wide[order(wide[[group_column]], wide$Ticker), ]
N0 <- sum(wide[[group_column]] == 0)
T0 <- sum(sub("^abnormal_return\\.", "", outcome_columns) < "2024-01")
if (N0 >= 10 && T0 >= 12 && nrow(wide) > N0) {
  Y <- as.matrix(wide[outcome_columns])
  estimate <- synthdid::synthdid_estimate(Y, N0, T0)
  scm_validation$sdid_estimate <- as.numeric(estimate)
  scm_validation$sdid_standard_error <- as.numeric(sqrt(vcov(estimate, method = "jackknife")))
  scm_validation$balanced_firms <- nrow(wide)
}
