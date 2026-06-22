design <- read.csv("data/gold/analytical/fact_exposure_score.csv")
stopifnot(all(design$FiscalYear[!is.na(design$FiscalYear)] <= 2022))
eligible_rows <- design$eligible_for_main_design %in% c(TRUE, "True", "TRUE")
stopifnot(all(design$four_years_observed[eligible_rows] == 4))

matching_covariates <- c("threshold_distance_log", "Firm_Size", "Leverage", "RD_Intensity", "ETR", "Intangible_Ratio")
baseline <- design[design$eligible_for_main_design %in% c(TRUE, "True", "TRUE"), ]
baseline <- baseline[complete.cases(baseline[matching_covariates]), ]
propensity_model <- glm(
  pillar_two_four_year_scope_proxy ~ threshold_distance_log + Firm_Size + Leverage +
    RD_Intensity + ETR + Intangible_Ratio,
  data = baseline,
  family = binomial()
)
baseline$propensity_score <- predict(propensity_model, type = "response")
baseline$overlap_weight <- ifelse(
  baseline$pillar_two_four_year_scope_proxy == 1,
  1 - baseline$propensity_score,
  baseline$propensity_score
)
weighted_market <- merge(market, baseline[c("Ticker", "overlap_weight")], by = "Ticker")
weighted_market <- weighted_market[weighted_market$overlap_weight > 0, ]
weighted_model <- lm(
  abnormal_return ~ DiD + factor(Ticker) + factor(Month),
  data = weighted_market,
  weights = overlap_weight
)
weighted_vcov <- cluster_vcov(weighted_model, weighted_market$Ticker)
weighted_did_validation <- extract_term(weighted_model, weighted_vcov, "DiD", "overlap_weighted_did")

cars <- read.csv("data/gold/analytical/fact_event_firm_car.csv")
event_data <- merge(
  cars,
  design[c("Ticker", "pillar_two_exposure_intensity", "threshold_distance_log", "Firm_Size", "Leverage")],
  by.x = "ticker",
  by.y = "Ticker"
)
exposure_event_validation <- do.call(
  rbind,
  lapply(split(event_data, list(event_data$event_id, event_data$window), drop = TRUE), function(group) {
    group <- group[complete.cases(group[c("car", "pillar_two_exposure_intensity", "threshold_distance_log", "Firm_Size", "Leverage")]), ]
    if (nrow(group) < 20 || length(unique(group$pillar_two_exposure_intensity)) < 3) return(NULL)
    model <- lm(car ~ pillar_two_exposure_intensity + threshold_distance_log + Firm_Size + Leverage, data = group)
    vcov_hc3 <- sandwich::vcovHC(model, type = "HC3")
    index <- which(names(coef(model)) == "pillar_two_exposure_intensity")
    estimate <- unname(coef(model)[index])
    standard_error <- sqrt(vcov_hc3[index, index])
    data.frame(
      event_id = group$event_id[1],
      window = group$window[1],
      estimate = estimate,
      std_error = standard_error,
      p_value = 2 * pnorm(abs(estimate / standard_error), lower.tail = FALSE),
      nobs = nrow(group)
    )
  })
)
write.csv(exposure_event_validation, "data/gold/statistical/r_validation/r_exposure_event_study.csv", row.names = FALSE)
