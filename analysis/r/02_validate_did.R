market <- read.csv("data/silver/fact_market_monthly.csv")
market <- market[market$trading_days >= 10, ]
market$time_index <- as.integer(factor(market$Month, levels = sort(unique(market$Month)))) - 1

cluster_vcov <- function(model, clusters) {
  X <- model.matrix(model)
  residual_values <- residuals(model)
  meat <- matrix(0, ncol(X), ncol(X))
  for (cluster in unique(clusters)) {
    index <- clusters == cluster
    score <- crossprod(X[index, , drop = FALSE], residual_values[index])
    meat <- meat + score %*% t(score)
  }
  bread <- MASS::ginv(crossprod(X))
  G <- length(unique(clusters))
  N <- nrow(X)
  K <- ncol(X)
  (G / (G - 1)) * ((N - 1) / (N - K)) * bread %*% meat %*% bread
}

extract_term <- function(model, vcov_matrix, term, label) {
  index <- which(names(coef(model)) == term)
  estimate <- unname(coef(model)[index])
  standard_error <- sqrt(vcov_matrix[index, index])
  p_value <- 2 * pnorm(abs(estimate / standard_error), lower.tail = FALSE)
  list(
    specification = label,
    estimate = estimate,
    std_error = standard_error,
    p_value = p_value,
    nobs = nobs(model),
    statistically_significant_5pct = p_value < 0.05
  )
}

fit_specification <- function(formula, data, label) {
  model <- lm(formula, data = data)
  vcov_matrix <- cluster_vcov(model, data$Ticker)
  extract_term(model, vcov_matrix, "DiD", label)
}

market_did <- fit_specification(
  abnormal_return ~ DiD + factor(Ticker) + factor(Month),
  market,
  "baseline_qqq"
)

did_robustness <- list(
  market_did,
  fit_specification(abnormal_return_spy ~ DiD + factor(Ticker) + factor(Month), market, "alternative_benchmark_spy"),
  fit_specification(abnormal_return_xlk ~ DiD + factor(Ticker) + factor(Month), market, "alternative_benchmark_xlk"),
  fit_specification(abnormal_return ~ DiD + pillar_two_in_scope_proxy:time_index + factor(Ticker) + factor(Month), market, "group_linear_trend")
)

firm <- read.csv("data/silver/fact_firm_financial_year.csv")
baseline <- firm[firm$FiscalYear == 2022, c("Ticker", "Firm_Size", "Leverage", "RD_Intensity", "ETR", "Intangible_Ratio")]
names(baseline)[-1] <- paste0("baseline_", names(baseline)[-1])
conditional <- merge(market, baseline, by = "Ticker")
conditional <- conditional[complete.cases(conditional), ]
did_robustness[[length(did_robustness) + 1]] <- fit_specification(
  abnormal_return ~ DiD + Post:baseline_Firm_Size + Post:baseline_Leverage +
    Post:baseline_RD_Intensity + Post:baseline_ETR + Post:baseline_Intangible_Ratio +
    factor(Ticker) + factor(Month),
  conditional,
  "baseline_covariate_post_interactions"
)

market$CalendarYear <- as.integer(substr(market$Month, 1, 4))
market$CalendarYearFactor <- relevel(factor(market$CalendarYear), ref = "2023")
dynamic_model <- lm(
  abnormal_return ~ pillar_two_in_scope_proxy * CalendarYearFactor + factor(Ticker) + factor(Month),
  data = market
)
dynamic_vcov <- cluster_vcov(dynamic_model, market$Ticker)
dynamic_terms <- grep("pillar_two_in_scope_proxy:CalendarYearFactor", names(coef(dynamic_model)), value = TRUE)
pre_terms <- dynamic_terms[grepl("2020|2021|2022", dynamic_terms)]
pre_indices <- match(pre_terms, names(coef(dynamic_model)))
pre_beta <- coef(dynamic_model)[pre_indices]
pre_cov <- dynamic_vcov[pre_indices, pre_indices, drop = FALSE]
wald_stat <- as.numeric(t(pre_beta) %*% MASS::ginv(pre_cov) %*% pre_beta)
pretrend_joint_p_value <- pchisq(wald_stat, df = length(pre_beta), lower.tail = FALSE)

did_dynamic <- data.frame(
  calendar_year = as.integer(sub(".*Factor", "", dynamic_terms)),
  estimate = coef(dynamic_model)[dynamic_terms],
  std_error = sqrt(diag(dynamic_vcov)[match(dynamic_terms, names(coef(dynamic_model)))]),
  reference_year = 2023
)
write.csv(did_dynamic, "data/gold/statistical/r_validation/r_did_dynamic_coefficients.csv", row.names = FALSE)
