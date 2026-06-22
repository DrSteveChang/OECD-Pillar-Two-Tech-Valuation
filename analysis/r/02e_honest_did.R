# 02e_honest_did.R
# HonestDiD sensitivity analysis — Rambachan & Roth (2023)
# Quantifies how robust the DiD conclusions are to violations of parallel trends.
# Requires did2s and HonestDiD packages.

library(did2s)
library(HonestDiD)

market <- tryCatch(read.csv("data/silver/fact_market_monthly.csv"), error = function(e) NULL)
if (is.null(market)) {
  cat("HonestDiD: market data not available, skipping.\n")
  return(invisible(NULL))
}

market <- market[market$trading_days >= 10, ]
market$Year <- as.integer(substr(market$Month, 1, 4))
market$treat <- market$pillar_two_in_scope_proxy

# Estimate base DiD event study using did2s (de Chaisemartin & D'Haultfoeuille style)
honest_es <- tryCatch({
  did2s(
    data = market,
    yname = "abnormal_return",
    first_stage = ~ 0 | Ticker + Year,
    second_stage = ~ i(Year, ref = 2023),
    treatment = "treat",
    cluster_var = "Ticker"
  )
}, error = function(e) {
  cat("HonestDiD: did2s estimation failed:", conditionMessage(e), "\n")
  NULL
})

if (!is.null(honest_es)) {
  coefs <- coef(honest_es)
  sigma <- vcov(honest_es)

  # Get pre-treatment coefficients for sensitivity analysis
  pre_terms <- grep("^Year::20(20|21|22)", names(coefs), value = TRUE)

  if (length(pre_terms) >= 2) {
    sens_result <- tryCatch({
      beta <- coefs[pre_terms]
      sigma_pre <- sigma[pre_terms, pre_terms]
      num_pre <- length(pre_terms)
      createSensitivityResults(
        betahat = beta,
        sigma = sigma_pre,
        numPrePeriods = num_pre,
        Mbarvec = c(0.5, 1, 1.5, 2)
      )
    }, error = function(e) {
      cat("HonestDiD sensitivity failed:", conditionMessage(e), "\n")
      NULL
    })

    if (!is.null(sens_result)) {
      sens_df <- as.data.frame(sens_result)
      write.csv(sens_df,
                "data/gold/statistical/r_validation/r_honest_did_sensitivity.csv",
                row.names = FALSE)
      cat("HonestDiD sensitivity analysis complete.\n")
    } else {
      write.csv(
        data.frame(note = "HonestDiD sensitivity could not be computed."),
        "data/gold/statistical/r_validation/r_honest_did_sensitivity.csv",
        row.names = FALSE
      )
    }
  } else {
    cat("HonestDiD: insufficient pre-treatment periods (need >= 2).\n")
    write.csv(
      data.frame(note = "Insufficient pre-treatment periods for HonestDiD (< 2)."),
      "data/gold/statistical/r_validation/r_honest_did_sensitivity.csv",
      row.names = FALSE
    )
  }
} else {
  cat("HonestDiD: writing placeholder.\n")
  write.csv(data.frame(note = "HonestDiD analysis not run (did2s failed)."),
            "data/gold/statistical/r_validation/r_honest_did_sensitivity.csv",
            row.names = FALSE)
}
