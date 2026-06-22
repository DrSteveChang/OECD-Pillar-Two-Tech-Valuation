# 04b_gsynth.R
# Generalized Synthetic Control Method — Xu (2017)
# Uses gsynth::gsynth() with interactive fixed effects for multiple treated units.
# Directly addresses the limitation of traditional SCM (single treated unit).

library(gsynth)

market <- tryCatch(read.csv("data/silver/fact_market_monthly.csv"), error = function(e) NULL)
if (is.null(market)) {
  cat("gsynth: market data not available, skipping.\n")
  return(invisible(NULL))
}

market <- market[market$trading_days >= 10, ]
market$MonthNum <- as.integer(factor(market$Month, levels = sort(unique(market$Month))))

# gsynth requires a balanced panel; filter to firms with complete data
panel_check <- table(market$Ticker, market$MonthNum)
complete_tickers <- rownames(panel_check)[rowSums(is.na(panel_check) | panel_check == 0) == 0]
market_balanced <- market[market$Ticker %in% complete_tickers, ]

if (length(complete_tickers) < 10) {
  cat("gsynth: insufficient complete-case firms (", length(complete_tickers),
      "), skipping.\n", sep = "")
  write.csv(data.frame(note = "Insufficient balanced panel firms for gsynth."),
            "data/gold/statistical/r_validation/r_gsynth_att.csv", row.names = FALSE)
  write.csv(data.frame(note = "Insufficient balanced panel firms for gsynth factors."),
            "data/gold/statistical/r_validation/r_gsynth_factors.csv", row.names = FALSE)
  return(invisible(NULL))
}

gsynth_result <- tryCatch(
  gsynth(
    abnormal_return ~ pillar_two_in_scope_proxy,
    data = market_balanced,
    index = c("Ticker", "MonthNum"),
    force = "two-way",
    CV = TRUE,
    r = c(0, 5),         # factor count range for CV
    EM = TRUE,
    se = TRUE,
    nboots = 200,
    parallel = FALSE,
    seed = 42
  ),
  error = function(e) {
    cat("gsynth: estimation failed:", conditionMessage(e), "\n")
    NULL
  }
)

if (!is.null(gsynth_result)) {
  # Extract ATT
  att <- gsynth_result$att
  att_df <- data.frame(
    time = seq_along(att) - 1 - gsynth_result$T0,
    att = att,
    ci_low = gsynth_result$att - 1.96 * gsynth_result$att.se,
    ci_high = gsynth_result$att + 1.96 * gsynth_result$att.se
  )
  write.csv(att_df, "data/gold/statistical/r_validation/r_gsynth_att.csv",
            row.names = FALSE)

  # Extract factor loadings
  if (!is.null(gsynth_result$factor)) {
    factors_df <- as.data.frame(gsynth_result$factor)
    factors_df$time <- seq_len(nrow(factors_df))
    write.csv(factors_df, "data/gold/statistical/r_validation/r_gsynth_factors.csv",
              row.names = FALSE)
  } else {
    write.csv(data.frame(note = "gsynth factor loadings not available."),
              "data/gold/statistical/r_validation/r_gsynth_factors.csv",
              row.names = FALSE)
  }

  cat("gsynth complete. ATT (average):", round(mean(att, na.rm = TRUE), 6),
      "Factors used:", gsynth_result$r.cv, "\n")
} else {
  cat("gsynth: writing placeholder.\n")
  write.csv(data.frame(note = "gsynth estimation failed."),
            "data/gold/statistical/r_validation/r_gsynth_att.csv", row.names = FALSE)
  write.csv(data.frame(note = "gsynth factor estimation failed."),
            "data/gold/statistical/r_validation/r_gsynth_factors.csv", row.names = FALSE)
}
