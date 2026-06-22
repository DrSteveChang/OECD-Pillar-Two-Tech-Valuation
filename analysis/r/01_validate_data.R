firm <- read.csv("data/silver/fact_firm_financial_year.csv")
market <- read.csv("data/silver/fact_market_monthly.csv")

stopifnot(!any(duplicated(firm[c("Ticker", "FiscalYear")])))
stopifnot(!any(duplicated(market[c("Ticker", "Month")])))
stopifnot(max(firm$FiscalYear) < 2026)
stopifnot(all(firm$pillar_two_in_scope_proxy %in% c(0, 1)))
stopifnot(all(market$pillar_two_in_scope_proxy %in% c(0, 1)))
stopifnot(!any(firm$FiscalYear < 2021))

data_validation <- list(
  status = "passed",
  firm_year_rows = nrow(firm),
  firm_count = length(unique(firm$Ticker)),
  market_month_rows = nrow(market),
  duplicate_firm_years = 0,
  duplicate_firm_months = 0
)

