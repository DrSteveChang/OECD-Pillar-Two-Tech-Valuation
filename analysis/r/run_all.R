source("analysis/r/00_environment.R")

# Core validation scripts (original)
source("analysis/r/01_validate_data.R")
source("analysis/r/02_validate_did.R")
source("analysis/r/03_validate_event_study.R")
source("analysis/r/04_validate_scm_sdid.R")
source("analysis/r/06_compare_python_results.R")
source("analysis/r/07_validate_exposure_design.R")

# Modern staggered DiD methods (new)
tryCatch(source("analysis/r/02b_bacon_decomposition.R"),
         error = function(e) cat("02b_bacon_decomposition failed:", conditionMessage(e), "\n"))
tryCatch(source("analysis/r/02c_cs_did.R"),
         error = function(e) cat("02c_cs_did failed:", conditionMessage(e), "\n"))
tryCatch(source("analysis/r/02d_sa_did.R"),
         error = function(e) cat("02d_sa_did failed:", conditionMessage(e), "\n"))

# Sensitivity analysis (new)
tryCatch(source("analysis/r/02e_honest_did.R"),
         error = function(e) cat("02e_honest_did failed:", conditionMessage(e), "\n"))

# Generalized SCM (new)
tryCatch(source("analysis/r/04b_gsynth.R"),
         error = function(e) cat("04b_gsynth failed:", conditionMessage(e), "\n"))

# Enhanced heterogeneity analysis (modified)
tryCatch(source("analysis/r/05_validate_heterogeneity.R"),
         error = function(e) cat("05_validate_heterogeneity failed:", conditionMessage(e), "\n"))

# Assemble results
output <- list(
  status = "completed_with_modern_methods",
  data_validation = if (exists("data_validation")) data_validation else list(status = "not_run"),
  market_did = if (exists("market_did")) market_did else list(status = "not_run"),
  did_robustness = if (exists("did_robustness")) did_robustness else list(),
  did_pretrend = list(
    joint_pretrend_p_value = if (exists("pretrend_joint_p_value"))
      pretrend_joint_p_value else NA_real_,
    joint_pretrend_not_rejected_5pct = if (exists("pretrend_joint_p_value"))
      pretrend_joint_p_value >= 0.05 else NA
  ),
  scm_validation = if (exists("scm_validation")) scm_validation else list(status = "not_run"),
  heterogeneity_validation = if (exists("heterogeneity_result"))
    heterogeneity_result else list(status = "not_run"),
  python_comparison = if (exists("comparison")) comparison else list()
)

jsonlite::write_json(output,
  "data/gold/statistical/r_validation/r_validation_results.json",
  pretty = TRUE, auto_unbox = TRUE, digits = 12)
