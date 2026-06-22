python <- jsonlite::fromJSON("data/gold/statistical/python/python_model_results.json")
python_did <- python$market_did
python_specs <- read.csv("data/gold/statistical/python/did_robustness_specifications.csv")
r_specs <- do.call(rbind, lapply(did_robustness, function(item) as.data.frame(item, stringsAsFactors = FALSE)))
spec_comparison <- merge(
  python_specs[c("specification", "estimate", "std_error", "p_value", "nobs")],
  r_specs[c("specification", "estimate", "std_error", "p_value", "nobs")],
  by = "specification",
  suffixes = c("_python", "_r")
)
python_events <- read.csv("data/gold/statistical/python/event_study_summary.csv")
event_comparison <- merge(
  python_events[c("event_id", "window", "difference", "p_value_bh", "p_value_holm")],
  event_summary_r[c("event_id", "window", "difference", "p_value_bh", "p_value_holm")],
  by = c("event_id", "window"),
  suffixes = c("_python", "_r")
)
comparison <- list(
  market_did_estimate_difference = abs(python_did$estimate - market_did$estimate),
  market_did_standard_error_difference = abs(python_did$std_error - market_did$std_error),
  market_did_p_value_difference = abs(python_did$p_value - market_did$p_value),
  market_did_nobs_match = python_did$nobs == market_did$nobs,
  pretrend_p_value_difference = abs(
    python_did$assumption_diagnostics$parallel_trends$joint_pretrend_p_value - pretrend_joint_p_value
  ),
  robustness_max_estimate_difference = max(abs(spec_comparison$estimate_python - spec_comparison$estimate_r)),
  robustness_max_standard_error_difference = max(abs(spec_comparison$std_error_python - spec_comparison$std_error_r)),
  event_max_difference_difference = max(abs(event_comparison$difference_python - event_comparison$difference_r)),
  event_max_bh_p_value_difference = max(abs(event_comparison$p_value_bh_python - event_comparison$p_value_bh_r)),
  event_max_holm_p_value_difference = max(abs(event_comparison$p_value_holm_python - event_comparison$p_value_holm_r))
)
