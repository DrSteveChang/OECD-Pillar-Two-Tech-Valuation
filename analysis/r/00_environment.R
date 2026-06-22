required <- c("jsonlite", "did", "Synth", "synthdid", "grf", "sandwich", "bacondecomp", "fixest", "gsynth", "HonestDiD", "did2s")
availability <- setNames(vapply(required, requireNamespace, logical(1), quietly = TRUE), required)
if (!all(availability)) {
  missing <- names(availability)[!availability]
  warning("Missing R packages: ", paste(missing, collapse = ", "),
          ". Some modern DiD / SCM methods will be skipped.")
}
dir.create("data/gold/statistical/r_validation", recursive = TRUE, showWarnings = FALSE)
