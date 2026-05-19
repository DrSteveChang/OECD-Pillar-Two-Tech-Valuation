# ==============================================================================
# Script: 01_twfe_robust_diagnostics.R
# Purpose: Diagnose TWFE failure using Callaway & Sant'Anna (2021) Estimator
# ==============================================================================

library(did)
library(tidyverse)

# Load the analytical dataset
# Note: Adjust the file path based on your working directory
df <- read.csv("/Users/boyanzhang/Downloads/Project/OECD-Pillar-Two-Tech-Valuation/data/analytical_panel_dataset.csv")

# Sanitize column names by stripping any hidden leading/trailing whitespaces
colnames(df) <- trimws(colnames(df))

# ------------------------------------------------------------------------------
# CRITICAL DATA PREP: Generate the exact columns required by the model
# ------------------------------------------------------------------------------
# 1. Create the logarithmic dependent variable
df$Log_Revenue <- log(df$TotalRevenue)

# 2. Convert company Ticker to a numeric ID (mandatory for the 'did' package)
df$id_numeric <- as.numeric(as.factor(df$Ticker))

# 3. Define the first year of treatment (Assuming FY2023. Control group is set to 0)
df$first_treat_year <- ifelse(df$Treatment_Group == 1, 2023, 0)

# ------------------------------------------------------------------------------
# MODEL ESTIMATION: Callaway & Sant'Anna (CS) Robust DiD
# ------------------------------------------------------------------------------
# Run robust estimation without covariates to avoid singularity/multicollinearity
cs_results <- att_gt(
  yname = "Log_Revenue",
  tname = "Year",
  idname = "id_numeric",
  gname = "first_treat_year",
  xformla = ~ 1,                        # Removed covariates to resolve GLM non-convergence
  data = df,
  control_group = "nevertreated",
  allow_unbalanced_panel = TRUE,        # Allow unbalanced panel to prevent dropping observations
  est_method = "dr"                     # Doubly Robust estimation
)

# Output aggregated event study results and dynamic treatment effects
# na.rm = TRUE ensures that NA values from unbalanced periods are ignored safely
es_cs <- aggte(cs_results, type = "dynamic", na.rm = TRUE)
summary(es_cs)

# ------------------------------------------------------------------------------
# VISUALIZATION & PHYSICAL EXPORT
# ------------------------------------------------------------------------------
# 1. Assign the generated plot object to variable 'p'
p <- ggdid(es_cs)

# 2. Export the plot as a high-resolution PDF (Recommended for lossless Overleaf integration)
ggsave("Figure_CS_Event_Study.pdf", plot = p, width = 8, height = 6, dpi = 300)

# 3. Export a PNG version for quick local preview
ggsave("Figure_CS_Event_Study.png", plot = p, width = 8, height = 6, dpi = 300)

print("--- Output Success ---")
print("Event study plots successfully exported as PDF and PNG to the working directory.")

# ------------------------------------------------------------------------------
# TEXT REPORT EXPORT: CS Dynamic Effects Summary
# ------------------------------------------------------------------------------
# Capture the console output of the event study summary and write to a .txt file
capture.output(
  summary(es_cs), 
  file = "Table_CS_Dynamic_Effects_Summary.txt"
)

print("Created Table_CS_Dynamic_Effects_Summary.txt")