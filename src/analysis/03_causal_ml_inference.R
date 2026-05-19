# ==============================================================================
# Script: 03_causal_ml_inference.R
# Purpose: First-Difference Causal Forest (Rigorously Aggregated & Balanced)
# Path Config: Hardcoded absolute data workspace path
# ==============================================================================

library(grf)
library(tidyverse)

# Define absolute dataset directory
DATA_PATH <- "/Users/boyanzhang/Downloads/Project/OECD-Pillar-Two-Tech-Valuation/data/analytical_panel_dataset.csv"

# Load the analytical dataset
df <- read.csv(DATA_PATH)
colnames(df) <- trimws(colnames(df))

# Secure variance boundaries and generate log revenue transformation
df$Log_Revenue <- log(ifelse(df$TotalRevenue <= 0 | is.na(df$TotalRevenue), 1, df$TotalRevenue))

# ------------------------------------------------------------------------------
# STEP 1: STRICT FIRM-YEAR UNIQUE AGGREGATION (KILLS MANY-TO-MANY ERRS)
# ------------------------------------------------------------------------------
df_secure <- df %>%
  group_by(Ticker, Year) %>%
  summarise(
    Log_Revenue = mean(Log_Revenue, na.rm = TRUE),
    Leverage = mean(Leverage, na.rm = TRUE),
    RD_Intensity = mean(RD_Intensity, na.rm = TRUE),
    ETR = mean(ETR, na.rm = TRUE),
    Treatment_Group = max(Treatment_Group, na.rm = TRUE),
    .groups = 'drop'
  )

# ------------------------------------------------------------------------------
# STEP 2: CROSS-SECTIONAL STRATUM EXTRACTION & TIMELINE ALIGNMENT
# ------------------------------------------------------------------------------
df_2022 <- df_secure %>% 
  filter(Year == 2022) %>% 
  select(Ticker, Log_Revenue_2022 = Log_Revenue)

df_2025 <- df_secure %>% 
  filter(Year == 2025) %>% 
  select(Ticker, Log_Revenue_2025 = Log_Revenue, Leverage, RD_Intensity, ETR, Treatment_Group)

# This join is now guaranteed to be a strict 1:1 single-key mapping
df_ml <- inner_join(df_2025, df_2022, by = "Ticker") %>%
  mutate(Delta_Log_Revenue = Log_Revenue_2025 - Log_Revenue_2022) %>%
  drop_na(Delta_Log_Revenue, Leverage, RD_Intensity, ETR, Treatment_Group)

print(paste("Clean 1:1 cross-sectional stratum generated. Total unique firms:", nrow(df_ml)))

# ------------------------------------------------------------------------------
# STEP 3: MATRIX PREPARATION & CAUSAL FOREST COMPILATION
# ------------------------------------------------------------------------------
X <- as.matrix(df_ml[, c("Leverage", "RD_Intensity", "ETR")])
Y <- as.numeric(df_ml$Delta_Log_Revenue)
W <- as.numeric(df_ml$Treatment_Group)

# Train the Causal Forest with Honest splitting enabled
cf_model <- causal_forest(X, Y, W, num.trees = 2000, seed = 42)

# Test the calibration and asymptotic normality
calibration <- test_calibration(cf_model)
print("--- First-Difference Causal Forest Calibration Test ---")
print(calibration)

# Extract Average Treatment Effect on the Treated (ATT)
att_cf <- average_treatment_effect(cf_model, target.sample = "treated")
print("--- First-Difference Causal Forest ATT ---")
print(att_cf)

# Calculate non-parametric variable importance
var_imp <- variable_importance(cf_model)
rownames(var_imp) <- colnames(X)
print("--- Variable Importance Matrix ---")
print(var_imp)

# ------------------------------------------------------------------------------
# STEP 4: EXPORT TEXT REPORT
# ------------------------------------------------------------------------------
sink("Table_Causal_Machine_Learning_Output.txt")
cat("=== First-Difference Causal Forest Calibration Results ===\n")
print(calibration)
cat("\n=== Estimated Average Treatment Effect on Treated (ATT) ===\n")
print(att_cf)
cat("\n=== Non-parametric Variable Importance Matrix ===\n")
print(var_imp)
sink()

print("--- Script 03 Finished: Clean Causal ML Output exported ---")