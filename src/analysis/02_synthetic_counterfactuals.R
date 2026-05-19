# ==============================================================================
# Script: 02_synthetic_counterfactuals.R
# Purpose: Structurally Balanced SCM & Robust Synthetic DiD (SDiD) Pipeline
# Path Config: Hardcoded absolute data workspace path
# ==============================================================================

library(Synth)
library(synthdid)
library(tidyverse)

# Define absolute dataset directory
DATA_PATH <- "/Users/boyanzhang/Downloads/Project/OECD-Pillar-Two-Tech-Valuation/data/analytical_panel_dataset.csv"

# Load the analytical dataset
df <- read.csv(DATA_PATH)
colnames(df) <- trimws(colnames(df))

# ------------------------------------------------------------------------------
# STEP 1: DEFINE CORE HORIZON & ENFORCE STRUCTURAL GRID COMPLETION
# ------------------------------------------------------------------------------
target_years <- 2018:2025
min_year <- min(target_years)
max_year <- max(target_years)

df_filtered <- df %>% 
  filter(Year %in% target_years) %>% 
  drop_na(Ticker, Year)

# Force a perfect rectangular grid (Every Ticker X Every Year)
df_grid <- df_filtered %>% 
  complete(Ticker, Year = target_years)

# ------------------------------------------------------------------------------
# STEP 2: LOCALIZED PANEL IMPUTATION & EXPLICIT AGGREGATION
# ------------------------------------------------------------------------------
df_balanced <- df_grid %>%
  group_by(Ticker) %>%
  fill(Treatment_Group, TotalRevenue, Leverage, RD_Intensity, ETR, .direction = "downup") %>%
  ungroup()

# Handle logarithmic conversion and secure variance boundaries
df_balanced$Log_Revenue <- log(ifelse(df_balanced$TotalRevenue <= 0 | is.na(df_balanced$TotalRevenue), 1, df_balanced$TotalRevenue))
df_balanced$DiD_Shock <- ifelse(df_balanced$Treatment_Group == 1 & df_balanced$Year >= 2023, 1, 0)

# ------------------------------------------------------------------------------
# PART A: NATIVE SYNTHETIC CONTROL METHOD (SCM) WITH PERMUTATION
# ------------------------------------------------------------------------------
df_agg <- df_balanced %>%
  mutate(Entity = ifelse(Treatment_Group == 1, "Treated_Agg", as.character(Ticker))) %>%
  group_by(Entity, Year) %>%
  summarise(
    Log_Revenue = mean(Log_Revenue, na.rm = TRUE),
    Leverage = mean(Leverage, na.rm = TRUE),
    RD_Intensity = mean(RD_Intensity, na.rm = TRUE),
    ETR = mean(ETR, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  as.data.frame()

df_agg$Entity_ID <- as.numeric(as.factor(df_agg$Entity))
entity_key <- df_agg %>% select(Entity, Entity_ID) %>% distinct()

target_id <- entity_key %>% filter(Entity == "Treated_Agg") %>% pull(Entity_ID)
control_ids <- entity_key %>% filter(Entity != "Treated_Agg") %>% pull(Entity_ID)

dataprep_main <- dataprep(
  foo = df_agg,
  predictors = c("Leverage", "RD_Intensity", "ETR"),
  predictors.op = "mean",
  time.predictors.prior = min_year:2022,
  dependent = "Log_Revenue",
  unit.variable = "Entity_ID",
  time.variable = "Year",
  treatment.identifier = target_id,
  controls.identifier = control_ids,
  time.plot = min_year:max_year,
  time.optimize.ssr = min_year:2022
)

synth_main <- synth(dataprep_main)
main_gaps <- dataprep_main$Y1plot - (dataprep_main$Y0plot %*% synth_main$solution.w)

main_df <- data.frame(
  Year = min_year:max_year,
  Gap = as.numeric(main_gaps),
  Type = "Treated",
  Group = "Treated_Agg"
)

# Placebo Loop with console suppression
placebo_list <- list()
for (p_id in control_ids) {
  p_controls <- control_ids[control_ids != p_id]
  p_name <- entity_key %>% filter(Entity_ID == p_id) %>% pull(Entity)
  tryCatch({
    dataprep_p <- dataprep(
      foo = df_agg,
      predictors = c("Leverage", "RD_Intensity", "ETR"),
      predictors.op = "mean",
      time.predictors.prior = min_year:2022,
      dependent = "Log_Revenue",
      unit.variable = "Entity_ID",
      time.variable = "Year",
      treatment.identifier = p_id,
      controls.identifier = p_controls,
      time.plot = min_year:max_year,
      time.optimize.ssr = min_year:2022
    )
    capture.output(synth_p <- synth(dataprep_p))
    p_gaps <- dataprep_p$Y1plot - (dataprep_p$Y0plot %*% synth_p$solution.w)
    placebo_list[[p_name]] <- data.frame(Year = min_year:max_year, Gap = as.numeric(p_gaps), Type = "Placebo", Group = p_name)
  }, error = function(e) { NULL })
}

plot_df <- bind_rows(main_df, bind_rows(placebo_list))

# P-Value Calculation
mspe_stats <- plot_df %>%
  group_by(Group, Type) %>%
  summarise(pre_mspe = mean(Gap[Year <= 2022]^2), post_mspe = mean(Gap[Year >= 2023]^2), .groups = 'drop') %>%
  mutate(ratio = post_mspe / pre_mspe)

treated_ratio <- mspe_stats %>% filter(Type == "Treated") %>% pull(ratio)
placebo_ratios <- mspe_stats %>% filter(Type == "Placebo") %>% pull(ratio)
exact_p_value <- mean(placebo_ratios >= treated_ratio)

print("--- SCM IN-SPACE PLACEBO INFERENCE ---")
print(paste("Empirical P-Value:", round(exact_p_value, 4)))
writeLines(paste("Empirical P-Value:", round(exact_p_value, 4)), con = "Table_SCM_Placebo_P_Value.txt")

# Fixed 'size' to 'linewidth' depreciation warning
p_scm <- ggplot(plot_df, aes(x = Year, y = Gap, group = Group)) +
  geom_line(data = filter(plot_df, Type == "Placebo"), aes(color = "Placebo Pool"), alpha = 0.3, linewidth = 0.5) +
  geom_line(data = filter(plot_df, Type == "Treated"), aes(color = "Treated Cohort"), linewidth = 1.2) +
  geom_vline(xintercept = 2022, linetype = "dashed", color = "red") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "black") +
  scale_color_manual(values = c("Placebo Pool" = "darkgrey", "Treated Cohort" = "#0072B2")) +
  theme_minimal() + labs(title = "SCM Placebo Effects (2018-2025 Window)", x = "Fiscal Year", y = "Log Revenue Gap")

ggsave("Figure_SCM_Placeboxes.pdf", plot = p_scm, width = 8, height = 6, dpi = 300)

# ------------------------------------------------------------------------------
# PART B: SYNTHETIC DIFFERENCE-IN-DIFFERENCES (SDiD)
# ------------------------------------------------------------------------------
# CRITICAL FIX: Explicitly aggregate at firm-year level to kill redundant mapping records
df_sdid_final <- df_balanced %>%
  group_by(Ticker, Year) %>%
  summarise(
    Log_Revenue = mean(Log_Revenue, na.rm = TRUE),
    DiD_Shock = max(DiD_Shock, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(Log_Revenue = replace_na(Log_Revenue, mean(Log_Revenue, na.rm = TRUE))) %>%
  as.data.frame()

# Setup SDiD matrix under strict mathematical rectangle verification
setup <- panel.matrices(df_sdid_final, 
                        unit = "Ticker", 
                        time = "Year", 
                        outcome = "Log_Revenue", 
                        treatment = "DiD_Shock")

# Run SDiD double shrinkage estimation
tau_sdid <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
se_sdid <- sqrt(vcov(tau_sdid, method = "jackknife"))

print("--- SYNTHETIC DIFFERENCE-IN-DIFFERENCES (SDiD) ---")
print(paste("SDiD Estimate (ATT):", round(as.numeric(tau_sdid), 4)))
print(paste("Jackknife Standard Error:", round(as.numeric(se_sdid), 4)))

sdid_report <- c(
  paste("SDiD ATT Estimate:", round(as.numeric(tau_sdid), 4)),
  paste("Jackknife Standard Error:", round(as.numeric(se_sdid), 4))
)
writeLines(sdid_report, con = "Table_SDiD_Results.txt")

pdf("Figure_SDiD_Parallel_Trends.pdf", width = 8, height = 6)
plot(tau_sdid)
dev.off()

print("--- Pipeline Execution Finished Successfully ---")