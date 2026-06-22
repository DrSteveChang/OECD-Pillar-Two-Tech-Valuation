cars <- read.csv("data/gold/analytical/fact_event_firm_car.csv")
event_summary_r <- do.call(
  rbind,
  lapply(split(cars, list(cars$event_id, cars$window), drop = TRUE), function(group) {
    treated <- group$car[group$pillar_two_in_scope_proxy == 1]
    control <- group$car[group$pillar_two_in_scope_proxy == 0]
    test <- t.test(treated, control)
    data.frame(
      event_id = group$event_id[1],
      window = group$window[1],
      difference = mean(treated) - mean(control),
      p_value = test$p.value,
      treated_n = length(treated),
      control_n = length(control)
    )
  })
)
event_summary_r$p_value_bh <- p.adjust(event_summary_r$p_value, method = "BH")
event_summary_r$p_value_holm <- p.adjust(event_summary_r$p_value, method = "holm")
write.csv(event_summary_r, "data/gold/statistical/r_validation/r_event_study_summary.csv", row.names = FALSE)
