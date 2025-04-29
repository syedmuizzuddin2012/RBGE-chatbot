rm(list=ls())

# a. Read CSV with correct settings ------
df <- read.csv(file.choose(), stringsAsFactors = FALSE, header = TRUE) #later numeric variables need to be as factors

# b. Load required packages ------
library(glmmTMB)
library(emmeans)
library(DHARMa)
library(ggplot2)
library(dplyr)

# c. Convert variables to categorical (factor) -------------------------------------------------------

df$Chunk.Size <- as.factor(df$Chunk.Size)
df$Overlap <- as.factor(df$Overlap)
df$kNN <- as.factor(df$kNN)

# --- 1. Fit the GLMM model for Accuracy, with random effect for Query ------

accuracy_model_simple <- glmmTMB(
  Accuracy ~ Chunk.Size + Overlap + kNN + (1 | Query),
  data = df,
  family = gaussian()
)

accuracy_model <- glmmTMB(
  Accuracy ~ Chunk.Size * Overlap * kNN + (1 | Query),
  data = df, family = gaussian()
)

anova(accuracy_model_simple, accuracy_model)

summary(accuracy_model_simple)
summary(accuracy_model)

# --- 2. Visualize raw data with ggplot boxplot, faceted by kNN --------
ggplot(df, aes(x = factor(Chunk.Size), y = Accuracy, fill = factor(Overlap))) +
  geom_boxplot(outlier.shape = NA) +
  facet_wrap(~ kNN, ncol = 1) +
  labs(
    title = "Accuracy Score Distribution",
    x = "Chunk Size",
    y = "Accuracy Score",
    fill = "Overlap"
  ) +
  theme_minimal()

# --- 3. Simulate and plot residuals to check model fit (assumptions) ---------
simulation_residuals_simple <- DHARMa::simulateResiduals(accuracy_model_simple)
plot(simulation_residuals_simple)

# --- 4. Get estimated marginal means for all combinations ---------
accuracy_emm <- emmeans(accuracy_model_simple, ~ Chunk.Size * Overlap * kNN)

# View them
plot(accuracy_emm)

# --- 5. Sort emmeans to find the best scoring combination ------
# View all combinations and sort
accuracy_emm_summary <- summary(accuracy_emm)
best_combo <- accuracy_emm_summary[which.max(accuracy_emm_summary$emmean), ]
best_combo


summary(accuracy_emm) %>% 
  arrange(desc(emmean)) 

# Step 1: Convert emmeans result to a dataframe
acc_df <- as.data.frame(accuracy_emm)

# Step 2: Create a readable label for each parameter combination
acc_df$Combo <- with(acc_df, paste("Chunk", Chunk.Size, "- Overlap", Overlap, "- kNN", kNN))

# Step 3: Sort Combo factor so they appear in order of descending score
acc_df$Combo <- factor(acc_df$Combo, levels = acc_df$Combo[order(acc_df$emmean)])

# Step 4: Plot using ggplot
ggplot(acc_df, aes(x = emmean, y = Combo)) +
  geom_point(size = 3, color = "red") +
  geom_errorbarh(aes(xmin = lower.CL, xmax = upper.CL), height = 0.2, color = "#0072B2") +
  labs(
    x = "Expected Accuracy Score",
    y = "Parameter Combination",
  ) +
  theme_minimal(base_size = 12)
# Let's say we found the best combo is the first one after sorting
# Now do pairwise comparisons: best vs all others

# --- 6. Compare the best combo to all others using "treatment vs control" ---
# This compares all other settings to the best one (reference = 1st row)

# Get pairwise comparisons
pairwise_comparisons <- pairs(accuracy_emm)

# Set the best combo label manually (or build it from best_combo object like before)
best_combo <- "Chunk.Size500 Overlap50 kNN1"

# Filter to only comparisons involving the best combo
best_vs_rest <- summary(pairwise_comparisons) %>%
  dplyr::filter(grepl(best_combo, contrast))

# View the filtered results
best_vs_rest
