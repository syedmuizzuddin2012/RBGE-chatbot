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

# --- 1. Fit the GLMM model for Completeness, with random effect for Query ------

completeness_model_simple <- glmmTMB(
  Completeness ~ Chunk.Size + Overlap + kNN + (1 | Query),
  data = df,
  family = gaussian()
)

completeness_model <- glmmTMB(
  Completeness ~ Chunk.Size * Overlap * kNN + (1 | Query),
  data = df, family = gaussian()
)

anova(completeness_model_simple, completeness_model)

summary(completeness_model_simple)
summary(completeness_model)

# --- 2. Visualize raw data with ggplot boxplot, faceted by kNN --------
ggplot(df, aes(x = factor(Chunk.Size), y = Completeness, fill = factor(Overlap))) +
  geom_boxplot(outlier.shape = NA) +
  facet_wrap(~ kNN, ncol = 1) +
  labs(
    x = "Chunk Size",
    y = "Completeness Score",
    fill = "Overlap"
  ) +
  theme_minimal()

# --- 3. Simulate and plot residuals to check model fit (assumptions) ---------
simulation_residuals_simple <- DHARMa::simulateResiduals(completeness_model_simple)
plot(simulation_residuals_simple)

# --- 4. Get estimated marginal means for all combinations ---------
completeness_emm <- emmeans(completeness_model_simple, ~ Chunk.Size * Overlap * kNN) #use main effect model with emmeans instead of interaction because want to know individual effects added tgt

# View them
plot(completeness_emm)

# --- 5. Sort emmeans to find the best scoring combination ------
# View all combinations and sort
completeness_emm_summary <- summary(completeness_emm)
best_combo <- completeness_emm_summary[which.max(completeness_emm_summary$emmean), ]
best_combo

# Sort predicted completeness scores
summary(completeness_emm) %>% 
  arrange(desc(emmean))  # Highest predicted score first
# Step 1: Convert emmeans result to a dataframe
com_df <- as.data.frame(completeness_emm)

# Step 2: Create a readable label for each parameter combination
com_df$Combo <- with(com_df, paste("Chunk", Chunk.Size, "- Overlap", Overlap, "- kNN", kNN))

# Step 3: Sort Combo factor so they appear in order of descending score
com_df$Combo <- factor(com_df$Combo, levels = com_df$Combo[order(com_df$emmean)])

# Step 4: Plot using ggplot
ggplot(com_df, aes(x = emmean, y = Combo)) +
  geom_point(size = 3, color = "red") +
  geom_errorbarh(aes(xmin = lower.CL, xmax = upper.CL), height = 0.2, color = "#0072B2") +
  labs(
    x = "Expected Completeness Score",
    y = "Parameter Combination",
  ) +
  theme_minimal(base_size = 12)


# Let's say we found the best combo is the first one after sorting
# Now do pairwise comparisons: best vs all others

# --- 6. Compare the best combo to all others using "treatment vs control" ---
# This compares all other settings to the best one (reference = 1st row)

# Get pairwise comparisons
pairs(completeness_emm)

best_combo <- "Chunk.Size500 Overlap50 kNN1"  # The best combo from emmeans

# Filter to only comparisons involving the best combo
best_vs_rest <- summary(pairwise_comparisons) %>%
  dplyr::filter(grepl(best_combo, contrast))

# View the filtered results
best_vs_rest
