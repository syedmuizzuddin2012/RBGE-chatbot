#1. library
#2. turn to factor
#3. glmmTMB relevance, accuracy, completeness
#4. boxplot facet wrap (raw data)
#5. dharma 
#6. plot emmeans (predicted data)
#7. pairwise


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

# --- 2. Visualize raw data with ggplot boxplot, faceted by kNN --------
ggplot(df, aes(x = factor(Chunk.Size), y = Relevance, fill = factor(Overlap))) +
  geom_boxplot(outlier.shape = NA) +
  facet_wrap(~ kNN, ncol = 1) +
  labs(
    x = "Chunk Size",
    y = "Relevance Score",
    fill = "Overlap (%)"
  ) +
  theme_minimal()

# --- 1. Fit the GLMM model for Relevance, with random effect for Query ------

anova(relevance_model_simple,relevance_model)

relevance_model_simple <- glmmTMB(
  Relevance ~ Chunk.Size + Overlap + kNN + (1 | Query),
  data = df,
  family = gaussian()
)

relevance_model <- glmmTMB(
  Relevance ~ Chunk.Size * Overlap * kNN + (1 | Query),
  data = df, family = gaussian()
)

summary(relevance_model_simple)
summary(relevance_model)

# ------- 3. Simulate and plot residuals to check model fit (assumptions) ------
# put it in appaendix instead...
simulation_residuals <- DHARMa::simulateResiduals(relevance_model)
plot(simulation_residuals)

simulation_residuals_simple <- DHARMa::simulateResiduals(relevance_model_simple)
plot(simulation_residuals_simple)

# --- 4. Get estimated marginal means for all combinations ---------
relevance_emm <- emmeans(relevance_model_simple, ~ Chunk.Size * Overlap * kNN) #use main effect model with emmeans instead of interaction because want to know individual effects added tgt

# Step 1: Convert emmeans result to a dataframe
emm_df <- as.data.frame(relevance_emm)

# Step 2: Create a readable label for each parameter combination
emm_df$Combo <- with(emm_df, paste("Chunk", Chunk.Size, "- Overlap", Overlap, "- kNN", kNN))

# Step 3: Sort Combo factor so they appear in order of descending score
emm_df$Combo <- factor(emm_df$Combo, levels = emm_df$Combo[order(emm_df$emmean)])

# Step 4: Plot using ggplot
ggplot(emm_df, aes(x = emmean, y = Combo)) +
  geom_point(size = 3, color = "red") +
  geom_errorbarh(aes(xmin = lower.CL, xmax = upper.CL), height = 0.2, color = "#0072B2") +
  labs(
    x = "Expected Relevance Score",
    y = "Parameter Combination",
  theme_minimal(base_size = 12))


# --- 5. Sort emmeans to find the best scoring combination ------
# View all combinations and sort
relevance_emm_summary <- summary(relevance_emm)
best_combo <- relevance_emm_summary[which.max(relevance_emm_summary$emmean), ]
best_combo

# Sort predicted relevance scores
summary(relevance_emm) %>% 
  arrange(desc(emmean))  # Highest predicted score first

# Let's say we found the best combo is the first one after sorting
# Now do pairwise comparisons: best vs all others

# --- 6. Compare the best combo to all others using "treatment vs control" ---
# This compares all other settings to the best one (reference = 1st row)

# Get pairwise comparisons
pairs(relevance_emm)

best_combo <- "Chunk.Size500 Overlap50 kNN1"  # The best combo from emmeans

# Filter to only comparisons involving the best combo
best_vs_rest <- summary(pairwise_comparisons) %>%
  dplyr::filter(grepl(best_combo, contrast))

# View the filtered results
best_vs_rest

