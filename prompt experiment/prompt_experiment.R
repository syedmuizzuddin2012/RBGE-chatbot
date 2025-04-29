#----------------------------------------------------------------------------------------------------------

# Load necessary libraries
library(glmmTMB)
library(emmeans)
library(ggplot2)
library(dplyr)
library(readxl)
library(simr)
library(lme4)  

# Read data
fullpromptsscored <- read_excel(file.choose())

# Ensure factors are correctly set
fullpromptsscored$prompt_style <- as.factor(fullpromptsscored$prompt_style)
fullpromptsscored$question <- as.factor(fullpromptsscored$question)

# Fit GLMMs for each score with question as random effect
model_prompt_completeness_accuracy <- glmmTMB(
  completeness_accuracy ~ prompt_style + (1 | question),
  data = fullpromptsscored,
  family = gaussian()
)

model_prompt_relevance_clarity <- glmmTMB(
  relevance_clarity ~ prompt_style + (1 | question),
  data = fullpromptsscored,
  family = gaussian()
)

model_prompt_emotional_expression <- glmmTMB(
  emotional_expression ~ prompt_style + (1 | question),
  data = fullpromptsscored,
  family = gaussian()
)

model_prompt_engagement <- glmmTMB(
  engagement ~ prompt_style + (1 | question),
  data = fullpromptsscored,
  family = gaussian()
)

# Summary of models (optional)
summary(model_prompt_completeness_accuracy)
summary(model_prompt_relevance_clarity)
summary(model_prompt_emotional_expression)
summary(model_prompt_engagement)


# power effect -------------------------------------------------------

# lmer model (lme4)
lme4_completeness <- lmer(completeness_accuracy ~ prompt_style + (1 | question), data = fullpromptsscored)
lme4_relevance <- lmer(relevance_clarity ~ prompt_style + (1 | question), data = fullpromptsscored)
lme4_emotional_expression <- lmer(emotional_expression ~ prompt_style + (1 | question), data = fullpromptsscored)
lme4_engagement <- lmer(engagement ~ prompt_style + (1 | question), data = fullpromptsscored)

# Now, perform power simulations using the simr package
# Set up power analysis for each model:
power_lme4_completeness <- powerSim(lme4_completeness, nsim = 100)
print(power_lme4_completeness)

power_lme4_relevance <- powerSim(lme4_relevance, nsim = 100)
print(power_lme4_relevance)

power_lme4_emotional_expression <- powerSim(lme4_emotional_expression, nsim = 100)
print(power_lme4_emotional_expression)

power_lme4_engagement <- powerSim(lme4_engagement, nsim = 100)
print(power_lme4_engagement)

# If you want to see more details of the power simulation, you can use:
power_lme4_completeness$simulations
