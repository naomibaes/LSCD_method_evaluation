# Authors: Naomi Baes and ChatGPT

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, levene
import matplotlib.pyplot as plt

# Set display precision for floating-point numbers
pd.options.display.float_format = '{:.10f}'.format

# Load your data
# Rationale: Load the dataset containing variables of interest for mixed model analysis.
data = pd.read_csv("output/baseline_averaged_valence_index_all-year.csv")

# Ensure that injection_ratio is treated as a continuous variable
# Rationale: Treating `injection_ratio` as continuous is necessary for accurate modeling of its effects.
data['injection_ratio'] = data['injection_ratio'].astype(float)

# Convert `target` to a categorical variable
# Rationale: `target` is a grouping variable and should be treated categorically for mixed models.
data['target'] = data['target'].astype('category')

# Null model to assess random effects necessity
# Rationale: A null model with random intercepts for `target` is used to calculate the Intraclass Correlation Coefficient (ICC),
# which measures how much variance is explained by grouping.
null_model = smf.mixedlm("avg_valence_index_negative ~ 1", data, groups=data['target'], re_formula="1").fit()
print("Null Model Summary:")
print(null_model.summary())

# Calculate ICC for `target`
# Rationale: ICC indicates the proportion of variance explained by the random grouping structure.
icc_target = null_model.cov_re.iloc[0, 0] / (null_model.cov_re.iloc[0, 0] + null_model.scale)
print(f"ICC for 'target': {icc_target}")

# Simplified model with random intercepts only
# Rationale: This model assumes that only the intercept varies by `target` without random slopes or interaction terms.
simplified_model = smf.mixedlm(
    "avg_valence_index_negative ~ injection_ratio", 
    data, 
    groups=data['target'], 
    re_formula="1"
).fit()
print("Simplified Model with Random Intercepts Summary:")
print(simplified_model.summary())

# Report variance estimates
# Rationale: To understand the contribution of random effects, report variance components for the random intercepts.
random_effects_variance = simplified_model.cov_re.iloc[0, 0]
residual_variance = simplified_model.scale
print(f"Random Effects Variance (Intercepts): {random_effects_variance}")
print(f"Residual Variance: {residual_variance}")

# Random slopes model
# Rationale: Testing random slopes allows for assessing whether the response to `injection_ratio` varies across `target`.
random_slopes_model = smf.mixedlm(
    "avg_valence_index_negative ~ injection_ratio", 
    data, 
    groups=data['target'], 
    re_formula="~injection_ratio"
).fit()
print("Random Slopes Model Summary:")
print(random_slopes_model.summary())

# Compare models using AIC and BIC
# Rationale: AIC and BIC balance model fit and complexity, guiding the selection of the most appropriate model.
print("\nModel Comparisons:")
print(f"Simplified Model AIC: {simplified_model.aic}")
print(f"Simplified Model BIC: {simplified_model.bic}")
print(f"Random Slopes Model AIC: {random_slopes_model.aic}")
print(f"Random Slopes Model BIC: {random_slopes_model.bic}")

# Select the best model based on AIC/BIC
# Rationale: The model with the lower AIC/BIC is preferred as it indicates better balance between fit and complexity.
best_model = simplified_model if simplified_model.aic < random_slopes_model.aic else random_slopes_model
print("\nBest Model Selected Based on AIC/BIC:")
print(best_model.summary())

# Diagnostics for the best model
# Rationale: Residual diagnostics ensure the chosen model meets assumptions such as normality and homoscedasticity.

# Extract residuals and fitted values
residuals = best_model.resid
fitted = best_model.fittedvalues

# Residuals vs. Fitted Plot
# Rationale: To check for patterns in residuals that might indicate model misspecification or heteroscedasticity.
plt.scatter(fitted, residuals)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Plot (Best Model)')
plt.axhline(0, color='red', linestyle='--')
plt.show()

# Residuals by group
# Rationale: To visualize whether residuals are centered around zero across groups (`target`).
for group in data['target'].unique():
    group_residuals = residuals[data['target'] == group]
    plt.hist(group_residuals, alpha=0.5, label=f"{group}")
plt.legend()
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals by Group')
plt.show()

# Normal Q-Q plot for residuals
# Rationale: To assess the normality of residuals, a key assumption of mixed models.
sm.qqplot(residuals, line='s')
plt.title('Normal Q-Q plot for residuals (Best Model)')
plt.show()

# Shapiro-Wilk test for normality of residuals
# Rationale: Quantitative test to confirm residual normality.
shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk test for normality (Best Model): Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")

# Levene’s test for homogeneity of variances
# Rationale: To assess homoscedasticity, another assumption of mixed models.
levene_test = levene(data['avg_valence_index_negative'], fitted)
print(f"Levene’s test for homogeneity of variances (Best Model): Statistic={levene_test.statistic}, p-value={levene_test.pvalue}")
