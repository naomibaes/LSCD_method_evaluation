import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Set display precision for floating-point numbers
pd.options.display.float_format = '{:.10f}'.format

# Load your data
data = pd.read_csv("output/baseline_final_combined.all-year.cds_mpnet.csv")

# Convert `inj_ratio` to float and `term` to category
data['inj_ratio'] = data['inj_ratio'].astype(float)
data['term'] = data['term'].astype('category')

# Null model to assess random effects necessity
null_model = smf.mixedlm("cosine_dissim_mean ~ 1", data, groups=data['term'], re_formula="1").fit()
print("Null Model Summary:")
print(null_model.summary())

# Calculate ICC for `term`
icc_term = null_model.cov_re.iloc[0, 0] / (null_model.cov_re.iloc[0, 0] + null_model.scale)
print(f"ICC for 'term': {icc_term}")

# Simplified model with random intercepts only
simplified_model = smf.mixedlm(
    "cosine_dissim_mean ~ inj_ratio", 
    data, 
    groups=data['term'], 
    re_formula="1"
).fit()
print("Simplified Model with Random Intercepts Summary:")
print(simplified_model.summary())

# Random slopes model
random_slopes_model = smf.mixedlm(
    "cosine_dissim_mean ~ inj_ratio", 
    data, 
    groups=data['term'], 
    re_formula="~inj_ratio"
).fit()
print("Random Slopes Model Summary:")
print(random_slopes_model.summary())

# Compare models using AIC and BIC
print("\nModel Comparisons:")
print(f"Simplified Model AIC: {simplified_model.aic}")
print(f"Simplified Model BIC: {simplified_model.bic}")
print(f"Random Slopes Model AIC: {random_slopes_model.aic}")
print(f"Random Slopes Model BIC: {random_slopes_model.bic}")

# Select the best model based on AIC/BIC
best_model = simplified_model if simplified_model.aic < random_slopes_model.aic else random_slopes_model
print("\nBest Model Selected Based on AIC/BIC:")
print(best_model.summary())

# Plotting residuals to check model assumptions
residuals = best_model.resid
fitted = best_model.fittedvalues

# Residuals vs. Fitted Plot
plt.scatter(fitted, residuals)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Plot (Best Model)')
plt.axhline(0, color='red', linestyle='--')
plt.show()

# Normal Q-Q plot for residuals
sm.qqplot(residuals, line='s')
plt.title('Normal Q-Q plot for residuals (Best Model)')
plt.show()

# Shapiro-Wilk test for normality of residuals
from scipy.stats import shapiro
shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk test for normality (Best Model): Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")
