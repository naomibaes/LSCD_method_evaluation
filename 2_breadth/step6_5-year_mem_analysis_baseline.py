import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, shapiro, levene
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("output/baseline_final_combined.5-year.cds_mpnet.csv")
data['inj_ratio'] = data['inj_ratio'].astype('category')

# Initial Model to assess random effects necessity
null_model = smf.mixedlm("cosine_dissim_mean ~ 1", data, groups=data['term'], re_formula="1").fit()
print("Null Model Summary:")
print(null_model.summary())

# Calculate ICC for term
icc_term = null_model.cov_re.iloc[0, 0] / (null_model.cov_re.iloc[0, 0] + null_model.scale)
print(f"ICC for 'term': {icc_term}")

# Full model with injection ratio
full_model = smf.mixedlm("cosine_dissim_mean ~ inj_ratio", data, groups=data['term'], re_formula="1").fit()
print("Full Model Summary:")
print(full_model.summary())

# Model comparison with and without 'epoch'
model_with_epoch = smf.mixedlm("cosine_dissim_mean ~ inj_ratio * C(epoch)", data, groups=data['term'], re_formula="1").fit()
print("Model with Epoch Summary:")
print(model_with_epoch.summary())

# Manually perform Likelihood Ratio Test
lr_stat = -2 * (full_model.llf - model_with_epoch.llf)
p_value = chi2.sf(lr_stat, df=1)  # df=1 should be adjusted based on the degrees of freedom difference between models
print(f"Likelihood Ratio Test Statistic: {lr_stat}, p-value: {p_value}")

# Diagnostic checks
residuals = full_model.resid
fitted = full_model.fittedvalues

# Residuals vs. Fitted Plot
plt.scatter(fitted, residuals)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Plot')
plt.axhline(0, color='red', linestyle='--')
plt.show()

# Normal Q-Q plot for residuals
sm.qqplot(residuals, line='s')
plt.title('Normal Q-Q plot for residuals')
plt.show()

# Shapiro-Wilk test for normality of residuals
shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk test for normality: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")

# Levene’s test for homogeneity of variances
levene_test = levene(data['cosine_dissim_mean'], fitted)
print(f"Levene’s test for homogeneity of variances: Statistic={levene_test.statistic}, p-value={levene_test.pvalue}")

# Additional OLS for comparison
ols_model = smf.ols("cosine_dissim_mean ~ inj_ratio", data=data).fit()
print("OLS Model Summary:")
print(ols_model.summary())
