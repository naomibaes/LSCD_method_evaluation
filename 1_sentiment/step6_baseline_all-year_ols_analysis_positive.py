# Authors: Naomi Baes and ChatGPT

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, levene
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data = pd.read_csv("output/baseline_averaged_valence_index_all-year.csv")

# Ensure that injection_ratio is treated as continuous and target as categorical
data['injection_ratio'] = data['injection_ratio'].astype(float)
data['target'] = data['target'].astype('category')

# Get unique targets for separate models
targets = data['target'].unique()

# Loop through each target and fit a separate model
for target in targets:
    print(f"\nRunning model for target: {target}\n")
    
    # Subset data for the specific target
    subset_data = data[data['target'] == target]
    
    # Fit a simple linear regression model
    model = smf.ols("avg_valence_index_positive ~ injection_ratio", data=subset_data).fit()
    
    # Print the summary of the model
    print(f"Summary for target: {target}")
    print(model.summary())
    
    # Residual diagnostics
    residuals = model.resid
    fitted = model.fittedvalues
    
    # Residuals vs. Fitted Plot
    plt.scatter(fitted, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Fitted Plot for {target}')
    plt.show()
    
    # Q-Q Plot for residuals
    sm.qqplot(residuals, line='s')
    plt.title(f'Normal Q-Q plot for residuals ({target})')
    plt.show()
    
    # Shapiro-Wilk test for normality
    shapiro_test = shapiro(residuals)
    print(f"Shapiro-Wilk test for normality (target: {target}):")
    print(f"Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")
    
    # Levene's test for homoscedasticity
    # Compare variance of residuals with variance of fitted values
    levene_test = levene(subset_data['avg_valence_index_positive'], fitted)
    print(f"Levene's test for homogeneity of variances (target: {target}):")
    print(f"Statistic={levene_test.statistic}, p-value={levene_test.pvalue}")
    
    # Plot distribution of residuals
    sns.histplot(residuals, kde=True, bins=15, color="blue", alpha=0.7)
    plt.title(f"Residuals Distribution for {target}")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.show()

print("\nAnalysis complete for all targets!")
