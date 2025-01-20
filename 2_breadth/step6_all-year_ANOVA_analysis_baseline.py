# Load necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene

# Load the data
data = pd.read_csv("output/baseline_final_combined.all-year.cds_mpnet.csv")

# Replace NA values in the inj_ratio column with 0
data['inj_ratio'] = data['inj_ratio'].fillna(0)

# Convert inj_ratio values to categorical labels
inj_ratio_labels = {
    0: 'zero',
    20: 'twenty',
    40: 'forty',
    60: 'sixty',
    80: 'eighty',
    100: 'hundred'
}
data['inj_ratio'] = data['inj_ratio'].replace(inj_ratio_labels)

# Convert cosine_dissim_mean to numeric
data['cosine_dissim_mean'] = pd.to_numeric(data['cosine_dissim_mean'], errors='coerce')

# Check for NaN values in key columns
print("Checking for NaN values before ANOVA:")
print(data.isna().sum())

# Filter data for target terms
target_terms = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]
data_filtered = data[data['term'].isin(target_terms)]

# Drop rows with NaN in relevant columns
data_filtered = data_filtered.dropna(subset=['cosine_dissim_mean', 'inj_ratio'])

# Check the filtered dataset
print("Filtered data shape:", data_filtered.shape)
print("Unique values in 'term':", data_filtered['term'].unique())
print("Unique values in 'inj_ratio':", data_filtered['inj_ratio'].unique())

# Check for zero variability in cosine_dissim_mean
if data_filtered['cosine_dissim_mean'].nunique() < 2:
    print("Warning: Not enough variability in 'cosine_dissim_mean' for ANOVA.")
else:
    # Convert term to categorical
    data_filtered['term'] = data_filtered['term'].astype('category')

    # Run repeated measures ANOVA
    anova_results = AnovaRM(data_filtered, 'cosine_dissim_mean', 'term', within=['inj_ratio']).fit()
    print(anova_results)

# Check the assumptions: Normality and Homogeneity of Variances
# Normality check for each term
for term in target_terms:
    group_data = data_filtered[data_filtered['term'] == term]['cosine_dissim_mean']
    stat, p_value = shapiro(group_data)
    print(f"Normality check for {term}: W-statistic={stat:.3f}, p-value={p_value:.3f}")

# Levene's test for homogeneity of variances across injection ratios
print("Levene's test for homogeneity of variances:")
stat, p = levene(*[data_filtered[data_filtered['inj_ratio'] == inj]['cosine_dissim_mean'] for inj in data_filtered['inj_ratio'].unique()])
print(f"Statistic: {stat}, p-value: {p}")

# Visualize the distribution of cosine_dissim_mean by inj_ratio, grouped by term
plt.figure(figsize=(12, 6))
sns.boxplot(x='inj_ratio', y='cosine_dissim_mean', hue='term', data=data_filtered)
plt.title('Distribution of cosine_dissim_mean by Injection Ratio, Grouped by Term')
plt.ylabel('cosine_dissim_mean')
plt.xlabel('Injection Ratio')
plt.legend(title='Term')
plt.show()
