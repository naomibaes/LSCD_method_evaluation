import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Load and filter data
data = pd.read_csv("output/baseline_final_combined.all-year.cds_mpnet.csv")
data['inj_ratio'] = data['inj_ratio'].replace('NA', 0)
data['inj_ratio'] = pd.to_numeric(data['inj_ratio'], errors='coerce').fillna(0).astype(int)
target_terms = ["trauma", "anxiety", "depression", "mental_health", "mental_illness", "abuse"]
data = data[data['term'].isin(target_terms)]
data = data.dropna(subset=['cosine_dissim_mean', 'inj_ratio', 'term'])

# Define the OLS model for ANOVA
model_anova = smf.ols("cosine_dissim_mean ~ C(inj_ratio)", data=data).fit()

# Perform ANOVA
anova_results = anova_lm(model_anova, typ=2)
print("ANOVA Results:")
print(anova_results)
