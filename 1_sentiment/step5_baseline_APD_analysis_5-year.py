import pandas as pd
from itertools import combinations

# Load the data from the CSV file
try:
    df = pd.read_csv("output/baseline_averaged_valence_index_5-year.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    raise SystemExit  # Exit the script if the file is not found

# Ensure numeric types for injection_ratio
df['injection_ratio'] = pd.to_numeric(df['injection_ratio'], errors='coerce').fillna(0).astype(int)

# List of target terms
target_terms = df['target'].unique()

# Initialize the result dictionaries
results_positive = {
    "Term": [],
    "0": [],
    "20": [],
    "40": [],
    "60": [],
    "80": [],
    "100": []
}

results_negative = {
    "Term": [],
    "0": [],
    "20": [],
    "40": [],
    "60": [],
    "80": [],
    "100": []
}

# Loop through each target term to calculate APD grouped by injection_ratio
for term in target_terms:
    # Filter data for the current term
    term_data = df[df['target'] == term]

    # Group by injection_ratio for positive valence
    avg_valence_positive = {
        inj_ratio: term_data[term_data['injection_ratio'] == inj_ratio]['avg_valence_index_positive'].mean()
        for inj_ratio in sorted(term_data['injection_ratio'].unique())
    }

    # Group by injection_ratio for negative valence
    avg_valence_negative = {
        inj_ratio: term_data[term_data['injection_ratio'] == inj_ratio]['avg_valence_index_negative'].mean()
        for inj_ratio in sorted(term_data['injection_ratio'].unique())
    }

    # Calculate APD for each injection ratio for positive valence
    apd_injection_positive = {
        inj_ratio: avg_valence_positive.get(inj_ratio, 0)
        for inj_ratio in [0, 20, 40, 60, 80, 100]
    }

    # Calculate APD for each injection ratio for negative valence
    apd_injection_negative = {
        inj_ratio: avg_valence_negative.get(inj_ratio, 0)
        for inj_ratio in [0, 20, 40, 60, 80, 100]
    }

    # Append results for the current term (positive)
    results_positive["Term"].append(term)
    for inj_ratio in [0, 20, 40, 60, 80, 100]:
        results_positive[str(inj_ratio)].append(apd_injection_positive[inj_ratio])

    # Append results for the current term (negative)
    results_negative["Term"].append(term)
    for inj_ratio in [0, 20, 40, 60, 80, 100]:
        results_negative[str(inj_ratio)].append(apd_injection_negative[inj_ratio])

# Combine results into DataFrames
apd_df_positive = pd.DataFrame(results_positive)
apd_df_negative = pd.DataFrame(results_negative)

# Print the resulting DataFrames
print("Positive Valence APD:")
print(apd_df_positive)
print("\nNegative Valence APD:")
print(apd_df_negative)

# Save the results to CSV files
apd_df_positive.to_csv("output/baseline_average_pairwise_distances_positive_sentiment_5-year.csv", index=False)
apd_df_negative.to_csv("output/baseline_average_pairwise_distances_negative_sentiment_5-year.csv", index=False)
