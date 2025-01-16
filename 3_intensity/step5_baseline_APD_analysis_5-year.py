# Authors: Naomi Baes and Chat GPT

import pandas as pd
from itertools import combinations

# Load the data from the CSV file
try:
    df = pd.read_csv("output/baseline_averaged_arousal_index_5-year.csv")
    print("Successfully read data from CSV file.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    raise SystemExit  # Exit the script if the file is not found

# Ensure numeric types for injection_ratio
df['injection_ratio'] = pd.to_numeric(df['injection_ratio'], errors='coerce').fillna(0).astype(int)

# List of target terms
target_terms = df['target'].unique()

# Initialize the result dictionaries
results_high = {
    "Term": [],
    "0": [],
    "20": [],
    "40": [],
    "60": [],
    "80": [],
    "100": []
}

results_low = {
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

    # Group by injection_ratio for high Arousal
    avg_arousal_high = {
        inj_ratio: term_data[term_data['injection_ratio'] == inj_ratio]['avg_arousal_index_high'].mean()
        for inj_ratio in sorted(term_data['injection_ratio'].unique())
    }

    # Group by injection_ratio for low Arousal
    avg_arousal_low = {
        inj_ratio: term_data[term_data['injection_ratio'] == inj_ratio]['avg_arousal_index_low'].mean()
        for inj_ratio in sorted(term_data['injection_ratio'].unique())
    }

    # Calculate APD for each injection ratio for high Arousal
    apd_injection_high = {
        inj_ratio: avg_arousal_high.get(inj_ratio, 0)
        for inj_ratio in [0, 20, 40, 60, 80, 100]
    }

    # Calculate APD for each injection ratio for low Arousal
    apd_injection_low = {
        inj_ratio: avg_arousal_low.get(inj_ratio, 0)
        for inj_ratio in [0, 20, 40, 60, 80, 100]
    }

    # Append results for the current term (high)
    results_high["Term"].append(term)
    for inj_ratio in [0, 20, 40, 60, 80, 100]:
        results_high[str(inj_ratio)].append(apd_injection_high[inj_ratio])

    # Append results for the current term (low)
    results_low["Term"].append(term)
    for inj_ratio in [0, 20, 40, 60, 80, 100]:
        results_low[str(inj_ratio)].append(apd_injection_low[inj_ratio])

# Combine results into DataFrames
apd_df_high = pd.DataFrame(results_high)
apd_df_low = pd.DataFrame(results_low)

# Print the resulting DataFrames
print("High Arouasl APD:")
print(apd_df_high)
print("\nLow Arouasl APD:")
print(apd_df_low)

# Save the results to CSV files
apd_df_high.to_csv("output/baseline_average_pairwise_distances_high_intensity_5-year.csv", index=False)
apd_df_low.to_csv("output/baseline_average_pairwise_distances_low_intensity_5-year.csv", index=False)
